#include "MainComponent.h"
#include "../core/AudioIO.h"
#include "../core/PitchDetect.h"
#include "../core/ParamPriority.h"
#include <juce_data_structures/juce_data_structures.h>

#if JUCE_WINDOWS
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
 #endif
 #include <windows.h>
 #include <dbghelp.h>
 #include <cstring>
 #include <cstdio>
 #include <vector>
#endif

namespace vms {

namespace {

#if JUCE_WINDOWS
static std::atomic<bool> gSymInit { false };
static void ensureSym(HANDLE proc) {
    if (!gSymInit.exchange(true)) { SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME); SymInitialize(proc, nullptr, TRUE); }
}

// POD-only, SEH-guarded stack walk given a CONTEXT. MSVC forbids C++ objects needing
// unwinding inside __try, so this is intentionally plain C. StackWalk64 mutates *pctx.
static void walkContextRaw(CONTEXT* pctx, HANDLE hThread, HANDLE proc, char* out, size_t outSize) {
    out[0] = 0;
    __try {
        STACKFRAME64 frame; ZeroMemory(&frame, sizeof(frame));
        DWORD machine;
       #if defined(_M_X64)
        machine = IMAGE_FILE_MACHINE_AMD64;
        frame.AddrPC.Offset = pctx->Rip; frame.AddrFrame.Offset = pctx->Rbp; frame.AddrStack.Offset = pctx->Rsp;
       #else
        machine = IMAGE_FILE_MACHINE_I386;
        frame.AddrPC.Offset = pctx->Eip; frame.AddrFrame.Offset = pctx->Ebp; frame.AddrStack.Offset = pctx->Esp;
       #endif
        frame.AddrPC.Mode = frame.AddrFrame.Mode = frame.AddrStack.Mode = AddrModeFlat;
        size_t pos = 0;
        for (int i = 0; i < 48 && pos + 256 < outSize; ++i) {
            if (!StackWalk64(machine, proc, hThread, &frame, pctx, nullptr,
                             SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
            if (frame.AddrPC.Offset == 0) break;
            char modName[64] = "";
            IMAGEHLP_MODULE64 mod; ZeroMemory(&mod, sizeof(mod)); mod.SizeOfStruct = sizeof(mod);
            if (SymGetModuleInfo64(proc, frame.AddrPC.Offset, &mod)) strncpy_s(modName, mod.ModuleName, _TRUNCATE);
            char symBuf[sizeof(SYMBOL_INFO) + 512]; ZeroMemory(symBuf, sizeof(symBuf));
            auto* sym = reinterpret_cast<SYMBOL_INFO*>(symBuf);
            sym->SizeOfStruct = sizeof(SYMBOL_INFO); sym->MaxNameLen = 511;
            DWORD64 disp = 0; int n;
            if (SymFromAddr(proc, frame.AddrPC.Offset, &disp, sym))
                n = _snprintf_s(out + pos, outSize - pos, _TRUNCATE, "    #%d  %s!%s+%lld\n", i, modName, sym->Name, (long long) disp);
            else
                n = _snprintf_s(out + pos, outSize - pos, _TRUNCATE, "    #%d  %s!0x%llx\n", i, modName, (unsigned long long) frame.AddrPC.Offset);
            if (n <= 0) break;
            pos += (size_t) n;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        strncpy_s(out, outSize, "(stack walk faulted)\n", _TRUNCATE);
    }
}

// Suspend a (different) thread, snapshot its context, walk, resume. For the hang watchdog.
static void walkStackRaw(HANDLE hThread, HANDLE proc, char* out, size_t outSize) {
    out[0] = 0;
    if (SuspendThread(hThread) == (DWORD) -1) { strncpy_s(out, outSize, "(suspend failed)\n", _TRUNCATE); return; }
    CONTEXT ctx; ZeroMemory(&ctx, sizeof(ctx)); ctx.ContextFlags = CONTEXT_FULL;
    if (GetThreadContext(hThread, &ctx)) walkContextRaw(&ctx, hThread, proc, out, outSize);
    ResumeThread(hThread);
}

// Best-effort message-thread stack (hang watchdog). SymXxx aren't thread-safe; only the
// single watchdog thread calls this, so they stay serialised.
juce::String captureThreadStack(void* hThreadV) {
    if (hThreadV == nullptr) return "(no thread handle)";
    HANDLE proc = GetCurrentProcess();
    ensureSym(proc);
    std::vector<char> buf(16 * 1024);
    walkStackRaw((HANDLE) hThreadV, proc, buf.data(), buf.size());
    juce::String s = juce::String::fromUTF8(buf.data());
    return s.isEmpty() ? "(no frames)" : s;
}

// ---- Crash handler: log native crashes (plugin access violations etc.) ----
// A hard fault in plugin code can't be caught by C++ try/catch and terminates the process
// before any "renderFailed" recovery can run. This unhandled-exception filter writes the
// crash + the FAULTING thread's stack to the log, using only POD + raw file I/O (the heap
// may be damaged), so we always learn WHERE it died.
static char gLogPathUtf8[1024] = { 0 };
static std::atomic<bool> gLogPathSet { false };
static char gLastActionBuf[512] = { 0 };   // mirror of lastAction for the crash log
static char gCurrentPluginUtf8[1024] = { 0 };   // plugin path of the running search (for the crash marker)
static char gMarkerPathUtf8[1024] = { 0 };      // <logdir>\crashed_plugin.txt

static void appendRaw(const char* text) {
    if (!gLogPathSet.load()) return;
    FILE* f = nullptr;
    if (fopen_s(&f, gLogPathUtf8, "a") == 0 && f != nullptr) { fputs(text, f); fclose(f); }
}

// Write the crashing plugin's path to a marker file (POD/raw, heap may be dead). On the next
// launch this -- and ONLY this -- flags the plugin for safe mode. A force-kill or close mid-search
// does NOT reach the crash handler, so it never false-flags a healthy plugin (e.g. Synth1).
static void writeCrashMarker() {
    if (gMarkerPathUtf8[0] == 0 || gCurrentPluginUtf8[0] == 0) return;
    FILE* f = nullptr;
    if (fopen_s(&f, gMarkerPathUtf8, "w") == 0 && f != nullptr) { fputs(gCurrentPluginUtf8, f); fclose(f); }
}

static LONG WINAPI vmsUnhandledFilter(EXCEPTION_POINTERS* ep) {
    __try {
        if (ep != nullptr && ep->ExceptionRecord != nullptr && ep->ContextRecord != nullptr) {
            HANDLE proc = GetCurrentProcess();
            ensureSym(proc);
            static char stk[16 * 1024];
            CONTEXT c = *ep->ContextRecord;
            walkContextRaw(&c, GetCurrentThread(), proc, stk, sizeof(stk));
            char hdr[1024];
            _snprintf_s(hdr, _TRUNCATE,
                "\n*** CRASH: exception 0x%08lx at 0x%p ***\n    last action: %s\n    faulting-thread stack:\n",
                (unsigned long) ep->ExceptionRecord->ExceptionCode,
                ep->ExceptionRecord->ExceptionAddress, gLastActionBuf);
            appendRaw(hdr);
            appendRaw(stk);
            appendRaw("\n");
            writeCrashMarker();   // a REAL crash -> flag this plugin for safe mode next launch
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) { }
    return EXCEPTION_EXECUTE_HANDLER;   // we've logged; let the process terminate
}
#else
juce::String captureThreadStack(void*) { return "(stack capture: this platform only on Windows)"; }
#endif

// Persisted settings (last-used folders) so the file chooser doesn't reset.
juce::PropertiesFile& settings() {
    static juce::PropertiesFile::Options opts = [] {
        juce::PropertiesFile::Options o;
        o.applicationName = "vst_match_studio";
        o.filenameSuffix = ".props";
        o.folderName = "vst_match_studio";
        o.osxLibrarySubFolder = "Application Support";
        return o;
    }();
    static juce::PropertiesFile pf(opts);
    return pf;
}
juce::File startDir(const juce::String& key, const juce::File& fallback) {
    const juce::String saved = settings().getValue(key);
    if (saved.isNotEmpty() && juce::File(saved).isDirectory()) return juce::File(saved);
    return fallback;
}
void rememberDir(const juce::String& key, const juce::File& f) {
    settings().setValue(key, f.getParentDirectory().getFullPathName());
    settings().saveIfNeeded();
}

// VST2 (.dll/.vst) vs VST3 (.vst3). VST2 plugins often crash with many live instances.
bool path_isVst2(const juce::String& p) {
    return p.endsWithIgnoreCase(".dll") || p.endsWithIgnoreCase(".vst");
}

// Per-plugin "safe mode" persistence (a plugin gets flagged after it crashes a search).
juce::String vst2SafeKey(const juce::String& path) { return "vst2safe_" + juce::String(path.hashCode64()); }
bool pluginNeedsSafeMode(const juce::String& path) { return settings().getBoolValue(vst2SafeKey(path), false); }
void markPluginSafeMode(const juce::String& path) { settings().setValue(vst2SafeKey(path), true); settings().saveIfNeeded(); }

// Wait for a background worker to finish WITHOUT a bare join() on the message thread.
// The worker tears down hosted plugin instances, and VST teardown needs the message
// thread (MessageManagerLock). A plain join() would block the message thread while the
// worker waits on that lock -> deadlock (the "freeze on Stop"). Keep pumping the message
// loop so the lock can be granted; the worker's completion callAsync clears `busy`.
void pumpUntilClear(std::atomic<bool>& busy) {
    auto* mm = juce::MessageManager::getInstanceWithoutCreating();
    while (busy.load() && mm != nullptr)
        mm->runDispatchLoopUntil(15);
}
} // namespace

// ---- ParamListModel --------------------------------------------------------
int MainComponent::ParamListModel::getNumRows() {
    return owner.liveInstance ? owner.liveInstance->numParams() : 0;
}

void MainComponent::ParamListModel::paintListBoxItem(int row, juce::Graphics& g, int w, int h, bool selected) {
    if (!owner.liveInstance || row < 0 || row >= (int) owner.freeFlags.size()) return;
    if (selected) g.fillAll(juce::Colours::white.withAlpha(0.05f));
    const bool isFree = owner.freeFlags[(size_t) row] != 0;
    const juce::String box = isFree ? "[x] " : "[ ] ";
    const auto info = owner.liveInstance->paramInfo(row);
    const auto cat = categorizeParam(info.name);

    // Category colour: core bright, secondary normal, mod dim, ignore very dim.
    juce::Colour catCol = (cat == ParamCat::Core)      ? juce::Colour(0xffe0e0e0)
                        : (cat == ParamCat::Secondary) ? juce::Colour(0xffb0b0b0)
                        : (cat == ParamCat::Mod)       ? juce::Colour(0xff707070)
                                                       : juce::Colour(0xff505050);
    if (isFree) catCol = juce::Colour(0xff35d0ce);
    g.setColour(catCol);
    g.setFont(13.0f);
    g.drawText(box + "[" + juce::String(paramCatName(cat)) + "] " + juce::String(row) + ": " + info.name,
               6, 0, w - 8, h, juce::Justification::centredLeft);
}

void MainComponent::ParamListModel::listBoxItemClicked(int row, const juce::MouseEvent& e) {
    const int n = (int) owner.freeFlags.size();
    if (row < 0 || row >= n) return;

    if (e.mods.isShiftDown() && owner.lastClickedRow >= 0 && owner.lastClickedRow < n) {
        // Extend selection: set anchor..row to the ANCHOR's current state.
        // Anchor stays fixed so the range can be re-extended.
        const char val = owner.freeFlags[(size_t) owner.lastClickedRow];
        owner.setFreeRange(owner.lastClickedRow, row, val);
    } else {
        // Plain click: toggle this row and make it the new anchor.
        owner.freeFlags[(size_t) row] = owner.freeFlags[(size_t) row] ? 0 : 1;
        owner.paramList.repaintRow(row);
        owner.lastClickedRow = row;
    }
    owner.setStatus("Free params: " + juce::String(owner.countFree())
                    + " / " + juce::String((int) owner.freeFlags.size()));
    owner.pushLiveParams();
}

// ---- MainComponent ---------------------------------------------------------
MainComponent::MainComponent() {
    // Simplified dashboard: load, one big "find best match" (all params, runs until Stop),
    // stop/save, A/B transport, a live-match meter. (Expert per-param controls removed -
    // the matcher always uses ALL parameters.)
    for (auto* b : { &btnLoadVst, &btnLoadWav, &btnStart, &btnStop, &btnSave,
                     &btnA, &btnB, &btnPlay })
        addAndMakeVisible(b);
    btnStart.setButtonText("FIND BEST MATCH");
    btnStart.setTooltip("Search ALL of the synth's parameters for the closest match to the target. "
                        "Runs until you press Stop; the editor knobs animate to the best patch as it improves.");

    btnLoadVst.onClick = [this] { loadVst(); };
    btnLoadWav.onClick = [this] { loadWav(); };
    btnStart.onClick   = [this] { startAuto(); };
    btnStop.onClick    = [this] { stopMatch(); };
    btnSave.onClick    = [this] { saveState(); };

    btnA.onClick = [this] {
        player.setSource(LivePlayer::Source::A);
        debug.setText(player.bufferLenA() > 0 ? "audio: A (target) selected" : "audio: A empty - load a target WAV",
                      juce::dontSendNotification);
    };
    btnB.onClick = [this] {
        player.setSource(LivePlayer::Source::B);
        debug.setText(player.bufferLenB() > 0 ? "audio: B (best) selected" : "audio: B empty - run a match first",
                      juce::dontSendNotification);
    };
    btnPlay.onClick = [this] {
        if (!player.isPlaying()) {
            if (!player.start()) { debug.setText("audio: output failed: " + player.lastError(), juce::dontSendNotification); return; }
            const int len = (player.getSource() == LivePlayer::Source::A) ? player.bufferLenA()
                                                                          : player.bufferLenB();
            if (len == 0) { debug.setText("audio: nothing to play on this side (load WAV / run match)", juce::dontSendNotification); return; }
            player.setPlaying(true);
            btnPlay.setButtonText("Stop");
            debug.setText("audio: playing - " + player.deviceInfo(), juce::dontSendNotification);
        } else {
            player.setPlaying(false);
            btnPlay.setButtonText("Play");
            debug.setText("audio: stopped", juce::dontSendNotification);
        }
    };

    addAndMakeVisible(lblNote); addAndMakeVisible(edNote); edNote.setText("60");
    lblNote.setText("Note:", juce::dontSendNotification);
    addAndMakeVisible(lblWorkers); addAndMakeVisible(edWorkers);
    edWorkers.setText(juce::String(juce::jlimit(1, 24, juce::SystemStats::getNumCpus() - 2)));
    edWorkers.setInputRestrictions(3, "0123456789");
    edWorkers.setJustification(juce::Justification::centred);

    addAndMakeVisible(lblExclude); addAndMakeVisible(edExclude);
    edExclude.setText("MIDI CC", juce::dontSendNotification);   // skip external-MIDI map params by default
    edExclude.setTooltip("Parameters whose name matches this regular expression are frozen (not searched). "
                         "Default 'MIDI CC' skips external-MIDI-input maps that don't shape the synth's own sound. "
                         "Use | for alternatives, e.g.  MIDI CC|Macro|Random LFO");

    addAndMakeVisible(btnMeter);
    btnMeter.setTooltip("While on, renders YOUR current patch and shows how close it is to the target "
                        "(loss + 0-100 score), updating as you turn the plugin's knobs.");

    // Big dashboard readouts.
    addAndMakeVisible(lblBest);
    lblBest.setColour(juce::Label::textColourId, juce::Colours::white);
    lblBest.setFont(juce::Font(juce::FontOptions(22.0f)).boldened());
    lblBest.setText("BEST: -", juce::dontSendNotification);
    addAndMakeVisible(lblProgress);
    lblProgress.setColour(juce::Label::textColourId, juce::Colour(0xffb0b0b0));
    lblProgress.setFont(juce::FontOptions(13.0f));

    addAndMakeVisible(spectro);
    addAndMakeVisible(heatmap);
    addAndMakeVisible(lossCurve);

    addAndMakeVisible(status);
    status.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(debug);
    debug.setColour(juce::Label::textColourId, juce::Colour(0xff8a8a8a));
    debug.setFont(juce::FontOptions(12.0f));
    debug.setText("audio/debug: idle", juce::dontSendNotification);
    setStatus("1. Load Synth   2. Load Target WAV   3. Find Best Match");

    // ---- Hang diagnostics: logger + heartbeat + watchdog -----------------
    logger.reset(juce::FileLogger::createDefaultAppLogger(
        "vst_match_studio", "vst_match_studio.log",
        "=== VST Match Studio session start ===", 4 * 1024 * 1024));
   #if JUCE_WINDOWS
    // Duplicate the (current = message) thread's pseudo-handle into a real handle
    // the watchdog can use later. The ctor runs on the message thread.
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                    reinterpret_cast<HANDLE*>(&msgThreadHandle), 0, FALSE, DUPLICATE_SAME_ACCESS);
    // Install the native crash handler so plugin access-violations are logged with a stack.
    if (logger != nullptr) {
        strncpy_s(gLogPathUtf8, logger->getLogFile().getFullPathName().toRawUTF8(), _TRUNCATE);
        gLogPathSet = true;
        const juce::File marker = logger->getLogFile().getParentDirectory().getChildFile("crashed_plugin.txt");
        strncpy_s(gMarkerPathUtf8, marker.getFullPathName().toRawUTF8(), _TRUNCATE);
    }
    SetUnhandledExceptionFilter(vmsUnhandledFilter);
   #endif
    lastBeatMs = juce::Time::getMillisecondCounter();
    logAction("app start");

    // Crash self-healing: a plugin is flagged for safe mode ONLY if the crash handler actually
    // fired last session (real access-violation) and left a marker file. A force-kill or close
    // mid-search never reaches the handler, so healthy plugins (Synth1) are never false-flagged.
   #if JUCE_WINDOWS
    {
        const juce::File marker(juce::String::fromUTF8(gMarkerPathUtf8));
        if (marker.existsAsFile()) {
            const juce::String crashed = marker.loadFileAsString().trim();
            if (crashed.isNotEmpty()) {
                markPluginSafeMode(crashed);
                logAction("recovered from REAL crash during '" + crashed + "' -> safe mode enabled for it");
            }
            marker.deleteFile();
        }
    }
   #endif
    watchdogThread = std::thread([this] {
      try {
        while (!watchdogStop.load()) {
            for (int i = 0; i < 10 && !watchdogStop.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (watchdogStop.load()) break;
            const auto now = juce::Time::getMillisecondCounter();
            const auto last = lastBeatMs.load();
            const juce::uint32 gap = (now >= last) ? (now - last) : 0;
            if (gap > 5000) {
                if (!hangLogged.exchange(true)) {
                    // Write the header + context FIRST and flush, so even if the stack
                    // walk is slow or faults we still have the cause (last action + flags).
                    juce::String a; { std::lock_guard<std::mutex> lk(actionMx); a = lastAction; }
                    juce::String hdr;
                    hdr << "*** HANG DETECTED: message thread unresponsive for " << (int) gap << " ms ***\n"
                        << "    last action : " << a << "\n"
                        << "    busy flags  : auto=" << (int) autoBusy.load() << " staged=" << (int) stagedBusy.load()
                        << " meter=" << (int) meterBusy.load() << " matching=" << (int) runner.isMatching();
                    if (logger) logger->logMessage(hdr);
                    // Then attempt the (riskier) stack walk and log it separately.
                    juce::String stk = captureThreadStack(msgThreadHandle);
                    if (logger) logger->logMessage(juce::String("    message-thread stack:\n") + stk);
                }
            } else {
                hangLogged = false;   // recovered (pump kept the loop alive)
            }
        }
      } catch (...) {
        if (logger) logger->logMessage("[bg ] watchdog thread exception (ignored)");
      }
    });

    setSize(1280, 800);
    startTimerHz(8);
}

void MainComponent::logAction(const juce::String& a) {
    { std::lock_guard<std::mutex> lk(actionMx); lastAction = a; }
   #if JUCE_WINDOWS
    strncpy_s(gLastActionBuf, a.toRawUTF8(), _TRUNCATE);   // mirror for the crash handler
   #endif
    if (logger == nullptr) return;
    const bool onMsg = juce::MessageManager::getInstanceWithoutCreating() != nullptr
                       && juce::MessageManager::getInstance()->isThisTheMessageThread();
    logger->logMessage(juce::String(onMsg ? "[msg] " : "[bg ] ") + a);
}

void MainComponent::timerCallback() {
    lastBeatMs = juce::Time::getMillisecondCounter();   // heartbeat: proves the message thread is alive
    if (btnMeter.getToggleState()) updateLiveMeter();
    if (!player.isPlaying()) { return; }
    const float outPk = player.consumeOutPeak();
    debug.setText("audio: " + player.deviceInfo()
                  + "  cb=" + juce::String(player.callbackCount())
                  + "  srcPeak=" + juce::String(player.currentSrcPeak(), 3)
                  + "  outPeak=" + juce::String(outPk, 3),
                  juce::dontSendNotification);
}

MainComponent::~MainComponent() {
    logAction("~MainComponent: shutting down");
    watchdogStop = true;
    if (watchdogThread.joinable()) watchdogThread.join();
    runner.stopMatching();
    joinStaged();
    joinAuto();
    if (meterThread.joinable()) meterThread.join();
    player.stop();
    editorWindow.reset();   // destroy editor (owned by window) before the plugin instance
   #if JUCE_WINDOWS
    if (msgThreadHandle != nullptr) { CloseHandle((HANDLE) msgThreadHandle); msgThreadHandle = nullptr; }
   #endif
    logAction("~MainComponent: done");
}

void MainComponent::joinStaged() {
    logAction("joinStaged: requestStop + pump");
    stagedMatcher.requestStop();
    if (stagedThread.joinable()) {
        pumpUntilClear(stagedBusy);   // keep message loop alive so plugin teardown can proceed
        stagedThread.join();
    }
    stagedBusy = false;
    logAction("joinStaged: worker joined");
}

void MainComponent::joinAuto() {
    logAction("joinAuto: requestStop + pump");
    autoMatcher.requestStop();
    if (autoThread.joinable()) {
        pumpUntilClear(autoBusy);   // keep message loop alive so plugin teardown can proceed
        autoThread.join();
    }
    autoBusy = false;
   #if JUCE_WINDOWS
    gCurrentPluginUtf8[0] = 0;   // no search running -> a later crash won't be attributed to a plugin
   #endif
    logAction("joinAuto: worker joined");
}

// FIND BEST MATCH: full-parameter block-coordinate search over ALL params, until Stop.
void MainComponent::startAuto() {
    if (liveInstance == nullptr) { setStatus("Load a synth first."); return; }
    if (targetBuf.getNumSamples() == 0) { setStatus("Load a target WAV first."); return; }
    if (autoBusy.load()) { setStatus("Already running - press Stop first."); return; }
    joinAuto();

    AutoConfig cfg;
    cfg.sampleRate = targetSr;
    cfg.durSec = targetBuf.getNumSamples() / targetSr;
    cfg.baseNote = edNote.getText().getIntValue();
    cfg.numWorkers = juce::jlimit(1, 32, edWorkers.getText().getIntValue());
    cfg.maxPasses = 0;             // UNLIMITED - run until Stop
    cfg.searchPerformance = true;
    cfg.excludeRegex = edExclude.getText();   // freeze params matching this regex (default "MIDI CC")

    // VST2 plugins commonly crash (hard access violation) once too many instances coexist
    // in one process - TAL NoiseMaker dies at the ~8th. The search keeps numWorkers instances
    // alive at once, so cap VST2 to a safe count. VST3 (e.g. Vital) is unaffected.
    // Safe mode is per-plugin (works for VST2 AND VST3): only plugins that have crashed a search
    // before run on the message thread (single-threaded). Others stay fast/parallel (Synth1/Vital).
    juce::String workersNote;
    const bool isVst2 = path_isVst2(vst3Path);
    const bool safeMode = pluginNeedsSafeMode(vst3Path);
    autoSafeMode = safeMode;
    VstTarget::setForceMessageThread(safeMode);
    if (safeMode) {
        workersNote = "  (safe mode: single-threaded on message thread - this plugin crashed before)";
        cfg.numWorkers = 1;
    }
    // Tell the crash handler which plugin is running, so a REAL crash flags exactly this one.
   #if JUCE_WINDOWS
    strncpy_s(gCurrentPluginUtf8, vst3Path.toRawUTF8(), _TRUNCATE);
   #endif

    const juce::String path = vst3Path; auto target = targetBuf;
    const double sr = targetSr; const double durSec = cfg.durSec;

    // Safe mode only: CLOSE the plugin editor for the search (a VST2 editor open while the
    // plugin renders on the message thread is fine, but safe mode keeps the live instance
    // untouched anyway; closing avoids any editor idle churn). Recreated with the result.
    if (safeMode) editorWindow.reset();

    autoBusy = true; autoBestLoss = 1.0e30;
    matchStartMs = juce::Time::getMillisecondCounter();
    lossCurve.clear();
    { std::lock_guard<std::mutex> lk(liveMx); liveAudio.setSize(0, 0); liveParams.clear(); }  // drop stale best
    setStatus("Searching ALL parameters for the best match... press Stop when satisfied." + workersNote);
    lblBest.setText("BEST: searching...", juce::dontSendNotification);
    logAction("startAuto: begin  workers=" + juce::String(cfg.numWorkers)
              + " note=" + juce::String(cfg.baseNote) + " exclude='" + cfg.excludeRegex + "'"
              + (isVst2 ? " [VST2]" : " [VST3]"));

    autoMatcher.onProgress = [this](int pass, int bi, int bc, const juce::String& nm, double, double) {
        juce::MessageManager::callAsync([this, pass, bi, bc, nm] {
            const int secs = (int) juce::jmax(0.0, (juce::Time::getMillisecondCounter() - matchStartMs) / 1000.0);
            juce::String t;
            if (pass == 0) {
                // Pre-search phases (find note/vel, rank params). bc>0 => show an x/total counter.
                t = nm + (bc > 0 ? "   " + juce::String(bi) + " / " + juce::String(bc) : juce::String());
            } else {
                // Block-coordinate descent: which tuning pass + which block of params.
                t = "Step 3/3: tuning all parameters   -   pass " + juce::String(pass)
                  + "   block " + juce::String(bi) + "/" + juce::String(bc) + "   [" + nm + "]";
            }
            lblProgress.setText(t + "      " + juce::String(secs) + "s elapsed", juce::dontSendNotification);
        });
    };
    // Heartbeat: even between improvements, show that the search is actively trying.
    // COALESCED: keep only the latest values; post one pending callAsync so a fast
    // search can't flood the message thread (that backlog is what froze the UI).
    autoMatcher.onTick = [this](long evals, double curBest) {
        { std::lock_guard<std::mutex> lk(liveMx); liveTickEvals = evals; liveTickLoss = curBest; }
        if (!tickPending.exchange(true))
            juce::MessageManager::callAsync([this] { applyLiveTick(); });
    };
    // COALESCED live-apply: store the latest improvement; post a single pending update.
    // The heavy work (render + spectrogram) runs at most once per drained message, never
    // once per improvement, so the message thread stays responsive.
    autoMatcher.onImprove = [this](const std::vector<float>& params, double loss, int note, int vel, float gate,
                                   const juce::AudioBuffer<float>& audio) {
        { std::lock_guard<std::mutex> lk(liveMx);
          liveParams = params; liveLoss = loss; liveNote = note; liveVel = vel; liveGate = gate;
          if (audio.getNumSamples() > 0) liveAudio = audio; }   // engine's render; avoid re-rendering live
        if (!improvePending.exchange(true))
            juce::MessageManager::callAsync([this] { applyLiveImprove(); });
    };

    autoThread = std::thread([this, path, target, sr, cfg] {
        auto factory = [path, sr]() -> std::unique_ptr<IRenderTarget> {
            juce::String e; return VstTarget::loadAny(juce::File(path), sr, 512, e); };
        AutoResult res = autoMatcher.run(factory, target, cfg);
        juce::MessageManager::callAsync([this, res, sr, cfg] {
            // Runs after the search finished -> all worker instances are destroyed, so it is
            // now safe to touch the live instance (and its editor) again.
            if (liveInstance != nullptr) {
                for (int i = 0; i < liveInstance->numParams() && i < (int) res.bestParams.size(); ++i)
                    liveInstance->setParamNotifying(i, res.bestParams[(size_t) i]);
                edNote.setText(juce::String(res.note), juce::dontSendNotification);
                // B = the engine's last best audio (avoid re-rendering the live instance: Synth1 and
                // other VST2 crash on reuse). Fall back to one render only if we have no audio yet.
                juce::AudioBuffer<float> a;
                { std::lock_guard<std::mutex> lk(liveMx); a = liveAudio; }
                if (a.getNumSamples() == 0)
                    a = liveInstance->render(res.note, res.velocity, cfg.durSec, res.gate);
                player.setBufferB(a, sr);
                // Recreate the editor we closed for a VST2 search, now showing the result patch.
                if (editorWindow == nullptr) {
                    if (auto* ap = liveInstance->getInstance())
                        if (auto* ed = ap->createEditorIfNeeded())
                            editorWindow = std::make_unique<PluginEditorWindow>(liveInstance->name(), ed);
                }
            }
            autoBusy = false;
            const double score = 100.0 * std::exp(-2.0 * res.bestLoss);
            setStatus("Stopped.  Best loss " + juce::String(res.bestLoss, 4) + "  (score "
                      + juce::String(score, 1) + "/100).  B = result - press Play, or Save Patch.");
        });
    });
}

// Message thread: apply the LATEST improvement only (coalesced). Heavy work lives here,
// but it runs at most once per drained message, so the UI thread never backlogs.
void MainComponent::applyLiveImprove() {
    improvePending = false;   // allow the next improvement to schedule a fresh update
    if (liveInstance == nullptr) return;
    std::vector<float> params; double loss; int note, vel; float gate;
    juce::AudioBuffer<float> audio;
    { std::lock_guard<std::mutex> lk(liveMx);
      params = liveParams; loss = liveLoss; note = liveNote; vel = liveVel; gate = liveGate;
      audio = liveAudio; }
    if (params.empty()) return;

    // Always-safe readouts.
    lossCurve.addPoint(loss);
    const double score = 100.0 * std::exp(-2.0 * loss);
    lblBest.setText("BEST:  loss " + juce::String(loss, 4) + "    score " + juce::String(score, 1) + "/100",
                    juce::dontSendNotification);

    // B preview + spectrogram come from the ENGINE's already-rendered audio - we do NOT re-render
    // the live instance, because some VST2 plugins (e.g. Synth1) crash on instance reuse, which
    // left B empty and broke playback. The engine workers already handle that (fresh-per-render).
    if (audio.getNumSamples() > 0) {
        player.setBufferB(audio, targetSr);
        auto spec = computeSpectrogram(audio, targetSr);
        spectro.setData(targetSpecCache, spec);
        heatmap.setData(targetSpecCache, spec);
    }

    // Safe mode: don't touch the live instance/editor during the search (editor closed, plugin
    // runs on the message thread). The editor is synced with the result at completion.
    if (autoSafeMode) return;

    // Fast mode: animate the hosted editor's knobs to the new best (no render - just param notify).
    edNote.setText(juce::String(note), juce::dontSendNotification);
    for (int i = 0; i < liveInstance->numParams() && i < (int) params.size(); ++i)
        liveInstance->setParamNotifying(i, params[(size_t) i]);
}

// Message thread: cheap heartbeat label (coalesced).
void MainComponent::applyLiveTick() {
    tickPending = false;
    long evals; double curBest;
    { std::lock_guard<std::mutex> lk(liveMx); evals = liveTickEvals; curBest = liveTickLoss; }
    if (curBest < 0) return;
    const double secs = juce::jmax(0.001, (juce::Time::getMillisecondCounter() - matchStartMs) / 1000.0);
    const double score = 100.0 * std::exp(-2.0 * curBest);
    lblBest.setText("BEST:  loss " + juce::String(curBest, 4) + "    score " + juce::String(score, 1)
                    + "/100        " + juce::String((juce::int64) (evals / secs)) + " tries/sec  ("
                    + juce::String(evals) + " total)", juce::dontSendNotification);
}

// Live match meter: snapshot the user's current patch, render it on a bg thread, and show
// the loss vs the target (updates as they turn the plugin's knobs). Skipped while a match runs.
void MainComponent::updateLiveMeter() {
    if (liveInstance == nullptr || targetBuf.getNumSamples() == 0) return;
    if (autoBusy.load() || stagedBusy.load() || runner.isMatching() || meterBusy.load()) return;
    if (meterThread.joinable()) meterThread.join();   // reap the finished previous render

    const int n = liveInstance->numParams();
    std::vector<float> params((size_t) n);
    for (int i = 0; i < n; ++i) params[(size_t) i] = liveInstance->getParam(i);
    const int note = edNote.getText().getIntValue();
    const double dur = targetBuf.getNumSamples() / targetSr;
    const juce::String path = vst3Path; const double sr = targetSr;
    auto target = targetBuf;

    meterBusy = true;
    meterThread = std::thread([this, params, note, dur, path, sr, target] {
        if (meterInst == nullptr || meterInst->renderFailed()) {
            juce::String e; meterInst = VstTarget::loadAny(juce::File(path), sr, 512, e);
        }
        double loss = -1.0; juce::AudioBuffer<float> a;
        if (meterInst != nullptr) {
            for (int i = 0; i < meterInst->numParams() && i < (int) params.size(); ++i)
                meterInst->setParam(i, params[(size_t) i]);
            a = meterInst->render(note, 100, dur, 0.7f);
            if (!meterInst->renderFailed() && Loss::peakAbs(a) > 1.0e-4f) {
                Loss L; loss = L.combined(target, a, 1.0f, 0.3f);
            }
        }
        juce::MessageManager::callAsync([this, loss, a] {
            if (loss < 0) setStatus("Live match meter: (silent / render failed - tweak the patch)");
            else {
                const double score = 100.0 * std::exp(-2.0 * loss);
                setStatus("Live match: loss=" + juce::String(loss, 4)
                          + "   score=" + juce::String(score, 1) + "/100   (turn knobs to improve)");
                player.setBufferB(a, targetSr);   // B = your current patch; press B then Play
            }
            meterBusy = false;
        });
    });
}

void MainComponent::setStatus(const juce::String& s) {
    juce::MessageManager::callAsync([this, s] { status.setText(s, juce::dontSendNotification); });
}

void MainComponent::loadVst() {
    const juce::File start = startDir("lastVstDir",
                                     juce::File("C:\\Program Files\\Common Files\\VST3"));
    const juce::String filter = VstTarget::vst2Supported() ? "*.vst3;*.dll;*.vst" : "*.vst3";
    chooser = std::make_unique<juce::FileChooser>("Select a VST3/VST2 instrument", start, filter);
    // Files only (NO canSelectDirectories): on Windows, combining files+dirs forces JUCE's
    // own non-native browser, which fetches a shell icon for every entry by extracting it
    // from the (often huge) plugin DLL -> the dialog becomes very slow. Files-only lets JUCE
    // use the fast NATIVE Windows dialog. Folder-bundle .vst3 (e.g. Dexed/M1) are still
    // loadable: open the bundle and pick Contents\x86_64-win\<name>.vst3 (the last dir is
    // remembered, so it's one click next time).
    const auto flags = juce::FileBrowserComponent::openMode
                     | juce::FileBrowserComponent::canSelectFiles;
    chooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
        const auto f = fc.getResult();
        if (f == juce::File()) return;
        rememberDir("lastVstDir", f);

        juce::String err;
        auto inst = VstTarget::loadAny(f, targetSr, 512, err);
        if (inst == nullptr) { setStatus("VST load failed: " + err); return; }

        // Tear down any previous editor window before replacing the instance.
        editorWindow.reset();
        vst3Path = f.getFullPathName();
        liveInstance = std::move(inst);

        if (auto* ap = liveInstance->getInstance()) {
            ap->prepareToPlay(targetSr, 512);
            if (auto* ed = ap->createEditorIfNeeded())
                editorWindow = std::make_unique<PluginEditorWindow>(liveInstance->name(), ed);
        }

        const int np = liveInstance->numParams();
        baseline.resize((size_t) np);
        freeFlags.assign((size_t) np, 0);
        for (int i = 0; i < np; ++i) baseline[(size_t) i] = liveInstance->getParam(i);

        // Determinism fix: per-note phase-randomization params (e.g. Vital's
        // "Oscillator N Phase Randomization", default 1) make the SAME patch render
        // differently every note -> spectral matching can't converge. Force them to 0
        // and freeze them so the synth renders deterministically. Without this, real
        // subtractive synths look "unmatchable"; with it Vital matches to ~98/100.
        int frozenRand = 0;
        for (int i = 0; i < np; ++i) {
            const auto nm = liveInstance->paramInfo(i).name.toLowerCase();
            const bool isPhaseRand = nm.contains("phase randomization")
                                  || nm.contains("random phase")
                                  || (nm.contains("phase") && nm.contains("rand"));
            if (isPhaseRand) {
                baseline[(size_t) i] = 0.0f;
                liveInstance->setParam(i, 0.0f);
                ++frozenRand;
            }
        }

        rebuildParamList();
        logAction("loadVst: loaded '" + liveInstance->name() + "' (" + juce::String(np) + " params) from " + vst3Path);
        setStatus("Loaded " + liveInstance->name() + " (" + juce::String(np) + " params). "
                  + (frozenRand > 0 ? "Froze " + juce::String(frozenRand)
                       + " phase-randomization param(s) -> deterministic. " : juce::String())
                  + "Click a param to free it; Shift+click for a range; or use Free first N / Free all.");
        resized();
    });
}

void MainComponent::loadWav() {
    const juce::File start = startDir("lastWavDir", juce::File());
    chooser = std::make_unique<juce::FileChooser>("Select target audio", start, "*.wav;*.aiff;*.aif;*.mp3;*.flac;*.ogg");
    const auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    chooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
        const auto f = fc.getResult();
        if (f == juce::File()) return;
        rememberDir("lastWavDir", f);
        double fileSr = 44100.0;
        juce::AudioBuffer<float> buf;
        if (!loadAudioFileMono(f, buf, fileSr)) { setStatus("Failed to load WAV: " + f.getFileName()); return; }

        // Work at a standard 44.1kHz so any hosted plugin accepts the rate
        // (e.g. Massive rejects unusual sample rates). Resample the target to match.
        const double renderSr = 44100.0;
        targetBuf = resampleMono(buf, fileSr, renderSr);
        targetSr = renderSr;

        const auto pitch = estimatePitch(targetBuf, targetSr);
        if (pitch.reliable) edNote.setText(juce::String(pitch.midiNote), juce::dontSendNotification);

        player.setBufferA(targetBuf, targetSr);
        updateTargetSpectrum();
        setStatus("Target loaded: " + f.getFileName() + "  dur=" + juce::String(targetBuf.getNumSamples() / targetSr, 2)
                  + "s  est.note=" + juce::String(pitch.midiNote));
    });
}

void MainComponent::updateTargetSpectrum() {
    if (targetBuf.getNumSamples() == 0) return;
    targetSpecCache = computeSpectrogram(targetBuf, targetSr);
    spectro.setData(targetSpecCache, {});
    heatmap.setData(targetSpecCache, {});
}

int MainComponent::countFree() const {
    int c = 0; for (char f : freeFlags) if (f) ++c; return c;
}

void MainComponent::pushLiveParams() {
    if (!runner.isMatching()) return;
    currentFreeParams.clear();
    for (int i = 0; i < (int) freeFlags.size(); ++i)
        if (freeFlags[(size_t) i]) currentFreeParams.push_back(i);
    runner.setFreeParams(currentFreeParams);   // applied live by the engine
}

void MainComponent::setFreeFirst(int n) {
    for (int i = 0; i < (int) freeFlags.size(); ++i) freeFlags[(size_t) i] = (i < n) ? 1 : 0;
    paramList.repaint();
    setStatus("Free params: " + juce::String(countFree()) + " / " + juce::String((int) freeFlags.size()));
    pushLiveParams();
}

void MainComponent::setAllFrozen() {
    std::fill(freeFlags.begin(), freeFlags.end(), (char) 0);
    paramList.repaint();
    setStatus("Free params: 0 / " + juce::String((int) freeFlags.size()));
    pushLiveParams();
}

void MainComponent::setAllFree() {
    std::fill(freeFlags.begin(), freeFlags.end(), (char) 1);
    paramList.repaint();
    setStatus("Free params: " + juce::String((int) freeFlags.size()) + " / " + juce::String((int) freeFlags.size()));
    pushLiveParams();
}

void MainComponent::invertFree() {
    for (auto& f : freeFlags) f = f ? 0 : 1;
    paramList.repaint();
    setStatus("Free params: " + juce::String(countFree()) + " / " + juce::String((int) freeFlags.size()));
    pushLiveParams();
}

void MainComponent::setFreeCore() {
    if (liveInstance == nullptr) return;
    for (int i = 0; i < (int) freeFlags.size(); ++i) {
        const auto cat = categorizeParam(liveInstance->paramInfo(i).name);
        freeFlags[(size_t) i] = (cat == ParamCat::Core || cat == ParamCat::Secondary) ? 1 : 0;
    }
    paramList.repaint();
    setStatus("Freed core tone params: " + juce::String(countFree())
              + " / " + juce::String((int) freeFlags.size()) + " (LFO/mod/MIDI-CC skipped)");
    pushLiveParams();
}

void MainComponent::setFreeRange(int a, int b, char value) {
    if (a > b) std::swap(a, b);
    a = juce::jmax(0, a);
    b = juce::jmin((int) freeFlags.size() - 1, b);
    for (int i = a; i <= b; ++i) freeFlags[(size_t) i] = value;
    paramList.repaint();
}

void MainComponent::rebuildParamList() {
    paramList.updateContent();
    paramList.repaint();
}

void MainComponent::startMatch() {
    if (liveInstance == nullptr) { setStatus("Load a VST3 first."); return; }
    if (targetBuf.getNumSamples() == 0) { setStatus("Load a target WAV first."); return; }

    currentFreeParams.clear();
    for (int i = 0; i < (int) freeFlags.size(); ++i)
        if (freeFlags[(size_t) i]) currentFreeParams.push_back(i);
    if (currentFreeParams.empty()) { setStatus("Select at least one free parameter."); return; }

    MatchConfig cfg;
    cfg.sampleRate = targetSr;
    cfg.durSec = targetBuf.getNumSamples() / targetSr;
    cfg.midiNote = edNote.getText().getIntValue();
    cfg.velocity = 100;
    const juce::int64 mv = edMax.getText().getLargeIntValue();
    cfg.maxFevals = (mv <= 0) ? 0 : juce::jmax((juce::int64) 100, mv);   // 0 = run until Stop
    cfg.sigma0 = sigmaSlider.getValue();
    cfg.envWeight = (float) envSlider.getValue();
    cfg.freeParams = currentFreeParams;
    cfg.baseParams = baseline;   // frozen baseline incl. forced phase-rand=0 (determinism)
    cfg.numWorkers = juce::jlimit(1, 64, edWorkers.getText().getIntValue());
    cfg.seed = 1;
    cfg.warmStart    = btnWarm.getToggleState();
    cfg.screening    = btnScreen.getToggleState();
    cfg.coarseToFine = btnCoarse.getToggleState();
    cfg.useBayesian  = btnBayes.getToggleState();
    cfg.useGenetic   = btnGenetic.getToggleState();   // GA over the free params
    if (cfg.useGenetic) {
        cfg.gaMutationRate  = (float) edMutRate.getText().getDoubleValue();
        cfg.gaMutationSigma = (float) edMutSigma.getText().getDoubleValue();
    }

    matchStartMs = juce::Time::getMillisecondCounter();
    lossCurve.clear();
    runner.onBest = [this](const BestUpdate& up) { applyBest(up); };
    runner.onFinished = [this](const juce::String& s) { setStatus(s); };
    runner.onStatus = [this](const juce::String& s) { setStatus(s); };
    runner.onProgress = [this](int gen, long evals, double bestLoss) {
        const double secs = juce::jmax(0.001, (juce::Time::getMillisecondCounter() - matchStartMs) / 1000.0);
        status.setText("running... gen " + juce::String(gen) + "  evals " + juce::String(evals)
                       + "  (" + juce::String(evals / secs, 1) + " eval/s)  best "
                       + juce::String(bestLoss, 4), juce::dontSendNotification);
    };
    runner.startMatching(vst3Path, targetBuf, cfg);
    setStatus("Starting: " + juce::String((int) currentFreeParams.size()) + " free params, "
              + juce::String(cfg.numWorkers) + " workers...");
}

void MainComponent::startStagedMatch() {
    if (liveInstance == nullptr) { setStatus("Load a VST3 first."); return; }
    if (targetBuf.getNumSamples() == 0) { setStatus("Load a target WAV first."); return; }
    if (stagedBusy.load()) { setStatus("Staged Auto already running."); return; }
    joinStaged();

    StagedConfig cfg;
    cfg.sampleRate = targetSr;
    cfg.durSec = targetBuf.getNumSamples() / targetSr;
    cfg.baseNote = edNote.getText().getIntValue();
    cfg.numWorkers = juce::jlimit(1, 64, edWorkers.getText().getIntValue());
    const juce::int64 mv = edMax.getText().getLargeIntValue();
    cfg.fevalsPerStage = (mv <= 0) ? 2000 : juce::jlimit((juce::int64) 200, (juce::int64) 20000, mv);
    cfg.passes = 0;            // UNLIMITED: keep re-running stages until the user presses Stop
    cfg.searchPerformance = true;

    const juce::String path = vst3Path;
    auto target = targetBuf;
    const double sr = targetSr;
    const double durSec = cfg.durSec;
    stagedBusy = true;
    matchStartMs = juce::Time::getMillisecondCounter();
    lossCurve.clear();
    setStatus("Staged Auto (unlimited): designing in human order... press Stop when satisfied.");

    stagedMatcher.onStage = [this](int si, int sc, const juce::String& nm, double loss) {
        juce::MessageManager::callAsync([this, si, sc, nm, loss] {
            const double secs = juce::jmax(0.001, (juce::Time::getMillisecondCounter() - matchStartMs) / 1000.0);
            status.setText("Staged (unlimited)  " + nm + "   loss=" + juce::String(loss, 4)
                           + "   " + juce::String((int) secs) + "s  (Stop to keep)", juce::dontSendNotification);
        });
    };
    // Live-apply each improvement: animate the editor knobs + refresh A/B (B) + spectrogram.
    stagedMatcher.onImprove = [this, durSec, sr](const std::vector<float>& params, double loss,
                                                 int note, int vel, float gate) {
        juce::MessageManager::callAsync([this, params, loss, note, vel, gate, durSec, sr] {
            if (liveInstance == nullptr) return;
            for (int i = 0; i < liveInstance->numParams() && i < (int) params.size(); ++i)
                liveInstance->setParamNotifying(i, params[(size_t) i]);
            edNote.setText(juce::String(note), juce::dontSendNotification);
            auto a = liveInstance->render(note, vel, durSec, gate);
            player.setBufferB(a, sr);
            auto spec = computeSpectrogram(a, targetSr);
            spectro.setData(targetSpecCache, spec);
            heatmap.setData(targetSpecCache, spec);
            lossCurve.addPoint(loss);
        });
    };

    // Run on a background std::thread; the engine builds its own worker instances, so
    // the message thread stays free for the editor/transport (mirrors MatchRunner).
    stagedThread = std::thread([this, path, target, sr, cfg] {
        auto factory = [path, sr]() -> std::unique_ptr<IRenderTarget> {
            juce::String e; return VstTarget::loadAny(juce::File(path), sr, 512, e); };
        StagedResult res = stagedMatcher.run(factory, target, cfg);
        juce::MessageManager::callAsync([this, res, cfg] {
            if (liveInstance != nullptr) {
                for (int i = 0; i < liveInstance->numParams() && i < (int) res.bestParams.size(); ++i)
                    liveInstance->setParamNotifying(i, res.bestParams[(size_t) i]);   // animate editor knobs
                edNote.setText(juce::String(res.note), juce::dontSendNotification);
                auto a = liveInstance->render(res.note, res.velocity, cfg.durSec, res.gate);
                player.setBufferB(a, cfg.sampleRate);
                auto bestSpec = computeSpectrogram(a, targetSr);
                spectro.setData(targetSpecCache, bestSpec);
                heatmap.setData(targetSpecCache, bestSpec);
            }
            stagedBusy = false;
            setStatus("Staged Auto stopped: best loss=" + juce::String(res.bestLoss, 4)
                      + "   note=" + juce::String(res.note) + " vel=" + juce::String(res.velocity)
                      + " gate=" + juce::String(res.gate, 2) + "   (B = result; Save Best State to keep)");
        });
    });
}

void MainComponent::stopMatch() {
    logAction("stopMatch: begin (Stop pressed)");
    runner.stopMatching();
    if (stagedBusy.load() || stagedThread.joinable()) joinStaged();
    if (autoBusy.load() || autoThread.joinable()) joinAuto();
    setStatus("Stopped.");
    logAction("stopMatch: done");
}

void MainComponent::applyBest(const BestUpdate& up) {
    // A/B: best -> source B.
    player.setBufferB(up.audio, targetSr);

    // Animate the hosted editor's knobs by applying the free params with notification.
    if (liveInstance != nullptr)
        for (int idx : currentFreeParams)
            if (idx >= 0 && idx < (int) up.fullParams.size())
                liveInstance->setParamNotifying(idx, up.fullParams[(size_t) idx]);

    // 3D waterfall + 2D heatmap: target (cached) vs current best + loss curve.
    auto bestSpec = computeSpectrogram(up.audio, targetSr);
    spectro.setData(targetSpecCache, bestSpec);
    heatmap.setData(targetSpecCache, bestSpec);
    lossCurve.addPoint(up.loss);

    const double secs = juce::jmax(0.001, (juce::Time::getMillisecondCounter() - matchStartMs) / 1000.0);
    const double rate = up.evaluations / secs;
    status.setText("gen " + juce::String(up.generation) + "  loss " + juce::String(up.loss, 4)
                   + "  evals " + juce::String(up.evaluations)
                   + "  (" + juce::String(rate, 1) + " eval/s)", juce::dontSendNotification);
}

void MainComponent::saveState() {
    if (liveInstance == nullptr) { setStatus("Nothing to save."); return; }
    chooser = std::make_unique<juce::FileChooser>("Save best state", juce::File(), "*.state");
    const auto flags = juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting;
    chooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
        auto f = fc.getResult();
        if (f == juce::File()) return;
        if (auto* ap = liveInstance->getInstance()) {
            juce::MemoryBlock mb;
            ap->getStateInformation(mb);
            f.withFileExtension("state").replaceWithData(mb.getData(), mb.getSize());
        }
        auto audio = liveInstance->render(edNote.getText().getIntValue(), 100,
                                          targetBuf.getNumSamples() / targetSr);
        writeWavMono(f.withFileExtension("wav"), audio, targetSr);
        setStatus("Saved " + f.withFileExtension("state").getFileName() + " (+ .wav)");
    });
}

// ---- layout ----------------------------------------------------------------
void MainComponent::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff1b1e24));
}

void MainComponent::resized() {
    auto r = getLocalBounds().reduced(8);

    // Two stacked readouts at the bottom: learning status + audio/debug.
    debug.setBounds(r.removeFromBottom(20));
    r.removeFromBottom(2);
    status.setBounds(r.removeFromBottom(22));
    r.removeFromBottom(6);

    // Left: 3D waterfall (top) + 2D spectrogram heatmap (bottom).
    auto left = r.removeFromLeft(juce::jmin(560, r.getWidth() / 2));
    spectro.setBounds(left.removeFromTop(left.getHeight() / 2));
    left.removeFromTop(6);
    heatmap.setBounds(left);
    r.removeFromLeft(8);

    // Right column = the dashboard.
    auto layoutRow = [](juce::Rectangle<int>& row, std::initializer_list<juce::Component*> comps, int gap = 6) {
        const int w = (row.getWidth() - gap * ((int) comps.size() - 1)) / (int) comps.size();
        for (auto* c : comps) { c->setBounds(row.removeFromLeft(w)); row.removeFromLeft(gap); }
    };

    // Row 1: load.
    auto row1 = r.removeFromTop(30);
    layoutRow(row1, { &btnLoadVst, &btnLoadWav });
    r.removeFromTop(6);

    // Row 2: note + workers + live meter.
    auto row2 = r.removeFromTop(24);
    lblNote.setBounds(row2.removeFromLeft(40));
    edNote.setBounds(row2.removeFromLeft(48)); row2.removeFromLeft(12);
    lblWorkers.setBounds(row2.removeFromLeft(56));
    edWorkers.setBounds(row2.removeFromLeft(44)); row2.removeFromLeft(12);
    btnMeter.setBounds(row2);
    r.removeFromTop(6);

    // Row 2b: exclude-by-regex (default "MIDI CC").
    auto row2b = r.removeFromTop(24);
    lblExclude.setBounds(row2b.removeFromLeft(108));
    edExclude.setBounds(row2b);
    r.removeFromTop(8);

    // Row 3: the BIG action button.
    btnStart.setBounds(r.removeFromTop(44));
    r.removeFromTop(6);
    auto row4 = r.removeFromTop(28);
    layoutRow(row4, { &btnStop, &btnSave });
    r.removeFromTop(8);

    // Dashboard readouts.
    lblBest.setBounds(r.removeFromTop(30));
    lblProgress.setBounds(r.removeFromTop(20));
    r.removeFromTop(6);

    // Transport row: A / B / Play.
    auto rowT = r.removeFromTop(26);
    layoutRow(rowT, { &btnA, &btnB, &btnPlay });
    r.removeFromTop(8);

    // Loss curve fills the rest of the right column.
    lossCurve.setBounds(r);
}

} // namespace vms
