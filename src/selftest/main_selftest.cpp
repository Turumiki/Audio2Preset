#include <juce_events/juce_events.h>
#include <juce_core/juce_core.h>
#include <iostream>

#include "SelfTest.h"
#include "../core/InternalSynthTarget.h"
#include "../core/VstTarget.h"
#include "../core/MatchingEngine.h"
#include "../core/StagedMatcher.h"
#include "../core/AutoMatcher.h"
#include "../core/AudioIO.h"
#include "../core/PitchDetect.h"
#include "../core/Loss.h"
#include "../core/ParamPriority.h"
#include <juce_dsp/juce_dsp.h>
#include <thread>
#include <atomic>
#include <random>
#include <mutex>

// Simple stdout logger.
class ConsoleLogger : public juce::Logger {
public:
    void logMessage(const juce::String& m) override { std::cout << m << std::endl; }
};

static bool buildVitalState(const juce::MemoryBlock& tmpl, const juce::MemoryBlock& vitalJson,
                            juce::MemoryBlock& outState);

static std::vector<int> parseIntList(const juce::String& s) {
    std::vector<int> out;
    for (auto& tok : juce::StringArray::fromTokens(s, ",", ""))
        if (tok.trim().isNotEmpty()) out.push_back(tok.getIntValue());
    return out;
}

// Search the MIDI note / velocity / note-length (gate) the synth should be played at,
// on a given base patch. note: synths transpose (Vital -2 semis); velocity shapes
// timbre; gate sets release timing. Returns the best (note, velocity, gate).
struct PerfResult { int note = 60; int velocity = 100; float gate = 0.7f; double loss = 1.0e30; };
static PerfResult performanceSearch(const std::function<std::unique_ptr<vms::IRenderTarget>()>& factory,
                                    const juce::AudioBuffer<float>& target,
                                    const juce::MemoryBlock& initState, const std::vector<float>& baseParams,
                                    int baseNote, double dur, int radius = 12) {
    PerfResult r; r.note = baseNote;
    auto inst = factory();
    if (!inst) return r;
    if (initState.getSize() > 0) inst->setStateData(initState.getData(), initState.getSize());
    for (int i = 0; i < inst->numParams() && i < (int) baseParams.size(); ++i) inst->setParam(i, baseParams[(size_t) i]);
    for (int p = 0; p < inst->numParams(); ++p) {  // force determinism
        const auto nm = inst->paramInfo(p).name.toLowerCase();
        if (nm.contains("phase randomization") || nm.contains("random phase")) inst->setParam(p, 0.0f);
    }
    vms::Loss L;
    auto score = [&](int n, int v, float g) -> double {
        auto a = inst->render(n, v, dur, g);
        return (vms::Loss::peakAbs(a) > 1.0e-4f) ? L.combined(target, a, 1.0f, 1.0f) : 1.0e9;
    };
    double bn = 1.0e30;
    for (int n = baseNote - radius; n <= baseNote + radius; ++n) { double l = score(n, 100, 0.7f); if (l < bn) { bn = l; r.note = n; } }
    double bvg = 1.0e30;
    for (int v : { 40, 70, 100, 127 }) for (float g : { 0.3f, 0.5f, 0.7f, 0.9f, 1.0f }) {
        double l = score(r.note, v, g); if (l < bvg) { bvg = l; r.velocity = v; r.gate = g; }
    }
    r.loss = bvg; return r;
}

// FULL-parameter auto matcher: block-coordinate descent over ALL params, cycled.
//   --match-auto <plugin> <wav> [--note N][--passes P][--block B][--feblock M][--avg N][--robust][--workers W][--exclude REGEX] --out <wav>
static int runMatchAuto(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& f) -> juce::String {
        const int i = args.indexOf(f); return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String(); };
    const int mi = args.indexOf("--match-auto");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String wav  = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    const juce::String outPath = valAfter("--out");
    if (vst3.isEmpty() || wav.isEmpty() || outPath.isEmpty()) {
        std::cout << "usage: --match-auto <plugin> <wav> [--note N][--passes P][--block B][--feblock M][--avg N][--robust][--workers W][--exclude REGEX] --out <wav>\n"; return 2; }

    juce::AudioBuffer<float> target; double sr = 44100.0;
    if (!vms::loadAudioFileMono(juce::File(wav), target, sr)) { std::cout << "ERROR: load wav\n"; return 3; }

    vms::AutoConfig cfg;
    cfg.sampleRate = sr; cfg.durSec = target.getNumSamples() / sr;
    cfg.baseNote = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : vms::estimatePitch(target, sr).midiNote;
    cfg.numWorkers = valAfter("--workers").isNotEmpty() ? juce::jmax(1, valAfter("--workers").getIntValue()) : 16;
    // VST2 plugins can hard-crash once too many instances coexist (TAL dies at ~8). The CLI
    // has no GUI window env (no cross-thread window crash), so it stays parallel but capped.
    if ((vst3.endsWithIgnoreCase(".dll") || vst3.endsWithIgnoreCase(".vst")) && cfg.numWorkers > 4) {
        std::cout << "VST2 detected: capping workers " << cfg.numWorkers << " -> 4 for stability\n";
        cfg.numWorkers = 4;
    }
    // --safe: force message-thread single-threaded rendering (for GUI-heavy plugins that crash
    // under worker-thread rendering, e.g. KORG Legacy VST3). Slower but stable.
    if (args.contains("--safe")) {
        vms::VstTarget::setForceMessageThread(true);
        cfg.numWorkers = 1;
        std::cout << "safe mode: message-thread single-threaded rendering\n";
    }
    if (valAfter("--block").isNotEmpty()) cfg.blockSize = juce::jmax(2, valAfter("--block").getIntValue());
    if (valAfter("--feblock").isNotEmpty()) cfg.fevalsPerBlock = valAfter("--feblock").getLargeIntValue();
    if (valAfter("--avg").isNotEmpty()) cfg.rendersPerEval = juce::jmax(1, valAfter("--avg").getIntValue());
    cfg.robustLoss = args.contains("--robust");
    cfg.maxPasses = valAfter("--passes").isNotEmpty() ? juce::jmax(1, valAfter("--passes").getIntValue()) : 3;
    if (valAfter("--exclude").isNotEmpty()) cfg.excludeRegex = valAfter("--exclude");

    auto factory = [vst3, sr]() -> std::unique_ptr<vms::IRenderTarget> {
        juce::String e; return vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e); };
    std::cout << "AUTO all-param block search  note=" << cfg.baseNote << " dur=" << cfg.durSec
              << " block=" << cfg.blockSize << " feblock=" << cfg.fevalsPerBlock << " passes=" << cfg.maxPasses
              << (cfg.robustLoss ? " robust" : "") << std::endl;

    vms::AutoResult res;
    std::atomic<bool> done { false };
    std::thread driver([&] {
        {
            vms::AutoMatcher m;
            m.onProgress = [](int pass, int bi, int bc, const juce::String& nm, double loss, double imp) {
                if (bi == 0 || imp > 0.0 || bi % 5 == 0)
                    std::cout << "  pass " << pass << " block " << bi << "/" << bc << " " << nm
                              << " loss=" << loss << (imp > 0 ? " (improved " + juce::String(imp, 4) + ")" : "") << std::endl;
            };
            res = m.run(factory, target, cfg);
            juce::String e; auto best = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
            if (best) {
                for (int i = 0; i < best->numParams() && i < (int) res.bestParams.size(); ++i) best->setParam(i, res.bestParams[(size_t) i]);
                auto a = best->render(res.note, res.velocity, cfg.durSec, res.gate);
                vms::writeWavMono(juce::File(outPath), a, sr);
                juce::MemoryBlock sm; best->getInstance()->getStateInformation(sm);
                juce::File(outPath).withFileExtension("state").replaceWithData(sm.getData(), sm.getSize());
                std::cout << "FINAL loss=" << res.bestLoss << " note=" << res.note << " peak=" << vms::Loss::peakAbs(a) << std::endl;
            }
        }
        done = true;
    });
    while (!done.load()) juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    driver.join();
    std::cout << "Done -> " << outPath << std::endl;
    return 0;
}

// PURE genetic algorithm over ALL parameters at once (no stages, no curation).
//   --match-ga <plugin> <wav> [--note N] [--maxfevals M] [--pop P] [--gate G] [--workers W] --out <wav>
static int runMatchGA(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& f) -> juce::String {
        const int i = args.indexOf(f); return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String(); };
    const int mi = args.indexOf("--match-ga");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String wav  = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    const juce::String outPath = valAfter("--out");
    if (vst3.isEmpty() || wav.isEmpty() || outPath.isEmpty()) {
        std::cout << "usage: --match-ga <plugin> <wav> [--note N] [--maxfevals M] [--pop P] [--gate G] [--workers W] --out <wav>\n"; return 2; }

    juce::AudioBuffer<float> target; double sr = 44100.0;
    if (!vms::loadAudioFileMono(juce::File(wav), target, sr)) { std::cout << "ERROR: load wav\n"; return 3; }

    juce::String e0; auto probe = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e0);
    if (!probe) { std::cout << "load failed: " << e0 << "\n"; return 4; }
    const int np = probe->numParams();
    probe.reset();

    vms::MatchConfig cfg;
    cfg.sampleRate = sr; cfg.durSec = target.getNumSamples() / sr;
    cfg.midiNote = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : vms::estimatePitch(target, sr).midiNote;
    cfg.velocity = 100;
    if (valAfter("--gate").isNotEmpty()) cfg.gateFrac = (float) valAfter("--gate").getDoubleValue();
    cfg.freeParams.resize((size_t) np); for (int i = 0; i < np; ++i) cfg.freeParams[(size_t) i] = i;  // ALL params
    cfg.maxFevals = valAfter("--maxfevals").isNotEmpty() ? valAfter("--maxfevals").getLargeIntValue() : 40000;
    cfg.numWorkers = valAfter("--workers").isNotEmpty() ? juce::jmax(1, valAfter("--workers").getIntValue()) : 16;
    cfg.useGenetic = true;
    if (valAfter("--pop").isNotEmpty()) cfg.gaPopulation = juce::jmax(8, valAfter("--pop").getIntValue());
    if (valAfter("--mutrate").isNotEmpty()) cfg.gaMutationRate = (float) valAfter("--mutrate").getDoubleValue();
    if (valAfter("--mutsigma").isNotEmpty()) cfg.gaMutationSigma = (float) valAfter("--mutsigma").getDoubleValue();
    if (valAfter("--immig").isNotEmpty()) cfg.gaImmigrants = (float) valAfter("--immig").getDoubleValue();
    if (args.contains("--adaptive")) cfg.gaAdaptive = true;
    if (valAfter("--avg").isNotEmpty()) cfg.rendersPerEval = juce::jmax(1, valAfter("--avg").getIntValue());
    cfg.forceDeterministic = true;

    auto factory = [vst3, sr]() -> std::unique_ptr<vms::IRenderTarget> {
        juce::String e; return vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e); };

    std::cout << "GA over ALL " << np << " params  note=" << cfg.midiNote << " dur=" << cfg.durSec
              << " pop=" << (cfg.gaPopulation > 0 ? cfg.gaPopulation : -1) << " budget=" << cfg.maxFevals
              << " workers=" << cfg.numWorkers << std::endl;

    vms::MatchResult res;
    std::atomic<bool> done { false };
    std::thread driver([&] {
        {
            vms::MatchingEngine engine(factory, target, cfg);
            res = engine.run({}, [](int gen, long evals, double best) {
                if (gen % 5 == 0) std::cout << "  gen " << gen << " evals " << evals << " best " << best << std::endl; });
            juce::String e; auto best = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
            if (best) {
                for (int i = 0; i < best->numParams() && i < (int) res.bestFullParams.size(); ++i)
                    best->setParam(i, res.bestFullParams[(size_t) i]);
                auto a = best->render(cfg.midiNote, cfg.velocity, cfg.durSec, cfg.gateFrac);
                vms::writeWavMono(juce::File(outPath), a, sr);
                juce::MemoryBlock sm; best->getInstance()->getStateInformation(sm);
                juce::File(outPath).withFileExtension("state").replaceWithData(sm.getData(), sm.getSize());
                std::cout << "FINAL loss=" << res.bestLoss << " gens=" << res.generations
                          << " evals=" << res.evaluations << " peak=" << vms::Loss::peakAbs(a) << std::endl;
            }
        }
        done = true;
    });
    while (!done.load()) juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    driver.join();
    std::cout << "Done -> " << outPath << std::endl;
    return 0;
}

// STAGED pure-parameter matcher: builds a patch from the INIT state using only the
// synth's own parameters, in the order a human sound designer works — oscillator/tone
// first, then filter, then amp envelope, then modulation, then FX. Each stage runs
// CMA-ES over just that stage's params (low dim => fast convergence), freezes the
// result, and moves on. No presets, no wavetable import ("that's cheating").
//   --match-staged <plugin> <wav> [--note N] [--fevals M] [--workers W] --out <wav>
static int runMatchStaged(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& f) -> juce::String {
        const int i = args.indexOf(f); return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String(); };
    const int mi = args.indexOf("--match-staged");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String wav  = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    const juce::String outPath = valAfter("--out");
    if (vst3.isEmpty() || wav.isEmpty() || outPath.isEmpty()) {
        std::cout << "usage: --match-staged <plugin> <wav> [--note N] [--fevals M] [--workers W] --out <wav>\n"; return 2; }

    juce::AudioBuffer<float> target; double sr = 44100.0;
    if (!vms::loadAudioFileMono(juce::File(wav), target, sr)) { std::cout << "ERROR: load wav\n"; return 3; }
    vms::StagedConfig scfg;
    scfg.sampleRate = sr; scfg.durSec = target.getNumSamples() / sr;
    scfg.baseNote = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue()
                                                    : vms::estimatePitch(target, sr).midiNote;
    scfg.fevalsPerStage = valAfter("--fevals").isNotEmpty() ? valAfter("--fevals").getLargeIntValue() : 2500;
    scfg.numWorkers = valAfter("--workers").isNotEmpty() ? juce::jmax(1, valAfter("--workers").getIntValue()) : 4;
    scfg.searchPerformance = !args.contains("--no-perf");
    if (valAfter("--passes").isNotEmpty()) scfg.passes = juce::jmax(1, valAfter("--passes").getIntValue());
    if (valAfter("--avg").isNotEmpty()) scfg.rendersPerEval = juce::jmax(1, valAfter("--avg").getIntValue());
    scfg.robustLoss = args.contains("--robust");

    auto factory = [vst3, sr]() -> std::unique_ptr<vms::IRenderTarget> {
        juce::String e; return vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e); };

    std::cout << "baseNote=" << scfg.baseNote << " dur=" << scfg.durSec << " passes=" << scfg.passes
              << (scfg.robustLoss ? " robust" : "") << " staged search\n";
    vms::StagedResult result;
    std::atomic<bool> done { false };
    std::thread driver([&] {
        vms::StagedMatcher m;
        m.onStage = [&](int si, int sc, const juce::String& nm, double loss) {
            std::cout << "stage " << si << "/" << sc << " " << nm << " -> loss=" << loss << std::endl; };
        result = m.run(factory, target, scfg);
        // Render + save final.
        juce::String e; auto best = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
        if (best) {
            for (int i = 0; i < best->numParams() && i < (int) result.bestParams.size(); ++i)
                best->setParam(i, result.bestParams[(size_t) i]);
            auto a = best->render(result.note, result.velocity, scfg.durSec, result.gate);
            vms::writeWavMono(juce::File(outPath), a, sr);
            juce::MemoryBlock sm; best->getInstance()->getStateInformation(sm);
            juce::File(outPath).withFileExtension("state").replaceWithData(sm.getData(), sm.getSize());
            std::cout << "FINAL loss=" << result.bestLoss << " note=" << result.note << " vel=" << result.velocity
                      << " gate=" << result.gate << " peak=" << vms::Loss::peakAbs(a) << " -> " << outPath << std::endl;
        }
        done = true;
    });
    while (!done.load()) juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    driver.join();
    return 0;
}

// Match a target WAV by SEARCHING OVER PRESETS (each .vital preset carries its own
// wavetable). Two phases: (1) screen every preset by single-render loss, (2) refine
// the top-K with CMA-ES over continuous tone params. This is how the matcher gets to
// GENERIC plugin matcher: searches a plugin's "state space" = {base states} x {params}.
// Base states come from two universal sources: the plugin's PROGRAM list
// (getNumPrograms/setCurrentProgram — e.g. KORG M1's 128 sounds) and/or PRESET FILES
// (.vital wavetables, or any saved JUCE .state). Phase 1 screens every base state by
// single-render loss; phase 2 refines the top-K with CMA-ES over auto-selected
// continuous tone params (or --params). Works for any VST/VST3 instrument.
//   --match <plugin> <wav> [--presets <dir>] [--programs] [--note N] [--screen S]
//           [--top K] [--params i,j] [--maxfree N] [--refine M] [--workers W] --out <wav>
static int runMatch(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& flag) -> juce::String {
        const int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String();
    };
    const int mi = juce::jmax(args.indexOf("--match"), args.indexOf("--match-presets"));
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String wav  = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    const juce::String presetsDir = valAfter("--presets");
    const juce::String outPath = valAfter("--out");
    const bool usePrograms = args.contains("--programs");
    if (vst3.isEmpty() || wav.isEmpty() || outPath.isEmpty() || (presetsDir.isEmpty() && !usePrograms)) {
        std::cout << "usage: --match <plugin> <wav> [--presets <dir>] [--programs] [--note N]"
                     " [--screen S] [--top K] [--params i,j] [--maxfree N] [--refine M] [--workers W] --out <wav>\n";
        return 2;
    }

    juce::AudioBuffer<float> target; double sr = 44100.0;
    if (!vms::loadAudioFileMono(juce::File(wav), target, sr)) {
        std::cout << "ERROR: could not load target wav: " << wav << "\n"; return 3;
    }
    const int screenMax = valAfter("--screen").isNotEmpty() ? juce::jmax(1, valAfter("--screen").getIntValue()) : 300;
    const int topK      = valAfter("--top").isNotEmpty()    ? juce::jmax(1, valAfter("--top").getIntValue())    : 6;
    const long refineFev = valAfter("--refine").isNotEmpty() ? valAfter("--refine").getLargeIntValue()          : 3000;
    const int workers   = valAfter("--workers").isNotEmpty() ? juce::jmax(1, valAfter("--workers").getIntValue()) : 4;
    const int maxFree   = valAfter("--maxfree").isNotEmpty() ? juce::jmax(1, valAfter("--maxfree").getIntValue()) : 16;

    auto factory = [vst3, sr]() -> std::unique_ptr<vms::IRenderTarget> {
        juce::String e; return vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
    };

    // A base-state candidate: either a preset FILE (state built on the fly) or a
    // pre-generated program STATE held in memory.
    struct Cand { juce::String label; bool isFile = false; juce::File file; juce::MemoryBlock state; };
    std::vector<Cand> cands;

    struct Scored { double loss; int idx; };
    std::vector<Scored> scored;
    double bestLoss = std::numeric_limits<double>::infinity();
    std::vector<float> bestParams; juce::MemoryBlock bestState; int bestIdx = -1;

    // ---- Program-state capture MUST run on the message (main) thread ----
    // M1's setCurrentProgram only takes effect on the message thread; capturing it
    // on a worker thread silently yields the default state for every program.
    if (usePrograms) {
        juce::String e; auto cap = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
        if (cap) {
            int nprog = cap->getInstance()->getNumPrograms();
            if (valAfter("--maxprog").isNotEmpty()) nprog = juce::jmin(nprog, valAfter("--maxprog").getIntValue());
            std::cout << "plugin exposes " << cap->getInstance()->getNumPrograms()
                      << " programs; capturing " << nprog << " (message thread)..." << std::endl;
            cap.reset();   // release the probe; capture each program on its own instance
            const int p0 = valAfter("--startprog").isNotEmpty() ? valAfter("--startprog").getIntValue() : 0;
            for (int p = p0; p < nprog; ++p) {
                {
                    juce::String ee; auto t = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, ee);
                    if (t) {
                        auto* in = t->getInstance();
                        in->setNonRealtime(true); in->prepareToPlay(sr, 512);
                        in->setCurrentProgram(p);
                        Cand c; c.label = "prog " + juce::String(p) + " (" + in->getProgramName(p) + ")";
                        in->getStateInformation(c.state);
                        cands.push_back(std::move(c));
                    }
                }
                juce::MessageManager::getInstance()->runDispatchLoopUntil(3);
                if ((p + 1) % 32 == 0) std::cout << "  captured " << (p + 1) << "/" << nprog << std::endl;
            }
            // NOTE: KORG M1 has a global in-process state — only the FIRST M1 instance
            // per process can switch programs, so batch capture yields identical states.
            // Such plugins need one render per OS process (see --programs --prog N).
        }
    }

    const int noteMain = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue()
                                                         : vms::estimatePitch(target, sr).midiNote;
    const double durMain = juce::jmax(1.0, target.getNumSamples() / sr);

    // ---- Program-only search runs ENTIRELY on the message thread ----
    // ROMplers like M1 ignore setStateInformation/program changes off the message
    // thread, and their sound lives in PCM (not params) so CMA-ES param refine is
    // both impossible (setParam clobbers) and unnecessary. So: screen all program
    // states sequentially on this (main) thread and output the best-matching program.
    if (usePrograms && presetsDir.isEmpty()) {
        std::cout << "note=" << noteMain << " dur=" << durMain << " candidates=" << cands.size()
                  << " (program search, message-thread)" << std::endl;
        vms::Loss L;
        for (size_t j = 0; j < cands.size(); ++j) {
            juce::String e; auto inst = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
            double loss = 1.0e9;
            if (inst && cands[j].state.getSize() > 0) {
                inst->setStateData(cands[j].state.getData(), cands[j].state.getSize());
                auto a = inst->render(noteMain, 100, durMain);
                if (!inst->renderFailed() && vms::Loss::peakAbs(a) > 1.0e-4f)
                    loss = L.combined(target, a, 1.0f, 0.3f);
            }
            scored.push_back({ loss, (int) j });
            if ((j + 1) % 25 == 0) std::cout << "  screened " << (j + 1) << "/" << cands.size() << std::endl;
        }
        std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) { return a.loss < b.loss; });
        std::cout << "top programs:" << std::endl;
        for (int i = 0; i < juce::jmin(topK, (int) scored.size()); ++i)
            std::cout << "  #" << (i + 1) << " loss=" << scored[(size_t) i].loss
                      << "  " << cands[(size_t) scored[(size_t) i].idx].label << std::endl;
        if (!scored.empty() && std::isfinite(scored[0].loss)) {
            const auto& bc = cands[(size_t) scored[0].idx];
            juce::String e; auto best = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
            if (best) {
                best->setStateData(bc.state.getData(), bc.state.getSize());
                auto a = best->render(noteMain, 100, durMain);
                vms::writeWavMono(juce::File(outPath), a, sr);
                juce::MemoryBlock sm; best->getInstance()->getStateInformation(sm);
                juce::File(outPath).withFileExtension("state").replaceWithData(sm.getData(), sm.getSize());
                std::cout << "BEST: " << bc.label << "  loss=" << scored[0].loss
                          << "  peak=" << vms::Loss::peakAbs(a) << " -> " << outPath << std::endl;
            }
        }
        return 0;
    }

    std::atomic<bool> done { false };
    std::thread driver([&] {
        {
            juce::String e0;
            auto probe = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e0);
            if (!probe) { std::cout << "load failed: " << e0 << "\n"; done = true; return; }
            const int np = probe->numParams();
            juce::MemoryBlock tmpl; probe->getInstance()->getStateInformation(tmpl);  // framing template
            const double dur = durMain;
            int note = noteMain; int vel = 100; float gate = 0.7f;
            if (!args.contains("--no-perf")) {   // search MIDI note/velocity/gate on the default patch
                auto pr = performanceSearch(factory, target, juce::MemoryBlock(), std::vector<float>(), noteMain, dur);
                note = pr.note; vel = pr.velocity; gate = pr.gate;
                std::cout << "PERFORMANCE -> note=" << note << " velocity=" << vel << " gate=" << gate
                          << "  loss=" << pr.loss << std::endl;
            }

            // ---- Build the preset-file candidates (programs captured above) ----
            if (presetsDir.isNotEmpty()) {
                juce::Array<juce::File> files;
                juce::File(presetsDir).findChildFiles(files, juce::File::findFiles, true, "*.vital");
                juce::File(presetsDir).findChildFiles(files, juce::File::findFiles, true, "*.state");
                std::vector<juce::File> picked;
                if (files.size() <= screenMax) { for (auto& f : files) picked.push_back(f); }
                else { const double step = (double) files.size() / screenMax;
                       for (int i = 0; i < screenMax; ++i) picked.push_back(files[(int) (i * step)]); }
                for (auto& f : picked) { Cand c; c.label = f.getFileName(); c.isFile = true; c.file = f; cands.push_back(std::move(c)); }
                std::cout << "found " << files.size() << " preset files, using " << picked.size() << std::endl;
            }
            if (cands.empty()) { std::cout << "no base states to search\n"; done = true; return; }

            // Resolve a candidate -> a loadable JUCE state blob.
            auto getState = [&](const Cand& c, juce::MemoryBlock& out) -> bool {
                if (!c.isFile) { out = c.state; return out.getSize() > 0; }
                juce::MemoryBlock raw;
                if (!c.file.loadFileAsData(raw) || raw.getSize() == 0) return false;
                if (c.file.hasFileExtension("vital")) return buildVitalState(tmpl, raw, out);
                out = raw; return true;  // already a JUCE .state
            };

            // ---- Free-param set: explicit --params, else GENERIC continuous tone params ----
            std::vector<int> freeParams;
            const juce::String pl = valAfter("--params");
            if (pl.isNotEmpty()) freeParams = parseIntList(pl);
            else {
                const char* kw[] = { "cutoff", "resonance", "attack", "decay", "sustain",
                                     "release", "level", "drive", "volume", "gain", "balance", "eg int" };
                for (int i = 0; i < np && (int) freeParams.size() < maxFree; ++i) {
                    const auto info = probe->paramInfo(i);
                    if (info.isDiscrete) continue;
                    const auto nm = info.name.toLowerCase();
                    for (auto* k : kw) if (nm.contains(k)) { freeParams.push_back(i); break; }
                }
            }
            std::cout << "note=" << note << " dur=" << dur << " candidates=" << cands.size()
                      << " refine free params=" << freeParams.size() << " workers=" << workers << std::endl;

            // ---- Phase 1: screen every candidate (parallel pool) ----
            std::cout << "screening..." << std::endl;
            scored.resize(cands.size());
            std::atomic<int> next { 0 }, doneCount { 0 };
            std::mutex loadMx;   // serialise plugin instantiation (concurrent loads can crash)
            // Some plugins (M1) ignore setStateInformation on a reused instance, so a
            // pooled-reuse screen scores every candidate identically. Use a fresh
            // instance per candidate when --fresh-screen is set (needed for M1).
            const bool freshScreen = args.contains("--fresh-screen") || usePrograms;
            auto screenWorker = [&]() {
                juce::String e; std::unique_ptr<vms::IRenderTarget> inst;
                if (!freshScreen) { std::lock_guard<std::mutex> lk(loadMx); inst = factory(); if (!inst) return; }
                vms::Loss L;
                for (;;) {
                    const int j = next.fetch_add(1);
                    if (j >= (int) cands.size()) break;
                    juce::MemoryBlock st; double loss = 1.0e9;
                    if (freshScreen) { std::lock_guard<std::mutex> lk(loadMx); inst = factory(); }
                    if (inst && getState(cands[(size_t) j], st)) {
                        inst->setStateData(st.getData(), st.getSize());
                        for (int p = 0; p < inst->numParams(); ++p) {  // force determinism
                            const auto nm = inst->paramInfo(p).name.toLowerCase();
                            if (nm.contains("phase randomization") || nm.contains("random phase")
                                || (nm.contains("phase") && nm.contains("rand"))) inst->setParam(p, 0.0f);
                        }
                        auto a = inst->render(note, vel, dur, gate);
                        if (!inst->renderFailed() && vms::Loss::peakAbs(a) > 1.0e-4f)
                            loss = L.combined(target, a, 1.0f, 0.3f);
                    }
                    scored[(size_t) j] = { loss, j };
                    const int dc = doneCount.fetch_add(1) + 1;
                    if (dc % 25 == 0) std::cout << "  screened " << dc << "/" << cands.size() << std::endl;
                }
            };
            std::vector<std::thread> pool;
            for (int t = 0; t < workers; ++t) pool.emplace_back(screenWorker);
            for (auto& th : pool) th.join();

            std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) { return a.loss < b.loss; });
            std::cout << "top screened:" << std::endl;
            for (int i = 0; i < juce::jmin(topK, (int) scored.size()); ++i)
                std::cout << "  #" << (i + 1) << " loss=" << scored[(size_t) i].loss
                          << "  " << cands[(size_t) scored[(size_t) i].idx].label << std::endl;

            // ---- Phase 2: refine top-K with CMA-ES ----
            for (int i = 0; i < juce::jmin(topK, (int) scored.size()); ++i) {
                if (!std::isfinite(scored[(size_t) i].loss)) continue;
                const int ci = scored[(size_t) i].idx;
                juce::MemoryBlock st;
                if (!getState(cands[(size_t) ci], st)) continue;

                vms::MatchConfig cfg;
                cfg.sampleRate = sr; cfg.durSec = dur; cfg.midiNote = note; cfg.velocity = vel; cfg.gateFrac = gate;
                cfg.freeParams = freeParams; cfg.maxFevals = refineFev; cfg.numWorkers = workers;
                cfg.sigma0 = 0.2; cfg.seed = 1; cfg.warmStart = false; // base state IS the warm start
                cfg.initStateData = st; cfg.forceDeterministic = true;

                vms::MatchingEngine engine(factory, target, cfg);
                auto res = engine.run();
                std::cout << "  refined #" << (i + 1) << " " << cands[(size_t) ci].label
                          << " -> loss=" << res.bestLoss << std::endl;
                if (res.bestLoss < bestLoss) {
                    bestLoss = res.bestLoss; bestParams = res.bestFullParams; bestState = st; bestIdx = ci;
                }
            }

            // ---- Render + save the overall best ----
            if (bestIdx >= 0) {
                juce::String e; auto best = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
                if (best) {
                    best->setStateData(bestState.getData(), bestState.getSize());
                    for (int p = 0; p < best->numParams() && p < (int) bestParams.size(); ++p)
                        best->setParam(p, bestParams[(size_t) p]);
                    auto audio = best->render(note, vel, dur, gate);
                    vms::writeWavMono(juce::File(outPath), audio, sr);
                    if (auto* in = best->getInstance()) {
                        juce::MemoryBlock sm; in->getStateInformation(sm);
                        juce::File(outPath).withFileExtension("state").replaceWithData(sm.getData(), sm.getSize());
                    }
                    std::cout << "BEST: " << cands[(size_t) bestIdx].label
                              << "  loss=" << bestLoss << "  peak=" << vms::Loss::peakAbs(audio) << std::endl;
                }
            }
        }
        done = true;
    });
    while (!done.load()) juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    driver.join();
    std::cout << "Done -> " << outPath << std::endl;
    return 0;
}

static int runHeadlessMatch(const juce::StringArray& args) {
    // args: --headless-match <vst3> <wav> [--params i,j..] [--note N] [--maxfevals M] --out <wav>
    auto valAfter = [&](const juce::String& flag) -> juce::String {
        const int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String();
    };

    const int mi = args.indexOf("--headless-match");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String wav  = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    const juce::String outPath = valAfter("--out");

    if (vst3.isEmpty() || wav.isEmpty() || outPath.isEmpty()) {
        std::cout << "usage: --headless-match <vst3> <wav> [--params i,j] [--note N] [--maxfevals M] --out <wav>\n";
        return 2;
    }

    juce::AudioBuffer<float> target;
    double sr = 44100.0;
    if (!vms::loadAudioFileMono(juce::File(wav), target, sr)) {
        std::cout << "ERROR: could not load target wav: " << wav << "\n";
        return 3;
    }

    juce::String err;
    auto probe = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, err);
    if (probe == nullptr) {
        std::cout << "ERROR: could not load VST3: " << err << "\n";
        return 4;
    }
    const int numParams = probe->numParams();

    vms::MatchConfig cfg;
    cfg.sampleRate = sr;
    cfg.durSec = target.getNumSamples() / sr;

    const juce::String noteStr = valAfter("--note");
    if (noteStr.isNotEmpty()) cfg.midiNote = noteStr.getIntValue();
    else cfg.midiNote = vms::estimatePitch(target, sr).midiNote;

    const juce::String mf = valAfter("--maxfevals");
    if (mf.isNotEmpty()) cfg.maxFevals = mf.getLargeIntValue();

    const juce::String pl = valAfter("--params");
    if (pl.isNotEmpty()) cfg.freeParams = parseIntList(pl);
    else { cfg.freeParams.resize(numParams); for (int i = 0; i < numParams; ++i) cfg.freeParams[(size_t) i] = i; }

    const juce::String wk = valAfter("--workers");
    cfg.numWorkers = wk.isNotEmpty() ? juce::jmax(1, wk.getIntValue())
                                     : juce::jmax(1, juce::SystemStats::getNumPhysicalCpus() - 1);

    // Determinism: force per-note phase-randomization params to 0 (auto by name,
    // same as the GUI) plus any explicit --baseparams overrides. Passed as the
    // engine's frozen baseline so even fresh worker instances render deterministically.
    {
        std::vector<float> base((size_t) numParams);
        for (int i = 0; i < numParams; ++i) base[(size_t) i] = probe->getParam(i);
        int frozenRand = 0;
        for (int i = 0; i < numParams; ++i) {
            const auto nm = probe->paramInfo(i).name.toLowerCase();
            if (nm.contains("phase randomization") || nm.contains("random phase")
                || (nm.contains("phase") && nm.contains("rand"))) {
                base[(size_t) i] = 0.0f; ++frozenRand;
            }
        }
        const int bi = args.indexOf("--baseparams");
        if (bi >= 0 && bi + 1 < args.size()) {
            for (auto& p : juce::StringArray::fromTokens(args[bi + 1], ",", "")) {
                auto kv = juce::StringArray::fromTokens(p, "=", "");
                if (kv.size() == 2) {
                    const int idx = kv[0].getIntValue();
                    if (idx >= 0 && idx < numParams) base[(size_t) idx] = (float) kv[1].getDoubleValue();
                }
            }
        }
        cfg.baseParams = base;
        std::cout << "froze " << frozenRand << " phase-randomization param(s) -> deterministic" << std::endl;
    }

    if (args.contains("--warm"))   cfg.warmStart = true;
    if (args.contains("--screen")) cfg.screening = true;
    if (args.contains("--coarse")) cfg.coarseToFine = true;
    if (args.contains("--bayes"))  cfg.useBayesian = true;
    if (valAfter("--avg").isNotEmpty()) cfg.rendersPerEval = juce::jmax(1, valAfter("--avg").getIntValue());

    std::cout << "Matching '" << vst3 << "' to '" << wav << "'\n"
              << "  note=" << cfg.midiNote << " dur=" << cfg.durSec
              << " params=" << (int) cfg.freeParams.size()
              << " maxfevals=" << cfg.maxFevals << " workers=" << cfg.numWorkers << std::endl;

    auto factory = [vst3, sr]() -> std::unique_ptr<vms::IRenderTarget> {
        juce::String e;
        auto v = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
        return v; // unique_ptr<VstTarget> -> unique_ptr<IRenderTarget>
    };

    // Run the engine on a background thread while the main (message) thread pumps
    // the dispatch loop. This mirrors the GUI architecture and prevents the
    // deadlock that occurs when plugin processing needs the message thread while
    // the engine blocks it. (Hosting many heavy plugins like Vital relies on this.)
    vms::MatchResult res;
    std::atomic<bool> done { false };
    std::thread engineThread([&] {
        // Inner scope: the engine (and its plugin workers) must be DESTROYED while
        // the main thread is still pumping messages (done == false), because some
        // plugins' destructors dispatch to the message thread. Setting done only
        // after teardown avoids a shutdown deadlock.
        {
            vms::MatchingEngine engine(factory, target, cfg);
            res = engine.run([](int gen, double loss, const std::vector<float>&, const juce::AudioBuffer<float>&) {
                std::cout << "  gen " << gen << " bestLoss=" << loss << std::endl;
            });

            juce::String e2;
            auto best = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e2);
            if (best) {
                for (int i = 0; i < best->numParams(); ++i)
                    best->setParam(i, res.bestFullParams[(size_t) i]);
                auto bestAudio = best->render(cfg.midiNote, cfg.velocity, cfg.durSec);
                vms::writeWavMono(juce::File(outPath), bestAudio, sr);
                if (auto* inst = best->getInstance()) {
                    juce::MemoryBlock mb;
                    inst->getStateInformation(mb);
                    juce::File(outPath).withFileExtension("state").replaceWithData(mb.getData(), mb.getSize());
                }
            }
        } // engine + best destruct here, with the message loop still pumping
        done = true;
    });

    while (!done.load())
        juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    engineThread.join();

    std::cout << "Done. bestLoss=" << res.bestLoss << " -> " << outPath << "\n";
    return 0;
}

// Oracle test: render a target FROM the plugin itself (a sound it can obviously
// reproduce), then recover those parameters from the default patch.
static int runVstRecover(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& flag) -> juce::String {
        const int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String();
    };
    const int mi = args.indexOf("--vst-recover");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    if (vst3.isEmpty()) {
        std::cout << "usage: --vst-recover <vst3> [--params i,j] [--note N] [--maxfevals M]"
                     " [--workers W] [--seed S] [--dur D]\n";
        return 2;
    }

    const double sr = 44100.0;
    const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
    const double dur = valAfter("--dur").isNotEmpty() ? valAfter("--dur").getDoubleValue() : 1.5;
    const long maxfev = valAfter("--maxfevals").isNotEmpty() ? valAfter("--maxfevals").getLargeIntValue() : 8000;
    const int workers = valAfter("--workers").isNotEmpty() ? juce::jmax(1, valAfter("--workers").getIntValue()) : 3;
    const unsigned seed = valAfter("--seed").isNotEmpty() ? (unsigned) valAfter("--seed").getIntValue() : 12345u;
    const int numRecover = valAfter("--num").isNotEmpty() ? juce::jmax(1, valAfter("--num").getIntValue()) : 8;
    const juce::String paramsArg = valAfter("--params");

    const juce::File outDir = juce::File::getCurrentWorkingDirectory().getChildFile("artifacts");
    outDir.createDirectory();

    vms::MatchResult res;
    std::atomic<bool> done { false };
    std::thread th([&] {
        {
            juce::String e;
            auto probe = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
            if (probe == nullptr) { std::cout << "load failed: " << e << "\n"; done = true; return; }
            const int np = probe->numParams();
            std::cout << "Loaded " << probe->name() << " (" << np << " params)" << std::endl;

            std::vector<float> def((size_t) np);
            for (int i = 0; i < np; ++i) def[(size_t) i] = probe->getParam(i);

            // --baseparams "idx=val,..." : force some params (e.g. Vital phase-rand=0)
            // for BOTH the target patch and the search baseline, so a normally
            // non-deterministic synth renders deterministically.
            {
                const int bi = args.indexOf("--baseparams");
                if (bi >= 0 && bi + 1 < args.size()) {
                    auto pairs = juce::StringArray::fromTokens(args[bi + 1], ",", "");
                    for (auto& p : pairs) {
                        auto kv = juce::StringArray::fromTokens(p, "=", "");
                        if (kv.size() == 2) {
                            const int idx = kv[0].getIntValue();
                            const float val = (float) kv[1].getDoubleValue();
                            if (idx >= 0 && idx < np) {
                                def[(size_t) idx] = val;
                                std::cout << "baseparam: p" << idx << " = " << val << std::endl;
                            }
                        }
                    }
                }
            }

            vms::Loss L;

            // Render a full param vector on a FRESH instance (always a clean
            // "first render" — safe even for plugins like Synth1 that crash on reuse).
            auto renderFreshFull = [&](const std::vector<float>& full) -> juce::AudioBuffer<float> {
                juce::String ee;
                auto t = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, ee);
                if (t == nullptr) return {};
                const int tn = t->numParams();
                for (int i = 0; i < tn && i < (int) full.size(); ++i) t->setParam(i, full[(size_t) i]);
                return t->render(note, 100, dur);
            };

            // Detect crash-on-reuse: render twice on the SAME probe instance.
            probe->render(note, 100, dur);
            probe->render(note, 100, dur);
            const bool crashProne = probe->renderFailed();
            std::cout << (crashProne ? "plugin crashes on instance reuse -> using fresh-per-render"
                                     : "plugin survives instance reuse") << std::endl;

            // Determinism via two fresh instances (true cross-render variation).
            auto a = renderFreshFull(def);
            auto b = renderFreshFull(def);
            float md = 0.0f;
            const int dn = juce::jmin(a.getNumSamples(), b.getNumSamples());
            for (int i = 0; i < dn; ++i)
                md = juce::jmax(md, std::abs(a.getReadPointer(0)[i] - b.getReadPointer(0)[i]));
            std::cout << "determinism maxdiff=" << md
                      << (md < 1.0e-5f ? " (deterministic)" : " (NON-deterministic -> loss floor expected)")
                      << std::endl;

            // Choose which parameters to vary/recover.
            std::vector<int> recover;
            if (paramsArg.isNotEmpty()) {
                recover = parseIntList(paramsArg);
            } else {
                std::cout << "selecting tone-shaping params (skipping MIDI CC / LFO / mod)..." << std::endl;
                std::vector<int> candidates;
                for (int i = 0; i < np; ++i) {
                    const auto cat = vms::categorizeParam(probe->paramInfo(i).name);
                    if (cat == vms::ParamCat::Core || cat == vms::ParamCat::Secondary)
                        candidates.push_back(i);
                }
                std::cout << "  " << candidates.size() << " tone-shaping params of " << np << std::endl;

                if (crashProne) {
                    // Impact-probing would need a fresh instance per param (too slow);
                    // just take the first N tone-shaping params.
                    for (int i = 0; i < juce::jmin(numRecover, (int) candidates.size()); ++i)
                        recover.push_back(candidates[(size_t) i]);
                } else {
                    std::vector<std::pair<double, int>> imp;
                    for (int idx : candidates) {
                        for (int i = 0; i < np; ++i) probe->setParam(i, def[(size_t) i]);
                        probe->setParam(idx, juce::jlimit(0.0f, 1.0f, def[(size_t) idx] + 0.4f));
                        auto r = probe->render(note, 100, dur);
                        imp.push_back({ L.combined(a, r, 1.0f), idx });
                    }
                    std::sort(imp.begin(), imp.end(), [](auto& x, auto& y) { return x.first > y.first; });
                    for (int i = 0; i < juce::jmin(numRecover, (int) imp.size()); ++i)
                        recover.push_back(imp[(size_t) i].second);
                }
                std::sort(recover.begin(), recover.end());
            }
            if (recover.empty()) { std::cout << "no impactful params found\n"; done = true; return; }
            std::cout << "recover params:" << std::endl;
            for (int x : recover)
                std::cout << "  p" << x << " [" << vms::paramCatName(vms::categorizeParam(probe->paramInfo(x).name))
                          << "] " << probe->paramInfo(x).name << std::endl;

            // Build the "true" patch. --perturb keeps it MUSICAL by varying around
            // the (audible) default instead of fully random (random FM = often silent).
            const float perturb = valAfter("--perturb").getFloatValue(); // 0 = full random
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> u(0.1f, 0.9f);
            std::uniform_real_distribution<float> pert(-perturb, perturb);
            std::vector<float> trueVals;
            std::vector<float> targetParams = def;
            for (int idx : recover) {
                const float v = (perturb > 0.0f) ? juce::jlimit(0.0f, 1.0f, def[(size_t) idx] + pert(rng))
                                                 : u(rng);
                trueVals.push_back(v); targetParams[(size_t) idx] = v;
            }
            auto target = renderFreshFull(targetParams);
            vms::writeWavMono(outDir.getChildFile("recover_target.wav"), target, sr);
            const bool robust = args.contains("--robust");
            std::cout << "target peak=" << vms::Loss::peakAbs(target) << " (audible if > ~0.05)" << std::endl;
            std::cout << "start loss (default vs target)=" << L.combined(target, a, 1.0f, 0.3f, robust) << std::endl;

            // Self-noise floor: render the SAME (true) patch again and measure loss.
            // This is the BEST loss any matcher could reach — for a non-deterministic
            // plugin it is > 0 and bounds achievable accuracy.
            auto target2 = renderFreshFull(targetParams);
            const double selfFloor = L.combined(target, target2, 1.0f, 0.3f, robust);
            std::cout << "*** self-noise floor (same patch, 2 renders) = " << selfFloor
                      << (robust ? " [robust metric]" : "") << " (best loss any matcher can reach) ***" << std::endl;

            // Recover from the default patch.
            vms::MatchConfig cfg;
            cfg.sampleRate = sr; cfg.durSec = dur; cfg.midiNote = note; cfg.velocity = 100;
            cfg.freeParams = recover; cfg.maxFevals = maxfev; cfg.numWorkers = workers;
            cfg.sigma0 = 0.3; cfg.seed = 1; cfg.baseParams = def; cfg.robustLoss = robust;
            if (args.contains("--warm"))  cfg.warmStart = true;
            if (args.contains("--bayes")) cfg.useBayesian = true;
            if (valAfter("--avg").isNotEmpty()) cfg.rendersPerEval = juce::jmax(1, valAfter("--avg").getIntValue());
            auto factory = [vst3, sr]() -> std::unique_ptr<vms::IRenderTarget> {
                juce::String ee; return vms::VstTarget::loadAny(juce::File(vst3), sr, 512, ee);
            };
            vms::MatchingEngine engine(factory, target, cfg);
            res = engine.run([](int g, double l, const std::vector<float>&, const juce::AudioBuffer<float>&) {
                std::cout << "  gen " << g << " loss=" << l << std::endl;
            });

            std::cout << "final loss=" << res.bestLoss << std::endl;
            std::cout << "param errors (|recovered-true|):" << std::endl;
            double sumErr = 0.0;
            for (size_t k = 0; k < recover.size(); ++k) {
                const float rec = res.bestFullParams[(size_t) recover[k]];
                const double er = std::abs(rec - trueVals[k]);
                sumErr += er;
                std::cout << "  p" << recover[k] << " (" << probe->paramInfo(recover[k]).name << "): true="
                          << trueVals[k] << " rec=" << rec << " err=" << er << std::endl;
            }
            const double meanErr = recover.empty() ? 0.0 : sumErr / recover.size();
            const double paramScore = 100.0 * (1.0 - meanErr);          // 100 = exact params
            const double soundScore = 100.0 * std::exp(-2.0 * res.bestLoss); // 100 = identical sound
            std::cout << "==== SCORE ====" << std::endl;
            std::cout << "param score = " << juce::String(paramScore, 1) << "/100  (mean param err "
                      << juce::String(meanErr, 4) << ")" << std::endl;
            std::cout << "sound score = " << juce::String(soundScore, 1) << "/100  (loss "
                      << juce::String(res.bestLoss, 4) << ")" << std::endl;

            auto ba = renderFreshFull(res.bestFullParams);
            vms::writeWavMono(outDir.getChildFile("recover_best.wav"), ba, sr);
            if (engine.crashedDuringRun())
                std::cout << "(note: plugin crashed during run; auto-recovered via fresh-per-render)" << std::endl;
        }
        done = true;
    });
    while (!done.load()) juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    th.join();
    std::cout << "Done. target=artifacts/recover_target.wav  best=artifacts/recover_best.wav" << std::endl;
    return 0;
}

// Diagnose: (1) the render-to-render spectral noise floor of a plugin, and
// (2) whether setParam actually changes the rendered sound vs that floor.
static int runVstDiag(const juce::StringArray& args) {
    const int mi = args.indexOf("--vst-diag");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    if (vst3.isEmpty()) { std::cout << "usage: --vst-diag <vst3>\n"; return 2; }
    const double sr = 44100.0; const int note = 60; const double dur = 1.5;

    // Run on the message thread (no background thread) to test message-thread render.
    {
        {
            juce::String e;
            auto v = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
            if (!v) { std::cout << "load failed: " << e << "\n"; return 4; }
            const int np = v->numParams();
            std::cout << "Loaded " << v->name() << " (" << np << " params)" << std::endl;
            std::vector<float> def((size_t) np);
            for (int i = 0; i < np; ++i) def[(size_t) i] = v->getParam(i);

            // --baseparams "idx=val,idx=val,..." : force some params before the determinism test.
            {
                const int bi = args.indexOf("--baseparams");
                if (bi >= 0 && bi + 1 < args.size()) {
                    auto pairs = juce::StringArray::fromTokens(args[bi + 1], ",", "");
                    for (auto& p : pairs) {
                        auto kv = juce::StringArray::fromTokens(p, "=", "");
                        if (kv.size() == 2) {
                            const int idx = kv[0].getIntValue();
                            const float val = (float) kv[1].getDoubleValue();
                            if (idx >= 0 && idx < np) {
                                def[(size_t) idx] = val;
                                std::cout << "baseparam: p" << idx << " = " << val << std::endl;
                            }
                        }
                    }
                }
            }

            vms::Loss L;
            auto renderDefault = [&] {
                for (int i = 0; i < np; ++i) v->setParam(i, def[(size_t) i]);
                return v->render(note, 100, dur);
            };
            std::cout << "rendering #1..." << std::endl;
            auto d1 = renderDefault();
            if (v->renderFailed()) { std::cout << "PLUGIN CRASHED on render #1 (SEH-caught). Unusable for matching.\n"; return 5; }
            std::cout << "  #1 ok (peak=" << vms::Loss::peakAbs(d1) << ")" << std::endl;

            // Reuse test (render #2 on SAME instance).
            std::cout << "render #2 on SAME instance..." << std::endl;
            auto d2 = renderDefault();
            std::cout << (v->renderFailed() ? "  reuse FAILED (crashed)" : "  reuse ok") << std::endl;

            // Fresh-instance test (render once on a NEW instance).
            std::cout << "render on a FRESH instance..." << std::endl;
            juce::String e2;
            auto v2 = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e2);
            if (v2) {
                auto d3 = v2->render(note, 100, dur);
                std::cout << (v2->renderFailed() ? "  fresh-instance FAILED (crashed)"
                              : ("  fresh-instance ok (peak=" + juce::String(vms::Loss::peakAbs(d3), 4) + ")").toStdString())
                          << std::endl;
            }
            if (v->renderFailed()) {
                std::cout << "=> SAME-instance reuse crashes; fresh-per-render is the workaround.\n";
                return 5;
            }
            const double floorDD = L.combined(d1, d2, 1.0f);
            std::cout << "noise floor (default vs default, same patch) = " << floorDD << std::endl;

            // Test several params: does a big change exceed the floor?
            std::cout << "param-change effect vs noise floor:" << std::endl;
            const int step = juce::jmax(1, np / 12);
            for (int idx = 0; idx < np; idx += step) {
                for (int i = 0; i < np; ++i) v->setParam(i, def[(size_t) i]);
                v->setParam(idx, 0.05f);
                auto lo = v->render(note, 100, dur);
                for (int i = 0; i < np; ++i) v->setParam(i, def[(size_t) i]);
                v->setParam(idx, 0.95f);
                auto hi = v->render(note, 100, dur);
                const double effect = L.combined(lo, hi, 1.0f);
                const double ratio = effect / juce::jmax(1.0e-9, floorDD);
                std::cout << "  p" << idx << ": effect(lo vs hi)=" << effect
                          << "  ratio_vs_floor=" << ratio
                          << (ratio > 3.0 ? "  <-- real effect" : "") << std::endl;
            }
        }
    }
    return 0;
}

// ---- Vital preset injection ------------------------------------------------
// Vital's wavetables aren't automatable params, but a .vital preset (plain JSON,
// wavetable included) CAN be loaded by wrapping it in the VST3 component-state
// framing that JUCE's host expects. We derive the framing from the LIVE plugin's
// own getStateInformation (so it matches the exact version), then splice the new
// JSON in and fix the chunk size fields.
static uint32_t readBE32(const uint8_t* p) {
    return ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) | ((uint32_t) p[2] << 8) | (uint32_t) p[3];
}
static void writeBE32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t) (v >> 24); p[1] = (uint8_t) (v >> 16); p[2] = (uint8_t) (v >> 8); p[3] = (uint8_t) v;
}
static int findBytesIn(const char* base, int n, const char* pat, int from = 0) {
    const int pl = (int) std::strlen(pat);
    for (int i = juce::jmax(0, from); i + pl <= n; ++i) { int k = 0; while (k < pl && base[i + k] == pat[k]) ++k; if (k == pl) return i; }
    return -1;
}

// Build a full JUCE VST3 state blob that loads `vitalJson`, using a captured
// plugin state `tmpl` (from getStateInformation) as the framing template.
static bool buildVitalState(const juce::MemoryBlock& tmpl, const juce::MemoryBlock& vitalJson,
                            juce::MemoryBlock& outState) {
    const char* tb = (const char*) tmpl.getData();
    const int tn = (int) tmpl.getSize();
    const int at = findBytesIn(tb, tn, "<IComponent>");
    const int bt = findBytesIn(tb, tn, "</IComponent>", at);
    if (at < 0 || bt < 0) return false;
    const int ps = at + 12;
    const juce::String payload = juce::String::fromUTF8(tb + ps, bt - ps);

    juce::MemoryBlock C;
    if (!C.fromBase64Encoding(payload)) return false;
    uint8_t* c = (uint8_t*) C.getData();
    const int cn = (int) C.getSize();

    const int ccnk = findBytesIn((const char*) c, cn, "CcnK");
    const int vita = findBytesIn((const char*) c, cn, "Vita");
    if (ccnk < 0 || vita < 0) return false;
    // First '{' after the Vita header = JSON start; the BE32 just before it is the chunk size.
    int jStart = -1;
    for (int i = vita; i < cn; ++i) if (c[i] == '{') { jStart = i; break; }
    if (jStart < 4) return false;
    // Last '}' = JSON end; trailing bytes (padding + JUCEPrivateData) are kept verbatim.
    int jEnd = -1;
    for (int i = cn - 1; i >= jStart; --i) if (c[i] == '}') { jEnd = i + 1; break; }
    if (jEnd < 0) return false;
    const int trailLen = cn - jEnd;

    // Assemble: [header up to jStart] + [new JSON] + [original trailing].
    const int newContentLen = (int) vitalJson.getSize() + trailLen;
    const int newCn = jStart + newContentLen;
    juce::MemoryBlock newC; newC.setSize((size_t) newCn);
    uint8_t* nc = (uint8_t*) newC.getData();
    memcpy(nc, c, (size_t) jStart);                                   // header
    memcpy(nc + jStart, vitalJson.getData(), vitalJson.getSize());    // new JSON
    memcpy(nc + jStart + vitalJson.getSize(), c + jEnd, (size_t) trailLen); // trailing

    // Fix size fields: BE32 right before JSON = content length from jStart;
    // CcnK size = total - (ccnk + 8).
    writeBE32(nc + (jStart - 4), (uint32_t) newContentLen);
    writeBE32(nc + (ccnk + 4), (uint32_t) (newCn - (ccnk + 8)));

    // Re-encode and reassemble the XML envelope (toBase64Encoding emits "len.data").
    const juce::String newPayload = newC.toBase64Encoding();
    juce::MemoryOutputStream os;
    os.write(tb, (size_t) ps);                       // up to and including "<IComponent>"
    os.write(newPayload.toRawUTF8(), newPayload.getNumBytesAsUTF8());
    os.write(tb + bt, (size_t) (tn - bt));           // "</IComponent>" .. end
    outState.setSize(0);
    outState.append(os.getData(), os.getDataSize());
    // Fix the JUCE "VC2!" header size field (LE32 at offset 4 = totalSize - 9),
    // which we inherited from the template and must recompute for the new length.
    if (outState.getSize() > 8) {
        uint8_t* o = (uint8_t*) outState.getData();
        if (o[0] == 'V' && o[1] == 'C' && o[2] == '2' && o[3] == '!') {
            const uint32_t v = (uint32_t) (outState.getSize() - 9);
            o[4] = (uint8_t) v; o[5] = (uint8_t) (v >> 8); o[6] = (uint8_t) (v >> 16); o[7] = (uint8_t) (v >> 24);
        }
    }
    return true;
}

// Inspect a plugin's program list (VST3 program-change units). Optionally select a
// program and render — for ROMplers (M1) whose waveforms are chosen by program.
static int runPrograms(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& f) -> juce::String {
        const int i = args.indexOf(f); return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String(); };
    const int mi = args.indexOf("--programs");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    if (vst3.isEmpty()) { std::cout << "usage: --programs <vst3> [--prog N] [--note M] [--out wav]\n"; return 2; }
    juce::String e; auto v = vms::VstTarget::loadAny(juce::File(vst3), 44100.0, 512, e);
    if (!v) { std::cout << "load failed: " << e << "\n"; return 4; }
    auto* inst = v->getInstance();
    inst->setNonRealtime(true); inst->prepareToPlay(44100.0, 512);
    std::cout << "outChannels=" << inst->getTotalNumOutputChannels()
              << " buses=" << inst->getBusCount(false)
              << " latencySamples=" << inst->getLatencySamples()
              << " tailLengthSec=" << inst->getTailLengthSeconds() << std::endl;
    // Some plugins only initialise their audio engine once an editor exists.
    std::unique_ptr<juce::AudioProcessorEditor> ed;
    if (args.contains("--editor")) {
        ed.reset(inst->createEditorIfNeeded());
        std::cout << "created editor=" << (ed != nullptr ? "yes" : "no") << std::endl;
    }
    const int n = inst->getNumPrograms();
    std::cout << "getNumPrograms = " << n << "  current=" << inst->getCurrentProgram() << std::endl;
    for (int i = 0; i < juce::jmin(n, 40); ++i)
        std::cout << "  prog " << i << ": " << inst->getProgramName(i) << std::endl;
    // Try driving sound selection via real MIDI bank-select + program-change messages
    // (some ROMplers only switch sound on MIDI PC, not setCurrentProgram).
    const juce::String mp = valAfter("--midiprog");
    if (mp.isNotEmpty()) {
        const juce::String inSt = valAfter("--in-state");
        if (inSt.isNotEmpty()) { juce::MemoryBlock s; if (juce::File(inSt).loadFileAsData(s)) { inst->setStateInformation(s.getData(), (int) s.getSize()); std::cout << "loaded state " << inSt << "\n"; } }
        inst->setNonRealtime(true);
        inst->prepareToPlay(44100.0, 512);
        const int prog = mp.getIntValue();
        const int bank = valAfter("--bank").isNotEmpty() ? valAfter("--bank").getIntValue() : 0;
        const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
        const int nOut = juce::jmax(2, inst->getTotalNumOutputChannels());
        juce::AudioBuffer<float> buf(nOut, 512); float pk = 0.0f;
        for (int b = 0; b < 120; ++b) {
            buf.clear(); juce::MidiBuffer m;
            if (b == 0) {
                if (!args.contains("--nopc")) {
                    m.addEvent(juce::MidiMessage::controllerEvent(1, 0, bank), 0);   // bank MSB
                    m.addEvent(juce::MidiMessage::controllerEvent(1, 32, 0), 1);     // bank LSB
                    m.addEvent(juce::MidiMessage::programChange(1, prog), 2);
                }
                m.addEvent(juce::MidiMessage::noteOn(1, note, (juce::uint8) 100), 10);
            }
            if (args.contains("--withreset") && b == 0) inst->reset();
            inst->processBlock(buf, m);
            for (int ch = 0; ch < buf.getNumChannels(); ++ch)
                pk = juce::jmax(pk, buf.getMagnitude(ch, 0, buf.getNumSamples()));
        }
        std::cout << "MIDI prog=" << prog << " bank=" << bank << " note=" << note << " peak=" << pk << std::endl;
        // Re-render capturing a mono buffer for spectral inspection.
        const juce::String outp = valAfter("--out");
        if (outp.isNotEmpty()) {
            inst->reset();
            juce::AudioBuffer<float> mono(1, 512 * 120);
            for (int b = 0; b < 120; ++b) {
                buf.clear(); juce::MidiBuffer m;
                if (b == 0) {
                    m.addEvent(juce::MidiMessage::controllerEvent(1, 0, bank), 0);
                    m.addEvent(juce::MidiMessage::programChange(1, prog), 2);
                    m.addEvent(juce::MidiMessage::noteOn(1, note, (juce::uint8) 100), 10);
                }
                inst->processBlock(buf, m);
                for (int i = 0; i < 512; ++i) { float s = 0; for (int ch = 0; ch < buf.getNumChannels(); ++ch) s += buf.getReadPointer(ch)[i]; mono.setSample(0, b * 512 + i, s / buf.getNumChannels()); }
            }
            vms::writeWavMono(juce::File(outp), mono, 44100.0);
        }
        return 0;
    }

    const juce::String saveSt = valAfter("--save-state");
    const juce::String pa = valAfter("--prog");
    if (pa.isNotEmpty()) {
        inst->setCurrentProgram(pa.getIntValue());
        // Round-trip the state so the program's PCM actually loads (live setCurrentProgram
        // then render is silent on M1; save+reload commits the sound).
        { juce::MemoryBlock mb; inst->getStateInformation(mb); inst->setStateInformation(mb.getData(), (int) mb.getSize()); }
        std::cout << "set program -> " << inst->getProgramName(pa.getIntValue()) << std::endl;
        const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
        const double dur = valAfter("--dur").isNotEmpty() ? valAfter("--dur").getDoubleValue() : 1.4;
        auto a = v->render(note, 100, dur);
        std::cout << "rendered peak=" << vms::Loss::peakAbs(a) << std::endl;
        const juce::String outp = valAfter("--out");
        if (outp.isNotEmpty()) vms::writeWavMono(juce::File(outp), a, 44100.0);
        // --target <wav>: print this program's loss vs a target (subprocess-per-program
        // M1 search — M1 only switches program on the FIRST instance per OS process).
        const juce::String tgt = valAfter("--target");
        if (tgt.isNotEmpty()) {
            juce::AudioBuffer<float> tb; double tsr = 44100.0;
            if (vms::loadAudioFileMono(juce::File(tgt), tb, tsr)) {
                vms::Loss L;
                const double loss = (vms::Loss::peakAbs(a) > 1.0e-4f) ? L.combined(tb, a, 1.0f, 0.3f) : 1.0e9;
                std::cout << "PROGLOSS prog=" << pa.getIntValue() << " loss=" << loss
                          << " name=" << inst->getProgramName(pa.getIntValue()) << std::endl;
            }
        }
    }
    if (saveSt.isNotEmpty()) {
        juce::MemoryBlock mb; inst->getStateInformation(mb);
        juce::File(saveSt).replaceWithData(mb.getData(), mb.getSize());
        std::cout << "saved state (" << mb.getSize() << " bytes) -> " << saveSt << std::endl;
    }
    return 0;
}

// Build a Vital state with a factory wavetable injected into oscillator 1 (and 2/3
// optionally), from a base .vital JSON. Returns false on failure.
static bool injectWavetable(const juce::String& baseJson, const juce::var& wavetableVar,
                            const juce::MemoryBlock& tmpl, juce::MemoryBlock& outState, int oscIndex = 0) {
    juce::var base = juce::JSON::parse(baseJson);
    if (!base.isObject()) return false;
    auto* settings = base["settings"].getDynamicObject();
    if (!settings) return false;
    juce::var wts = settings->getProperty("wavetables");
    auto* arr = wts.getArray();
    if (!arr || oscIndex >= arr->size()) return false;
    arr->set(oscIndex, wavetableVar);
    const juce::String outJson = juce::JSON::toString(base, true);
    juce::MemoryBlock jb; jb.append(outJson.toRawUTF8(), outJson.getNumBytesAsUTF8());
    return buildVitalState(tmpl, jb, outState);
}

// Inject a curated set of modulation ROUTINGS (source->destination) into a base .vital
// JSON. The AMOUNT of each is the VST param "Modulation N Amount" (slot N), so the staged
// search can dial each in (0 = off). Routing destination names are Vital's internal names
// harvested from the factory presets.
static juce::String injectModRoutings(const juce::String& baseJson) {
    static const char* routes[][2] = {
        { "lfo_1", "filter_1_cutoff" }, { "lfo_1", "osc_1_wave_frame" }, { "lfo_1", "osc_1_level" },
        { "lfo_2", "osc_1_transpose" }, { "lfo_2", "osc_1_wave_frame" }, { "env_2", "filter_1_cutoff" },
        { "env_2", "osc_1_level" },     { "env_3", "osc_2_level" },      { "lfo_1", "filter_1_resonance" },
        { "lfo_2", "osc_1_distortion_amount" },
    };
    const int nR = (int) (sizeof(routes) / sizeof(routes[0]));
    juce::var base = juce::JSON::parse(baseJson);
    auto* settings = base.isObject() ? base["settings"].getDynamicObject() : nullptr;
    if (!settings) return baseJson;
    juce::Array<juce::var> mods;
    for (int i = 0; i < nR; ++i) {
        auto* o = new juce::DynamicObject();
        o->setProperty("source", juce::String(routes[i][0]));
        o->setProperty("destination", juce::String(routes[i][1]));
        mods.add(juce::var(o));
    }
    settings->setProperty("modulations", mods);
    return juce::JSON::toString(base, true);
}

// Extract the plugin's native preset JSON (the .vital content) from its current state.
static juce::String extractVitalJson(juce::AudioPluginInstance* inst) {
    juce::MemoryBlock tmpl; inst->getStateInformation(tmpl);
    const char* tb = (const char*) tmpl.getData(); const int tn = (int) tmpl.getSize();
    const int at = findBytesIn(tb, tn, "<IComponent>");
    const int bt = findBytesIn(tb, tn, "</IComponent>", at);
    if (at < 0 || bt < 0) return {};
    juce::MemoryBlock C;
    if (!C.fromBase64Encoding(juce::String::fromUTF8(tb + at + 12, bt - (at + 12)))) return {};
    const char* c = (const char*) C.getData(); const int cn = (int) C.getSize();
    int js = -1; for (int i = 0; i < cn; ++i) if (c[i] == '{') { js = i; break; }
    int je = -1; for (int i = cn - 1; i >= js; --i) if (c[i] == '}') { je = i + 1; break; }
    if (js < 0 || je < 0) return {};
    return juce::String::fromUTF8(c + js, je - js);
}

// Test: extract default JSON, inject a factory .vitaltable into oscillator 1, render.
static int runInjectWt(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& f) -> juce::String {
        const int i = args.indexOf(f); return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String(); };
    const int mi = args.indexOf("--inject-wt");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String wt = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    if (vst3.isEmpty() || wt.isEmpty()) { std::cout << "usage: --inject-wt <vst3> <vitaltable> [--note N] [--out wav]\n"; return 2; }
    juce::String e; auto v = vms::VstTarget::loadAny(juce::File(vst3), 44100.0, 512, e);
    if (!v) { std::cout << "load failed: " << e << "\n"; return 4; }

    const juce::String baseJson = extractVitalJson(v->getInstance());
    if (baseJson.isEmpty()) { std::cout << "extractVitalJson FAILED\n"; return 5; }
    std::cout << "base JSON chars=" << baseJson.length() << std::endl;
    juce::var base = juce::JSON::parse(baseJson);
    juce::var wtv  = juce::JSON::parse(juce::File(wt).loadFileAsString());
    if (!base.isObject() || !wtv.isObject()) { std::cout << "JSON parse FAILED\n"; return 5; }

    auto* settings = base["settings"].getDynamicObject();
    if (!settings) { std::cout << "no settings object\n"; return 5; }
    juce::var wts = settings->getProperty("wavetables");
    if (auto* arr = wts.getArray()) {
        if (arr->size() > 0) arr->set(0, wtv);   // replace oscillator 1's wavetable
        std::cout << "wavetables array size=" << arr->size() << ", injected into [0]" << std::endl;
    } else { std::cout << "no wavetables array\n"; return 5; }

    const juce::String outJson = juce::JSON::toString(base, true);
    juce::MemoryBlock jb; jb.append(outJson.toRawUTF8(), outJson.getNumBytesAsUTF8());
    juce::MemoryBlock tmpl; v->getInstance()->getStateInformation(tmpl);
    juce::MemoryBlock state;
    if (!buildVitalState(tmpl, jb, state)) { std::cout << "buildVitalState FAILED\n"; return 5; }
    v->setStateData(state.getData(), state.getSize());

    const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
    auto a = v->render(note, 100, 1.0);
    std::cout << "rendered peak=" << vms::Loss::peakAbs(a) << std::endl;
    const juce::String outp = valAfter("--out");
    if (outp.isNotEmpty()) vms::writeWavMono(juce::File(outp), a, 44100.0);
    return 0;
}

// FULL Vital match: search the factory WAVETABLE library (inject each, screen by loss),
// then staged-optimise the synth parameters on top of the best wavetable(s). Uses ALL of
// Vital's tone-shaping elements (its real wavetables + osc/filter/env/fx/mod-amount params).
//   --match-full <vst3> <wav> --wavetables <dir> [--note N] [--screen S] [--top K]
//                [--fevals M] [--workers W] --out <wav>
static int runMatchFull(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& f) -> juce::String {
        const int i = args.indexOf(f); return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String(); };
    const int mi = args.indexOf("--match-full");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String wav  = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    const juce::String wtDir = valAfter("--wavetables");
    const juce::String outPath = valAfter("--out");
    if (vst3.isEmpty() || wav.isEmpty() || wtDir.isEmpty() || outPath.isEmpty()) {
        std::cout << "usage: --match-full <vst3> <wav> --wavetables <dir> [--note N] [--screen S] [--top K] [--fevals M] [--workers W] --out <wav>\n"; return 2; }

    juce::AudioBuffer<float> target; double sr = 44100.0;
    if (!vms::loadAudioFileMono(juce::File(wav), target, sr)) { std::cout << "ERROR: load wav\n"; return 3; }
    const int screenMax = valAfter("--screen").isNotEmpty() ? juce::jmax(1, valAfter("--screen").getIntValue()) : 200;
    const int topK = valAfter("--top").isNotEmpty() ? juce::jmax(1, valAfter("--top").getIntValue()) : 4;
    const long fevals = valAfter("--fevals").isNotEmpty() ? valAfter("--fevals").getLargeIntValue() : 1500;
    const int workers = valAfter("--workers").isNotEmpty() ? juce::jmax(1, valAfter("--workers").getIntValue()) : 4;
    const double dur = target.getNumSamples() / sr;
    const int baseNote = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue()
                                                         : vms::estimatePitch(target, sr).midiNote;

    // Wavetable sources: .vitaltable files AND wavetables harvested from .vital presets
    // (Vital's built-in factory wavetables live inside presets, not as files on disk).
    struct WtCand { juce::File file; bool isPreset; };
    juce::Array<juce::File> tableFiles, presetFiles;
    juce::File(wtDir).findChildFiles(tableFiles, juce::File::findFiles, true, "*.vitaltable");
    juce::File(wtDir).findChildFiles(presetFiles, juce::File::findFiles, true, "*.vital");
    std::vector<WtCand> all;
    for (auto& f : tableFiles)  all.push_back({ f, false });
    for (auto& f : presetFiles) all.push_back({ f, true });
    if (all.empty()) { std::cout << "no .vitaltable/.vital under " << wtDir << "\n"; return 3; }
    std::vector<WtCand> wts;
    if ((int) all.size() <= screenMax) wts = all;
    else { const double st = (double) all.size() / screenMax; for (int i = 0; i < screenMax; ++i) wts.push_back(all[(size_t) (int) (i * st)]); }
    // Parse a candidate into its oscillator-1 wavetable var.
    auto getWtVar = [](const WtCand& c) -> juce::var {
        juce::var v = juce::JSON::parse(c.file.loadFileAsString());
        if (!c.isPreset) return v;                       // .vitaltable IS a wavetable element
        if (auto* s = v["settings"].getDynamicObject()) {
            juce::var wtsv = s->getProperty("wavetables");
            if (auto* arr = wtsv.getArray()) if (arr->size() > 0) return (*arr)[0];
        }
        return juce::var();
    };
    std::cout << "found " << tableFiles.size() << " .vitaltable + " << presetFiles.size()
              << " presets; screening " << wts.size() << " wavetables  note=" << baseNote << " dur=" << dur << std::endl;

    struct Scored { double loss; int idx; };
    std::vector<Scored> scored;
    double bestLoss = std::numeric_limits<double>::infinity();
    std::vector<float> bestParams; juce::MemoryBlock bestState; int bestIdx = -1; int bestNote = baseNote, bestVel = 100; float bestGate = 0.7f;

    std::atomic<bool> done { false };
    std::thread driver([&] {
        {
            juce::String e0; auto probe = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e0);
            if (!probe) { std::cout << "load failed: " << e0 << "\n"; done = true; return; }
            juce::String baseJson = extractVitalJson(probe->getInstance());
            juce::MemoryBlock tmpl; probe->getInstance()->getStateInformation(tmpl);
            probe.reset();
            if (baseJson.isEmpty()) { std::cout << "extractVitalJson failed\n"; done = true; return; }
            // Inject candidate LFO/env routings; the staged LFO/MOD stage tunes the amounts.
            baseJson = injectModRoutings(baseJson);
            std::cout << "injected modulation routings (amounts searched in LFO/MOD stage)" << std::endl;

            // ---- Phase 1: screen wavetables (inject -> render default -> loss) ----
            std::cout << "screening wavetables..." << std::endl;
            scored.resize(wts.size());
            for (size_t j = 0; j < wts.size(); ++j) {
                double loss = 1.0e9;
                juce::var wtv = getWtVar(wts[j]);
                juce::MemoryBlock st;
                if (wtv.isObject() && injectWavetable(baseJson, wtv, tmpl, st)) {
                    juce::String e; auto inst = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
                    if (inst) {
                        inst->setStateData(st.getData(), st.getSize());
                        for (int p = 0; p < inst->numParams(); ++p) {
                            const auto nm = inst->paramInfo(p).name.toLowerCase();
                            if (nm.contains("phase randomization") || nm.contains("random phase")) inst->setParam(p, 0.0f);
                        }
                        vms::Loss L; auto a = inst->render(baseNote, 100, dur);
                        if (!inst->renderFailed() && vms::Loss::peakAbs(a) > 1.0e-4f) loss = L.combined(target, a, 1.0f, 0.3f);
                    }
                }
                scored[j] = { loss, (int) j };
                if ((j + 1) % 20 == 0) std::cout << "  screened " << (j + 1) << "/" << wts.size() << std::endl;
            }
            std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) { return a.loss < b.loss; });
            std::cout << "top wavetables:" << std::endl;
            for (int i = 0; i < juce::jmin(topK, (int) scored.size()); ++i)
                std::cout << "  #" << (i + 1) << " loss=" << scored[(size_t) i].loss
                          << "  " << wts[(size_t) scored[(size_t) i].idx].file.getFileName() << std::endl;

            // ---- Phase 2: staged param optimisation on each top-K wavetable ----
            for (int i = 0; i < juce::jmin(topK, (int) scored.size()); ++i) {
                if (!std::isfinite(scored[(size_t) i].loss)) continue;
                const auto& wf = wts[(size_t) scored[(size_t) i].idx];
                juce::var wtv = getWtVar(wf);
                juce::MemoryBlock st;
                if (!wtv.isObject() || !injectWavetable(baseJson, wtv, tmpl, st)) continue;

                // Factory that produces instances pre-loaded with this wavetable.
                auto wtFactory = [vst3, sr, st]() -> std::unique_ptr<vms::IRenderTarget> {
                    juce::String e; auto v = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
                    if (v) v->setStateData(st.getData(), st.getSize());
                    return v;
                };
                vms::StagedConfig scfg;
                scfg.sampleRate = sr; scfg.durSec = dur; scfg.baseNote = baseNote;
                scfg.fevalsPerStage = fevals; scfg.numWorkers = workers; scfg.searchPerformance = true;
                if (valAfter("--passes").isNotEmpty()) scfg.passes = juce::jmax(1, valAfter("--passes").getIntValue());
                vms::StagedMatcher m;
                m.onStage = [i](int si, int sc, const juce::String& nm, double loss) {
                    std::cout << "    [#" << (i + 1) << "] " << si << "/" << sc << " " << nm
                              << " loss=" << loss << std::endl; };
                auto res = m.run(wtFactory, target, scfg);
                std::cout << "  refined #" << (i + 1) << " " << wf.file.getFileName() << " -> loss=" << res.bestLoss << std::endl;
                if (res.bestLoss < bestLoss) {
                    bestLoss = res.bestLoss; bestParams = res.bestParams; bestState = st;
                    bestIdx = scored[(size_t) i].idx; bestNote = res.note; bestVel = res.velocity; bestGate = res.gate;
                }
            }

            // ---- Render + save the best ----
            if (bestIdx >= 0) {
                juce::String e; auto best = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
                if (best) {
                    best->setStateData(bestState.getData(), bestState.getSize());
                    for (int p = 0; p < best->numParams() && p < (int) bestParams.size(); ++p) best->setParam(p, bestParams[(size_t) p]);
                    auto a = best->render(bestNote, bestVel, dur, bestGate);
                    vms::writeWavMono(juce::File(outPath), a, sr);
                    juce::MemoryBlock sm; best->getInstance()->getStateInformation(sm);
                    juce::File(outPath).withFileExtension("state").replaceWithData(sm.getData(), sm.getSize());
                    std::cout << "BEST wavetable: " << wts[(size_t) bestIdx].file.getFileName()
                              << "  loss=" << bestLoss << "  note=" << bestNote << " peak=" << vms::Loss::peakAbs(a) << std::endl;
                }
            }
        }
        done = true;
    });
    while (!done.load()) juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    driver.join();
    std::cout << "Done -> " << outPath << std::endl;
    return 0;
}

// Test: wrap a .vital into a loadable state, apply it, render -> verify timbre changes.
static int runWrapTest(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& f) -> juce::String {
        const int i = args.indexOf(f); return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String(); };
    const int mi = args.indexOf("--wrap-test");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String vital = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    if (vst3.isEmpty() || vital.isEmpty()) { std::cout << "usage: --wrap-test <vst3> <vital> [--note N] [--out wav]\n"; return 2; }
    juce::String e; auto v = vms::VstTarget::loadAny(juce::File(vst3), 44100.0, 512, e);
    if (!v) { std::cout << "load failed: " << e << "\n"; return 4; }
    juce::MemoryBlock vj; if (!juce::File(vital).loadFileAsData(vj)) { std::cout << "cannot read vital\n"; return 3; }
    juce::MemoryBlock tmpl; v->getInstance()->getStateInformation(tmpl);
    juce::MemoryBlock state;
    if (!buildVitalState(tmpl, vj, state)) { std::cout << "buildVitalState FAILED\n"; return 5; }
    std::cout << "built state bytes=" << state.getSize() << " (from .vital " << vj.getSize() << ")\n";
    const juce::String outState = valAfter("--out-state");
    if (outState.isNotEmpty()) { juce::File(outState).replaceWithData(state.getData(), state.getSize()); std::cout << "wrote state " << outState << "\n"; }
    const bool ok = v->setStateData(state.getData(), state.getSize());
    std::cout << "setStateData ok=" << (ok ? "yes" : "no") << std::endl;
    // Did params actually change? (50=Env Decay, 104=Filter Cutoff, 382=Osc1 Level)
    for (int p : { 48, 50, 104, 382, 462 })
        if (p < v->numParams())
            std::cout << "  p" << p << " " << v->paramInfo(p).name << " = " << v->getParam(p) << std::endl;
    const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 50;
    auto a = v->render(note, 100, 0.8);
    std::cout << "rendered peak=" << vms::Loss::peakAbs(a) << std::endl;
    const juce::String outp = valAfter("--out");
    if (outp.isNotEmpty()) vms::writeWavMono(juce::File(outp), a, 44100.0);
    return 0;
}

// Decode a saved JUCE VST3 .state: extract the IComponent payload (JUCE
// MemoryBlock::toBase64Encoding "len.data" form) and dump the plugin's raw
// component bytes, to learn the plugin's native state format (JSON? compressed?).
static int runDecodeState(const juce::StringArray& args) {
    const int mi = args.indexOf("--decode-state");
    const juce::String sf = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    if (sf.isEmpty()) { std::cout << "usage: --decode-state <statefile> [--out <bin>]\n"; return 2; }
    juce::MemoryBlock raw;
    if (!juce::File(sf).loadFileAsData(raw)) { std::cout << "cannot read\n"; return 3; }
    // Byte-scan for the markers (the file has a binary prefix that breaks toString()).
    const char* base = (const char*) raw.getData();
    const int n = (int) raw.getSize();
    auto findBytes = [&](const char* pat, int from) -> int {
        const int pl = (int) std::strlen(pat);
        for (int i = from; i + pl <= n; ++i) { int k = 0; while (k < pl && base[i + k] == pat[k]) ++k; if (k == pl) return i; }
        return -1;
    };
    const int at = findBytes("<IComponent>", 0);
    const int bt = findBytes("</IComponent>", at < 0 ? 0 : at);
    if (at < 0 || bt < 0) { std::cout << "no IComponent tag\n"; return 3; }
    const int ps = at + 12;
    const juce::String payload = juce::String::fromUTF8(base + ps, bt - ps);
    std::cout << "payload chars=" << payload.length() << " head=" << payload.substring(0, 30) << std::endl;
    juce::MemoryBlock decoded;
    if (!decoded.fromBase64Encoding(payload)) { std::cout << "fromBase64Encoding FAILED\n"; return 4; }
    std::cout << "decoded bytes=" << decoded.getSize() << std::endl;
    const auto* d = (const unsigned char*) decoded.getData();
    juce::String hex, asc;
    for (int i = 0; i < juce::jmin(48, (int) decoded.getSize()); ++i) {
        hex << juce::String::toHexString((int) d[i]).paddedLeft('0', 2) << " ";
        asc << (juce::CharacterFunctions::isPrintable((char) d[i]) ? (char) d[i] : '.');
    }
    std::cout << "first hex: " << hex << "\nfirst asc: " << asc << std::endl;
    const bool gzip = decoded.getSize() > 2 && d[0] == 0x1f && d[1] == 0x8b;
    const bool json = decoded.getSize() > 0 && (d[0] == '{' || d[0] == ' ' || d[0] == '\n');
    std::cout << "looks gzip=" << (gzip ? "yes" : "no") << " json=" << (json ? "yes" : "no") << std::endl;
    const juce::String outp = [&]{ const int i = args.indexOf("--out"); return (i>=0&&i+1<args.size())?args[i+1]:juce::String(); }();
    if (outp.isNotEmpty()) { juce::File(outp).replaceWithData(decoded.getData(), decoded.getSize()); std::cout << "wrote " << outp << "\n"; }
    return 0;
}

// Load a saved plugin .state and print parameter values (verify what a match chose).
static int runReadState(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& flag) -> juce::String {
        const int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String();
    };
    const int mi = args.indexOf("--read-state");
    const juce::String vst3  = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String state = (mi + 2 < args.size()) ? args[mi + 2] : juce::String();
    if (vst3.isEmpty() || state.isEmpty()) { std::cout << "usage: --read-state <vst3> <state> [--params i,j]\n"; return 2; }

    juce::String err;
    auto v = vms::VstTarget::loadAny(juce::File(vst3), 44100.0, 512, err);
    if (!v) { std::cout << "load failed: " << err << "\n"; return 4; }
    auto* inst = v->getInstance();
    if (!inst) { std::cout << "no instance\n"; return 4; }

    juce::MemoryBlock mb;
    if (!juce::File(state).loadFileAsData(mb)) { std::cout << "cannot read state file\n"; return 3; }
    inst->setStateInformation(mb.getData(), (int) mb.getSize());

    const int np = v->numParams();

    // Optional param overrides + render (verify a single param's audible effect).
    {
        const int bi = args.indexOf("--setparams");
        if (bi >= 0 && bi + 1 < args.size()) {
            for (auto& p : juce::StringArray::fromTokens(args[bi + 1], ",", "")) {
                auto kv = juce::StringArray::fromTokens(p, "=", "");
                if (kv.size() == 2) {
                    const int idx = kv[0].getIntValue();
                    if (idx >= 0 && idx < np) v->setParam(idx, (float) kv[1].getDoubleValue());
                }
            }
        }
        const juce::String outPath = valAfter("--out");
        if (outPath.isNotEmpty()) {
            const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
            const double dur = valAfter("--dur").isNotEmpty() ? valAfter("--dur").getDoubleValue() : 1.0;
            auto buf = v->render(note, 100, dur);
            vms::writeWavMono(juce::File(outPath), buf, 44100.0);
            std::cout << "rendered (state" << (bi >= 0 ? "+overrides" : "") << ") peak="
                      << vms::Loss::peakAbs(buf) << " -> " << outPath << "\n";
        }
        // --target <wav>: score this loaded state against a target (subprocess M1 search).
        const juce::String tgt = valAfter("--target");
        if (tgt.isNotEmpty()) {
            const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
            const double dur = valAfter("--dur").isNotEmpty() ? valAfter("--dur").getDoubleValue() : 1.4;
            juce::AudioBuffer<float> tb; double tsr = 44100.0;
            if (vms::loadAudioFileMono(juce::File(tgt), tb, tsr)) {
                auto a = v->render(note, 100, dur);
                vms::Loss L;
                const double loss = (vms::Loss::peakAbs(a) > 1.0e-4f) ? L.combined(tb, a, 1.0f, 0.3f) : 1.0e9;
                std::cout << "STATELOSS loss=" << loss << " peak=" << vms::Loss::peakAbs(a) << std::endl;
            }
        }
    }

    std::vector<int> idxs;
    const juce::String pl = valAfter("--params");
    if (pl.isNotEmpty()) idxs = parseIntList(pl);
    else { for (int i = 0; i < np; ++i) idxs.push_back(i); }

    for (int idx : idxs) {
        if (idx < 0 || idx >= np) { std::cout << "  p" << idx << ": <out of range>\n"; continue; }
        std::cout << "  p" << idx << ": " << v->paramInfo(idx).name
                  << " = " << v->getParam(idx) << "\n";
    }
    return 0;
}

// List parameter index -> name (all, or a given subset via --params).
static int runListParams(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& flag) -> juce::String {
        const int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String();
    };
    const int mi = args.indexOf("--list-params");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    if (vst3.isEmpty()) { std::cout << "usage: --list-params <vst3> [--params i,j,..]\n"; return 2; }

    juce::String err;
    auto v = vms::VstTarget::loadAny(juce::File(vst3), 44100.0, 512, err);
    if (!v) { std::cout << "load failed: " << err << "\n"; return 4; }
    const int np = v->numParams();
    std::cout << "Loaded " << v->name() << " (" << np << " params)" << std::endl;

    std::vector<int> idxs;
    const juce::String pl = valAfter("--params");
    if (pl.isNotEmpty()) idxs = parseIntList(pl);
    else { for (int i = 0; i < np; ++i) idxs.push_back(i); }

    for (int idx : idxs) {
        if (idx < 0 || idx >= np) { std::cout << "  p" << idx << ": <out of range>\n"; continue; }
        const auto info = v->paramInfo(idx);
        std::cout << "  p" << idx << ": " << info.name
                  << "  (value=" << v->getParam(idx) << (info.isDiscrete ? ", discrete" : "") << ")\n";
    }
    return 0;
}

// Inspect a WAV: duration, fundamental, centroid, and RMS envelope shape.
static int runWavInfo(const juce::StringArray& args) {
    const int mi = args.indexOf("--wav-info");
    const juce::String wav = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    if (wav.isEmpty()) { std::cout << "usage: --wav-info <wav>\n"; return 2; }
    juce::AudioBuffer<float> buf; double sr = 44100.0;
    if (!vms::loadAudioFileMono(juce::File(wav), buf, sr)) { std::cout << "load failed\n"; return 3; }
    const int n = buf.getNumSamples();
    std::cout << "samples=" << n << " sr=" << sr << " dur=" << (n / sr) << "s peak=" << vms::Loss::peakAbs(buf) << std::endl;

    // Dominant frequency from a big FFT in the middle.
    double binHz = 0.0;
    auto mags = vms::Loss::magnitudeSpectrum(buf, 14, sr, binHz);
    int pk = 1; for (int k = 2; k < (int) mags.size(); ++k) if (mags[(size_t) k] > mags[(size_t) pk]) pk = k;
    const double f0 = pk * binHz;
    const int midi = (f0 > 0) ? (int) std::lround(69.0 + 12.0 * std::log2(f0 / 440.0)) : 0;
    double num = 0, den = 0;
    for (int k = 1; k < (int) mags.size(); ++k) { num += k * binHz * mags[(size_t) k]; den += mags[(size_t) k]; }
    std::cout << "dominant freq=" << f0 << "Hz (~MIDI " << midi << ")  spectral centroid="
              << (den > 0 ? num / den : 0) << "Hz" << std::endl;
    auto est = vms::estimatePitch(buf, sr);
    std::cout << "estimatePitch=" << est.frequencyHz << "Hz MIDI " << est.midiNote
              << (est.reliable ? " (reliable)" : " (unreliable)") << std::endl;

    // RMS envelope over 12 segments (is it sustained? percussive? sweeping?).
    std::cout << "RMS envelope: ";
    const int seg = 12;
    for (int s = 0; s < seg; ++s) {
        const int a = (int) ((long long) s * n / seg), b = (int) ((long long) (s + 1) * n / seg);
        double e = 0; for (int i = a; i < b; ++i) e += (double) buf.getReadPointer(0)[i] * buf.getReadPointer(0)[i];
        std::cout << juce::String(std::sqrt(e / juce::jmax(1, b - a)), 3) << " ";
    }
    std::cout << std::endl;
    return 0;
}

// Log-frequency / log-magnitude spectrogram resampled to a fixed [T x K] grid,
// values in [0,1]. Compact, fixed-size input for the neural inverse-synth model.
static std::vector<float> specFeature(const juce::AudioBuffer<float>& buf, double sr, int T, int K) {
    const int order = 10, fftSize = 1 << order, hop = fftSize / 2;
    const int n = buf.getNumSamples();
    std::vector<float> out((size_t) T * K, 0.0f);
    if (n < fftSize) return out;
    juce::dsp::FFT fft(order);
    std::vector<float> win((size_t) fftSize);
    for (int i = 0; i < fftSize; ++i)
        win[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * juce::MathConstants<float>::pi * i / (fftSize - 1));
    std::vector<int> starts;
    for (int s = 0; s + fftSize <= n; s += hop) starts.push_back(s);
    if (starts.empty()) return out;

    const int numBins = fftSize / 2 + 1;
    const double logMin = std::log10(40.0), logMax = std::log10(18000.0);
    std::vector<std::vector<float>> frames;
    std::vector<float> data((size_t) fftSize * 2, 0.0f);
    float gMax = 1.0e-9f;
    for (int st : starts) {
        std::fill(data.begin(), data.end(), 0.0f);
        for (int i = 0; i < fftSize; ++i) data[(size_t) i] = buf.getReadPointer(0)[st + i] * win[(size_t) i];
        fft.performRealOnlyForwardTransform(data.data());
        std::vector<float> row((size_t) K, 0.0f);
        for (int k = 0; k < K; ++k) {
            const double f = std::pow(10.0, logMin + (double) k / (K - 1) * (logMax - logMin));
            int b = juce::jlimit(1, numBins - 1, (int) (f * fftSize / sr));
            const float re = data[(size_t) (2 * b)], im = data[(size_t) (2 * b + 1)];
            const float m = std::sqrt(re * re + im * im);
            row[(size_t) k] = m; gMax = juce::jmax(gMax, m);
        }
        frames.push_back(std::move(row));
    }
    // Resample the variable number of frames to exactly T, normalise + log-compress.
    const float inv = 1.0f / gMax;
    for (int t = 0; t < T; ++t) {
        const int fi = juce::jlimit(0, (int) frames.size() - 1,
                                    (int) ((long long) t * (frames.size() - 1) / juce::jmax(1, T - 1)));
        for (int k = 0; k < K; ++k) {
            const float db = 20.0f * std::log10(juce::jmax(1.0e-6f, frames[(size_t) fi][(size_t) k] * inv));
            out[(size_t) (t * K + k)] = juce::jlimit(0.0f, 1.0f, (db + 80.0f) / 80.0f);
        }
    }
    return out;
}

// A1: generate a (spectrogram feature -> parameter) dataset for neural inverse
// synthesis. Renders the plugin with random values on a chosen param set.
static int runGenDataset(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& flag) -> juce::String {
        const int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String();
    };
    const int mi = args.indexOf("--gen-dataset");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String outDir = valAfter("--out");
    if (vst3.isEmpty() || outDir.isEmpty()) {
        std::cout << "usage: --gen-dataset <vst3> --out <dir> [--n N] [--num K | --params i,j]"
                     " [--note M] [--dur D] [--T 96] [--K 64] [--avg R] [--workers W]\n";
        return 2;
    }
    const double sr = 44100.0;
    const int N = valAfter("--n").isNotEmpty() ? valAfter("--n").getIntValue() : 10000;
    const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
    const double dur = valAfter("--dur").isNotEmpty() ? valAfter("--dur").getDoubleValue() : 1.5;
    const int T = valAfter("--T").isNotEmpty() ? valAfter("--T").getIntValue() : 96;
    const int K = valAfter("--K").isNotEmpty() ? valAfter("--K").getIntValue() : 64;
    const int avg = valAfter("--avg").isNotEmpty() ? juce::jmax(1, valAfter("--avg").getIntValue()) : 1;
    const int W = valAfter("--workers").isNotEmpty() ? juce::jmax(1, valAfter("--workers").getIntValue())
                                                     : juce::jmax(1, juce::SystemStats::getNumPhysicalCpus() - 1);

    juce::String e;
    auto probe = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
    if (!probe) { std::cout << "load failed: " << e << "\n"; return 4; }
    const int np = probe->numParams();
    std::vector<float> def((size_t) np);
    for (int i = 0; i < np; ++i) def[(size_t) i] = probe->getParam(i);

    // Choose the parameter set the model will predict.
    std::vector<int> pset;
    if (valAfter("--params").isNotEmpty()) pset = parseIntList(valAfter("--params"));
    else {
        const int K2 = valAfter("--num").isNotEmpty() ? valAfter("--num").getIntValue() : 24;
        for (int i = 0; i < np && (int) pset.size() < K2; ++i)
            if (vms::categorizeParam(probe->paramInfo(i).name) == vms::ParamCat::Core) pset.push_back(i);
    }
    const int P = (int) pset.size();
    if (P == 0) { std::cout << "no params selected\n"; return 5; }

    juce::File dir(outDir); dir.createDirectory();
    juce::String meta = "plugin=" + vst3 + "\nnote=" + juce::String(note) + " dur=" + juce::String(dur)
                      + " sr=" + juce::String(sr) + "\nT=" + juce::String(T) + " K=" + juce::String(K)
                      + " featDim=" + juce::String(T * K) + "\nN=" + juce::String(N)
                      + " paramCount=" + juce::String(P) + "\nparamIndices=";
    for (int idx : pset) meta += juce::String(idx) + ",";
    meta += "\nparamNames=";
    for (int idx : pset) meta += probe->paramInfo(idx).name + ",";
    dir.getChildFile("meta.txt").replaceWithText(meta);
    std::cout << "Dataset: " << probe->name() << "  N=" << N << "  feat=" << T << "x" << K
              << "  params=" << P << "  workers=" << W << "  avg=" << avg << std::endl;

    juce::FileOutputStream featStream(dir.getChildFile("features.f32"));
    juce::FileOutputStream parStream(dir.getChildFile("params.f32"));
    featStream.setPosition(0); featStream.truncate();
    parStream.setPosition(0); parStream.truncate();
    std::mutex writeMx;
    std::atomic<int> done { 0 }, next { 0 };

    auto factory = [vst3, sr]() -> std::unique_ptr<vms::IRenderTarget> {
        juce::String ee; return vms::VstTarget::loadAny(juce::File(vst3), sr, 512, ee);
    };

    auto worker = [&](int wIdx) {
        auto inst = factory();
        if (!inst) return;
        std::mt19937 rng(1234u + (unsigned) wIdx);
        std::uniform_real_distribution<float> u(0.05f, 0.95f);
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= N) break;
            std::vector<float> pv((size_t) P);
            for (int i2 = 0; i2 < inst->numParams(); ++i2) inst->setParam(i2, def[(size_t) i2]);
            for (int k = 0; k < P; ++k) { pv[(size_t) k] = u(rng); inst->setParam(pset[(size_t) k], pv[(size_t) k]); }

            std::vector<float> feat((size_t) (T * K), 0.0f);
            int good = 0;
            for (int r = 0; r < avg; ++r) {
                auto audio = inst->render(note, 100, dur);
                if (inst->renderFailed()) { inst = factory(); if (!inst) return;
                    for (int i2 = 0; i2 < inst->numParams(); ++i2) inst->setParam(i2, def[(size_t) i2]);
                    for (int k = 0; k < P; ++k) inst->setParam(pset[(size_t) k], pv[(size_t) k]);
                    continue; }
                auto f = specFeature(audio, sr, T, K);
                for (size_t z = 0; z < f.size(); ++z) feat[z] += f[z];
                ++good;
            }
            if (good == 0) continue;
            for (auto& z : feat) z /= (float) good;

            {
                std::lock_guard<std::mutex> lk(writeMx);
                featStream.write(feat.data(), feat.size() * sizeof(float));
                parStream.write(pv.data(), pv.size() * sizeof(float));
            }
            const int d = done.fetch_add(1) + 1;
            if (d % 500 == 0) std::cout << "  " << d << "/" << N << std::endl;
        }
    };

    // Run the worker pool on a driver thread while MAIN pumps the message loop —
    // some plugins (Vital) need the message thread to instantiate; a plain join()
    // on main would deadlock.
    std::atomic<bool> finished { false };
    std::thread driver([&] {
        std::vector<std::thread> pool;
        for (int t = 0; t < W; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
        featStream.flush(); parStream.flush();
        finished.store(true);
    });
    while (!finished.load())
        juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    driver.join();
    std::cout << "Done. " << done.load() << " examples -> " << dir.getFullPathName() << std::endl;
    return 0;
}

// Render a plugin with specific param values (on top of default), save WAV and
// optionally the NN input feature. If --vals omitted, uses random values (seeded)
// and prints them. Used to demo the neural inverse model end-to-end.
static int runRenderParams(const juce::StringArray& args) {
    auto valAfter = [&](const juce::String& flag) -> juce::String {
        const int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args[i + 1] : juce::String();
    };
    const int mi = args.indexOf("--render-params");
    const juce::String vst3 = (mi + 1 < args.size()) ? args[mi + 1] : juce::String();
    const juce::String outWav = valAfter("--out");
    const juce::String idxStr = valAfter("--idx");
    if (vst3.isEmpty() || outWav.isEmpty() || idxStr.isEmpty()) {
        std::cout << "usage: --render-params <vst3> --idx i,j --vals v,v [--rand] [--note N] [--dur D]"
                     " [--T 96] [--K 64] [--featout f] --out wav\n";
        return 2;
    }
    const double sr = 44100.0;
    const int note = valAfter("--note").isNotEmpty() ? valAfter("--note").getIntValue() : 60;
    const double dur = valAfter("--dur").isNotEmpty() ? valAfter("--dur").getDoubleValue() : 1.2;
    const int T = valAfter("--T").isNotEmpty() ? valAfter("--T").getIntValue() : 96;
    const int K = valAfter("--K").isNotEmpty() ? valAfter("--K").getIntValue() : 64;
    const auto idx = parseIntList(idxStr);
    std::vector<float> vals;
    for (auto& t : juce::StringArray::fromTokens(valAfter("--vals"), ",", "")) if (t.trim().isNotEmpty()) vals.push_back(t.getFloatValue());

    int rc = 0;
    std::atomic<bool> done { false };
    std::thread th([&] {
        juce::String e;
        auto v = vms::VstTarget::loadAny(juce::File(vst3), sr, 512, e);
        if (!v) { std::cout << "load failed: " << e << "\n"; rc = 4; done = true; return; }
        const int np = v->numParams();
        std::vector<float> def((size_t) np);
        for (int i = 0; i < np; ++i) def[(size_t) i] = v->getParam(i);
        for (int i = 0; i < np; ++i) v->setParam(i, def[(size_t) i]);

        if (args.contains("--rand")) {
            std::mt19937 rng(valAfter("--seed").isNotEmpty() ? (unsigned) valAfter("--seed").getIntValue() : 99u);
            std::uniform_real_distribution<float> u(0.05f, 0.95f);
            vals.clear();
            std::cout << "vals=";
            for (size_t k = 0; k < idx.size(); ++k) { float x = u(rng); vals.push_back(x); std::cout << x << ","; }
            std::cout << std::endl;
        }
        for (size_t k = 0; k < idx.size() && k < vals.size(); ++k) v->setParam(idx[k], vals[(size_t) k]);
        auto audio = v->render(note, 100, dur);
        vms::writeWavMono(juce::File(outWav), audio, sr);
        std::cout << "rendered peak=" << vms::Loss::peakAbs(audio) << " -> " << outWav << std::endl;

        const juce::String featout = valAfter("--featout");
        if (featout.isNotEmpty()) {
            auto feat = specFeature(audio, sr, T, K);
            juce::File featFile(featout);
            juce::FileOutputStream fs(featFile); fs.setPosition(0); fs.truncate();
            fs.write(feat.data(), feat.size() * sizeof(float)); fs.flush();
            std::cout << "feature -> " << featout << " (" << T << "x" << K << ")" << std::endl;
        }
        done = true;
    });
    while (!done.load()) juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    th.join();
    return rc;
}

int main(int argc, char* argv[]) {
    juce::ScopedJuceInitialiser_GUI juceInit; // message manager for plugin hosting
    ConsoleLogger logger;
    juce::Logger::setCurrentLogger(&logger);

    juce::StringArray args;
    for (int i = 1; i < argc; ++i) args.add(juce::String(argv[i]));

    int rc = 0;
    if (args.contains("--match-auto")) {
        rc = runMatchAuto(args);
    } else if (args.contains("--match-ga")) {
        rc = runMatchGA(args);
    } else if (args.contains("--match-staged")) {
        rc = runMatchStaged(args);
    } else if (args.contains("--match") || args.contains("--match-presets")) {
        rc = runMatch(args);   // checked first: it uses --programs as a SOURCE flag
    } else if (args.contains("--render-params")) {
        rc = runRenderParams(args);
    } else if (args.contains("--gen-dataset")) {
        rc = runGenDataset(args);
    } else if (args.contains("--wav-info")) {
        rc = runWavInfo(args);
    } else if (args.contains("--programs")) {
        rc = runPrograms(args);
    } else if (args.contains("--match-full")) {
        rc = runMatchFull(args);
    } else if (args.contains("--inject-wt")) {
        rc = runInjectWt(args);
    } else if (args.contains("--wrap-test")) {
        rc = runWrapTest(args);
    } else if (args.contains("--decode-state")) {
        rc = runDecodeState(args);
    } else if (args.contains("--read-state")) {
        rc = runReadState(args);
    } else if (args.contains("--list-params")) {
        rc = runListParams(args);
    } else if (args.contains("--vst-diag")) {
        rc = runVstDiag(args);
    } else if (args.contains("--vst-recover")) {
        rc = runVstRecover(args);
    } else if (args.contains("--headless-match")) {
        rc = runHeadlessMatch(args);
    } else {
        // Default to selftest for this console target.
        juce::String vst3Path;
        const int vi = args.indexOf("--vst3");
        if (vi >= 0 && vi + 1 < args.size()) vst3Path = args[vi + 1];

        const juce::File artifacts = juce::File::getCurrentWorkingDirectory().getChildFile("artifacts");
        const int failures = vms::runSelfTest(vst3Path, artifacts);
        rc = (failures == 0) ? 0 : 1;
    }

    juce::Logger::setCurrentLogger(nullptr);
    return rc;
}
