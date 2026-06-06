#include "VstTarget.h"
#include <functional>
#include <atomic>

namespace vms {

// When true, plugin lifecycle+render is forced onto the message thread ("safe mode") for
// plugins that crash under worker-thread rendering (VST2 like TAL touching its window during
// processing, or GUI-heavy VST3 like KORG Legacy). Read at instance-CREATE time and latched
// per-instance. Off by default so well-behaved plugins (Synth1, Vital) stay fast/parallel.
static std::atomic<bool> sForceMsgThread { false };
void VstTarget::setForceMessageThread(bool on) { sForceMsgThread = on; }

namespace {
// Run a function on the message thread and block until it finishes. VST2 plugins create
// (and destroy) their window during instantiation; doing that on a worker thread makes the
// window's messages dispatch on that worker -> CallWindowProcW crashes inside the plugin.
// Marshalling lifecycle to the message thread fixes it. Rendering stays on worker threads.
void runOnMessageThreadSync(std::function<void()> fn) {
    auto* mm = juce::MessageManager::getInstanceWithoutCreating();
    if (mm == nullptr || mm->isThisTheMessageThread()) { fn(); return; }
    mm->callFunctionOnMessageThread([](void* p) -> void* {
        (*static_cast<std::function<void()>*>(p))();
        return nullptr;
    }, &fn);
}
} // namespace

VstTarget::VstTarget(std::unique_ptr<juce::AudioPluginInstance> inst, double sampleRate, int bs, bool msgThread)
    : instance(std::move(inst)), sr(sampleRate), blockSize(bs), msgThreadMode(msgThread) {
    params = instance->getParameters();
}

VstTarget::~VstTarget() {
    // Safe mode: destroy the instance (and its window) on the message thread, mirroring creation.
    // Fast mode lets the unique_ptr destroy normally (well-behaved plugins).
    if (msgThreadMode && instance != nullptr) {
        auto* raw = instance.release();
        runOnMessageThreadSync([raw] { std::unique_ptr<juce::AudioPluginInstance> kill(raw); });
    }
}

int VstTarget::numParams() { return params.size(); }

ParamInfo VstTarget::paramInfo(int i) {
    ParamInfo info;
    if (i >= 0 && i < params.size()) {
        info.name = params[i]->getName(64);
        info.isDiscrete = params[i]->isDiscrete();
    }
    return info;
}

float VstTarget::getParam(int i) {
    if (i >= 0 && i < params.size()) return params[i]->getValue();
    return 0.0f;
}

void VstTarget::setParam(int i, float v01) {
    if (i >= 0 && i < params.size())
        params[i]->setValue(juce::jlimit(0.0f, 1.0f, v01)); // no host notification
}

void VstTarget::setParamNotifying(int i, float v01) {
    if (i >= 0 && i < params.size()) {
        auto* p = params[i];
        const float v = juce::jlimit(0.0f, 1.0f, v01);
        p->beginChangeGesture();
        p->setValueNotifyingHost(v);
        p->endChangeGesture();
    }
}

juce::String VstTarget::name() const {
    return instance ? instance->getName() : juce::String("VST");
}

bool VstTarget::setStateData(const void* data, size_t size) {
    if (!instance || data == nullptr || size == 0) return false;
    instance->setStateInformation(data, (int) size);
    // Re-cache the parameter pointers (state load can swap the param set on some
    // plugins) and force a re-prepare so the new patch's voices initialise.
    prepared = false;
    return true;
}

void VstTarget::ensurePrepared() {
    if (!prepared) {
        instance->setNonRealtime(true);
        instance->prepareToPlay(sr, blockSize);
        prepared = true;
    }
}

// Guard the plugin's processBlock against hard crashes (access violations). Some
// plug-ins (e.g. Synth1) corrupt themselves under repeated offline rendering; SEH
// lets us abort that render instead of taking down the whole app. Windows-only;
// the helper creates no unwinding C++ objects so __try is permitted here.
static bool safeProcessBlock(juce::AudioPluginInstance* inst,
                             juce::AudioBuffer<float>& buf, juce::MidiBuffer& midi) {
#if defined(_WIN32)
    __try {
        inst->processBlock(buf, midi);
        return true;
    } __except (1 /* EXCEPTION_EXECUTE_HANDLER */) {
        return false;
    }
#else
    inst->processBlock(buf, midi);
    return true;
#endif
}

juce::AudioBuffer<float> VstTarget::render(int midiNote, int velocity, double durSec, float gateFrac) {
    return renderChord(std::vector<int>{ midiNote }, velocity, durSec, gateFrac);
}

juce::AudioBuffer<float> VstTarget::renderChord(const std::vector<int>& notes, int velocity,
                                                double durSec, float gateFrac) {
    // Safe mode: prepareToPlay/processBlock touch the plugin's window; off the message thread
    // (worker render) those dispatch window messages cross-thread -> crash (TAL, KORG Legacy).
    // Marshal the whole render to the message thread. Fast mode renders on the worker.
    if (msgThreadMode) {
        auto* mm = juce::MessageManager::getInstanceWithoutCreating();
        if (mm != nullptr && !mm->isThisTheMessageThread()) {
            juce::AudioBuffer<float> out;
            runOnMessageThreadSync([&] { out = renderChordImpl(notes, velocity, durSec, gateFrac); });
            return out;
        }
    }
    return renderChordImpl(notes, velocity, durSec, gateFrac);
}

juce::AudioBuffer<float> VstTarget::renderChordImpl(const std::vector<int>& notes, int velocity,
                                                    double durSec, float gateFrac) {
    ensurePrepared();
    instance->reset();
    lastRenderFailed = false;

    const int latency = juce::jmax(0, instance->getLatencySamples());
    const int total = juce::jmax(1, (int) std::llround(durSec * sr));
    const int renderLen = total + latency; // render extra to compensate latency
    const int gate = (int) (total * juce::jlimit(0.01f, 1.0f, gateFrac));

    const int numOut = juce::jmax(1, instance->getTotalNumOutputChannels());
    const int procCh = juce::jmax(numOut, instance->getTotalNumInputChannels());

    juce::AudioBuffer<float> mono(1, total);
    mono.clear();

    // Over-allocate the channel STORAGE (4x blockSize samples) but hand the plugin
    // a view reporting only blockSize. Some plugins write beyond the requested
    // sample count; the cushion absorbs that instead of corrupting the heap
    // (observed: Synth1 crashing on a later render).
    const int cushSamples = juce::jmax(blockSize * 4, 4096);
    juce::AudioBuffer<float> storage(procCh, cushSamples);
    storage.clear();

    const float vel = juce::jlimit(0, 127, velocity) / 127.0f;
    const int numBlocks = (renderLen + blockSize - 1) / blockSize;

    for (int b = 0; b < numBlocks; ++b) {
        const int blockStart = b * blockSize;
        juce::AudioBuffer<float> block(storage.getArrayOfWritePointers(), procCh, 0, blockSize);
        block.clear();

        juce::MidiBuffer midi;
        if (blockStart == 0)
            for (int n : notes) midi.addEvent(juce::MidiMessage::noteOn(1, n, vel), 0);     // chord on
        if (gate >= blockStart && gate < blockStart + blockSize)
            for (int n : notes) midi.addEvent(juce::MidiMessage::noteOff(1, n), gate - blockStart);

        if (!safeProcessBlock(instance.get(), block, midi)) {
            lastRenderFailed = true;   // plugin crashed mid-render; bail with what we have
            break;
        }

        for (int i = 0; i < blockSize; ++i) {
            const int outIdx = blockStart + i - latency;
            if (outIdx >= 0 && outIdx < total) {
                float s = 0.0f;
                for (int ch = 0; ch < numOut; ++ch) s += block.getReadPointer(ch)[i];
                mono.getWritePointer(0)[outIdx] = s / (float) numOut;
            }
        }
    }
    return mono;
}

std::unique_ptr<VstTarget> VstTarget::loadVst3(const juce::File& file, double sr,
                                               int blockSize, juce::String& error) {
    juce::AudioPluginFormatManager fm;
    fm.addDefaultFormats();

    juce::VST3PluginFormat vst3;
    juce::OwnedArray<juce::PluginDescription> found;
    vst3.findAllTypesForFile(found, file.getFullPathName());

    if (found.isEmpty()) {
        error = "No VST3 types found in: " + file.getFullPathName();
        return nullptr;
    }

    juce::String createError;
    std::unique_ptr<juce::AudioPluginInstance> inst;
    const bool msgThread = sForceMsgThread.load();
    // Safe mode: create on the message thread (GUI-heavy VST3 like KORG Legacy crash off-thread).
    if (msgThread)
        runOnMessageThreadSync([&] { inst = fm.createPluginInstance(*found[0], sr, blockSize, createError); });
    else
        inst = fm.createPluginInstance(*found[0], sr, blockSize, createError);

    if (inst == nullptr) {
        error = "createPluginInstance failed: " + createError;
        return nullptr;
    }

    if (inst->getTotalNumOutputChannels() <= 0) {
        error = "Plugin reports no output channels (not an instrument?)";
        return nullptr;
    }

    return std::make_unique<VstTarget>(std::move(inst), sr, blockSize, msgThread);
}

bool VstTarget::vst2Supported() {
   #if JUCE_PLUGINHOST_VST
    return true;
   #else
    return false;
   #endif
}

std::unique_ptr<VstTarget> VstTarget::loadVst2(const juce::File& file, double sr,
                                               int blockSize, juce::String& error) {
   #if JUCE_PLUGINHOST_VST
    juce::AudioPluginFormatManager fm;
    fm.addDefaultFormats();

    juce::VSTPluginFormat vst2;
    juce::OwnedArray<juce::PluginDescription> found;
    vst2.findAllTypesForFile(found, file.getFullPathName());
    if (found.isEmpty()) {
        error = "No VST2 types found in: " + file.getFullPathName();
        return nullptr;
    }

    juce::String createError;
    std::unique_ptr<juce::AudioPluginInstance> inst;
    const bool msgThread = sForceMsgThread.load();
    // Safe mode: create on the message thread (VST2 builds its window in the ctor; off-thread
    // -> crash). Fast mode: create on the calling thread (well-behaved plugins, e.g. Synth1).
    if (msgThread)
        runOnMessageThreadSync([&] { inst = fm.createPluginInstance(*found[0], sr, blockSize, createError); });
    else
        inst = fm.createPluginInstance(*found[0], sr, blockSize, createError);
    if (inst == nullptr) {
        error = "createPluginInstance failed: " + createError;
        return nullptr;
    }
    if (inst->getTotalNumOutputChannels() <= 0) {
        error = "Plugin reports no output channels (not an instrument?)";
        return nullptr;
    }
    return std::make_unique<VstTarget>(std::move(inst), sr, blockSize, msgThread);
   #else
    juce::ignoreUnused(file, sr, blockSize);
    error = "VST2 support not built. Place the VST2 SDK (aeffect.h, aeffectx.h) in "
            "thirdparty/vst2sdk/pluginterfaces/vst2.x/ and re-run CMake.";
    return nullptr;
   #endif
}

std::unique_ptr<VstTarget> VstTarget::loadAny(const juce::File& file, double sr,
                                              int blockSize, juce::String& error) {
    const juce::String ext = file.getFileExtension().toLowerCase();
    if (ext == ".vst3") return loadVst3(file, sr, blockSize, error);
    if (ext == ".dll" || ext == ".vst") return loadVst2(file, sr, blockSize, error);
    // Unknown extension: try VST3 then VST2.
    auto v = loadVst3(file, sr, blockSize, error);
    if (v) return v;
    return loadVst2(file, sr, blockSize, error);
}

} // namespace vms
