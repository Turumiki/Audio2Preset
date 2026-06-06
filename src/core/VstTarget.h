#pragma once
#include "IRenderTarget.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace vms {

// Wraps a juce::AudioPluginInstance for deterministic offline single-note
// rendering. Parameters are set without host notification (fast, no editor
// callbacks). instance->reset() is called per render to avoid tail bleed.
class VstTarget : public IRenderTarget {
public:
    VstTarget(std::unique_ptr<juce::AudioPluginInstance> inst, double sr, int blockSize = 512, bool msgThread = false);
    ~VstTarget() override;

    int numParams() override;
    ParamInfo paramInfo(int i) override;
    float getParam(int i) override;
    void setParam(int i, float v01) override;
    // For the live-monitoring instance only: notify the host so the plugin's own
    // editor updates its knobs. Must be called on the message thread.
    void setParamNotifying(int i, float v01);
    juce::AudioBuffer<float> render(int midiNote, int velocity, double durSec, float gateFrac = 0.7f) override;
    juce::AudioBuffer<float> renderChord(const std::vector<int>& notes, int velocity, double durSec, float gateFrac = 0.7f) override;
    double sampleRate() const override { return sr; }
    juce::String name() const override;

    juce::AudioPluginInstance* getInstance() { return instance.get(); }
    bool renderFailed() const override { return lastRenderFailed; } // true if last render crashed (SEH-caught)
    bool setStateData(const void* data, size_t size) override; // load a saved patch state (e.g. .vital JSON)

    // Scan a VST3 file and instantiate its first type. Returns nullptr + error on
    // failure (caller logs and degrades to the internal synth — never crashes).
    static std::unique_ptr<VstTarget> loadVst3(const juce::File& file, double sr,
                                               int blockSize, juce::String& error);

    // Load a VST2 (.dll/.vst) — only available if built with the VST2 SDK.
    static std::unique_ptr<VstTarget> loadVst2(const juce::File& file, double sr,
                                               int blockSize, juce::String& error);

    // Dispatch by extension: .vst3 -> VST3, .dll/.vst -> VST2.
    static std::unique_ptr<VstTarget> loadAny(const juce::File& file, double sr,
                                              int blockSize, juce::String& error);

    static bool vst2Supported();

    // "Safe mode": force plugin create/destroy/render onto the message thread (for plugins that
    // crash under worker-thread rendering - VST2 like TAL, or GUI-heavy VST3 like KORG Legacy).
    // Off by default (fast parallel). Read at instance-CREATE time; enabled per-plugin after a crash.
    static void setForceMessageThread(bool on);

private:
    std::unique_ptr<juce::AudioPluginInstance> instance;
    double sr;
    int blockSize;
    juce::Array<juce::AudioProcessorParameter*> params;
    bool prepared = false;
    bool lastRenderFailed = false;
    bool msgThreadMode = false;   // this instance marshals create/destroy/render to the message thread

    void ensurePrepared();
    // Actual render; renderChord() marshals this to the message thread for VST2.
    juce::AudioBuffer<float> renderChordImpl(const std::vector<int>& notes, int velocity,
                                             double durSec, float gateFrac);
};

} // namespace vms
