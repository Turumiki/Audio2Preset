#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <vector>

namespace vms {

struct ParamInfo {
    juce::String name;
    bool isDiscrete = false;
};

// Abstraction that decouples the matching engine from VST hosting.
// Two implementations exist: InternalSynthTarget (dependency-free oracle) and
// VstTarget (wraps a juce::AudioPluginInstance).
class IRenderTarget {
public:
    virtual ~IRenderTarget() = default;

    virtual int numParams() = 0;
    virtual ParamInfo paramInfo(int i) = 0;

    // Get/set normalised parameter value in [0,1].
    virtual float getParam(int i) = 0;
    virtual void setParam(int i, float v01) = 0;

    // Deterministic offline render of a single note. Returns mono buffer at the
    // engine sample rate. Latency is compensated by the implementation. gateFrac is
    // the fraction of durSec the note is held before noteOff (release timing).
    virtual juce::AudioBuffer<float> render(int midiNote, int velocity, double durSec,
                                            float gateFrac = 0.7f) = 0;

    // Render a CHORD: all notes sound together (polyphonic). Default plays just the
    // first note; VstTarget overrides to send every note (real chord). Used to match
    // polyphonic targets (e.g. the Windows TADA fanfare).
    virtual juce::AudioBuffer<float> renderChord(const std::vector<int>& notes, int velocity,
                                                 double durSec, float gateFrac = 0.7f) {
        return render(notes.empty() ? 60 : notes[0], velocity, durSec, gateFrac);
    }

    virtual double sampleRate() const = 0;
    virtual juce::String name() const = 0;

    // True if the most recent render() aborted because the plugin crashed
    // (SEH-caught). Internal targets never fail.
    virtual bool renderFailed() const { return false; }

    // Apply a saved plugin state blob (e.g. a Vital .vital preset JSON, which
    // includes the wavetable). No-op for targets without state. Returns false if
    // unsupported. Used by preset/wavetable search to start from a real patch.
    virtual bool setStateData(const void* /*data*/, size_t /*size*/) { return false; }
};

} // namespace vms
