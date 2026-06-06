#pragma once
#include "IRenderTarget.h"

namespace vms {

// Dependency-free, fully deterministic reference synth used as the self-test
// oracle. Phase resets on note-on; there are no free-running oscillators, noise
// sources, or async LFOs, so identical parameters yield bit-reproducible output.
//
// Signal path: osc1 + osc2 (sine/saw/square, detune, mix) -> amp ADSR ->
// TPT state-variable lowpass (cutoff, resonance, env amount driven by the
// amp-envelope shape).
class InternalSynthTarget : public IRenderTarget {
public:
    enum ParamId {
        Osc1Wave = 0,  // discrete: sine/saw/square
        Osc2Wave,      // discrete: sine/saw/square
        Osc2Detune,    // cents, [-50, +50]
        OscMix,        // 0 = osc1 only ... 1 = osc2 only
        AmpAttack,     // seconds
        AmpDecay,      // seconds
        AmpSustain,    // [0,1]
        AmpRelease,    // seconds
        FiltCutoff,    // Hz (log mapped)
        FiltReso,      // [0,1]
        FiltEnvAmount, // [0,1] amount of env applied to cutoff
        NumParams
    };

    explicit InternalSynthTarget(double sr = 44100.0);

    int numParams() override { return NumParams; }
    ParamInfo paramInfo(int i) override;
    float getParam(int i) override { return params[(size_t) i]; }
    void setParam(int i, float v01) override;
    juce::AudioBuffer<float> render(int midiNote, int velocity, double durSec, float gateFrac = 0.7f) override;
    double sampleRate() const override { return sr; }
    juce::String name() const override { return "InternalSynth"; }

    void setSampleRate(double newSr) { sr = newSr; }

    // Fill all parameters with deterministic pseudo-random values for a given
    // seed (used to generate known target patches in the self-test).
    void randomize(unsigned seed);

private:
    double sr;
    std::vector<float> params;

    // Mapping helpers from normalised [0,1] to engineering units.
    static int waveIndex(float v01) { return juce::jlimit(0, 2, (int) (v01 * 2.999f)); }
    static double mapLog(float v01, double lo, double hi)
        { return lo * std::pow(hi / lo, (double) v01); }

    static double oscSample(int wave, double phase); // phase in [0,1)
};

} // namespace vms
