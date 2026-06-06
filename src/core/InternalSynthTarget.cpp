#include "InternalSynthTarget.h"
#include <random>

namespace vms {

InternalSynthTarget::InternalSynthTarget(double sampleRate) : sr(sampleRate) {
    params.assign((size_t) NumParams, 0.5f);
    // Sensible, audible defaults.
    params[Osc1Wave]      = 0.4f;   // saw-ish
    params[Osc2Wave]      = 0.0f;   // sine
    params[Osc2Detune]    = 0.5f;   // no detune
    params[OscMix]        = 0.3f;
    params[AmpAttack]     = 0.05f;
    params[AmpDecay]      = 0.3f;
    params[AmpSustain]    = 0.7f;
    params[AmpRelease]    = 0.3f;
    params[FiltCutoff]    = 0.6f;
    params[FiltReso]      = 0.2f;
    params[FiltEnvAmount] = 0.4f;
}

ParamInfo InternalSynthTarget::paramInfo(int i) {
    static const char* names[] = {
        "Osc1 Wave", "Osc2 Wave", "Osc2 Detune", "Osc Mix",
        "Amp Attack", "Amp Decay", "Amp Sustain", "Amp Release",
        "Filter Cutoff", "Filter Reso", "Filter Env Amount"
    };
    ParamInfo info;
    info.name = (i >= 0 && i < NumParams) ? names[i] : "?";
    info.isDiscrete = (i == Osc1Wave || i == Osc2Wave);
    return info;
}

void InternalSynthTarget::setParam(int i, float v01) {
    if (i >= 0 && i < NumParams)
        params[(size_t) i] = juce::jlimit(0.0f, 1.0f, v01);
}

void InternalSynthTarget::randomize(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.05f, 0.95f);
    for (auto& p : params) p = u(rng);
}

double InternalSynthTarget::oscSample(int wave, double phase) {
    switch (wave) {
        case 0: return std::sin(2.0 * juce::MathConstants<double>::pi * phase); // sine
        case 1: return 2.0 * phase - 1.0;                                       // saw (naive)
        case 2: return (phase < 0.5) ? 1.0 : -1.0;                              // square
        default: return 0.0;
    }
}

juce::AudioBuffer<float> InternalSynthTarget::render(int midiNote, int velocity, double durSec, float gateFrac) {
    const int total = juce::jmax(1, (int) std::llround(durSec * sr));
    juce::AudioBuffer<float> out(1, total);
    out.clear();
    float* y = out.getWritePointer(0);

    const double baseFreq = 440.0 * std::pow(2.0, (midiNote - 69) / 12.0);
    const double cents = juce::jmap((double) params[Osc2Detune], 0.0, 1.0, -50.0, 50.0);
    const double f1 = baseFreq;
    const double f2 = baseFreq * std::pow(2.0, cents / 1200.0);
    const int w1 = waveIndex(params[Osc1Wave]);
    const int w2 = waveIndex(params[Osc2Wave]);
    const double mix = params[OscMix];
    const double vel = juce::jlimit(0, 127, velocity) / 127.0;

    // ADSR (seconds).
    const double atk = juce::jmap((double) params[AmpAttack],  0.0, 1.0, 0.001, 1.0);
    const double dec = juce::jmap((double) params[AmpDecay],   0.0, 1.0, 0.001, 1.5);
    const double sus = params[AmpSustain];
    const double rel = juce::jmap((double) params[AmpRelease], 0.0, 1.0, 0.001, 1.5);

    // Gate: hold for gateFrac of the buffer, then release (exercises full ADSR).
    const int gateSamples = (int) (total * juce::jlimit(0.01f, 1.0f, gateFrac));

    // Filter setup (TPT state-variable lowpass).
    const double cutoffBase = mapLog(params[FiltCutoff], 30.0, 18000.0);
    const double envAmt = params[FiltEnvAmount];
    const double reso = juce::jmap((double) params[FiltReso], 0.0, 1.0, 0.5, 8.0); // Q
    double ic1eq = 0.0, ic2eq = 0.0; // SVF integrator states

    double phase1 = 0.0, phase2 = 0.0;
    const double inc1 = f1 / sr, inc2 = f2 / sr;

    double env = 0.0;          // amp envelope value
    int stage = 0;            // 0=A,1=D,2=S,3=R
    double relStart = 0.0;    // env value at release onset

    for (int n = 0; n < total; ++n) {
        const double t = n / sr;

        // Amp envelope (deterministic, sample-accurate stages).
        if (n < gateSamples) {
            const double tg = t;
            if (tg < atk) { env = tg / atk; stage = 0; }
            else if (tg < atk + dec) {
                const double dd = (tg - atk) / dec;
                env = 1.0 - (1.0 - sus) * dd; stage = 1;
            } else { env = sus; stage = 2; }
            relStart = env;
        } else {
            const double tr = (n - gateSamples) / sr;
            env = relStart * std::max(0.0, 1.0 - tr / rel);
            stage = 3;
        }
        juce::ignoreUnused(stage);

        // Oscillators.
        const double o1 = oscSample(w1, phase1);
        const double o2 = oscSample(w2, phase2);
        double s = (1.0 - mix) * o1 + mix * o2;

        // Filter cutoff modulated by envelope shape.
        double cutoff = cutoffBase * std::pow(4.0, envAmt * env); // up to 2 octaves up
        cutoff = juce::jlimit(20.0, sr * 0.45, cutoff);
        const double g = std::tan(juce::MathConstants<double>::pi * cutoff / sr);
        const double k = 1.0 / reso;
        const double a1 = 1.0 / (1.0 + g * (g + k));
        const double a2 = g * a1;
        const double v3 = s - ic2eq;
        const double v1 = a1 * ic1eq + a2 * v3;
        const double v2 = ic2eq + g * v1;
        ic1eq = 2.0 * v1 - ic1eq;
        ic2eq = 2.0 * v2 - ic2eq;
        const double lp = v2; // lowpass output

        y[n] = (float) (lp * env * vel * 0.5);

        phase1 += inc1; if (phase1 >= 1.0) phase1 -= 1.0;
        phase2 += inc2; if (phase2 >= 1.0) phase2 -= 1.0;
    }

    return out;
}

} // namespace vms
