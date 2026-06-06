#include "Features.h"
#include "PitchDetect.h"
#include "Loss.h"
#include <cmath>

namespace vms {

TargetFeatures extractFeatures(const juce::AudioBuffer<float>& target, double sr) {
    TargetFeatures f;
    const int n = target.getNumSamples();
    if (n < 16) return f;

    f.midiNote = estimatePitch(target, sr).midiNote;

    // Spectral centroid from a mid-buffer magnitude spectrum.
    double binHz = 0.0;
    auto mags = Loss::magnitudeSpectrum(target, 13, sr, binHz);
    double num = 0.0, den = 0.0;
    for (size_t k = 1; k < mags.size(); ++k) {
        const double fHz = k * binHz;
        num += fHz * mags[k];
        den += mags[k];
    }
    f.spectralCentroidHz = (den > 1e-9) ? (num / den) : 1000.0;
    f.centroidNorm = juce::jlimit(0.0, 1.0, f.spectralCentroidHz / (sr * 0.5));

    // RMS envelope (frame-wise) for attack / sustain / release estimates.
    const int frame = 512, hop = 128;
    std::vector<double> env;
    const float* p = target.getReadPointer(0);
    for (int s = 0; s + frame <= n; s += hop) {
        double e = 0.0;
        for (int i = 0; i < frame; ++i) e += (double) p[s + i] * p[s + i];
        env.push_back(std::sqrt(e / frame));
    }
    if (env.empty()) return f;

    double peak = 0.0; int peakIdx = 0;
    for (size_t i = 0; i < env.size(); ++i) if (env[i] > peak) { peak = env[i]; peakIdx = (int) i; }
    if (peak < 1e-9) return f;

    const double framesPerSec = sr / hop;

    // Attack: time from start to 90% of peak.
    int aIdx = 0;
    for (size_t i = 0; i < env.size(); ++i) { if (env[i] >= 0.9 * peak) { aIdx = (int) i; break; } }
    f.attackSec = juce::jmax(0.001, aIdx / framesPerSec);

    // Sustain: average of the last quarter relative to peak.
    const int q0 = (int) (env.size() * 0.75);
    double sus = 0.0; int cnt = 0;
    for (int i = q0; i < (int) env.size(); ++i) { sus += env[i]; ++cnt; }
    f.sustainLevel = (cnt > 0) ? juce::jlimit(0.0, 1.0, (sus / cnt) / peak) : 0.7;

    // Release: time after the peak to fall to 20% of peak (proxy for tail length).
    int rIdx = (int) env.size() - 1;
    for (int i = peakIdx; i < (int) env.size(); ++i) { if (env[i] <= 0.2 * peak) { rIdx = i; break; } }
    f.releaseSec = juce::jmax(0.001, (rIdx - peakIdx) / framesPerSec);

    return f;
}

} // namespace vms
