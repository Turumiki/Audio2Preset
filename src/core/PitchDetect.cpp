#include "PitchDetect.h"
#include <vector>
#include <cmath>

namespace vms {

PitchResult estimatePitch(const juce::AudioBuffer<float>& buf, double sampleRate) {
    PitchResult r;
    const int n = buf.getNumSamples();
    if (n < 2048) return r;

    // Analyse a window in the sustain region (skip the attack transient).
    const int start = juce::jmin(n / 4, n - 1);
    const int len = juce::jmin(n - start, (int) (sampleRate * 0.1)); // ~100 ms
    if (len < 1024) return r;

    const float* p = buf.getReadPointer(0) + start;

    // Search f0 in 40 Hz .. 2000 Hz.
    const int minLag = (int) (sampleRate / 2000.0);
    const int maxLag = juce::jmin(len - 1, (int) (sampleRate / 40.0));

    // Normalised autocorrelation (difference-function style for robustness).
    double bestVal = 1.0e30;
    int bestLag = -1;
    double energy0 = 0.0;
    for (int i = 0; i < len; ++i) energy0 += (double) p[i] * p[i];
    if (energy0 < 1.0e-9) return r;

    std::vector<double> nsdf;
    for (int lag = minLag; lag <= maxLag; ++lag) {
        double diff = 0.0;
        const int m = len - lag;
        for (int i = 0; i < m; ++i) {
            const double d = (double) p[i] - (double) p[i + lag];
            diff += d * d;
        }
        diff /= m;
        if (diff < bestVal) { bestVal = diff; bestLag = lag; }
    }

    if (bestLag > 0) {
        r.frequencyHz = sampleRate / bestLag;
        r.midiNote = hzToMidi(r.frequencyHz);
        r.reliable = true;
    }
    return r;
}

} // namespace vms
