#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace vms {

// Loss = multi-scale log-amplitude STFT distance + env_weight * log-RMS envelope
// distance (DDSP-style multi-scale spectral loss). Inputs are peak-normalised
// before comparison so timbre/shape is matched rather than absolute level.
class Loss {
public:
    Loss();

    // Peak-normalise a (mono) buffer in place to unit peak. No-op if silent.
    static void peakNormalize(juce::AudioBuffer<float>& buf);
    static float peakAbs(const juce::AudioBuffer<float>& buf);

    // Multi-scale log-STFT L1 distance (averaged per bin/frame across scales).
    float multiScaleStft(const juce::AudioBuffer<float>& a,
                         const juce::AudioBuffer<float>& b);

    // log-RMS envelope MSE over the overlapping frame count.
    static float logRmsEnvelope(const juce::AudioBuffer<float>& a,
                                const juce::AudioBuffer<float>& b);

    // Perceptual distance: spectral-centroid (brightness) + spectral-flatness
    // (tonal vs noisy) trajectories compared over time. Cheap, perceptually
    // salient, and the flatness term further discourages "cover it with noise".
    float perceptual(const juce::AudioBuffer<float>& a, const juce::AudioBuffer<float>& b);

    // Combined loss. Both buffers are copied & peak-normalised internally.
    // robust=true uses a TIME-AVERAGED spectrum (LTAS) instead of frame-by-frame STFT, so
    // non-deterministic synths (Synth1: free phases / unison beating) whose per-frame
    // spectra fluctuate still score ~0 for the same patch — letting the optimiser work.
    float combined(const juce::AudioBuffer<float>& target,
                   const juce::AudioBuffer<float>& candidate,
                   float envWeight = 1.0f, float percWeight = 1.0f, bool robust = false);

    // Time-averaged log-magnitude spectrum (LTAS), log-frequency binned to `bands`,
    // normalised. Robust to per-frame phase/beating fluctuation.
    std::vector<float> avgLogSpectrum(const juce::AudioBuffer<float>& buf, int bands = 96);

    // Magnitude spectrum (linear) of the whole mono buffer using one Hann-windowed
    // FFT of the given order. Used by the analyzer overlay and the FFT peak test.
    static std::vector<float> magnitudeSpectrum(const juce::AudioBuffer<float>& buf,
                                                int fftOrder, double sampleRate,
                                                double& binHz);

private:
    // One STFT scale's mean log-magnitude L1 distance.
    float stftScale(const float* a, int na, const float* b, int nb, int fftSize);

    std::vector<juce::dsp::FFT> ffts;     // for sizes 512,1024,2048
    std::vector<std::vector<float>> windows;
    std::vector<int> sizes { 512, 1024, 2048 };
};

} // namespace vms
