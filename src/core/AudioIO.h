#pragma once
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_basics/juce_audio_basics.h>

namespace vms {

// Load a WAV/AIFF file as a mono buffer (channels averaged), resampled is NOT
// performed — the file's own sample rate is returned. Returns false on failure.
bool loadAudioFileMono(const juce::File& file, juce::AudioBuffer<float>& out, double& sampleRate);

// Write a mono buffer to a 24-bit WAV.
bool writeWavMono(const juce::File& file, const juce::AudioBuffer<float>& buf, double sampleRate);

// Resample a mono buffer from srcSr to dstSr (Lagrange interpolation).
juce::AudioBuffer<float> resampleMono(const juce::AudioBuffer<float>& in, double srcSr, double dstSr);

} // namespace vms
