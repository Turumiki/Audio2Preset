#pragma once
#include <juce_audio_basics/juce_audio_basics.h>

namespace vms {

// Coarse perceptual features extracted from the (peak-normalised) target, used
// to warm-start parameters whose names are recognisable.
struct TargetFeatures {
    int    midiNote = 60;
    double spectralCentroidHz = 1000.0;  // brightness
    double centroidNorm = 0.1;           // centroid / nyquist, [0,1]
    double attackSec = 0.01;             // time to reach near-peak
    double releaseSec = 0.2;             // decay time after the peak region
    double sustainLevel = 0.7;           // late RMS / peak RMS, [0,1]
};

TargetFeatures extractFeatures(const juce::AudioBuffer<float>& target, double sampleRate);

} // namespace vms
