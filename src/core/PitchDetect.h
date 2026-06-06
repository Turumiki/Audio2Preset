#pragma once
#include <juce_audio_basics/juce_audio_basics.h>

namespace vms {

// Lightweight autocorrelation-based pitch estimate over the sustain portion of a
// mono buffer. Returns the nearest MIDI note (musician can override in the GUI).
struct PitchResult {
    double frequencyHz = 0.0;
    int midiNote = 60;
    bool reliable = false;
};

PitchResult estimatePitch(const juce::AudioBuffer<float>& buf, double sampleRate);

inline int hzToMidi(double hz) {
    if (hz <= 0.0) return 60;
    return (int) std::lround(69.0 + 12.0 * std::log2(hz / 440.0));
}

} // namespace vms
