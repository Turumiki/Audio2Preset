#pragma once
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_basics/juce_audio_basics.h>

namespace vms {

// A/B looping player. Holds two mono buffers (A = target, B = current best),
// loudness-matched to a common RMS so switching is a fair comparison. Audio
// callback uses a try-lock; if a swap is in progress it outputs the previous
// block's tail of silence rather than blocking.
class LivePlayer : public juce::AudioIODeviceCallback {
public:
    enum class Source { A, B };

    LivePlayer();
    ~LivePlayer() override;

    bool start();        // opens default output device
    void stop();
    juce::String lastError() const { return lastErr; }
    int bufferLenA() const { return bufA.getNumSamples(); }
    int bufferLenB() const { return bufB.getNumSamples(); }
    juce::String deviceInfo() const;

    // Live diagnostics: did the audio callback fire, and what peak did we write?
    int callbackCount() const { return cbCount.load(); }
    float consumeOutPeak() { return outPeak.exchange(0.0f); }
    float currentSrcPeak() const;   // peak of the currently-selected buffer

    void setBufferA(const juce::AudioBuffer<float>& mono, double sr);
    void setBufferB(const juce::AudioBuffer<float>& mono, double sr);
    void setSource(Source s) { source = s; }
    Source getSource() const { return source; }
    void setPlaying(bool p) { playing = p; if (p) readPos = 0; }
    bool isPlaying() const { return playing; }

    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData, int numInputChannels,
                                          float* const* outputChannelData, int numOutputChannels,
                                          int numSamples,
                                          const juce::AudioIODeviceCallbackContext& context) override;
    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override {}

private:
    juce::AudioDeviceManager deviceManager;
    juce::CriticalSection lock;
    juce::AudioBuffer<float> bufA, bufB; // RMS-normalised mono
    std::atomic<Source> source { Source::A };
    std::atomic<bool> playing { false };
    double readPos = 0.0;                 // fractional, for resampling
    double srcRateA = 44100.0, srcRateB = 44100.0;
    std::atomic<double> deviceRate { 44100.0 };
    bool started = false;
    juce::String lastErr;
    std::atomic<int> cbCount { 0 };
    std::atomic<float> outPeak { 0.0f };

    static void rmsNormalise(juce::AudioBuffer<float>& buf, float targetRms = 0.15f);
};

} // namespace vms
