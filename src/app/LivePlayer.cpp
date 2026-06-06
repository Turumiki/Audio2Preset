#include "LivePlayer.h"

namespace vms {

LivePlayer::LivePlayer() {}
LivePlayer::~LivePlayer() { stop(); }

bool LivePlayer::start() {
    if (started) return true;
    lastErr = deviceManager.initialiseWithDefaultDevices(0, 2);
    if (lastErr.isNotEmpty()) return false;
    if (deviceManager.getCurrentAudioDevice() == nullptr) {
        lastErr = "no default output device opened";
        return false;
    }
    deviceManager.addAudioCallback(this);
    started = true;
    return true;
}

juce::String LivePlayer::deviceInfo() const {
    if (auto* dev = deviceManager.getCurrentAudioDevice())
        return dev->getName() + " @ " + juce::String(dev->getCurrentSampleRate(), 0) + "Hz, out="
               + juce::String(dev->getActiveOutputChannels().countNumberOfSetBits());
    return "(no device)";
}

void LivePlayer::stop() {
    if (!started) return;
    deviceManager.removeAudioCallback(this);
    deviceManager.closeAudioDevice();
    started = false;
}

void LivePlayer::rmsNormalise(juce::AudioBuffer<float>& buf, float targetRms) {
    if (buf.getNumSamples() == 0) return;
    double sum = 0.0;
    const float* p = buf.getReadPointer(0);
    for (int i = 0; i < buf.getNumSamples(); ++i) sum += (double) p[i] * p[i];
    const double rms = std::sqrt(sum / buf.getNumSamples());
    if (rms > 1.0e-7) buf.applyGain((float) (targetRms / rms));
    // Hard-limit to avoid clipping after RMS match.
    const float pk = buf.getMagnitude(0, 0, buf.getNumSamples());
    if (pk > 0.99f) buf.applyGain(0.99f / pk);
}

void LivePlayer::setBufferA(const juce::AudioBuffer<float>& mono, double sr) {
    juce::AudioBuffer<float> tmp; tmp.makeCopyOf(mono); rmsNormalise(tmp);
    const juce::ScopedLock sl(lock);
    bufA = std::move(tmp);
    if (sr > 0.0) srcRateA = sr;
}

void LivePlayer::setBufferB(const juce::AudioBuffer<float>& mono, double sr) {
    juce::AudioBuffer<float> tmp; tmp.makeCopyOf(mono); rmsNormalise(tmp);
    const juce::ScopedLock sl(lock);
    bufB = std::move(tmp);
    if (sr > 0.0) srcRateB = sr;
}

void LivePlayer::audioDeviceAboutToStart(juce::AudioIODevice* device) {
    if (device != nullptr && device->getCurrentSampleRate() > 0.0)
        deviceRate.store(device->getCurrentSampleRate());
}

void LivePlayer::audioDeviceIOCallbackWithContext(const float* const*, int,
                                                  float* const* outputChannelData,
                                                  int numOutputChannels, int numSamples,
                                                  const juce::AudioIODeviceCallbackContext&) {
    for (int ch = 0; ch < numOutputChannels; ++ch)
        if (outputChannelData[ch]) juce::FloatVectorOperations::clear(outputChannelData[ch], numSamples);

    cbCount.fetch_add(1);

    if (!playing.load()) return;

    const juce::ScopedTryLock stl(lock);
    if (!stl.isLocked()) return;

    const bool isA = (source.load() == Source::A);
    const juce::AudioBuffer<float>& src = isA ? bufA : bufB;
    const int n = src.getNumSamples();
    if (n == 0) return;
    const float* in = src.getReadPointer(0);

    // Resample from the buffer's source rate to the device rate (linear interp),
    // so playback pitch is correct regardless of the WAV / render sample rate.
    const double dr = deviceRate.load();
    const double srcRate = isA ? srcRateA : srcRateB;
    const double ratio = (dr > 0.0) ? (srcRate / dr) : 1.0;

    float pk = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        if (readPos >= n) readPos -= n;            // loop
        const int i0 = (int) readPos;
        const int i1 = (i0 + 1 < n) ? i0 + 1 : 0;
        const float frac = (float) (readPos - i0);
        const float s = in[i0] * (1.0f - frac) + in[i1] * frac;
        pk = juce::jmax(pk, std::abs(s));
        for (int ch = 0; ch < numOutputChannels; ++ch)
            if (outputChannelData[ch]) outputChannelData[ch][i] = s;
        readPos += ratio;
    }
    float prev = outPeak.load();
    while (pk > prev && !outPeak.compare_exchange_weak(prev, pk)) {}
}

float LivePlayer::currentSrcPeak() const {
    const juce::ScopedLock sl(lock);
    const juce::AudioBuffer<float>& src = (source.load() == Source::A) ? bufA : bufB;
    return src.getNumSamples() > 0 ? src.getMagnitude(0, 0, src.getNumSamples()) : 0.0f;
}

} // namespace vms
