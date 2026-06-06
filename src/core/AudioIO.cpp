#include "AudioIO.h"

namespace vms {

bool loadAudioFileMono(const juce::File& file, juce::AudioBuffer<float>& out, double& sampleRate) {
    juce::AudioFormatManager fm;
    fm.registerBasicFormats();   // WAV, AIFF, FLAC, Ogg
   #if JUCE_USE_MP3AUDIOFORMAT
    fm.registerFormat(new juce::MP3AudioFormat(), false);   // MP3 (JUCE built-in decoder)
   #endif
    std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(file));
    if (reader == nullptr) return false;

    const int numSamples = (int) reader->lengthInSamples;
    if (numSamples <= 0) return false;
    sampleRate = reader->sampleRate;

    juce::AudioBuffer<float> tmp((int) reader->numChannels, numSamples);
    reader->read(&tmp, 0, numSamples, 0, true, true);

    out.setSize(1, numSamples);
    out.clear();
    float* dst = out.getWritePointer(0);
    const int nch = tmp.getNumChannels();
    for (int ch = 0; ch < nch; ++ch) {
        const float* src = tmp.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i) dst[i] += src[i] / (float) nch;
    }
    return true;
}

juce::AudioBuffer<float> resampleMono(const juce::AudioBuffer<float>& in, double srcSr, double dstSr) {
    if (in.getNumSamples() == 0 || srcSr <= 0.0 || dstSr <= 0.0 || std::abs(srcSr - dstSr) < 1.0) {
        juce::AudioBuffer<float> copy; copy.makeCopyOf(in); return copy;
    }
    const double ratio = srcSr / dstSr;                      // input samples per output sample
    const int outLen = juce::jmax(1, (int) (in.getNumSamples() / ratio));
    juce::AudioBuffer<float> out(1, outLen);
    juce::LagrangeInterpolator interp;
    interp.process(ratio, in.getReadPointer(0), out.getWritePointer(0), outLen);
    return out;
}

bool writeWavMono(const juce::File& file, const juce::AudioBuffer<float>& buf, double sampleRate) {
    file.deleteFile();
    std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
    if (stream == nullptr) return false;

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer(
        wav.createWriterFor(stream.get(), sampleRate, 1, 24, {}, 0));
    if (writer == nullptr) return false;
    stream.release(); // writer takes ownership

    return writer->writeFromAudioSampleBuffer(buf, 0, buf.getNumSamples());
}

} // namespace vms
