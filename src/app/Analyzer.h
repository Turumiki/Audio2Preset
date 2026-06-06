#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>

namespace vms {

// STFT spectrogram matrix [frame][freqBin], normalised log-magnitude in [0,1],
// log-spaced frequency bins. Used by the 3D waterfall view.
using SpecMatrix = std::vector<std::vector<float>>;
SpecMatrix computeSpectrogram(const juce::AudioBuffer<float>& buf, double sampleRate);

// 3D waterfall spectrogram: x = frequency (log), depth = time, height = magnitude.
// Target drawn grey (reference), current best drawn cyan (converging toward it).
class Spectrogram3D : public juce::Component {
public:
    void setData(SpecMatrix target, SpecMatrix best);
    void paint(juce::Graphics& g) override;
private:
    SpecMatrix targetSpec, bestSpec;
    void drawWaterfall(juce::Graphics& g, const SpecMatrix& m, juce::Colour base);
};

// Classic 2D spectrogram heatmap (x = time, y = frequency, colour = level).
// Shows target (top) and current best (bottom) stacked for comparison.
class SpectrogramHeat : public juce::Component {
public:
    void setData(const SpecMatrix& target, const SpecMatrix& best);
    void paint(juce::Graphics& g) override;
    static juce::Colour heatColour(float v01);
private:
    juce::Image targetImg, bestImg;
    static juce::Image buildImage(const SpecMatrix& m);
};

// Overlaid magnitude spectra: target (grey) vs current best (cyan), log-frequency
// x-axis, dB y-axis.
class SpectrumView : public juce::Component {
public:
    void setSpectra(std::vector<float> target, std::vector<float> best, double binHz);
    void paint(juce::Graphics& g) override;
private:
    std::vector<float> targetMags, bestMags;
    double binHz = 1.0;
    void drawCurve(juce::Graphics& g, const std::vector<float>& mags, juce::Colour c);
};

// Loss-over-generations curve (auto-scaling y).
class LossCurveView : public juce::Component {
public:
    void addPoint(double loss);
    void clear() { points.clear(); repaint(); }
    void paint(juce::Graphics& g) override;
private:
    std::vector<double> points;
};

} // namespace vms
