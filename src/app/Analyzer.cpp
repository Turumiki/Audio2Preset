#include "Analyzer.h"
#include <juce_dsp/juce_dsp.h>
#include <cmath>

namespace vms {

SpecMatrix computeSpectrogram(const juce::AudioBuffer<float>& buf, double sr) {
    const int order = 10, fftSize = 1 << order, hop = fftSize / 2;
    const int n = buf.getNumSamples();
    if (n < fftSize) return {};

    juce::dsp::FFT fft(order);
    std::vector<float> win((size_t) fftSize);
    for (int i = 0; i < fftSize; ++i)
        win[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * juce::MathConstants<float>::pi * i / (fftSize - 1));

    std::vector<int> starts;
    for (int s = 0; s + fftSize <= n; s += hop) starts.push_back(s);
    if (starts.empty()) return {};

    const int Fmax = 48;
    std::vector<int> sel;
    if ((int) starts.size() <= Fmax) sel = starts;
    else for (int i = 0; i < Fmax; ++i)
        sel.push_back(starts[(size_t) ((long long) i * (starts.size() - 1) / (Fmax - 1))]);

    const int K = 96;
    const double logMin = std::log10(40.0), logMax = std::log10(18000.0);
    const int numBins = fftSize / 2 + 1;

    SpecMatrix frames;
    std::vector<float> data((size_t) fftSize * 2, 0.0f);
    float globalPeak = 1.0e-9f;

    for (int st : sel) {
        std::fill(data.begin(), data.end(), 0.0f);
        const float* p = buf.getReadPointer(0);
        for (int i = 0; i < fftSize; ++i) data[(size_t) i] = p[st + i] * win[(size_t) i];
        fft.performRealOnlyForwardTransform(data.data());

        std::vector<float> row((size_t) K, 0.0f);
        for (int k = 0; k < K; ++k) {
            const double frac = (double) k / (K - 1);
            const double freq = std::pow(10.0, logMin + frac * (logMax - logMin));
            int b = (int) (freq * fftSize / sr);
            b = juce::jlimit(1, numBins - 1, b);
            const float re = data[(size_t) (2 * b)], im = data[(size_t) (2 * b + 1)];
            const float mag = std::sqrt(re * re + im * im);
            row[(size_t) k] = mag;
            globalPeak = juce::jmax(globalPeak, mag);
        }
        frames.push_back(std::move(row));
    }
    for (auto& r : frames)
        for (auto& v : r) {
            const float db = 20.0f * std::log10(juce::jmax(1.0e-6f, v / globalPeak));
            v = juce::jlimit(0.0f, 1.0f, (db + 80.0f) / 80.0f);
        }
    return frames;
}

void Spectrogram3D::setData(SpecMatrix target, SpecMatrix best) {
    targetSpec = std::move(target);
    bestSpec = std::move(best);
    repaint();
}

void Spectrogram3D::drawWaterfall(juce::Graphics& g, const SpecMatrix& m, juce::Colour base) {
    const int F = (int) m.size();
    if (F < 1) return;
    const int K = (int) m[0].size();
    if (K < 2) return;

    const float W = (float) getWidth(), H = (float) getHeight();
    const float mL = 10.0f, mB = 20.0f, mT = 10.0f, mR = 10.0f;
    const float plotW = W - mL - mR, plotH = H - mT - mB;
    const float depthX = plotW * 0.30f, depthY = -plotH * 0.42f;
    const float usableW = plotW - depthX;
    const float magH = plotH * 0.55f;

    // Painter's algorithm: oldest frame (f=0) at the back, newest at the front.
    for (int f = 0; f < F; ++f) {
        const float t = (F > 1) ? (float) f / (F - 1) : 0.0f;
        const float sx = mL + (1.0f - t) * depthX;
        const float sy = (H - mB) + (1.0f - t) * depthY;

        juce::Path p;
        for (int k = 0; k < K; ++k) {
            const float x = sx + (float) k / (K - 1) * usableW;
            const float y = sy - m[(size_t) f][(size_t) k] * magH;
            if (k == 0) p.startNewSubPath(x, y);
            else        p.lineTo(x, y);
        }
        g.setColour(base.withAlpha(0.25f + 0.7f * t));
        g.strokePath(p, juce::PathStrokeType(1.0f));
    }
}

void Spectrogram3D::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff0e1014));

    // Frequency reference lines (front plane).
    g.setColour(juce::Colours::white.withAlpha(0.06f));
    const float W = (float) getWidth(), H = (float) getHeight();
    const float mL = 10.0f, mB = 20.0f;
    const float plotW = W - 20.0f;
    const float depthX = (plotW) * 0.30f, usableW = (plotW) - depthX;
    const double logMin = std::log10(40.0), logMax = std::log10(18000.0);
    for (double f : { 100.0, 1000.0, 10000.0 }) {
        const float frac = (float) ((std::log10(f) - logMin) / (logMax - logMin));
        const float x = mL + frac * usableW;
        g.drawVerticalLine((int) x, H - mB - 4, H - mB);
    }

    drawWaterfall(g, targetSpec, juce::Colours::grey);
    drawWaterfall(g, bestSpec, juce::Colour(0xff35d0ce));

    g.setColour(juce::Colours::grey);     g.setFont(12.0f);
    g.drawText("target", getLocalBounds().removeFromTop(16).reduced(6, 0), juce::Justification::topLeft);
    g.setColour(juce::Colour(0xff35d0ce));
    g.drawText("best", getLocalBounds().removeFromTop(32).reduced(6, 0), juce::Justification::topLeft);
    g.setColour(juce::Colours::white.withAlpha(0.4f));
    g.drawText("freq -> / depth = time / height = level", getLocalBounds().removeFromBottom(16).reduced(6, 0),
               juce::Justification::bottomRight);
}

void SpectrumView::setSpectra(std::vector<float> target, std::vector<float> best, double bin) {
    targetMags = std::move(target);
    bestMags = std::move(best);
    binHz = (bin > 0.0) ? bin : 1.0;
    repaint();
}

void SpectrumView::drawCurve(juce::Graphics& g, const std::vector<float>& mags, juce::Colour c) {
    if (mags.size() < 2) return;
    const float w = (float) getWidth();
    const float h = (float) getHeight();
    const double fMin = 20.0, fMax = 20000.0;
    const double logMin = std::log10(fMin), logMax = std::log10(fMax);
    const float dbFloor = -90.0f, dbTop = 0.0f;

    // Normalise to peak for shape comparison.
    float peak = 1.0e-9f;
    for (float m : mags) peak = juce::jmax(peak, m);

    juce::Path path;
    bool started = false;
    for (size_t k = 1; k < mags.size(); ++k) {
        const double freq = k * binHz;
        if (freq < fMin || freq > fMax) continue;
        const float x = (float) ((std::log10(freq) - logMin) / (logMax - logMin)) * w;
        const float ratio = juce::jmax(1.0e-6f, mags[k] / peak);
        const float db = juce::jmax(dbFloor, 20.0f * std::log10(ratio));
        const float y = juce::jmap(db, dbFloor, dbTop, h, 0.0f);
        if (!started) { path.startNewSubPath(x, y); started = true; }
        else path.lineTo(x, y);
    }
    g.setColour(c);
    g.strokePath(path, juce::PathStrokeType(1.5f));
}

void SpectrumView::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff121418));
    g.setColour(juce::Colours::white.withAlpha(0.08f));
    for (double f : { 100.0, 1000.0, 10000.0 }) {
        const double logMin = std::log10(20.0), logMax = std::log10(20000.0);
        const float x = (float) ((std::log10(f) - logMin) / (logMax - logMin)) * getWidth();
        g.drawVerticalLine((int) x, 0.0f, (float) getHeight());
    }
    drawCurve(g, targetMags, juce::Colours::grey);
    drawCurve(g, bestMags, juce::Colour(0xff35d0ce));
    g.setColour(juce::Colours::grey);
    g.setFont(12.0f);
    g.drawText("target", getLocalBounds().removeFromTop(16).reduced(6, 0), juce::Justification::topLeft);
    g.setColour(juce::Colour(0xff35d0ce));
    g.drawText("best", getLocalBounds().removeFromTop(32).reduced(6, 0), juce::Justification::topLeft);
}

// ---- 2D spectrogram heatmap -----------------------------------------------
juce::Colour SpectrogramHeat::heatColour(float v) {
    v = juce::jlimit(0.0f, 1.0f, v);
    // Inferno-ish ramp through 5 stops.
    static const float stops[5][3] = {
        {  0.0f,   0.0f,   8.0f },
        { 60.0f,  12.0f,  90.0f },
        {160.0f,  30.0f,  90.0f },
        {240.0f, 110.0f,  30.0f },
        {250.0f, 250.0f, 180.0f }
    };
    const float x = v * 4.0f;
    const int i = juce::jlimit(0, 3, (int) x);
    const float f = x - i;
    auto lerp = [&](int c) { return stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f; };
    return juce::Colour((juce::uint8) lerp(0), (juce::uint8) lerp(1), (juce::uint8) lerp(2));
}

juce::Image SpectrogramHeat::buildImage(const SpecMatrix& m) {
    const int F = (int) m.size();
    if (F < 1) return {};
    const int K = (int) m[0].size();
    juce::Image img(juce::Image::RGB, F, K, false);
    for (int f = 0; f < F; ++f)
        for (int k = 0; k < K; ++k)
            img.setPixelAt(f, K - 1 - k, heatColour(m[(size_t) f][(size_t) k])); // low freq at bottom
    return img;
}

void SpectrogramHeat::setData(const SpecMatrix& target, const SpecMatrix& best) {
    targetImg = buildImage(target);
    bestImg = buildImage(best);
    repaint();
}

void SpectrogramHeat::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff0e1014));
    g.setImageResamplingQuality(juce::Graphics::mediumResamplingQuality);

    auto area = getLocalBounds().reduced(4);
    const int labelW = 8;
    area.removeFromLeft(labelW);
    auto top = area.removeFromTop(area.getHeight() / 2);
    auto gap = top.removeFromBottom(4); juce::ignoreUnused(gap);
    auto bottom = area;

    auto drawOne = [&](juce::Image& img, juce::Rectangle<int> r, const juce::String& label) {
        if (img.isValid())
            g.drawImage(img, r.toFloat(), juce::RectanglePlacement::stretchToFit);
        else { g.setColour(juce::Colours::darkgrey); g.drawRect(r); }
        g.setColour(juce::Colours::white.withAlpha(0.85f));
        g.setFont(12.0f);
        g.drawText(label, r.reduced(4, 2), juce::Justification::topLeft);
    };
    drawOne(targetImg, top, "target");
    drawOne(bestImg, bottom, "best");
}

void LossCurveView::addPoint(double loss) {
    points.push_back(loss);
    repaint();
}

void LossCurveView::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff121418));
    if (points.size() < 2) {
        g.setColour(juce::Colours::grey);
        g.setFont(12.0f);
        g.drawText("loss", getLocalBounds().reduced(6), juce::Justification::topLeft);
        return;
    }
    double lo = points[0], hi = points[0];
    for (double v : points) { lo = juce::jmin(lo, v); hi = juce::jmax(hi, v); }
    if (hi - lo < 1.0e-9) hi = lo + 1.0e-9;

    const float w = (float) getWidth(), h = (float) getHeight();
    juce::Path path;
    for (size_t i = 0; i < points.size(); ++i) {
        const float x = (float) i / (float) (points.size() - 1) * w;
        const float y = juce::jmap((float) points[i], (float) lo, (float) hi, h - 2, 2.0f);
        if (i == 0) path.startNewSubPath(x, y); else path.lineTo(x, y);
    }
    g.setColour(juce::Colour(0xffe0a030));
    g.strokePath(path, juce::PathStrokeType(1.5f));
    g.setColour(juce::Colours::grey);
    g.setFont(12.0f);
    g.drawText("loss " + juce::String(points.back(), 4), getLocalBounds().reduced(6),
               juce::Justification::topLeft);
}

} // namespace vms
