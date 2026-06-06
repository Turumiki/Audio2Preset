#include "Loss.h"
#include <cmath>

namespace vms {

Loss::Loss() {
    for (int sz : sizes) {
        const int order = (int) std::round(std::log2((double) sz));
        ffts.emplace_back(order);
        std::vector<float> w((size_t) sz);
        for (int i = 0; i < sz; ++i)
            w[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * juce::MathConstants<float>::pi * i / (sz - 1));
        windows.push_back(std::move(w));
    }
}

float Loss::peakAbs(const juce::AudioBuffer<float>& buf) {
    float pk = 0.0f;
    for (int ch = 0; ch < buf.getNumChannels(); ++ch)
        pk = juce::jmax(pk, buf.getMagnitude(ch, 0, buf.getNumSamples()));
    return pk;
}

void Loss::peakNormalize(juce::AudioBuffer<float>& buf) {
    const float pk = peakAbs(buf);
    if (pk > 1.0e-9f)
        buf.applyGain(1.0f / pk);
}

float Loss::stftScale(const float* a, int na, const float* b, int nb, int fftSize) {
    // Locate matching FFT/window for this size.
    int idx = -1;
    for (size_t i = 0; i < sizes.size(); ++i)
        if (sizes[i] == fftSize) { idx = (int) i; break; }
    if (idx < 0) return 0.0f;

    auto& fft = ffts[(size_t) idx];
    const auto& win = windows[(size_t) idx];
    const int hop = fftSize / 4;
    const int numBins = fftSize / 2 + 1;

    std::vector<float> fa((size_t) fftSize * 2, 0.0f);
    std::vector<float> fb((size_t) fftSize * 2, 0.0f);

    const int maxN = juce::jmax(na, nb);

    // Pass 1: collect magnitudes per frame; track each signal's global max so we
    // can normalise both spectrograms to [0,1]. Quiet/silent frames then weigh ~0
    // (no noise amplification), and the LINEAR term is dominated by the strong
    // partials — so broadband noise can't cheaply "cover" the spectrum.
    std::vector<float> magsA, magsB;
    magsA.reserve(1 << 16); magsB.reserve(1 << 16);
    float gMaxA = 1.0e-9f, gMaxB = 1.0e-9f;

    for (int start = 0; start + fftSize <= maxN + hop; start += hop) {
        std::fill(fa.begin(), fa.end(), 0.0f);
        std::fill(fb.begin(), fb.end(), 0.0f);
        for (int i = 0; i < fftSize; ++i) {
            const int s = start + i;
            if (s < na) fa[(size_t) i] = a[s] * win[(size_t) i];
            if (s < nb) fb[(size_t) i] = b[s] * win[(size_t) i];
        }
        fft.performRealOnlyForwardTransform(fa.data());
        fft.performRealOnlyForwardTransform(fb.data());

        for (int k = 0; k < numBins; ++k) {
            const float reA = fa[(size_t) (2 * k)], imA = fa[(size_t) (2 * k + 1)];
            const float reB = fb[(size_t) (2 * k)], imB = fb[(size_t) (2 * k + 1)];
            const float magA = std::sqrt(reA * reA + imA * imA);
            const float magB = std::sqrt(reB * reB + imB * imB);
            magsA.push_back(magA); magsB.push_back(magB);
            gMaxA = juce::jmax(gMaxA, magA);
            gMaxB = juce::jmax(gMaxB, magB);
        }
    }

    // Pass 2: normalised linear + log L1 (DDSP-style; linear term anchors peaks).
    const float invA = 1.0f / gMaxA, invB = 1.0f / gMaxB;
    const float eps = 1.0e-4f;
    const double logWeight = 0.5;
    double acc = 0.0;
    const size_t count = magsA.size();
    for (size_t i = 0; i < count; ++i) {
        const float nA = magsA[i] * invA;
        const float nB = magsB[i] * invB;
        acc += std::fabs(nA - nB)
             + logWeight * std::fabs(std::log(nA + eps) - std::log(nB + eps));
    }
    return (count > 0) ? (float) (acc / (double) count) : 0.0f;
}

float Loss::multiScaleStft(const juce::AudioBuffer<float>& a,
                           const juce::AudioBuffer<float>& b) {
    const float* pa = a.getReadPointer(0);
    const float* pb = b.getReadPointer(0);
    const int na = a.getNumSamples();
    const int nb = b.getNumSamples();
    double sum = 0.0;
    for (int sz : sizes) sum += stftScale(pa, na, pb, nb, sz);
    return (float) (sum / (double) sizes.size());
}

float Loss::logRmsEnvelope(const juce::AudioBuffer<float>& a,
                           const juce::AudioBuffer<float>& b) {
    const int frame = 1024, hop = 256;
    auto envOf = [&](const juce::AudioBuffer<float>& buf) {
        const float* p = buf.getReadPointer(0);
        const int n = buf.getNumSamples();
        std::vector<float> e;
        for (int start = 0; start + frame <= n + hop; start += hop) {
            double s = 0.0; int cnt = 0;
            for (int i = 0; i < frame; ++i) {
                const int idx = start + i;
                if (idx < n) { s += (double) p[idx] * p[idx]; ++cnt; }
            }
            const double rms = (cnt > 0) ? std::sqrt(s / cnt) : 0.0;
            e.push_back((float) std::log(rms + 1.0e-5));
        }
        return e;
    };
    auto ea = envOf(a), eb = envOf(b);
    const size_t m = juce::jmin(ea.size(), eb.size());
    if (m == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < m; ++i) { const double d = ea[i] - eb[i]; acc += d * d; }
    return (float) (acc / (double) m);
}

float Loss::perceptual(const juce::AudioBuffer<float>& a, const juce::AudioBuffer<float>& b) {
    const int fftSize = 1024;
    int idx = -1;
    for (size_t i = 0; i < sizes.size(); ++i) if (sizes[i] == fftSize) { idx = (int) i; break; }
    if (idx < 0) return 0.0f;
    auto& fft = ffts[(size_t) idx];
    const auto& win = windows[(size_t) idx];
    const int hop = 512, numBins = fftSize / 2 + 1;

    auto traj = [&](const juce::AudioBuffer<float>& buf, std::vector<float>& cen, std::vector<float>& flat) {
        const float* p = buf.getReadPointer(0);
        const int n = buf.getNumSamples();
        std::vector<float> fa((size_t) fftSize * 2, 0.0f);
        for (int start = 0; start + fftSize <= n; start += hop) {
            std::fill(fa.begin(), fa.end(), 0.0f);
            for (int i = 0; i < fftSize; ++i) fa[(size_t) i] = p[start + i] * win[(size_t) i];
            fft.performRealOnlyForwardTransform(fa.data());
            double num = 0, den = 0, logsum = 0, arsum = 0; int cnt = 0;
            for (int k = 1; k < numBins; ++k) {
                const float re = fa[(size_t) (2 * k)], im = fa[(size_t) (2 * k + 1)];
                const float m = std::sqrt(re * re + im * im);
                num += (double) k * m; den += m;
                logsum += std::log(m + 1.0e-9); arsum += m; ++cnt;
            }
            cen.push_back(den > 1.0e-9 ? (float) (num / den / numBins) : 0.0f);   // brightness 0..1
            const float gm = (cnt > 0) ? std::exp((float) (logsum / cnt)) : 0.0f;
            const float am = (cnt > 0) ? (float) (arsum / cnt) : 0.0f;
            flat.push_back(am > 1.0e-9f ? gm / am : 0.0f);                          // flatness 0..1
        }
    };
    std::vector<float> ca, fa_, cb, fb;
    traj(a, ca, fa_); traj(b, cb, fb);
    const size_t m = juce::jmin(ca.size(), cb.size());
    if (m == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < m; ++i) acc += std::fabs(ca[i] - cb[i]) + std::fabs(fa_[i] - fb[i]);
    return (float) (acc / (double) m);
}

// Time-averaged log-magnitude spectrum, log-frequency binned. Averaging across all frames
// smooths out the render-to-render fluctuation (phase, unison beating) of non-deterministic
// synths, so two renders of the SAME patch give nearly identical vectors (low self-noise).
std::vector<float> Loss::avgLogSpectrum(const juce::AudioBuffer<float>& buf, int bands) {
    const int fftSize = 2048, hop = 512;
    int idx = -1;
    for (size_t i = 0; i < sizes.size(); ++i) if (sizes[i] == fftSize) { idx = (int) i; break; }
    if (idx < 0) return {};
    auto& fft = ffts[(size_t) idx];
    const auto& win = windows[(size_t) idx];
    const int numBins = fftSize / 2 + 1;
    const float* p = buf.getReadPointer(0);
    const int n = buf.getNumSamples();

    std::vector<double> acc((size_t) numBins, 0.0); int frames = 0;
    std::vector<float> fa((size_t) fftSize * 2, 0.0f);
    for (int start = 0; start + fftSize <= n; start += hop) {
        std::fill(fa.begin(), fa.end(), 0.0f);
        for (int i = 0; i < fftSize; ++i) fa[(size_t) i] = p[start + i] * win[(size_t) i];
        fft.performRealOnlyForwardTransform(fa.data());
        for (int k = 0; k < numBins; ++k) {
            const float re = fa[(size_t) (2 * k)], im = fa[(size_t) (2 * k + 1)];
            acc[(size_t) k] += std::sqrt(re * re + im * im);
        }
        ++frames;
    }
    if (frames == 0) return std::vector<float>((size_t) bands, 0.0f);
    for (auto& v : acc) v /= frames;   // time-average magnitude per bin

    // Log-frequency binning into `bands` (bin 1..numBins-1 -> log buckets).
    std::vector<float> out((size_t) bands, 0.0f);
    std::vector<int> cnt((size_t) bands, 0);
    const double lo = std::log((double) 1), hi = std::log((double) (numBins - 1));
    for (int k = 1; k < numBins; ++k) {
        const int bnd = juce::jlimit(0, bands - 1, (int) ((std::log((double) k) - lo) / (hi - lo) * (bands - 1)));
        out[(size_t) bnd] += (float) acc[(size_t) k]; cnt[(size_t) bnd]++;
    }
    for (int b = 0; b < bands; ++b) if (cnt[(size_t) b] > 0) out[(size_t) b] /= cnt[(size_t) b];
    // log + normalise by max.
    float mx = 1.0e-9f; for (float v : out) mx = juce::jmax(mx, v);
    for (auto& v : out) v = std::log(v / mx + 1.0e-4f);
    return out;
}

float Loss::combined(const juce::AudioBuffer<float>& target,
                     const juce::AudioBuffer<float>& candidate,
                     float envWeight, float percWeight, bool robust) {
    juce::AudioBuffer<float> a, b;
    a.makeCopyOf(target);
    b.makeCopyOf(candidate);
    // Normalise BOTH by the TARGET's peak (shared reference) — NOT each independently.
    // Independent peak-normalisation let near-silent/thin candidates get blown up to
    // unit level and score deceptively well (a degenerate local optimum). Sharing the
    // reference keeps the comparison level-aware: a quiet candidate stays quiet and is
    // penalised, forcing the optimiser to produce real sound at the right level.
    const float ref = juce::jmax(peakAbs(a), 1.0e-9f);
    a.applyGain(1.0f / ref);
    b.applyGain(1.0f / ref);
    const float env  = logRmsEnvelope(a, b);
    if (robust) {
        // Time-averaged spectrum (L1) instead of frame-by-frame STFT: invariant to the
        // per-frame fluctuation of non-deterministic synths.
        auto sa = avgLogSpectrum(a), sb = avgLogSpectrum(b);
        const size_t m = juce::jmin(sa.size(), sb.size());
        double spec = 0.0; for (size_t i = 0; i < m; ++i) spec += std::fabs(sa[i] - sb[i]);
        spec = (m > 0) ? spec / (double) m : 0.0;
        return (float) spec + envWeight * env;
    }
    const float stft = multiScaleStft(a, b);
    const float perc = (percWeight > 0.0f) ? perceptual(a, b) : 0.0f;
    return stft + envWeight * env + percWeight * perc;
}

std::vector<float> Loss::magnitudeSpectrum(const juce::AudioBuffer<float>& buf,
                                           int fftOrder, double sampleRate,
                                           double& binHz) {
    const int fftSize = 1 << fftOrder;
    const int numBins = fftSize / 2 + 1;
    juce::dsp::FFT fft(fftOrder);
    std::vector<float> data((size_t) fftSize * 2, 0.0f);
    const float* p = buf.getReadPointer(0);
    const int n = buf.getNumSamples();

    // Window a slice from the (assumed steady) middle of the buffer.
    const int start = juce::jmax(0, (n - fftSize) / 2);
    for (int i = 0; i < fftSize; ++i) {
        const float w = 0.5f - 0.5f * std::cos(2.0f * juce::MathConstants<float>::pi * i / (fftSize - 1));
        const int s = start + i;
        data[(size_t) i] = (s < n) ? p[s] * w : 0.0f;
    }
    fft.performRealOnlyForwardTransform(data.data());

    std::vector<float> mags((size_t) numBins, 0.0f);
    for (int k = 0; k < numBins; ++k) {
        const float re = data[(size_t) (2 * k)], im = data[(size_t) (2 * k + 1)];
        mags[(size_t) k] = std::sqrt(re * re + im * im);
    }
    binHz = sampleRate / fftSize;
    return mags;
}

} // namespace vms
