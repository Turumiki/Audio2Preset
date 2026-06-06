// Minimal Bayesian optimisation (Gaussian-process surrogate + Expected
// Improvement) for bounded minimisation on [0,1]^N. Dependency-free.
//
// Intended for LOW dimension (<= ~15) and EXPENSIVE evaluations (e.g. a plugin
// that must be re-instantiated per render): BO reaches a good optimum in far
// fewer evaluations than population methods there. Batch suggestions keep the
// worker pool busy. GP fit is O(n^3); keep the sample budget modest (<= ~250).

#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace bo {

class BayesOpt {
public:
    BayesOpt(int dim, unsigned seed) : N(dim), rng(seed) {}

    void addSample(const std::vector<double>& x, double y) {
        X.push_back(x); Y.push_back(y); dirty = true;
        if (y < bestY) { bestY = y; bestX = x; }
    }
    int count() const { return (int) X.size(); }
    double bestValue() const { return bestY; }
    const std::vector<double>& best() const { return bestX; }

    std::vector<double> randomPoint() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> x((size_t) N);
        for (auto& v : x) v = u(rng);
        return x;
    }

    // Top-k Expected-Improvement candidates (sampled), for parallel evaluation.
    std::vector<std::vector<double>> suggestBatch(int k, int nCandidates = 768) {
        fit();
        std::vector<std::pair<double, std::vector<double>>> cands;
        cands.reserve((size_t) nCandidates);
        for (int c = 0; c < nCandidates; ++c) {
            auto cand = (c % 3 == 0 && !bestX.empty()) ? perturb(bestX, 0.12) : randomPoint();
            double mu, s2; predict(cand, mu, s2);
            const double ei = expectedImprovement(mu, std::sqrt(std::max(1.0e-12, s2)));
            cands.push_back({ ei, std::move(cand) });
        }
        std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<std::vector<double>> out;
        for (int i = 0; i < k && i < (int) cands.size(); ++i) out.push_back(cands[(size_t) i].second);
        return out;
    }

private:
    int N;
    std::mt19937 rng;
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    std::vector<std::vector<double>> L;   // Cholesky factor
    std::vector<double> alpha;
    double yMean = 0.0;
    double bestY = 1.0e30;
    std::vector<double> bestX;
    bool dirty = true;

    const double lengthscale = 0.2;
    const double sigmaF = 1.0;
    const double sigmaN = 1.0e-3;

    double kernel(const std::vector<double>& a, const std::vector<double>& b) const {
        double d2 = 0.0;
        for (int i = 0; i < N; ++i) { const double e = a[(size_t) i] - b[(size_t) i]; d2 += e * e; }
        return sigmaF * std::exp(-d2 / (2.0 * lengthscale * lengthscale));
    }

    std::vector<double> perturb(const std::vector<double>& base, double s) {
        std::normal_distribution<double> g(0.0, s);
        std::vector<double> x((size_t) N);
        for (int i = 0; i < N; ++i)
            x[(size_t) i] = std::min(1.0, std::max(0.0, base[(size_t) i] + g(rng)));
        return x;
    }

    void fit() {
        if (!dirty) return;
        dirty = false;
        const int n = (int) X.size();
        if (n == 0) return;

        yMean = 0.0; for (double v : Y) yMean += v; yMean /= n;

        std::vector<std::vector<double>> K((size_t) n, std::vector<double>((size_t) n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                K[(size_t) i][(size_t) j] = kernel(X[(size_t) i], X[(size_t) j]) + (i == j ? sigmaN : 0.0);

        // Cholesky: K = L L^T
        L.assign((size_t) n, std::vector<double>((size_t) n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                double s = K[(size_t) i][(size_t) j];
                for (int k = 0; k < j; ++k) s -= L[(size_t) i][(size_t) k] * L[(size_t) j][(size_t) k];
                if (i == j) L[(size_t) i][(size_t) j] = std::sqrt(std::max(1.0e-12, s));
                else        L[(size_t) i][(size_t) j] = s / L[(size_t) j][(size_t) j];
            }

        // alpha = K^-1 (Y - mean)
        std::vector<double> z((size_t) n);
        for (int i = 0; i < n; ++i) {
            double s = Y[(size_t) i] - yMean;
            for (int k = 0; k < i; ++k) s -= L[(size_t) i][(size_t) k] * z[(size_t) k];
            z[(size_t) i] = s / L[(size_t) i][(size_t) i];
        }
        alpha.assign((size_t) n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            double s = z[(size_t) i];
            for (int k = i + 1; k < n; ++k) s -= L[(size_t) k][(size_t) i] * alpha[(size_t) k];
            alpha[(size_t) i] = s / L[(size_t) i][(size_t) i];
        }
    }

    void predict(const std::vector<double>& x, double& mu, double& s2) {
        const int n = (int) X.size();
        if (n == 0) { mu = 0.0; s2 = 1.0; return; }
        std::vector<double> ks((size_t) n);
        for (int i = 0; i < n; ++i) ks[(size_t) i] = kernel(x, X[(size_t) i]);
        mu = yMean;
        for (int i = 0; i < n; ++i) mu += ks[(size_t) i] * alpha[(size_t) i];
        // v = L^-1 ks ; var = k(x,x) - v.v
        std::vector<double> v((size_t) n);
        for (int i = 0; i < n; ++i) {
            double s = ks[(size_t) i];
            for (int k = 0; k < i; ++k) s -= L[(size_t) i][(size_t) k] * v[(size_t) k];
            v[(size_t) i] = s / L[(size_t) i][(size_t) i];
        }
        double dot = 0.0; for (int i = 0; i < n; ++i) dot += v[(size_t) i] * v[(size_t) i];
        s2 = std::max(1.0e-12, kernel(x, x) - dot);
    }

    double expectedImprovement(double mu, double s) const {
        const double impr = bestY - mu;          // minimisation
        const double z = impr / s;
        const double Phi = 0.5 * std::erfc(-z / std::sqrt(2.0));
        const double phi = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.14159265358979323846);
        return impr * Phi + s * phi;
    }
};

} // namespace bo
