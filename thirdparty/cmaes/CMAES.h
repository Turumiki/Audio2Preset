// Self-contained header-only CMA-ES (mu/mu_w recombination, rank-mu + rank-one
// covariance update, CSA step-size control) with an internal symmetric Jacobi
// eigensolver. No external dependencies (only the C++ standard library).
//
// Rationale: vendoring Hansen's c-cmaes adds build/network friction. The handoff
// spec explicitly permits a self-written ES fallback. This implementation follows
// the canonical CMA-ES (Hansen, "The CMA Evolution Strategy: A Tutorial").
//
// Search domain is bounded to [0,1]^N. Out-of-bounds samples are mirrored back
// into the box (reflection) which avoids the boundary pile-up bias of clamping.
//
// Usage:
//   cmaes::CMAES opt(N, x0, sigma0, seed);
//   while (!opt.shouldStop(maxEvals)) {
//       auto pop = opt.ask();                 // vector<vector<double>>, each in [0,1]^N
//       std::vector<double> fit(pop.size());
//       for (size_t k=0;k<pop.size();++k) fit[k] = evaluate(pop[k]);
//       opt.tell(fit);
//   }
//   auto best = opt.bestEverX(); double bestF = opt.bestEverF();

#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

namespace cmaes {

class CMAES {
public:
    // lambdaOverride > 0 forces the population size (used by IPOP restarts).
    CMAES(int dim, const std::vector<double>& x0, double sigma0, unsigned seed = 1u,
          int lambdaOverride = 0)
        : N(dim), sigma(sigma0), rng(seed), gauss(0.0, 1.0)
    {
        mean = x0;
        if ((int)mean.size() != N) mean.assign(N, 0.5);

        // Selection / recombination parameters.
        lambda = (lambdaOverride > 0) ? lambdaOverride
                                      : 4 + (int)std::floor(3.0 * std::log((double)N));
        if (lambda < 5) lambda = 5;
        mu = lambda / 2;

        weights.resize(mu);
        double wsum = 0.0;
        for (int i = 0; i < mu; ++i) {
            weights[i] = std::log(mu + 0.5) - std::log((double)(i + 1));
            wsum += weights[i];
        }
        for (auto& w : weights) w /= wsum;
        double wsq = 0.0;
        for (auto w : weights) wsq += w * w;
        mueff = 1.0 / wsq;

        // Adaptation constants.
        cc    = (4.0 + mueff / N) / (N + 4.0 + 2.0 * mueff / N);
        cs    = (mueff + 2.0) / (N + mueff + 5.0);
        c1    = 2.0 / ((N + 1.3) * (N + 1.3) + mueff);
        cmu   = std::min(1.0 - c1,
                         2.0 * (mueff - 2.0 + 1.0 / mueff) / ((N + 2.0) * (N + 2.0) + mueff));
        damps = 1.0 + 2.0 * std::max(0.0, std::sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs;
        chiN  = std::sqrt((double)N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N));

        pc.assign(N, 0.0);
        ps.assign(N, 0.0);

        // Covariance C = B * diag(D^2) * B^T, initialised to identity.
        C.assign(N, std::vector<double>(N, 0.0));
        B.assign(N, std::vector<double>(N, 0.0));
        D.assign(N, 1.0);
        for (int i = 0; i < N; ++i) { C[i][i] = 1.0; B[i][i] = 1.0; }

        eigeneval = 0;
        countEval = 0;
        generation = 0;
        bestF = std::numeric_limits<double>::infinity();
        bestX = mean;

        population.assign(lambda, std::vector<double>(N, 0.0));
        zStore.assign(lambda, std::vector<double>(N, 0.0));
    }

    int populationSize() const { return lambda; }
    long evaluations() const { return countEval; }
    int gen() const { return generation; }
    double stepSize() const { return sigma; }

    // Sample a new population. Returned points are mirrored into [0,1]^N.
    const std::vector<std::vector<double>>& ask() {
        for (int k = 0; k < lambda; ++k) {
            for (int i = 0; i < N; ++i) zStore[k][i] = gauss(rng);
            // y = B * (D .* z)
            std::vector<double> Dz(N);
            for (int i = 0; i < N; ++i) Dz[i] = D[i] * zStore[k][i];
            for (int i = 0; i < N; ++i) {
                double yi = 0.0;
                for (int j = 0; j < N; ++j) yi += B[i][j] * Dz[j];
                double x = mirror01(mean[i] + sigma * yi);
                if (!std::isfinite(x)) x = 0.5; // bulletproof against internal blow-up
                population[k][i] = x;
            }
        }
        return population;
    }

    // Provide fitness for each member of the last ask() population (minimisation).
    void tell(const std::vector<double>& fitness) {
        countEval += lambda;
        ++generation;

        std::vector<int> idx(lambda);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        if (fitness[idx[0]] < bestF) {
            bestF = fitness[idx[0]];
            bestX = population[idx[0]];
        }

        // New mean = weighted average of mu best points.
        std::vector<double> oldMean = mean;
        std::fill(mean.begin(), mean.end(), 0.0);
        for (int i = 0; i < mu; ++i)
            for (int d = 0; d < N; ++d)
                mean[d] += weights[i] * population[idx[i]][d];

        // y_w = (mean - oldMean)/sigma ;  z_w in the C^{-1/2} frame.
        std::vector<double> yw(N), Cinv_yw(N);
        for (int d = 0; d < N; ++d) yw[d] = (mean[d] - oldMean[d]) / sigma;
        // C^{-1/2} * yw = B * diag(1/D) * B^T * yw
        std::vector<double> Bt_yw(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j) s += B[j][i] * yw[j];
            Bt_yw[i] = s / D[i];
        }
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j) s += B[i][j] * Bt_yw[j];
            Cinv_yw[i] = s;
        }

        // Update evolution path ps (conjugate).
        double csFactor = std::sqrt(cs * (2.0 - cs) * mueff);
        for (int d = 0; d < N; ++d)
            ps[d] = (1.0 - cs) * ps[d] + csFactor * Cinv_yw[d];

        double psNorm = norm(ps);
        bool hsig = (psNorm / std::sqrt(1.0 - std::pow(1.0 - cs, 2.0 * countEval / lambda)) / chiN)
                    < (1.4 + 2.0 / (N + 1.0));

        // Update evolution path pc.
        double ccFactor = std::sqrt(cc * (2.0 - cc) * mueff);
        for (int d = 0; d < N; ++d)
            pc[d] = (1.0 - cc) * pc[d] + (hsig ? 1.0 : 0.0) * ccFactor * yw[d];

        // Rank-one + rank-mu covariance update.
        double deltaHsig = (1.0 - (hsig ? 1.0 : 0.0)) * cc * (2.0 - cc);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                double rankOne = pc[i] * pc[j];
                double rankMu = 0.0;
                for (int k = 0; k < mu; ++k) {
                    double yik = (population[idx[k]][i] - oldMean[i]) / sigma;
                    double yjk = (population[idx[k]][j] - oldMean[j]) / sigma;
                    rankMu += weights[k] * yik * yjk;
                }
                double val = (1.0 - c1 - cmu) * C[i][j]
                           + c1 * (rankOne + deltaHsig * C[i][j])
                           + cmu * rankMu;
                C[i][j] = val;
                C[j][i] = val;
            }
        }

        // Step-size update (CSA).
        sigma *= std::exp((cs / damps) * (psNorm / chiN - 1.0));

        // Periodic eigendecomposition to keep B, D in sync with C.
        if (countEval - eigeneval > lambda / (c1 + cmu) / N / 10.0) {
            eigeneval = countEval;
            enforceSymmetry();
            eigen(C, B, D);
            for (int i = 0; i < N; ++i)
                D[i] = std::sqrt(std::max(D[i], 1e-20));
        }

        // Guard against numerical degeneracy / blow-up.
        if (!std::isfinite(sigma) || sigma <= 0.0) sigma = 1e-9;
        if (sigma > 1.0e3) sigma = 1.0e3;
        for (int d = 0; d < N; ++d)
            if (!std::isfinite(mean[d])) { mean[d] = 0.5; sigma = 0.3; }
    }

    bool shouldStop(long maxEvals) const {
        if (countEval >= maxEvals) return true;
        if (sigma < 1e-12) return true;
        return false;
    }

    const std::vector<double>& bestEverX() const { return bestX; }
    double bestEverF() const { return bestF; }
    const std::vector<double>& currentMean() const { return mean; }

private:
    int N, lambda, mu;
    double sigma, mueff, cc, cs, c1, cmu, damps, chiN;
    long countEval, eigeneval;
    int generation;
    double bestF;

    std::vector<double> mean, weights, pc, ps, D, bestX;
    std::vector<std::vector<double>> C, B, population, zStore;

    std::mt19937 rng;
    std::normal_distribution<double> gauss;

    static double mirror01(double x) {
        // Reflect repeatedly into [0,1].
        if (x >= 0.0 && x <= 1.0) return x;
        double period = 2.0;
        double t = std::fmod(x, period);
        if (t < 0.0) t += period;
        return (t <= 1.0) ? t : (period - t);
    }

    static double norm(const std::vector<double>& v) {
        double s = 0.0; for (double x : v) s += x * x; return std::sqrt(s);
    }

    void enforceSymmetry() {
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                C[i][j] = C[j][i] = 0.5 * (C[i][j] + C[j][i]);
    }

    // Symmetric eigendecomposition via cyclic Jacobi rotations.
    // On return: A = V * diag(eval) * V^T, columns of V are eigenvectors.
    static void eigen(const std::vector<std::vector<double>>& Ain,
                      std::vector<std::vector<double>>& V,
                      std::vector<double>& eval) {
        int n = (int)Ain.size();
        std::vector<std::vector<double>> A = Ain;
        V.assign(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) V[i][i] = 1.0;

        const int maxSweeps = 100;
        for (int sweep = 0; sweep < maxSweeps; ++sweep) {
            double off = 0.0;
            for (int p = 0; p < n; ++p)
                for (int q = p + 1; q < n; ++q)
                    off += A[p][q] * A[p][q];
            if (off < 1e-30) break;

            for (int p = 0; p < n; ++p) {
                for (int q = p + 1; q < n; ++q) {
                    if (std::fabs(A[p][q]) < 1e-300) continue;
                    double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                    double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
                    double c = std::cos(phi), s = std::sin(phi);

                    for (int k = 0; k < n; ++k) {
                        double akp = A[k][p], akq = A[k][q];
                        A[k][p] = c * akp - s * akq;
                        A[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; ++k) {
                        double apk = A[p][k], aqk = A[q][k];
                        A[p][k] = c * apk - s * aqk;
                        A[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; ++k) {
                        double vkp = V[k][p], vkq = V[k][q];
                        V[k][p] = c * vkp - s * vkq;
                        V[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        eval.resize(n);
        for (int i = 0; i < n; ++i) eval[i] = A[i][i];
    }
};

} // namespace cmaes
