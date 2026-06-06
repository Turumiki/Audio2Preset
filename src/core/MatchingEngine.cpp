#include "MatchingEngine.h"
#include "../../thirdparty/cmaes/CMAES.h"
#include "../../thirdparty/bo/BayesOpt.h"
#include <thread>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

namespace vms {

std::unique_ptr<IRenderTarget> MatchingEngine::makeWorker() {
    auto w = factory();
    // Start the worker from a caller-supplied patch state (e.g. a .vital preset,
    // which carries its own wavetable) before it is used / its baseline is read.
    if (w != nullptr && cfg.initStateData.getSize() > 0)
        w->setStateData(cfg.initStateData.getData(), cfg.initStateData.getSize());
    return w;
}

MatchingEngine::MatchingEngine(TargetFactory f, juce::AudioBuffer<float> target, MatchConfig c)
    : factory(std::move(f)), cfg(c) {
    targetBuf.makeCopyOf(target);
    Loss::peakNormalize(targetBuf);

    const int nWorkers = juce::jmax(1, cfg.numWorkers);
    for (int i = 0; i < nWorkers; ++i) {
        workers.push_back(makeWorker());
        losses.push_back(std::make_unique<Loss>());
    }
    workerFresh.assign((size_t) nWorkers, 0);

    auto& w0 = *workers[0];
    baseline.resize((size_t) w0.numParams());
    for (int i = 0; i < w0.numParams(); ++i) baseline[(size_t) i] = w0.getParam(i);
    // Optional caller-supplied frozen baseline (e.g. force osc phase-randomization
    // to 0 so a normally non-deterministic synth renders deterministically).
    if (! cfg.baseParams.empty())
        for (size_t i = 0; i < baseline.size() && i < cfg.baseParams.size(); ++i)
            baseline[i] = cfg.baseParams[i];
    // Auto-zero per-note phase-randomization params (by name) so the synth renders
    // deterministically regardless of what the loaded preset set them to.
    if (cfg.forceDeterministic)
        for (int i = 0; i < w0.numParams(); ++i) {
            const auto nm = w0.paramInfo(i).name.toLowerCase();
            if (nm.contains("phase randomization") || nm.contains("random phase")
                || (nm.contains("phase") && nm.contains("rand")))
                baseline[(size_t) i] = 0.0f;
        }
    bestFull = baseline;   // working best starts at the defaults

    fullDurSec = cfg.durSec;
    setEvalScale(fullDurSec);
    features = extractFeatures(targetBuf, cfg.sampleRate);
}

void MatchingEngine::setEvalScale(double durSec) {
    evalDurSec = juce::jlimit(0.05, fullDurSec, durSec);
    const int total = targetBuf.getNumSamples();
    const int len = juce::jlimit(1, total, (int) std::llround(evalDurSec * cfg.sampleRate));
    evalTarget.setSize(1, len);
    evalTarget.copyFrom(0, 0, targetBuf, 0, 0, len);
}

double MatchingEngine::evalCandidate(int workerIdx, const std::vector<double>& freeVals,
                                     juce::AudioBuffer<float>* outAudio) {
    const size_t wi = (size_t) workerIdx;

    // Honor a stop request BEFORE any expensive work (fresh-instance creation can
    // re-load the plugin DLL, which is slow). This keeps Stop responsive.
    if (stopFlag.load()) return 1.0e30;

    auto applyAndRender = [&]() -> juce::AudioBuffer<float> {
        auto* w = workers[wi].get();
        if (w == nullptr) return {};
        for (int i = 0; i < w->numParams(); ++i) w->setParam(i, bestFull[(size_t) i]);
        for (size_t k = 0; k < cfg.freeParams.size(); ++k)
            w->setParam(cfg.freeParams[k], (float) freeVals[k]);
        return w->render(cfg.midiNote, cfg.velocity, evalDurSec, cfg.gateFrac);
    };

    // One rendered+scored attempt (handles instance reuse crashes -> fresh-per-render).
    auto renderOnce = [&](juce::AudioBuffer<float>& outBuf) -> double {
        if (workerFresh[wi]) workers[wi] = makeWorker();
        auto audio = applyAndRender();
        if (workers[wi] != nullptr && workers[wi]->renderFailed()) {
            sawCrash.store(true);
            workerFresh[wi] = 1;
            workers[wi] = makeWorker();
            audio = applyAndRender();
            if (workers[wi] == nullptr || workers[wi]->renderFailed()) { outBuf = audio; return 1.0e9; }
        }
        const double l = losses[wi]->combined(evalTarget, audio, cfg.envWeight, cfg.percWeight, cfg.robustLoss);
        outBuf = std::move(audio);
        return l;
    };

    // Average the loss over N renders to denoise non-deterministic plugins (the
    // per-render noise otherwise corrupts the optimiser's candidate ranking).
    const int K = juce::jmax(1, cfg.rendersPerEval);
    juce::AudioBuffer<float> firstAudio;
    double sum = 0.0; int got = 0;
    for (int r = 0; r < K && !stopFlag.load(); ++r) {
        juce::AudioBuffer<float> a;
        const double l = renderOnce(a);
        if (l >= 1.0e9) { if (outAudio) *outAudio = a; return 1.0e9; }
        sum += l; ++got;
        if (r == 0) firstAudio = std::move(a);
    }
    if (outAudio) *outAudio = std::move(firstAudio);
    return got > 0 ? sum / got : 1.0e30;
}

// --- Warm start: nudge recognisable parameters from target features --------
void MatchingEngine::applyWarmStart() {
    auto& w = *workers[0];
    auto setIfFree = [&](int idx, float v) {
        // Only adjust params the user actually freed.
        if (std::find(cfg.freeParams.begin(), cfg.freeParams.end(), idx) != cfg.freeParams.end())
            bestFull[(size_t) idx] = juce::jlimit(0.0f, 1.0f, v);
    };

    // Rough mappings of measured time/brightness into normalised [0,1].
    const float attackN  = juce::jlimit(0.0, 1.0, features.attackSec / 1.0);
    const float releaseN = juce::jlimit(0.0, 1.0, features.releaseSec / 1.5);
    const float sustainN = (float) features.sustainLevel;
    const float cutoffN  = (float) juce::jlimit(0.0, 1.0, std::sqrt(features.centroidNorm)); // perceptual-ish

    for (int i = 0; i < w.numParams(); ++i) {
        const juce::String n = w.paramInfo(i).name.toLowerCase();
        if (n.contains("attack"))                                 setIfFree(i, attackN);
        else if (n.contains("release"))                           setIfFree(i, releaseN);
        else if (n.contains("sustain"))                           setIfFree(i, sustainN);
        else if (n.contains("cutoff") || n.contains("freq")
                 || (n.contains("filter") && n.contains("cut")))  setIfFree(i, cutoffN);
    }
}

// --- Sensitivity screening: keep only impactful parameters -----------------
std::vector<int> MatchingEngine::runScreening(const std::vector<int>& freeIn) {
    const int dim = (int) freeIn.size();
    if (dim <= 1) return freeIn;

    std::vector<double> baseFree((size_t) dim);
    for (int i = 0; i < dim; ++i) baseFree[(size_t) i] = bestFull[(size_t) freeIn[(size_t) i]];

    const double baseLoss = evalCandidate(0, baseFree, nullptr);
    const double delta = 0.15;

    std::vector<double> impact((size_t) dim, 0.0);
    std::atomic<int> next { 0 };
    const int nThreads = (int) workers.size();
    auto probe = [&](int wIdx) {
        for (;;) {
            const int j = next.fetch_add(1);
            if (j >= dim) break;
            std::vector<double> vp = baseFree, vm = baseFree;
            vp[(size_t) j] = juce::jlimit(0.0, 1.0, baseFree[(size_t) j] + delta);
            vm[(size_t) j] = juce::jlimit(0.0, 1.0, baseFree[(size_t) j] - delta);
            const double lp = evalCandidate(wIdx, vp, nullptr);
            const double lm = evalCandidate(wIdx, vm, nullptr);
            impact[(size_t) j] = juce::jmax(std::fabs(lp - baseLoss), std::fabs(lm - baseLoss));
        }
    };
    if (nThreads <= 1) probe(0);
    else {
        std::vector<std::thread> pool;
        for (int t = 0; t < nThreads; ++t) pool.emplace_back(probe, t);
        for (auto& th : pool) th.join();
    }

    std::vector<int> order(dim);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return impact[a] > impact[b]; });

    const double maxImpact = impact[order[0]];
    std::vector<int> kept;
    if (cfg.screenKeep > 0) {
        for (int i = 0; i < juce::jmin(cfg.screenKeep, dim); ++i)
            kept.push_back(freeIn[(size_t) order[(size_t) i]]);
    } else {
        const double thresh = juce::jmax(1.0e-6, 0.05 * maxImpact);
        for (int i = 0; i < dim; ++i)
            if (impact[order[(size_t) i]] >= thresh) kept.push_back(freeIn[(size_t) order[(size_t) i]]);
        if (kept.empty()) kept.push_back(freeIn[(size_t) order[0]]);
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

void MatchingEngine::setFreeParams(const std::vector<int>& newFree) {
    std::lock_guard<std::mutex> lk(freeMx);
    pendingFree = newFree;
    freeDirty.store(true);
}

// --- IPOP CMA-ES search that re-reads free params live ----------------------
void MatchingEngine::searchLoop(long budgetRemaining, double& globalBest, int& totalGen,
                                long& totalEval, const BestCallback& onBest,
                                const ProgressCallback& onProgress) {
    const int nThreads = (int) workers.size();
    int baseLambda = 0;
    long localUsed = 0;
    int restart = 0;
    std::mt19937 rng(cfg.seed + 13u);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    auto renderFull = [&](const std::vector<double>& freeVals) -> juce::AudioBuffer<float> {
        auto applyTo = [&](IRenderTarget* w) {
            for (int i = 0; i < w->numParams(); ++i) w->setParam(i, bestFull[(size_t) i]);
            for (size_t k = 0; k < cfg.freeParams.size(); ++k)
                w->setParam(cfg.freeParams[k], (float) freeVals[k]);
        };
        if (workerFresh[0]) {
            auto t = makeWorker();
            if (t == nullptr) return {};
            applyTo(t.get());
            return t->render(cfg.midiNote, cfg.velocity, fullDurSec, cfg.gateFrac);
        }
        applyTo(workers[0].get());
        auto a = workers[0]->render(cfg.midiNote, cfg.velocity, fullDurSec, cfg.gateFrac);
        if (workers[0]->renderFailed()) {
            sawCrash.store(true); workerFresh[0] = 1; workers[0] = makeWorker();
            if (workers[0] != nullptr) { applyTo(workers[0].get()); a = workers[0]->render(cfg.midiNote, cfg.velocity, fullDurSec, cfg.gateFrac); }
        }
        return a;
    };

    while (localUsed < budgetRemaining && !stopFlag.load()) {
        // Apply a pending live change to the free-param set.
        if (freeDirty.exchange(false)) {
            std::lock_guard<std::mutex> lk(freeMx);
            cfg.freeParams = pendingFree;
            baseLambda = 0; restart = 0;
        }

        const int dim = (int) cfg.freeParams.size();
        if (dim == 0) {
            // Nothing to optimise (user froze everything) — idle until they add some.
            if (onProgress) onProgress(totalGen, totalEval, globalBest);
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            continue;
        }

        // Multi-start: every 3rd restart explores from a fresh RANDOM point to
        // escape the current basin (helps rugged landscapes, e.g. FM). Other
        // restarts continue from the global best. Global best is never overwritten
        // unless a start actually improves it.
        const bool randomStart = (restart > 0 && (restart % 3 == 2));
        std::vector<double> x0((size_t) dim);
        for (int i = 0; i < dim; ++i)
            x0[(size_t) i] = randomStart ? uni01(rng)
                                         : juce::jlimit(0.0f, 1.0f, bestFull[(size_t) cfg.freeParams[(size_t) i]]);

        const int maxStagnGen = 25 + 5 * dim;
        const int lambdaOverride = (baseLambda == 0) ? 0 : (baseLambda << juce::jmin(restart, 4));
        cmaes::CMAES opt(dim, x0, cfg.sigma0, cfg.seed + (unsigned) restart, lambdaOverride);
        if (baseLambda == 0) baseLambda = opt.populationSize();

        double localBest = std::numeric_limits<double>::infinity();
        int stagn = 0;

        while (!opt.shouldStop(budgetRemaining - localUsed) && !stopFlag.load() && !freeDirty.load()) {
            const auto& pop = opt.ask();
            const int lambda = (int) pop.size();
            std::vector<double> fit((size_t) lambda, 1.0e30);

            std::atomic<int> next { 0 };
            auto worker = [&](int wIdx) {
                for (;;) {
                    if (stopFlag.load()) break;
                    const int k = next.fetch_add(1);
                    if (k >= lambda) break;
                    double f = evalCandidate(wIdx, pop[(size_t) k], nullptr);
                    if (!std::isfinite(f)) f = 1.0e9;
                    fit[(size_t) k] = f;
                }
            };
            if (nThreads <= 1) worker(0);
            else {
                std::vector<std::thread> pool;
                for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
                for (auto& th : pool) th.join();
            }
            if (stopFlag.load()) break;

            opt.tell(fit);
            ++totalGen;

            if (opt.bestEverF() < globalBest - 1.0e-9) {
                globalBest = opt.bestEverF();
                const auto& bestFree = opt.bestEverX();
                for (size_t k = 0; k < cfg.freeParams.size(); ++k)        // commit into bestFull
                    bestFull[(size_t) cfg.freeParams[k]] = (float) bestFree[k];
                if (onBest) {
                    auto fullAudio = renderFull(bestFree);
                    onBest(totalGen, globalBest, bestFull, fullAudio);
                }
            }
            if (onProgress)
                onProgress(totalGen, totalEval + opt.evaluations(), globalBest);

            if (opt.bestEverF() < localBest - 1.0e-6) { localBest = opt.bestEverF(); stagn = 0; }
            else ++stagn;
            if (stagn > maxStagnGen) break;   // restart with a larger population
        }

        localUsed += opt.evaluations();
        totalEval += opt.evaluations();

        // Memetic local polish: PARALLEL pattern search around the global best.
        // Each round evaluates every coordinate's +/-step probe across ALL workers
        // at once, applies the single best improving move, and shrinks the step when
        // none helps. Parallel (keeps worker utilisation) and bounded.
        if (!stopFlag.load() && !freeDirty.load() && localUsed < budgetRemaining && dim > 0) {
            std::vector<double> base((size_t) dim);
            for (int k = 0; k < dim; ++k)
                base[(size_t) k] = juce::jlimit(0.0f, 1.0f, bestFull[(size_t) cfg.freeParams[(size_t) k]]);
            double cur = globalBest;
            double step = 0.08;
            for (int round = 0; round < 6 && step > 0.004 && !stopFlag.load()
                                 && !freeDirty.load() && localUsed < budgetRemaining; ++round) {
                std::vector<std::vector<double>> probes;
                probes.reserve((size_t) dim * 2);
                for (int k = 0; k < dim; ++k)
                    for (double s : { step, -step }) {
                        auto t = base; t[(size_t) k] = juce::jlimit(0.0, 1.0, base[(size_t) k] + s);
                        probes.push_back(std::move(t));
                    }
                const int P = (int) probes.size();
                std::vector<double> fit((size_t) P, 1.0e30);
                std::atomic<int> next { 0 };
                auto pw = [&](int wIdx) {
                    for (;;) {
                        if (stopFlag.load()) break;
                        const int j = next.fetch_add(1);
                        if (j >= P) break;
                        const double l = evalCandidate(wIdx, probes[(size_t) j], nullptr);
                        fit[(size_t) j] = std::isfinite(l) ? l : 1.0e9;
                    }
                };
                if (nThreads <= 1) pw(0);
                else {
                    std::vector<std::thread> pool;
                    for (int t = 0; t < nThreads; ++t) pool.emplace_back(pw, t);
                    for (auto& th : pool) th.join();
                }
                localUsed += P; totalEval += P; ++totalGen;

                int bestJ = -1; double bestL = cur;
                for (int j = 0; j < P; ++j) if (fit[(size_t) j] < bestL - 1.0e-9) { bestL = fit[(size_t) j]; bestJ = j; }
                if (bestJ >= 0) {
                    base = probes[(size_t) bestJ]; cur = bestL; globalBest = bestL;
                    for (int k = 0; k < dim; ++k)
                        bestFull[(size_t) cfg.freeParams[(size_t) k]] = (float) base[(size_t) k];
                    if (onBest) { auto fa = renderFull(base); onBest(totalGen, globalBest, bestFull, fa); }
                } else {
                    step *= 0.5;
                }
                if (onProgress) onProgress(totalGen, totalEval, globalBest);
            }
        }
        ++restart;
    }
}

void MatchingEngine::searchBayes(long budgetRemaining, double& globalBest, int& totalGen,
                                 long& totalEval, const BestCallback& onBest,
                                 const ProgressCallback& onProgress) {
    const int dim = (int) cfg.freeParams.size();
    if (dim == 0) return;
    const int nThreads = (int) workers.size();

    auto renderFull = [&](const std::vector<double>& freeVals) -> juce::AudioBuffer<float> {
        auto applyTo = [&](IRenderTarget* w) {
            for (int i = 0; i < w->numParams(); ++i) w->setParam(i, bestFull[(size_t) i]);
            for (size_t k = 0; k < cfg.freeParams.size(); ++k)
                w->setParam(cfg.freeParams[k], (float) freeVals[k]);
        };
        if (workerFresh[0]) { auto t = makeWorker(); if (!t) return {}; applyTo(t.get());
                              return t->render(cfg.midiNote, cfg.velocity, fullDurSec, cfg.gateFrac); }
        applyTo(workers[0].get());
        auto a = workers[0]->render(cfg.midiNote, cfg.velocity, fullDurSec, cfg.gateFrac);
        if (workers[0]->renderFailed()) { sawCrash.store(true); workerFresh[0] = 1; workers[0] = makeWorker();
            if (workers[0]) { applyTo(workers[0].get()); a = workers[0]->render(cfg.midiNote, cfg.velocity, fullDurSec, cfg.gateFrac); } }
        return a;
    };

    auto evalBatch = [&](const std::vector<std::vector<double>>& pts, std::vector<double>& out) {
        out.assign(pts.size(), 1.0e30);
        std::atomic<int> next { 0 };
        auto w = [&](int wIdx) {
            for (;;) {
                if (stopFlag.load()) break;
                const int j = next.fetch_add(1);
                if (j >= (int) pts.size()) break;
                const double l = evalCandidate(wIdx, pts[(size_t) j], nullptr);
                out[(size_t) j] = std::isfinite(l) ? l : 1.0e9;
            }
        };
        if (nThreads <= 1) w(0);
        else { std::vector<std::thread> pool; for (int t = 0; t < nThreads; ++t) pool.emplace_back(w, t);
               for (auto& th : pool) th.join(); }
    };

    auto commitIfBetter = [&](const std::vector<double>& freeVals, double l) {
        if (l < globalBest - 1.0e-9) {
            globalBest = l;
            for (size_t k = 0; k < cfg.freeParams.size(); ++k)
                bestFull[(size_t) cfg.freeParams[k]] = (float) freeVals[k];
            if (onBest) { auto fa = renderFull(freeVals); onBest(totalGen, globalBest, bestFull, fa); }
        }
    };

    bo::BayesOpt opt(dim, cfg.seed + 5u);

    // Initial design: the current best plus random points.
    std::vector<std::vector<double>> init;
    { std::vector<double> cur((size_t) dim);
      for (int i = 0; i < dim; ++i) cur[(size_t) i] = juce::jlimit(0.0f, 1.0f, bestFull[(size_t) cfg.freeParams[(size_t) i]]);
      init.push_back(cur); }
    const int nInit = juce::jmax(8, 2 * dim);
    for (int i = 0; i < nInit; ++i) init.push_back(opt.randomPoint());

    long used = 0;
    std::vector<double> fit;
    evalBatch(init, fit); used += (long) init.size(); totalEval += (long) init.size(); ++totalGen;
    for (size_t i = 0; i < init.size(); ++i) { opt.addSample(init[i], fit[i]); commitIfBetter(init[i], fit[i]); }
    if (onProgress) onProgress(totalGen, totalEval, globalBest);

    const int maxSamples = 250;   // GP is O(n^3); cap the surrogate size
    while (used < budgetRemaining && !stopFlag.load() && !freeDirty.load() && opt.count() < maxSamples) {
        const int k = juce::jlimit(1, juce::jmax(1, nThreads), maxSamples - opt.count());
        auto batch = opt.suggestBatch(k);
        if (batch.empty()) break;
        std::vector<double> bf;
        evalBatch(batch, bf);
        used += (long) batch.size(); totalEval += (long) batch.size(); ++totalGen;
        for (size_t i = 0; i < batch.size(); ++i) { opt.addSample(batch[i], bf[i]); commitIfBetter(batch[i], bf[i]); }
        if (onProgress) onProgress(totalGen, totalEval, globalBest);
    }
}

// Pure genetic algorithm over the full free-param genome. Parallel fitness via evalCandidate
// (crash-safe, one worker per thread). Tournament selection + BLX-alpha crossover + per-gene
// Gaussian mutation + elitism. Runs until the eval budget is spent or stop is requested.
void MatchingEngine::searchGenetic(long budgetRemaining, double& globalBest, int& totalGen,
                                   long& totalEval, const BestCallback& onBest,
                                   const ProgressCallback& onProgress) {
    const int dim = (int) cfg.freeParams.size();
    if (dim == 0) return;
    const int nThreads = (int) workers.size();
    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    const int pop = cfg.gaPopulation > 0 ? cfg.gaPopulation
                                         : juce::jlimit(24, 200, (int) (4 * std::sqrt((double) dim)) + 16);

    auto clamp01 = [](float v) { return juce::jlimit(0.0f, 1.0f, v); };
    using Genome = std::vector<double>;

    // Initialise: one genome from the current best (frozen baseline's free values), rest random.
    std::vector<Genome> P((size_t) pop, Genome((size_t) dim));
    for (int i = 0; i < dim; ++i) P[0][(size_t) i] = juce::jlimit(0.0f, 1.0f, bestFull[(size_t) cfg.freeParams[(size_t) i]]);
    for (int g = 1; g < pop; ++g) for (int i = 0; i < dim; ++i) P[(size_t) g][(size_t) i] = uni01(rng);

    std::vector<double> fit((size_t) pop, 1.0e30);
    auto evalAll = [&](std::vector<Genome>& gs, std::vector<double>& f) {
        std::atomic<int> next { 0 };
        const int n = (int) gs.size();
        auto worker = [&](int wIdx) {
            for (;;) {
                if (stopFlag.load()) break;
                const int k = next.fetch_add(1);
                if (k >= n) break;
                double v = evalCandidate(wIdx, gs[(size_t) k], nullptr);
                f[(size_t) k] = std::isfinite(v) ? v : 1.0e9;
            }
        };
        if (nThreads <= 1) worker(0);
        else { std::vector<std::thread> tp; for (int t = 0; t < nThreads; ++t) tp.emplace_back(worker, t); for (auto& th : tp) th.join(); }
        totalEval += n;
    };

    evalAll(P, fit);
    long used = pop;

    auto bestIdx = [&]() { int b = 0; for (int i = 1; i < pop; ++i) if (fit[(size_t) i] < fit[(size_t) b]) b = i; return b; };
    auto commitBest = [&](int b) {
        if (fit[(size_t) b] < globalBest) {
            globalBest = fit[(size_t) b];
            for (int i = 0; i < dim; ++i) bestFull[(size_t) cfg.freeParams[(size_t) i]] = (float) P[(size_t) b][(size_t) i];
            if (onBest) { juce::AudioBuffer<float> a; evalCandidate(0, P[(size_t) b], &a); onBest(totalGen, globalBest, bestFull, a); }
        }
    };
    commitBest(bestIdx());
    if (onProgress) onProgress(totalGen, totalEval, globalBest);

    auto tournament = [&]() { int a = (int) (uni01(rng) * pop), b = (int) (uni01(rng) * pop);
                              a = juce::jmin(a, pop - 1); b = juce::jmin(b, pop - 1);
                              return fit[(size_t) a] <= fit[(size_t) b] ? a : b; };

    // Adaptive mutation + random immigrants: when the best stagnates, widen the mutation
    // (rate & sigma) and inject more fresh random genomes to escape local optima; when it
    // improves, relax back toward the base settings.
    const float baseRate = juce::jlimit(0.0f, 1.0f, cfg.gaMutationRate);
    const float baseSigma = juce::jmax(1.0e-4f, cfg.gaMutationSigma);
    const float baseImmig = juce::jlimit(0.0f, 0.9f, cfg.gaImmigrants);
    float curRate = baseRate, curSigma = baseSigma, curImmig = baseImmig;
    double prevBest = globalBest;
    int stagn = 0;

    while (used < budgetRemaining && !stopFlag.load()) {
        // Rank for elitism.
        std::vector<int> order((size_t) pop); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return fit[(size_t) a] < fit[(size_t) b]; });

        std::vector<Genome> child((size_t) pop, Genome((size_t) dim));
        const int elite = juce::jlimit(0, pop - 1, cfg.gaElite);
        for (int e = 0; e < elite; ++e) child[(size_t) e] = P[(size_t) order[(size_t) e]];

        // A fraction of the new generation are random immigrants (more when stagnating).
        const int immig = juce::jlimit(0, pop - elite, (int) std::lround(curImmig * pop));
        for (int c = pop - immig; c < pop; ++c)
            for (int i = 0; i < dim; ++i) child[(size_t) c][(size_t) i] = uni01(rng);

        for (int c = elite; c < pop - immig; ++c) {
            const int p1 = tournament(), p2 = tournament();
            for (int i = 0; i < dim; ++i) {
                // BLX-alpha crossover.
                const double x1 = P[(size_t) p1][(size_t) i], x2 = P[(size_t) p2][(size_t) i];
                const double lo = juce::jmin(x1, x2), hi = juce::jmax(x1, x2), d = hi - lo;
                const double a = 0.5;
                double v = lo - a * d + uni01(rng) * (d + 2 * a * d);
                if (uni01(rng) < curRate) v += gauss(rng) * curSigma;   // adaptive Gaussian mutation
                child[(size_t) c][(size_t) i] = clamp01((float) v);
            }
        }

        std::vector<double> cfit((size_t) pop, 1.0e30);
        evalAll(child, cfit);
        used += pop;
        P.swap(child); fit.swap(cfit);
        ++totalGen;
        commitBest(bestIdx());

        // Adaptive explore/exploit control — OPT-IN (gaAdaptive). Testing showed that for
        // these targets it hurt (aggressive widening thrashes; immigrants waste evals), so
        // the default GA is plain BLX + elitism + fixed mutation, which converged best.
        if (cfg.gaAdaptive) {
            if (globalBest < prevBest - 1.0e-6) {
                prevBest = globalBest; stagn = 0;
                curRate  = juce::jmax(baseRate,  curRate  * 0.8f);
                curSigma = juce::jmax(baseSigma, curSigma * 0.8f);
                curImmig = juce::jmax(baseImmig, curImmig * 0.7f);
            } else if (++stagn >= 6) {
                stagn = 0;
                curRate  = juce::jmin(0.6f, curRate  * 1.4f);
                curSigma = juce::jmin(0.5f, curSigma * 1.4f);
                curImmig = juce::jmin(0.4f, curImmig + 0.05f);
            }
        }
        if (onProgress) onProgress(totalGen, totalEval, globalBest);
    }
}

MatchResult MatchingEngine::run(const BestCallback& onBest, const ProgressCallback& onProgress) {
    MatchResult result;
    fullDurSec = cfg.durSec;

    if (cfg.warmStart) applyWarmStart();

    if (cfg.screening && cfg.freeParams.size() > 1) {
        setEvalScale(cfg.coarseToFine ? cfg.coarseDurSec : fullDurSec);
        auto kept = runScreening(cfg.freeParams);
        if (!kept.empty()) cfg.freeParams = kept;
    }

    double globalBest = std::numeric_limits<double>::infinity();
    int totalGen = 0;
    long totalEval = 0;

    const long budget = (cfg.maxFevals <= 0) ? std::numeric_limits<long>::max() : (long) cfg.maxFevals;

    // Optional coarse phase (short render) before the live full-scale loop.
    if (cfg.coarseToFine && fullDurSec > cfg.coarseDurSec + 1.0e-6 && !cfg.freeParams.empty()) {
        const long coarseBudget = (cfg.maxFevals <= 0) ? 3000 : juce::jmax((long) 200, budget / 3);
        setEvalScale(cfg.coarseDurSec);
        searchLoop(coarseBudget, globalBest, totalGen, totalEval, onBest, onProgress);
        globalBest = std::numeric_limits<double>::infinity();   // loss not comparable across scales
    }

    setEvalScale(fullDurSec);
    if (cfg.useGenetic) {
        // Pure GA over all free params (no CMA-ES / Bayesian).
        searchGenetic(budget - totalEval, globalBest, totalGen, totalEval, onBest, onProgress);
    } else {
        // Bayesian warm start (sample-efficient) finds a good basin, then CMA-ES +
        // memetic polish refines from it.
        if (cfg.useBayesian)
            searchBayes(budget - totalEval, globalBest, totalGen, totalEval, onBest, onProgress);
        searchLoop(budget - totalEval, globalBest, totalGen, totalEval, onBest, onProgress);
    }

    result.bestLoss = globalBest;
    result.generations = totalGen;
    result.evaluations = totalEval;
    result.bestFullParams = bestFull;   // frozen + optimised values, full vector
    return result;
}

} // namespace vms
