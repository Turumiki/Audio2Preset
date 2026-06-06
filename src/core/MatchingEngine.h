#pragma once
#include "IRenderTarget.h"
#include "Loss.h"
#include "Features.h"
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>

namespace vms {

struct MatchConfig {
    int midiNote = 60;
    int velocity = 100;
    float gateFrac = 0.7f;          // fraction of durSec the note is held (release timing)
    double durSec = 2.0;
    double sampleRate = 44100.0;
    std::vector<int> freeParams;   // indices optimised; others stay frozen
    std::vector<float> baseParams; // if non-empty, overrides the frozen baseline (e.g. force phase-rand=0)
    juce::MemoryBlock initStateData; // if non-empty, loaded into every worker first (e.g. a .vital preset incl. wavetable)
    bool forceDeterministic = false; // auto-zero "phase randomization" params (by name) in the baseline
    double sigma0 = 0.25;
    long maxFevals = 4000;
    float envWeight = 1.0f;
    float percWeight = 0.3f;        // perceptual (centroid+flatness trajectory) weight (gentle: don't derail spectral search)
    bool robustLoss = false;        // time-averaged-spectrum loss (for non-deterministic synths, e.g. Synth1)
    unsigned seed = 1u;
    int numWorkers = 1;            // offline worker instances (>=1)

    // ---- "Smart" search options (all off by default) --------------------
    bool warmStart = false;        // set initial values from target features + param names
    bool screening = false;        // rank params by sensitivity, optimise only impactful ones
    int  screenKeep = 0;           // keep top-K after screening (0 = auto by threshold)
    bool coarseToFine = false;     // cheap short-render exploration, then full-render refine
    double coarseDurSec = 0.5;     // render length used during the coarse phase
    bool useBayesian = false;      // surrogate (GP) optimiser instead of CMA-ES (low-dim only)
    int  rendersPerEval = 1;       // average loss over N renders (denoises non-deterministic plugins)

    // ---- Pure genetic algorithm over ALL free params (alternative to CMA-ES) -----
    bool useGenetic = false;       // population GA instead of CMA-ES
    int  gaPopulation = 0;         // 0 = auto (4 + 3*log? -> use a sane default ~ max(48, 4*sqrt(dim)))
    float gaMutationRate = 0.12f;  // per-gene probability of a Gaussian mutation (base; adapts up on stagnation)
    float gaMutationSigma = 0.15f; // mutation step in [0,1] param space (base; adapts up on stagnation)
    float gaImmigrants = 0.0f;     // fraction of each generation replaced by random genomes (opt-in; hurt in tests)
    bool gaAdaptive = false;       // widen mutation+immigrants on stagnation (opt-in; hurt in tests)
    int  gaElite = 2;              // best N copied unchanged each generation
};

struct MatchResult {
    double bestLoss = std::numeric_limits<double>::infinity();
    std::vector<float> bestFullParams;     // full normalised param vector
    int generations = 0;
    long evaluations = 0;
};

// Runs CMA-ES over the free parameters of render targets produced by a factory,
// matching a (peak-normalised) target buffer via the multi-scale STFT + envelope
// loss. The factory lets the engine build N independent worker instances for
// parallel offline rendering (one VST/synth instance per worker thread).
class MatchingEngine {
public:
    using TargetFactory = std::function<std::unique_ptr<IRenderTarget>()>;
    // gen, bestLoss, bestFullParams, bestAudio (called on improvement, background thread)
    using BestCallback = std::function<void(int, double, const std::vector<float>&,
                                            const juce::AudioBuffer<float>&)>;
    // gen, evals, currentBestLoss (called every generation, even without improvement)
    using ProgressCallback = std::function<void(int, long, double)>;

    MatchingEngine(TargetFactory factory, juce::AudioBuffer<float> target, MatchConfig cfg);

    // Synchronous run on the calling thread. Returns the best result found.
    MatchResult run(const BestCallback& onBest = {}, const ProgressCallback& onProgress = {});

    void requestStop() { stopFlag = true; }
    bool stopRequested() const { return stopFlag.load(); }

    // Live-edit the optimised parameter set DURING a run. Frozen params keep their
    // current best value; newly freed params start from it. Thread-safe.
    void setFreeParams(const std::vector<int>& newFree);
    // True if any worker plugin crashed during the run (auto-recovered via fresh
    // instance-per-render). Surface this so the UI can warn the user.
    bool crashedDuringRun() const { return sawCrash.load(); }

private:
    TargetFactory factory;
    juce::AudioBuffer<float> targetBuf;     // peak-normalised
    MatchConfig cfg;
    std::atomic<bool> stopFlag { false };

    std::vector<std::unique_ptr<IRenderTarget>> workers;
    std::vector<std::unique_ptr<Loss>> losses;  // one per worker (FFT state is not shared)
    std::vector<char> workerFresh;              // 1 = recreate instance before each render
    std::atomic<bool> sawCrash { false };       // any worker plugin crashed during the run
    std::vector<float> baseline;                // initial full-param defaults
    std::vector<float> bestFull;                // working best of ALL params (frozen ones kept here)

    // Live free-param editing.
    std::mutex freeMx;
    std::vector<int> pendingFree;
    std::atomic<bool> freeDirty { false };

    // Evaluation scale (coarse-to-fine): renders + target may be shortened.
    juce::AudioBuffer<float> evalTarget;        // current comparison target
    double evalDurSec = 0.0;
    double fullDurSec = 0.0;
    TargetFeatures features;

    // factory() + apply cfg.initStateData (so recreated/crash-recovered workers
    // keep the preset/wavetable they were started from).
    std::unique_ptr<IRenderTarget> makeWorker();

    double evalCandidate(int workerIdx, const std::vector<double>& freeVals,
                         juce::AudioBuffer<float>* outAudio);

    // Smart-search helpers.
    void setEvalScale(double durSec);           // build truncated evalTarget + evalDurSec
    void applyWarmStart();                       // set baseline values from features + names
    std::vector<int> runScreening(const std::vector<int>& freeIn); // sensitivity reduction

    // IPOP CMA-ES search that re-reads cfg.freeParams live (restarts on change) and
    // keeps bestFull updated. Runs until budgetRemaining is spent or stop requested.
    void searchLoop(long budgetRemaining, double& globalBest, int& totalGen,
                    long& totalEval, const BestCallback& onBest,
                    const ProgressCallback& onProgress);

    // Bayesian optimisation (GP + Expected Improvement) over the current free set.
    // Sample-efficient for low dim / expensive evals; used as a warm start before
    // the CMA-ES refine when cfg.useBayesian is set.
    void searchBayes(long budgetRemaining, double& globalBest, int& totalGen,
                     long& totalEval, const BestCallback& onBest,
                     const ProgressCallback& onProgress);

    // Pure genetic algorithm over the full free-param genome (population, tournament
    // selection, BLX-alpha crossover, Gaussian mutation, elitism). Parallel fitness eval.
    void searchGenetic(long budgetRemaining, double& globalBest, int& totalGen,
                       long& totalEval, const BestCallback& onBest,
                       const ProgressCallback& onProgress);
};

} // namespace vms
