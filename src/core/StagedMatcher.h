#pragma once
#include "IRenderTarget.h"
#include "MatchingEngine.h"
#include <functional>
#include <atomic>
#include <vector>

namespace vms {

// Builds a patch from the INIT state using ONLY the synth's own parameters, in the
// order a human sound designer works: performance (MIDI note/velocity/gate) → oscillator
// → filter → amp envelope → modulation → FX → polish. Each stage runs CMA-ES over just
// that stage's params (low dim => fast), freezes the result, and feeds the next. No
// presets, no wavetable import. Shared by the CLI (--match-staged) and the GUI.
struct StagedConfig {
    int baseNote = 60;             // starting MIDI note (perf stage searches around it)
    double durSec = 1.0;
    double sampleRate = 44100.0;
    long fevalsPerStage = 2000;    // scaled per-stage by a budget weight
    int numWorkers = 4;
    bool searchPerformance = true; // discover MIDI performance (note/velocity/gate) first
    int passes = 1;                // re-run the whole stage sequence N times (0 = until requestStop)
    int rendersPerEval = 1;        // average loss over N renders — denoises non-deterministic synths (Synth1)
    bool robustLoss = false;       // time-averaged-spectrum loss (invariant to render fluctuation)
};

struct StagedResult {
    std::vector<float> bestParams;        // full normalised param vector
    double bestLoss = 0.0;
    int note = 60; int velocity = 100; float gate = 0.7f;
};

class StagedMatcher {
public:
    using Factory = std::function<std::unique_ptr<IRenderTarget>()>;

    // Progress callback: (stageIndex, stageCount, stageName, lossAfterStage). Background thread.
    std::function<void(int, int, const juce::String&, double)> onStage;
    // Called whenever the best improves: (fullParams, loss, note, velocity, gate). Background thread.
    // Lets the GUI apply the patch live (animate knobs, update A/B) during an unlimited run.
    std::function<void(const std::vector<float>&, double, int, int, float)> onImprove;
    // Stop promptly: set our flag AND stop the currently-running stage engine (else a join
    // would block until the in-progress stage spends its whole budget — looked like a hang).
    void requestStop() { stopFlag = true; if (auto* e = activeEngine.load()) e->requestStop(); }

    StagedResult run(Factory factory, const juce::AudioBuffer<float>& target, const StagedConfig& cfg);

private:
    std::atomic<bool> stopFlag { false };
    std::atomic<MatchingEngine*> activeEngine { nullptr };
};

} // namespace vms
