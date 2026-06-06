#pragma once
#include "IRenderTarget.h"
#include "MatchingEngine.h"
#include <functional>
#include <atomic>
#include <vector>

namespace vms {

// Full-parameter automatic matcher. Premise: ALL parameters are in play (never excluded —
// excluding a parameter caps the reachable optimum). To make an all-params search tractable
// it uses BLOCK-COORDINATE DESCENT: rank every parameter by impact, split into low-dim blocks,
// CMA-ES each block (others frozen at the current best), cycle the blocks forever, and on a
// stagnant pass widen the search / restart a block to escape local optima. Runs until stop.
struct AutoConfig {
    int baseNote = 60;
    double durSec = 1.0;
    double sampleRate = 44100.0;
    int numWorkers = 16;
    int blockSize = 24;            // params optimised together per step (low-dim => fast)
    long fevalsPerBlock = 600;     // CMA-ES budget per block visit
    int rendersPerEval = 1;        // average renders (denoise non-deterministic synths)
    bool robustLoss = false;       // time-averaged-spectrum loss
    bool searchPerformance = true; // find MIDI note/velocity/gate first
    int maxPasses = 0;             // 0 = run until requestStop(); >0 = stop after N block-cycle passes (CLI/testing)
    juce::String excludeRegex;     // params whose NAME matches this regex are frozen, not searched
                                   // (default in the GUI: "MIDI CC" - external-MIDI maps, no own sound)
};

struct AutoResult {
    std::vector<float> bestParams;
    double bestLoss = 0.0;
    int note = 60; int velocity = 100; float gate = 0.7f;
};

class AutoMatcher {
public:
    using Factory = std::function<std::unique_ptr<IRenderTarget>()>;

    // (pass, blockIndex, blockCount, blockLabel, bestLoss, improvedThisBlock). Background thread.
    std::function<void(int, int, int, const juce::String&, double, double)> onProgress;
    // (fullParams, loss, note, vel, gate, bestAudio) whenever the best improves. Background thread.
    // bestAudio is the engine's already-rendered mono buffer (may be empty) - the GUI uses it for
    // the A/B preview instead of re-rendering the live instance (crash-prone for VST2 reuse, e.g. Synth1).
    std::function<void(const std::vector<float>&, double, int, int, float,
                       const juce::AudioBuffer<float>&)> onImprove;
    // (totalEvals, currentBestLoss) streamed continuously during block search (proves the
    // search is actively trying, even between improvements). Background thread.
    std::function<void(long, double)> onTick;
    void requestStop() { stopFlag = true; if (auto* e = activeEngine.load()) e->requestStop(); }

    AutoResult run(Factory factory, const juce::AudioBuffer<float>& target, const AutoConfig& cfg);

private:
    std::atomic<bool> stopFlag { false };
    std::atomic<MatchingEngine*> activeEngine { nullptr };
};

} // namespace vms
