#pragma once
#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include "../core/MatchingEngine.h"
#include <functional>
#include <atomic>

namespace vms {

struct BestUpdate {
    int generation = 0;
    double loss = 0.0;
    long evaluations = 0;
    std::vector<float> fullParams;
    juce::AudioBuffer<float> audio;
};

// Drives MatchingEngine on a background thread, building N VstTarget workers from
// a VST3 path. Best updates are throttled and marshalled to the message thread.
class MatchRunner : public juce::Thread {
public:
    MatchRunner();
    ~MatchRunner() override;

    std::function<void(const BestUpdate&)> onBest;          // called on message thread
    std::function<void(const juce::String&)> onFinished;    // called on message thread
    std::function<void(const juce::String&)> onStatus;      // progress text, message thread
    std::function<void(int, long, double)> onProgress;      // gen, evals, bestLoss; message thread

    void startMatching(const juce::String& vst3Path,
                       const juce::AudioBuffer<float>& target, const MatchConfig& cfg);
    void stopMatching();
    bool isMatching() const { return isThreadRunning(); }

    // Live-edit the optimised parameter set while a match is running (no-op if idle).
    void setFreeParams(const std::vector<int>& freeParams);

    void run() override;

private:
    juce::String vst3Path;
    juce::AudioBuffer<float> targetBuf;
    MatchConfig cfg;
    std::unique_ptr<MatchingEngine> engine;
    std::atomic<MatchingEngine*> liveEngine { nullptr }; // valid only while running
    juce::uint32 lastPostMs = 0;
    juce::uint32 lastProgressMs = 0;
};

} // namespace vms
