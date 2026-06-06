#include "MatchRunner.h"
#include "../core/VstTarget.h"

namespace vms {

MatchRunner::MatchRunner() : juce::Thread("vms-match") {}
MatchRunner::~MatchRunner() { stopMatching(); }

void MatchRunner::startMatching(const juce::String& path,
                                const juce::AudioBuffer<float>& target, const MatchConfig& c) {
    stopMatching();
    vst3Path = path;
    targetBuf.makeCopyOf(target);
    cfg = c;
    lastPostMs = 0;
    startThread();
}

void MatchRunner::stopMatching() {
    if (engine) engine->requestStop();
    signalThreadShouldExit();

    // PUMP the message loop while waiting for the worker to finish its in-flight
    // render. A plain blocking wait would freeze the GUI AND can deadlock when a
    // plugin (e.g. Synth1 in fresh-per-render mode) needs the message thread to
    // instantiate. Pumping keeps the message thread alive (no freeze, no deadlock).
    // Never force-kill (that corrupts a plugin mid-call and crashes).
    if (juce::MessageManager::getInstanceWithoutCreating() != nullptr
        && juce::MessageManager::getInstance()->isThisTheMessageThread()) {
        const juce::uint32 t0 = juce::Time::getMillisecondCounter();
        while (isThreadRunning() && juce::Time::getMillisecondCounter() - t0 < 30000)
            juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
    } else {
        waitForThreadToExit(30000);
    }

    if (!isThreadRunning()) {
        liveEngine.store(nullptr);
        engine.reset();   // safe: thread has exited; on the message thread
    }
}

void MatchRunner::run() {
    const juce::String path = vst3Path;
    const double sr = cfg.sampleRate;

    auto factory = [path, sr]() -> std::unique_ptr<IRenderTarget> {
        juce::String e;
        return VstTarget::loadAny(juce::File(path), sr, 512, e);
    };

    // Verify at least one instance loads before constructing the engine.
    {
        juce::String e;
        auto probe = VstTarget::loadAny(juce::File(path), sr, 512, e);
        if (probe == nullptr) {
            const juce::String msg = "Failed to load VST3 workers: " + e;
            juce::MessageManager::callAsync([cb = onFinished, msg] { if (cb) cb(msg); });
            return;
        }
    }

    {
        const int nw = juce::jmax(1, cfg.numWorkers);
        const juce::String msg = "Loading " + juce::String(nw) + " plugin instance(s)... "
                                 "(heavy synths can take a while)";
        juce::MessageManager::callAsync([cb = onStatus, msg] { if (cb) cb(msg); });
    }
    const juce::uint32 loadStart = juce::Time::getMillisecondCounter();

    engine = std::make_unique<MatchingEngine>(factory, targetBuf, cfg);
    liveEngine.store(engine.get());   // expose for live param edits from the message thread

    {
        const double secs = (juce::Time::getMillisecondCounter() - loadStart) / 1000.0;
        const juce::String msg = "Workers loaded in " + juce::String(secs, 1) + "s. Optimizing...";
        juce::MessageManager::callAsync([cb = onStatus, msg] { if (cb) cb(msg); });
    }

    auto best = engine->run([this](int gen, double loss, const std::vector<float>& full,
                                   const juce::AudioBuffer<float>& audio) {
        if (threadShouldExit()) return;
        const juce::uint32 now = juce::Time::getMillisecondCounter();
        if (now - lastPostMs < 120 && loss > 1.0e-4) return; // throttle
        lastPostMs = now;

        BestUpdate up;
        up.generation = gen;
        up.loss = loss;
        up.fullParams = full;
        up.audio.makeCopyOf(audio);
        juce::MessageManager::callAsync([cb = onBest, up = std::move(up)] { if (cb) cb(up); });
    },
    [this](int gen, long evals, double bestLoss) {
        // Heartbeat: fires every generation even without improvement, so the UI
        // never looks frozen while the optimizer explores.
        if (threadShouldExit()) return;
        const juce::uint32 now = juce::Time::getMillisecondCounter();
        if (now - lastProgressMs < 200) return; // throttle
        lastProgressMs = now;
        juce::MessageManager::callAsync([cb = onProgress, gen, evals, bestLoss] {
            if (cb) cb(gen, evals, bestLoss);
        });
    });

    liveEngine.store(nullptr);   // stop accepting live edits; engine about to be idle

    const juce::String done = "Finished: bestLoss=" + juce::String(best.bestLoss, 5)
                            + " gens=" + juce::String(best.generations)
                            + " evals=" + juce::String(best.evaluations);
    juce::MessageManager::callAsync([cb = onFinished, done] { if (cb) cb(done); });
}

void MatchRunner::setFreeParams(const std::vector<int>& freeParams) {
    if (auto* e = liveEngine.load())
        e->setFreeParams(freeParams);
}

} // namespace vms
