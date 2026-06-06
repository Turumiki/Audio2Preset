#include "AutoMatcher.h"
#include "Loss.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <regex>

namespace vms {

namespace {
bool isPhaseRand(const juce::String& n) {
    return n.contains("phase randomization") || n.contains("random phase")
        || (n.contains("phase") && n.contains("rand"));
}
juce::String blockLabel(IRenderTarget* probe, const std::vector<int>& blk) {
    // Name the block after its first param (helps the UI show "what's being tuned").
    if (blk.empty() || probe == nullptr) return "block";
    auto nm = probe->paramInfo(blk[0]).name;
    return nm.substring(0, 18) + (blk.size() > 1 ? " +" + juce::String((int) blk.size() - 1) : juce::String());
}

// Some synths (KORG Legacy: MonoPoly/MS-20/ARP/Polysix) load with EVERY parameter at 0 and
// expose no host program -> dead silence (master volume, VCO level, filter cutoff, amp EG all
// zero). A search can't bootstrap sound from silence. Name-match the gain/tone params and set
// them to sensible audible values so there's a signal to optimise. These stay SEARCHABLE (the
// matcher can still move them); we only change the STARTING point. Applied ONLY when the default
// renders silent, so synths with a real default patch (Vital etc.) are untouched.
void applyAudibleBaseline(IRenderTarget* probe, std::vector<float>& base) {
    const int np = probe->numParams();
    for (int i = 0; i < np; ++i) {
        const auto n = probe->paramInfo(i).name.toLowerCase();
        const bool lfo = n.contains("lfo") || n.contains("mg ") || n.contains("mod g");
        if (n.contains("master volume") || n.contains("output volume")
            || (n.contains("volume") && !lfo) || n.contains("output gain") || n.contains("main level"))
            base[(size_t) i] = 0.9f;
        else if (n.contains("level") && (n.contains("vco") || n.contains("osc") || n.contains("dco")
                 || n.contains("operator") || n.contains("vc1") || n.contains("vco1")))
            base[(size_t) i] = 0.75f;
        else if (n.contains("cutoff"))               base[(size_t) i] = 0.85f;
        else if (n.contains("sustain"))              base[(size_t) i] = 0.8f;   // VCF + VCA EG sustain
        else if (n.contains("decay") && !lfo)        base[(size_t) i] = 0.45f;
        else if (n.contains("attack") && !lfo)       base[(size_t) i] = 0.05f;
        else if (n.contains("total voices") || n == "voices" || n.contains("polyphony"))
            base[(size_t) i] = 0.5f;
    }
}
} // namespace

AutoResult AutoMatcher::run(Factory factory, const juce::AudioBuffer<float>& target, const AutoConfig& cfg) {
    stopFlag = false;
    AutoResult result; result.note = cfg.baseNote;

    auto probe = factory();
    if (!probe) return result;
    const int np = probe->numParams();
    const double dur = cfg.durSec;

    // Baseline = defaults with phase-randomization forced to 0 (determinism where possible).
    std::vector<float> baseline((size_t) np);
    for (int i = 0; i < np; ++i) baseline[(size_t) i] = probe->getParam(i);
    for (int i = 0; i < np; ++i)
        if (isPhaseRand(probe->paramInfo(i).name.toLowerCase())) baseline[(size_t) i] = 0.0f;

    // Silent-default rescue: if the init patch makes no sound (KORG Legacy synths init all params
    // to 0), force a name-matched audible baseline so the search has a signal to optimise.
    for (int i = 0; i < np; ++i) probe->setParam(i, baseline[(size_t) i]);
    {
        auto a = probe->render(cfg.baseNote, 100, dur, 0.7f);
        if (!probe->renderFailed() && Loss::peakAbs(a) < 5.0e-3f) {
            applyAudibleBaseline(probe.get(), baseline);
            for (int i = 0; i < np; ++i) probe->setParam(i, baseline[(size_t) i]);
            auto a2 = probe->render(cfg.baseNote, 100, dur, 0.7f);
            if (onProgress) onProgress(0, 0, 0, "Silent default detected -> forced audible baseline (peak "
                                       + juce::String(Loss::peakAbs(a2), 4) + ")", -1.0, 0.0);
        }
    }

    int perfNote = cfg.baseNote, perfVel = 100; float perfGate = 0.7f;

    // ---- PERFORMANCE: MIDI note / velocity / gate (crash-safe for Synth1) ----
    if (onProgress) onProgress(0, 0, 0, "Step 1/3: finding best MIDI note / velocity / gate", -1.0, 0.0);
    if (cfg.searchPerformance) {
        auto inst = factory();
        if (inst) {
            for (int i = 0; i < inst->numParams() && i < np; ++i) inst->setParam(i, baseline[(size_t) i]);
            Loss L;
            auto score = [&](int n, int v, float g) -> double {
                auto a = inst->render(n, v, dur, g);
                if (inst->renderFailed()) { inst = factory(); if (!inst) return 1.0e9;
                    for (int i = 0; i < inst->numParams() && i < np; ++i) inst->setParam(i, baseline[(size_t) i]);
                    a = inst->render(n, v, dur, g); if (inst->renderFailed()) return 1.0e9; }
                return (Loss::peakAbs(a) > 1.0e-4f) ? L.combined(target, a, 1.0f, 1.0f, cfg.robustLoss) : 1.0e9;
            };
            double bn = 1.0e30;
            for (int n = cfg.baseNote - 12; n <= cfg.baseNote + 12 && !stopFlag.load(); ++n) { double l = score(n, 100, 0.7f); if (l < bn) { bn = l; perfNote = n; } }
            double bvg = 1.0e30;
            for (int v : { 40, 70, 100, 127 }) for (float g : { 0.3f, 0.5f, 0.7f, 0.9f, 1.0f }) {
                if (stopFlag.load()) break;
                double l = score(perfNote, v, g); if (l < bvg) { bvg = l; perfVel = v; perfGate = g; }
            }
            if (onProgress) onProgress(0, 0, 0, "PERFORMANCE note=" + juce::String(perfNote)
                                       + " vel=" + juce::String(perfVel) + " gate=" + juce::String(perfGate, 2), bvg, 0.0);
        }
    }

    // ---- EXCLUDE by regex (default "MIDI CC"): freeze matching params, never search them ----
    // Params like "MIDI CC 17" / "Macro Control" map EXTERNAL MIDI input; they don't shape the
    // synth's own sound, so searching them only wastes budget. Matching names stay at baseline.
    std::vector<int> searchable; int excluded = 0;
    {
        bool haveRx = false; std::regex rx;
        const std::string pat = cfg.excludeRegex.trim().toStdString();
        if (!pat.empty()) { try { rx = std::regex(pat, std::regex::icase); haveRx = true; } catch (...) { haveRx = false; } }
        for (int i = 0; i < np; ++i) {
            if (haveRx && std::regex_search(probe->paramInfo(i).name.toStdString(), rx)) { ++excluded; continue; }
            searchable.push_back(i);
        }
    }
    if (searchable.empty()) { searchable.resize((size_t) np); std::iota(searchable.begin(), searchable.end(), 0); excluded = 0; }
    const int nSearch = (int) searchable.size();
    if (onProgress) onProgress(0, 0, nSearch, "Excluded " + juce::String(excluded)
                               + " param(s) by regex; searching " + juce::String(nSearch), -1.0, 0.0);

    // ---- IMPACT RANK: perturb each searchable param once, measure |loss change| (parallel) ----
    // Orders the blocks so the most influential parameters are tuned first.
    if (onProgress) onProgress(0, 0, nSearch, "Step 2/3: ranking parameters by impact", -1.0, 0.0);
    std::vector<double> impact((size_t) np, 0.0);
    {
        const int nW = juce::jmax(1, cfg.numWorkers);
        std::vector<std::unique_ptr<IRenderTarget>> ws; std::vector<std::unique_ptr<Loss>> ls;
        for (int i = 0; i < nW; ++i) { ws.push_back(factory()); ls.push_back(std::make_unique<Loss>()); }
        auto renderWith = [&](IRenderTarget* w, const std::vector<float>& p) {
            for (int i = 0; i < w->numParams() && i < (int) p.size(); ++i) w->setParam(i, p[(size_t) i]);
            return w->render(perfNote, perfVel, dur, perfGate);
        };
        auto baseAudio = ws[0] ? renderWith(ws[0].get(), baseline) : juce::AudioBuffer<float>();
        std::atomic<int> next { 0 };
        auto worker = [&](int wi) {
            if (!ws[(size_t) wi]) return; Loss& L = *ls[(size_t) wi];
            for (;;) {
                if (stopFlag.load()) break;
                const int k = next.fetch_add(1); if (k >= nSearch) break;
                const int i = searchable[(size_t) k];
                auto p = baseline; p[(size_t) i] = (baseline[(size_t) i] < 0.5f) ? baseline[(size_t) i] + 0.3f
                                                                                 : baseline[(size_t) i] - 0.3f;
                auto a = renderWith(ws[(size_t) wi].get(), p);
                impact[(size_t) i] = (Loss::peakAbs(a) > 1.0e-4f) ? L.combined(baseAudio, a, 1.0f, 0.3f, cfg.robustLoss) : 0.0;
                if (onProgress && (k % 32) == 0)
                    onProgress(0, next.load(), nSearch, "Step 2/3: ranking parameters by impact", -1.0, 0.0);
            }
        };
        std::vector<std::thread> tp; for (int t = 0; t < nW; ++t) tp.emplace_back(worker, t);
        for (auto& th : tp) th.join();
    }

    std::vector<int> order = searchable;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return impact[(size_t) a] > impact[(size_t) b]; });
    // Split impact-ordered (searchable) params into blocks. Excluded params are never added.
    const int bs = juce::jmax(2, cfg.blockSize);
    std::vector<std::vector<int>> blocks;
    for (int i = 0; i < (int) order.size(); i += bs) {
        std::vector<int> blk(order.begin() + i, order.begin() + juce::jmin((int) order.size(), i + bs));
        blocks.push_back(std::move(blk));
    }

    std::vector<juce::String> blkNames;
    for (auto& b : blocks) blkNames.push_back(blockLabel(probe.get(), b));
    probe.reset();

    // ---- BLOCK-COORDINATE DESCENT, cycled until stop, with restart on stagnation ----
    std::vector<float> bestFull = baseline;
    double bestLoss = -1.0;
    std::mt19937 rng(1234u);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    int pass = 0, stagnantPasses = 0;
    long totalEvals = 0;   // cumulative evaluations across all blocks (for tries/sec readout)
    int tickCount = 0;

    while (!stopFlag.load() && (cfg.maxPasses <= 0 || pass < cfg.maxPasses)) {
        ++pass;
        const double passStart = bestLoss;
        // On stagnation, widen exploration (bigger sigma) and occasionally restart a block.
        const double sigma = (stagnantPasses == 0) ? 0.15 : juce::jmin(0.5, 0.15 + 0.12 * stagnantPasses);

        for (int bi = 0; bi < (int) blocks.size() && !stopFlag.load(); ++bi) {
            const auto& blk = blocks[(size_t) bi];
            if (blk.empty()) continue;
            MatchConfig mc;
            mc.sampleRate = cfg.sampleRate; mc.durSec = dur; mc.midiNote = perfNote;
            mc.velocity = perfVel; mc.gateFrac = perfGate;
            mc.freeParams = blk; mc.baseParams = bestFull;
            mc.maxFevals = juce::jmax((juce::int64) 100, (juce::int64) cfg.fevalsPerBlock);
            mc.numWorkers = cfg.numWorkers; mc.sigma0 = sigma; mc.seed = 1u + (unsigned) (pass * 131 + bi);
            mc.envWeight = 1.0f; mc.forceDeterministic = true;
            mc.rendersPerEval = juce::jmax(1, cfg.rendersPerEval); mc.robustLoss = cfg.robustLoss;

            MatchingEngine engine(factory, target, mc);
            activeEngine = &engine;
            if (stopFlag.load()) { engine.requestStop(); }

            const long evalsBefore = totalEvals;
            // Stream improvements live (animate the editor / refresh A-B as they happen)...
            auto onBestCb = [&](int, double loss, const std::vector<float>& full,
                                const juce::AudioBuffer<float>& audio) {
                if (bestLoss < 0 || loss < bestLoss) {
                    bestFull = full; bestLoss = loss;
                    if (onImprove) onImprove(bestFull, bestLoss, perfNote, perfVel, perfGate, audio);
                }
            };
            // ...and stream a heartbeat every generation so the UI shows it is actively trying.
            auto onProgCb = [&](int, long evals, double curBest) {
                if (onTick && (++tickCount % 2 == 0))
                    onTick(evalsBefore + evals, (curBest < bestLoss && curBest >= 0) ? curBest : bestLoss);
            };
            auto res = engine.run(onBestCb, onProgCb);
            activeEngine = nullptr;
            totalEvals = evalsBefore + res.evaluations;

            double improved = 0.0;
            if (res.bestLoss < bestLoss || bestLoss < 0) {
                improved = (bestLoss < 0) ? 0.0 : (bestLoss - res.bestLoss);
                bestFull = res.bestFullParams; bestLoss = res.bestLoss;
                if (onImprove) onImprove(bestFull, bestLoss, perfNote, perfVel, perfGate, juce::AudioBuffer<float>());
            }
            if (onProgress) onProgress(pass, bi + 1, (int) blocks.size(), blkNames[(size_t) bi], bestLoss, improved);
        }

        const bool improvedThisPass = (bestLoss < passStart - 1.0e-6) || passStart < 0;
        stagnantPasses = improvedThisPass ? 0 : (stagnantPasses + 1);
    }

    result.bestParams = bestFull;
    result.bestLoss = (bestLoss < 0) ? 0.0 : bestLoss;
    result.note = perfNote; result.velocity = perfVel; result.gate = perfGate;
    return result;
}

} // namespace vms
