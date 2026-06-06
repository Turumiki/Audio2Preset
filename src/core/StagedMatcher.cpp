#include "StagedMatcher.h"
#include "Loss.h"
#include <algorithm>

namespace vms {

namespace {
struct Stage { const char* name; std::vector<const char*> kw; double budget; };
const std::vector<Stage>& stageDefs() {
    // GENERIC keyword groups that match BOTH Vital ("oscillator 1 level", "filter 1 cutoff",
    // "envelope 1 ...", "LFO 1 ...") and Synth1 ("osc1 shape", "filter freq", "amp attack",
    // "lfo1 destination", "delay/chorus/equalizer") and similar subtractive synths. The
    // per-stage dim cap keeps high-count matches (e.g. Vital's ~90 "osc" params) tractable.
    // Keywords are ordered SPECIFIC (Vital-curated) -> GENERIC. When a stage matches more
    // params than the dim cap, params are kept by the index of the FIRST keyword they match,
    // so the curated terms win for Vital while the generic terms still pick up Synth1 etc.
    static const std::vector<Stage> s = {
        { "OSC (tone)", { "oscillator 1 wave frame","oscillator 1 distortion","oscillator 1 spectral morph",
                          "oscillator 1 frequency morph","oscillator 1 level","oscillator 1 unison",
                          "oscillator 2 switch","oscillator 2 level","oscillator 2 transpose","oscillator 2 distortion",
                          "oscillator 3 switch","oscillator 3 level","sample level","noise level",
                          // generic (Synth1 etc.)
                          "osc","shape","pulse","sync","ring","fm","wave","unison","detune" }, 1.5 },
        { "FILTER",     { "filter 1 cutoff","filter 1 resonance","filter 1 drive","filter 1 blend","filter 1 mix",
                          "filter 1 model","filter freq","filter","cutoff","reso" }, 1.0 },
        { "AMP ENV",    { "envelope 1 attack","envelope 1 decay","envelope 1 sustain","envelope 1 release",
                          "envelope 1 hold","envelope 1 delay","amp attack","amp decay","amp sustain","amp release",
                          "amp gain","amp","gain","volume" }, 1.0 },
        { "LFO/MOD",    { "modulation 1 amount","modulation 2 amount","modulation 3 amount","modulation 4 amount",
                          "modulation 5 amount","modulation 6 amount","modulation 7 amount","modulation 8 amount",
                          "modulation 9 amount","modulation 10 amount","lfo 1","lfo 2","lfo","mod env","envelope 2","envelope 3" }, 1.2 },
        { "FX",         { "reverb","delay switch","delay mix","delay feedback","chorus","distortion switch",
                          "distortion drive","distortion mix","phaser","flanger","delay","distortion","equalizer","eq " }, 0.9 },
        { "POLISH",     { "oscillator 1 level","oscillator 2 level","filter 1 cutoff","filter 1 resonance","filter 1 drive",
                          "envelope 1 attack","envelope 1 decay","envelope 1 sustain","envelope 1 release",
                          "filter freq","amp attack","amp decay","amp sustain","amp release","volume","gain" }, 1.2 },
    };
    return s;
}

bool isPhaseRand(const juce::String& lname) {
    return lname.contains("phase randomization") || lname.contains("random phase")
        || (lname.contains("phase") && lname.contains("rand"));
}
} // namespace

StagedResult StagedMatcher::run(Factory factory, const juce::AudioBuffer<float>& target, const StagedConfig& cfg) {
    stopFlag = false;
    StagedResult result;
    result.note = cfg.baseNote;

    auto probe = factory();
    if (!probe) return result;
    const int np = probe->numParams();
    const auto& stages = stageDefs();

    // Init baseline = defaults with phase-randomization forced to 0.
    std::vector<float> baseline((size_t) np);
    for (int i = 0; i < np; ++i) baseline[(size_t) i] = probe->getParam(i);
    for (int i = 0; i < np; ++i)
        if (isPhaseRand(probe->paramInfo(i).name.toLowerCase())) baseline[(size_t) i] = 0.0f;

    // Precompute each stage's param indices (probe is released before the engine loop).
    // Cap per-stage dimensionality so CMA-ES stays tractable (broad FX keywords can match
    // dozens of params). Prefer enable "switch" + "mix/dry wet/amount/cutoff/drive" params.
    const int kStageDimCap = 26;
    std::vector<std::vector<int>> stageParams(stages.size());
    for (size_t si = 0; si < stages.size(); ++si) {
        // rank = index of the FIRST stage keyword the param matches (earlier = more specific).
        std::vector<std::pair<int,int>> matched;  // (rank, paramIndex)
        for (int i = 0; i < np; ++i) {
            const auto nm = probe->paramInfo(i).name.toLowerCase();
            for (int k = 0; k < (int) stages[si].kw.size(); ++k)
                if (nm.contains(stages[si].kw[(size_t) k])) { matched.push_back({ k, i }); break; }
        }
        std::stable_sort(matched.begin(), matched.end(), [](auto& a, auto& b) { return a.first < b.first; });
        if ((int) matched.size() > kStageDimCap) matched.resize(kStageDimCap);
        std::vector<int> idxs; for (auto& m : matched) idxs.push_back(m.second);
        std::sort(idxs.begin(), idxs.end());
        stageParams[si] = idxs;
    }
    probe.reset();

    const double dur = cfg.durSec;
    int perfNote = cfg.baseNote, perfVel = 100; float perfGate = 0.7f;

    // ---- PERFORMANCE: a SINGLE MIDI note + velocity + gate ----
    // The synth always plays one note; richness/chords come from the synth's own
    // oscillators (osc2/3 transpose, unison) tuned by the OSC stage — there is no
    // notion of "chord" in the MIDI input.
    if (cfg.searchPerformance) {
        auto inst = factory();
        if (inst) {
            for (int i = 0; i < inst->numParams() && i < (int) baseline.size(); ++i) inst->setParam(i, baseline[(size_t) i]);
            Loss L;
            // Crash-safe render: some plugins (Synth1) crash on instance REUSE — recreate
            // a fresh instance (with the baseline applied) whenever a render fails.
            auto score = [&](int n, int v, float g) -> double {
                auto a = inst->render(n, v, dur, g);
                if (inst->renderFailed()) {
                    inst = factory();
                    if (!inst) return 1.0e9;
                    for (int i = 0; i < inst->numParams() && i < (int) baseline.size(); ++i) inst->setParam(i, baseline[(size_t) i]);
                    a = inst->render(n, v, dur, g);
                    if (inst->renderFailed()) return 1.0e9;
                }
                return (Loss::peakAbs(a) > 1.0e-4f) ? L.combined(target, a, 1.0f, 1.0f) : 1.0e9;
            };
            double bn = 1.0e30;
            for (int n = cfg.baseNote - 12; n <= cfg.baseNote + 12 && !stopFlag.load(); ++n) { double l = score(n, 100, 0.7f); if (l < bn) { bn = l; perfNote = n; } }
            double bvg = 1.0e30;
            for (int v : { 40, 70, 100, 127 }) for (float g : { 0.3f, 0.5f, 0.7f, 0.9f, 1.0f }) {
                if (stopFlag.load()) break;
                double l = score(perfNote, v, g); if (l < bvg) { bvg = l; perfVel = v; perfGate = g; }
            }
            if (onStage) onStage(0, (int) stages.size(), "PERFORMANCE note=" + juce::String(perfNote)
                                 + " vel=" + juce::String(perfVel) + " gate=" + juce::String(perfGate, 2), bvg);
        }
    }

    // ---- Staged parameter optimisation (multiple passes: stages interact, so re-running
    // the sequence keeps improving; the accept-if-better guard prevents any regression) ----
    std::vector<float> bestFull = baseline;
    double lastLoss = -1.0;
    const bool unlimited = (cfg.passes <= 0);   // 0 => loop until requestStop()
    const int passes = juce::jmax(1, cfg.passes);
    for (int pass = 0; (unlimited || pass < passes) && !stopFlag.load(); ++pass) {
        // Later passes narrow the search (smaller sigma) to refine rather than re-explore.
        const double sigma = (pass == 0) ? 0.25 : 0.12;
        for (size_t si = 0; si < stages.size() && !stopFlag.load(); ++si) {
            const auto& free = stageParams[si];
            if (free.empty()) continue;
            MatchConfig mc;
            mc.sampleRate = cfg.sampleRate; mc.durSec = dur; mc.midiNote = perfNote;
            mc.velocity = perfVel; mc.gateFrac = perfGate;
            mc.freeParams = free; mc.baseParams = bestFull;
            mc.maxFevals = juce::jmax((juce::int64) 200, (juce::int64) (cfg.fevalsPerStage * stages[si].budget));
            mc.numWorkers = cfg.numWorkers; mc.sigma0 = sigma; mc.seed = 1 + pass; mc.envWeight = 1.0f;
            mc.forceDeterministic = true; mc.rendersPerEval = juce::jmax(1, cfg.rendersPerEval);
            mc.robustLoss = cfg.robustLoss;
            MatchingEngine engine(factory, target, mc);
            activeEngine = &engine;
            if (stopFlag.load()) engine.requestStop();   // stop set between creating and running
            auto res = engine.run();
            activeEngine = nullptr;
            if (res.bestLoss < lastLoss || lastLoss < 0) {
                bestFull = res.bestFullParams; lastLoss = res.bestLoss;
                if (onImprove) onImprove(bestFull, lastLoss, perfNote, perfVel, perfGate);  // live apply
            }
            if (onStage) onStage((int) si + 1, (int) stages.size(),
                                 "p" + juce::String(pass + 1) + " " + stages[si].name, lastLoss);
        }
    }

    result.bestParams = bestFull;
    result.bestLoss = (lastLoss < 0) ? 0.0 : lastLoss;
    result.note = perfNote; result.velocity = perfVel; result.gate = perfGate;
    return result;
}

} // namespace vms
