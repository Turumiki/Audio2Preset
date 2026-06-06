#include "ParamPriority.h"

namespace vms {

ParamCat categorizeParam(const juce::String& nameIn) {
    const juce::String n = nameIn.toLowerCase();
    auto has = [&](const char* s) { return n.contains(s); };

    // --- Ignore: no effect on the rendered tone ---------------------------
    if (has("midi cc") || has("mpe") || has("bypass") || has("oversampl")
        || has("polyphon") || has("voice prior") || has("voice count") || has("version")
        || has("preset") || has("stereo routing") || has("legato") || has("velocity track"))
        return ParamCat::Ignore;

    // --- Mod: indirect (only matters if a routing/depth is active) --------
    if (has("lfo") || has("modulation") || n.startsWith("mod ") || has(" mod ")
        || has("macro") || has("random") || has("step seq") || has("sequencer")
        || has("mod wheel") || has("aftertouch"))
        return ParamCat::Mod;

    // --- Core: the tone-shaping parameters --------------------------------
    if (has("osc") || has("filter") || has("cutoff") || has("resonance") || has("reso")
        || has("envelope") || has("attack") || has("decay") || has("sustain") || has("release")
        || has("wave") || has("transpose") || has("tune") || has("pitch")
        || has("level") || has("gain") || has("volume") || has("amp") || has("mix")
        || has("drive") || has("unison") || has("detune") || has("spread")
        || has("fm") || has("sync") || has("ring") || has("sub") || has("noise")
        || has("keytrack") || has("key track") || has("semi") || has("octave"))
        return ParamCat::Core;

    // --- Everything else: present but secondary ---------------------------
    return ParamCat::Secondary;
}

} // namespace vms
