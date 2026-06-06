#pragma once
#include <juce_core/juce_core.h>

namespace vms {

// Heuristic priority for which parameters to optimise first, based on the
// parameter name. Synths (esp. Vital) expose thousands of params that don't
// shape the core tone (MIDI CC slots, mod-matrix routing). Touch the tone-
// shaping params first; treat LFO/mod as secondary; skip the inert ones.
enum class ParamCat {
    Core = 0,       // oscillator / filter / envelope / level / pitch — shapes the tone
    Secondary = 1,  // phase / warp / fx / pan — affects sound, less central
    Mod = 2,        // LFO / modulation matrix / macro / random — indirect
    Ignore = 3      // MIDI CC, MPE, bypass, voice/version housekeeping — no tone effect
};

ParamCat categorizeParam(const juce::String& name);

inline const char* paramCatName(ParamCat c) {
    switch (c) {
        case ParamCat::Core:      return "core";
        case ParamCat::Secondary: return "2nd";
        case ParamCat::Mod:       return "mod";
        default:                  return "skip";
    }
}

} // namespace vms
