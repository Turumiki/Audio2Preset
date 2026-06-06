#pragma once
#include <juce_core/juce_core.h>

namespace vms {

// Runs all headless gates (M1..M6) and returns the number of failures.
// SKIP results (e.g. no VST3 available) are non-fatal and not counted.
// vst3Path may be empty -> M5 attempts to auto-discover a plugin.
int runSelfTest(const juce::String& vst3Path, const juce::File& artifactsDir);

} // namespace vms
