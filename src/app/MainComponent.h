#pragma once
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "../core/VstTarget.h"
#include "../core/Loss.h"
#include "../core/StagedMatcher.h"
#include "../core/AutoMatcher.h"
#include "LivePlayer.h"
#include "MatchRunner.h"
#include "Analyzer.h"
#include <thread>
#include <atomic>
#include <mutex>

namespace vms {

// The integrated studio: hosted synth editor (live instance, left), controls +
// parameter free/freeze list + analyzer + loss curve + A/B transport (right).
class MainComponent : public juce::Component, private juce::Timer {
public:
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    // ---- parameter free/freeze list -------------------------------------
    class ParamListModel : public juce::ListBoxModel {
    public:
        explicit ParamListModel(MainComponent& o) : owner(o) {}
        int getNumRows() override;
        void paintListBoxItem(int row, juce::Graphics& g, int w, int h, bool selected) override;
        void listBoxItemClicked(int row, const juce::MouseEvent&) override;
        MainComponent& owner;
    };

    void loadVst();
    void loadWav();
    void startMatch();
    void startStagedMatch();   // human-order pure-parameter auto sound design
    void stopMatch();
    void saveState();
    void applyBest(const BestUpdate& up);
    void updateTargetSpectrum();
    void setFreeFirst(int n);
    void setAllFrozen();
    void setAllFree();
    void setFreeCore();
    void invertFree();
    void setFreeRange(int a, int b, char value);
    int countFree() const;
    void pushLiveParams();   // forward current free set to a running match
    void rebuildParamList();
    void setStatus(const juce::String& s);

    // Plugin editor shown in its own non-modal window (avoids native-window overlap).
    class PluginEditorWindow : public juce::DocumentWindow {
    public:
        PluginEditorWindow(const juce::String& name, juce::AudioProcessorEditor* ed)
            : DocumentWindow(name, juce::Colours::black, DocumentWindow::closeButton) {
            setUsingNativeTitleBar(true);
            setContentOwned(ed, true);   // window owns the editor and sizes to it
            setResizable(true, false);
            setVisible(true);
        }
        void closeButtonPressed() override { setVisible(false); } // hide, keep instance alive
    };

    // Hosted live instance (editor display + knob animation).
    std::unique_ptr<VstTarget> liveInstance;
    std::unique_ptr<PluginEditorWindow> editorWindow;

    juce::String vst3Path;
    juce::AudioBuffer<float> targetBuf;
    double targetSr = 44100.0;
    std::vector<float> baseline;       // full normalised params at load
    std::vector<char> freeFlags;       // per param: 1 = optimise
    std::vector<int> currentFreeParams;

    LivePlayer player;
    MatchRunner runner;
    Loss loss;

    // Controls.
    juce::TextButton btnLoadVst { "Load VST3" };
    juce::TextButton btnLoadWav { "Load Target WAV" };
    juce::TextButton btnStart { "Start" };
    juce::TextButton btnStaged { "Staged Auto" };
    juce::TextButton btnStop { "Stop" };
    juce::TextButton btnSave { "Save Best State" };
    juce::TextButton btnFreeFirst { "Free first" };
    juce::TextEditor edFreeN;
    juce::TextButton btnFreeCore { "Free core" };
    juce::TextButton btnFreeAll { "Free all" };
    juce::TextButton btnFreezeAll { "Freeze all" };
    juce::TextButton btnInvert { "Invert" };
    juce::TextButton btnA { "A: target" };
    juce::TextButton btnB { "B: best" };
    juce::TextButton btnPlay { "Play" };
    juce::ToggleButton btnMeter { "Live match meter" };   // live loss as you tweak knobs

    juce::Label lblNote { {}, "Note:" };
    juce::TextEditor edNote;
    juce::Label lblMax { {}, "maxfevals(0=inf):" };
    juce::TextEditor edMax;
    juce::Label lblSigma { {}, "sigma0:" };
    juce::Slider sigmaSlider;
    juce::Label lblEnv { {}, "env_weight:" };
    juce::Slider envSlider;
    juce::Label lblWorkers { {}, "workers:" };
    juce::TextEditor edWorkers;
    juce::Label lblExclude { {}, "exclude (regex):" };
    juce::TextEditor edExclude;   // param-name regex to skip (default "MIDI CC")

    // Smart-search toggles.
    juce::ToggleButton btnWarm   { "Warm start" };
    juce::ToggleButton btnScreen { "Screening" };
    juce::ToggleButton btnCoarse { "Coarse->fine" };
    juce::ToggleButton btnBayes  { "Bayesian" };
    juce::ToggleButton btnGenetic { "Genetic (GA)" };   // Start uses a genetic algorithm
    juce::Label lblMut { {}, "mut rate/sigma:" };
    juce::TextEditor edMutRate;    // GA per-gene mutation probability
    juce::TextEditor edMutSigma;   // GA mutation step

    juce::ListBox paramList;
    ParamListModel paramModel { *this };
    int lastClickedRow = -1;   // for shift-click range selection

    Spectrogram3D spectro;          // 3D waterfall (left pane, top)
    SpectrogramHeat heatmap;        // 2D spectrogram (left pane, bottom)
    LossCurveView lossCurve;
    SpecMatrix targetSpecCache;     // cached target spectrogram

    juce::Label status;   // learning / match progress
    juce::Label debug;    // audio device + playback meter diagnostics
    juce::TooltipWindow tooltip;

    juce::uint32 matchStartMs = 0;   // for evals/sec readout

    std::unique_ptr<juce::FileChooser> chooser;

    // Staged ("Staged Auto") pure-parameter sound design on a background thread.
    StagedMatcher stagedMatcher;
    std::thread stagedThread;
    std::atomic<bool> stagedBusy { false };
    void joinStaged();   // request stop + join (called on stop / destruct / relaunch)

    // Full-parameter auto matcher (block-coordinate over ALL params, runs until Stop).
    AutoMatcher autoMatcher;
    std::thread autoThread;
    std::atomic<bool> autoBusy { false };
    double autoBestLoss = 1.0e30;
    bool autoSafeMode = false;   // current search runs the VST2 on the message thread (slow but crash-safe)
    void startAuto();
    void joinAuto();

    // Coalesced live-apply: the background search can fire improvements/heartbeats far
    // faster than the message thread can render+draw them. Instead of posting one heavy
    // callAsync per event (which floods and freezes the UI), we keep only the LATEST
    // values and post a single pending update; the message thread always applies the
    // newest state and never backlogs.
    std::mutex liveMx;
    std::vector<float> liveParams;
    juce::AudioBuffer<float> liveAudio;   // engine-rendered best audio (avoids crashy live re-render)
    double liveLoss = 0.0; int liveNote = 60, liveVel = 100; float liveGate = 0.7f;
    std::atomic<bool> improvePending { false };
    long liveTickEvals = 0; double liveTickLoss = 0.0;
    std::atomic<bool> tickPending { false };
    void applyLiveImprove();   // message thread: render + animate + A/B + spectrogram (latest only)
    void applyLiveTick();      // message thread: cheap label update (latest only)
    juce::Label lblBest;       // big "BEST loss / score / elapsed"
    juce::Label lblProgress;   // "pass N  block i/M  <name>  improved"

    // ---- Hang/crash diagnostics ------------------------------------------
    // A watchdog thread logs WHY the app froze: if the message thread stops
    // updating its heartbeat (the classic deadlock-on-Stop), it records the
    // last action, the busy flags, and the message thread's stack trace.
    std::unique_ptr<juce::FileLogger> logger;
    void logAction(const juce::String& a);          // timestamped line -> log file
    std::atomic<juce::uint32> lastBeatMs { 0 };      // message thread heartbeat (timerCallback)
    std::mutex actionMx;
    juce::String lastAction;                          // most recent action (for the hang report)
    std::thread watchdogThread;
    std::atomic<bool> watchdogStop { false };
    std::atomic<bool> hangLogged { false };          // one report per hang episode
    void* msgThreadHandle = nullptr;                 // real handle to the message thread (for stack walk)

    // Live match meter: renders the user's current patch on a bg thread and shows the loss.
    std::unique_ptr<VstTarget> meterInst;
    std::thread meterThread;
    std::atomic<bool> meterBusy { false };
    void updateLiveMeter();   // called from the timer when the meter toggle is on

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};

} // namespace vms
