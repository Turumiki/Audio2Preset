#include <juce_gui_extra/juce_gui_extra.h>
#include "MainComponent.h"
#include "../selftest/SelfTest.h"

class VstMatchApplication : public juce::JUCEApplication {
public:
    const juce::String getApplicationName() override { return "vst_match_studio"; }
    const juce::String getApplicationVersion() override { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override { return true; }

    void initialise(const juce::String& commandLine) override {
        if (commandLine.contains("--selftest")) {
            const auto artifacts = juce::File::getCurrentWorkingDirectory().getChildFile("artifacts");
            artifacts.createDirectory();
            std::unique_ptr<juce::FileLogger> logger(
                new juce::FileLogger(artifacts.getChildFile("selftest_gui.log"), "vst_match_studio selftest"));
            juce::Logger::setCurrentLogger(logger.get());
            const int failures = vms::runSelfTest({}, artifacts);
            juce::Logger::setCurrentLogger(nullptr);
            setApplicationReturnValue(failures == 0 ? 0 : 1);
            quit();
            return;
        }
        mainWindow = std::make_unique<MainWindow>(getApplicationName());
    }

    void shutdown() override { mainWindow = nullptr; }
    void systemRequestedQuit() override { quit(); }

private:
    class MainWindow : public juce::DocumentWindow {
    public:
        explicit MainWindow(const juce::String& name)
            : DocumentWindow(name,
                             juce::Desktop::getInstance().getDefaultLookAndFeel()
                                 .findColour(juce::ResizableWindow::backgroundColourId),
                             DocumentWindow::allButtons) {
            setUsingNativeTitleBar(true);
            setContentOwned(new vms::MainComponent(), true);
            setResizable(true, true);
            centreWithSize(getWidth(), getHeight());
            setVisible(true);
        }
        void closeButtonPressed() override {
            juce::JUCEApplication::getInstance()->systemRequestedQuit();
        }
        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainWindow)
    };

    std::unique_ptr<MainWindow> mainWindow;
};

START_JUCE_APPLICATION(VstMatchApplication)
