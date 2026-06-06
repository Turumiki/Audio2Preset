cmake_minimum_required(VERSION 3.22)

project(vst_match_studio VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- JUCE (pinned stable JUCE 8 tag) --------------------------------------
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

# ---- Shared core sources (compiled into each target) ----------------------
set(VMS_CORE_SOURCES
    src/core/InternalSynthTarget.cpp
    src/core/Loss.cpp
    src/core/PitchDetect.cpp
    src/core/Features.cpp
    src/core/ParamPriority.cpp
    src/core/MatchingEngine.cpp
    src/core/StagedMatcher.cpp
    src/core/AutoMatcher.cpp
    src/core/VstTarget.cpp
    src/core/AudioIO.cpp
)

# JUCE config flags shared by all targets.
set(VMS_DEFS
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_AU=0
    JUCE_USE_CURL=0
    JUCE_WEB_BROWSER=0
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_USE_MP3AUDIOFORMAT=1
)

# ---- Optional VST2 hosting ------------------------------------------------
# JUCE can host VST2 if the legacy SDK headers (pluginterfaces/vst2.x/aeffect.h
# + aeffectx.h) are available. Steinberg no longer distributes them, so the user
# must drop them in thirdparty/vst2sdk/ (or set VST2_SDK_PATH). If absent, VST2
# is disabled and the app builds VST3-only.
set(VMS_VST2_SDK "" CACHE PATH "Folder containing pluginterfaces/vst2.x/aeffect.h")
if(NOT VMS_VST2_SDK)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vst2sdk/pluginterfaces/vst2.x/aeffect.h")
        set(VMS_VST2_SDK "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vst2sdk")
    elseif(DEFINED ENV{VST2_SDK_PATH} AND EXISTS "$ENV{VST2_SDK_PATH}/pluginterfaces/vst2.x/aeffect.h")
        set(VMS_VST2_SDK "$ENV{VST2_SDK_PATH}")
    endif()
endif()

set(VMS_VST2_DEFS JUCE_PLUGINHOST_VST=0)
set(VMS_VST2_LIBS "")
if(VMS_VST2_SDK AND EXISTS "${VMS_VST2_SDK}/pluginterfaces/vst2.x/aeffect.h")
    juce_set_vst2_sdk_path("${VMS_VST2_SDK}")
    set(VMS_VST2_DEFS JUCE_PLUGINHOST_VST=1 VMS_HAS_VST2=1)
    set(VMS_VST2_LIBS juce_vst2_sdk)
    message(STATUS "VST2 hosting ENABLED (SDK: ${VMS_VST2_SDK})")
else()
    message(STATUS "VST2 hosting DISABLED -- place aeffect.h/aeffectx.h in thirdparty/vst2sdk/pluginterfaces/vst2.x/")
endif()

# ===========================================================================
# Phase 1 target: headless self-test / matcher console app
# ===========================================================================
juce_add_console_app(vst_match_selftest
    PRODUCT_NAME "vst_match_selftest"
)

target_sources(vst_match_selftest PRIVATE
    ${VMS_CORE_SOURCES}
    src/selftest/SelfTest.cpp
    src/selftest/main_selftest.cpp
)

target_include_directories(vst_match_selftest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

target_compile_definitions(vst_match_selftest PRIVATE
    ${VMS_DEFS}
    ${VMS_VST2_DEFS}
    JUCE_STANDALONE_APPLICATION=1
)

target_link_libraries(vst_match_selftest PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_formats
    juce::juce_audio_basics
    juce::juce_dsp
    juce::juce_events
    juce::juce_core
    ${VMS_VST2_LIBS}
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
)

# ===========================================================================
# Phase 2/3 target: the integrated GUI studio
# ===========================================================================
juce_add_gui_app(vst_match_studio
    PRODUCT_NAME "vst_match_studio"
    NEEDS_CURL FALSE
    NEEDS_WEB_BROWSER FALSE
)

target_sources(vst_match_studio PRIVATE
    ${VMS_CORE_SOURCES}
    src/selftest/SelfTest.cpp
    src/app/Main.cpp
    src/app/MainComponent.cpp
    src/app/Analyzer.cpp
    src/app/MatchRunner.cpp
    src/app/LivePlayer.cpp
)

target_include_directories(vst_match_studio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

target_compile_definitions(vst_match_studio PRIVATE
    ${VMS_DEFS}
    ${VMS_VST2_DEFS}
    JUCE_STANDALONE_APPLICATION=1
)

target_link_libraries(vst_match_studio PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_audio_formats
    juce::juce_audio_devices
    juce::juce_audio_basics
    juce::juce_dsp
    juce::juce_gui_extra
    juce::juce_gui_basics
    juce::juce_data_structures
    juce::juce_events
    juce::juce_core
    ${VMS_VST2_LIBS}
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
)

# dbghelp: lets the hang-watchdog walk the message thread's stack on a freeze.
if(WIN32)
    target_link_libraries(vst_match_studio PRIVATE dbghelp)
endif()
