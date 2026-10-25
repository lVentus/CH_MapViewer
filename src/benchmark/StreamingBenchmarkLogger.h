#pragma once

#include "streaming/runtime/GraphPageStreamer.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace chmv::benchmark {

struct StreamingFrameTiming {
    double updateStreamingCpuMs = 0.0;
    double gpuProcessSubmitCpuMs = 0.0;
    double gpuFaultReadbackCpuMs = 0.0;
    double drawSubmitCpuMs = 0.0;
};

class StreamingBenchmarkLogger {
public:
    StreamingBenchmarkLogger() = default;
    ~StreamingBenchmarkLogger();

    StreamingBenchmarkLogger(const StreamingBenchmarkLogger&) = delete;
    StreamingBenchmarkLogger& operator=(const StreamingBenchmarkLogger&) = delete;

    void Start(const std::filesystem::path& datasetPath, std::string runtimeMode);
    void Stop();

    void ObserveStreamingFrame(
        std::uint64_t viewSequence,
        const streaming::runtime::GraphPageStreamingStats& stats,
        const StreamingFrameTiming& timing);

    void RecordFullResidentInstall(std::uint64_t nodes, std::uint64_t edges,
                                   double gpuUploadMs, double totalInstallMs);
    void RecordEvent(std::string eventName, std::string details = {});

    [[nodiscard]] const std::filesystem::path& Path() const { return path_; }

private:
    struct Counters {
        double plannerMs = 0.0;
        std::uint64_t rootIoBytes = 0;
        double rootIoMs = 0.0;
        std::uint64_t sourceBytes = 0;
        double pageLoadMs = 0.0;
        double refinementLoadMs = 0.0;
        std::uint64_t rootUploadBytes = 0;
        std::uint64_t backingUploadBytes = 0;
        double rootUploadMs = 0.0;
        double backingUploadMs = 0.0;
        std::uint64_t pageLoads = 0;
        std::uint64_t refinementLoads = 0;
        std::uint64_t pageEvictions = 0;
        std::uint64_t refinementEvictions = 0;
        std::uint64_t gpuRootEvictions = 0;
        std::uint64_t gpuBackingEvictions = 0;
    };

    struct Transition {
        std::uint64_t id = 0;
        std::uint64_t viewSequence = 0;
        std::chrono::steady_clock::time_point begin{};
        std::chrono::steady_clock::time_point lastSample{};
        Counters baseline{};
        double updateCpuMs = 0.0;
        double processSubmitCpuMs = 0.0;
        double readbackCpuMs = 0.0;
        double drawSubmitCpuMs = 0.0;
        double gpuFilterMs = 0.0;
        double gpuRefineMs = 0.0;
        double gpuComposeMs = 0.0;
        std::uint64_t frames = 0;
        std::uint32_t cleanFrames = 0;
    };

    static Counters Capture(const streaming::runtime::GraphPageStreamingStats& stats);
    static std::uint64_t Delta(std::uint64_t value, std::uint64_t baseline);
    static double Delta(double value, double baseline);
    static std::string JsonEscape(std::string_view value);
    static std::string RuntimeSource(const streaming::runtime::GraphPageStreamingStats& stats);

    void BeginTransition(std::uint64_t viewSequence,
                         const streaming::runtime::GraphPageStreamingStats& stats);
    void FinishTransition(std::string_view status,
                          const streaming::runtime::GraphPageStreamingStats& stats);
    void Queue(std::string line);
    void WriterMain(std::stop_token stopToken);

    std::filesystem::path path_;
    std::string runtimeMode_;
    bool enabled_ = false;
    std::uint64_t nextTransitionId_ = 1;
    std::uint64_t lastFinishedSequence_ = 0;
    std::optional<Transition> transition_;
    std::optional<Counters> lastCounters_;

    std::mutex mutex_;
    std::condition_variable_any condition_;
    std::deque<std::string> pending_;
    std::size_t pendingBytes_ = 0;
    std::jthread writer_;
};

} // namespace chmv::benchmark
