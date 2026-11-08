#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace chmv::benchmark {

// Every filter x expansion, without and with segment culling, at a camera that stays put. Each
// step is selected, left until its view is complete and its pass has settled, then benchmarked;
// Application does the selecting and measuring, this keeps the order, the waiting and the
// summary (written as a text table next to benchmark.csv when the last step is done).
class BenchmarkSuite {
public:
    struct Step {
        std::size_t filter = 0;
        std::size_t expansion = 0;
        bool cullSegments = false;
    };

    // What a step measured, for the summary; negative times were not measured.
    struct Outcome {
        std::string filter;
        std::string expansion;
        bool cullSegments = false;
        double passWallMs = -1.0;
        double passGpuMs = -1.0;
        double drawWallMs = -1.0;
        double cpuTotalMs = -1.0;
        std::uint64_t segments = 0;
        std::string note;
    };

    enum class Wait { Waiting, Ready, TimedOut };

    void Start(std::vector<Step> steps, std::string run, std::filesystem::path summaryPath);
    void Cancel();
    [[nodiscard]] bool Running() const { return running_; }
    [[nodiscard]] const Step& Current() const { return steps_[next_]; }
    [[nodiscard]] std::size_t Index() const { return next_; }
    [[nodiscard]] std::size_t Count() const { return steps_.size(); }
    [[nodiscard]] const std::string& Run() const { return run_; }
    // Called after every frame of the current step: `settled` means the view is complete and the
    // frame ran no pass. Ready once that held for a few frames in a row (GPU results arrive
    // late); timed out when it never does.
    [[nodiscard]] Wait Advance(bool settled, double seconds);
    // Records the current step and moves on; after the last one writes the summary.
    void Finish(Outcome outcome);
    [[nodiscard]] const std::string& Summary() const { return summary_; }

private:
    void WriteSummary();

    std::vector<Step> steps_;
    std::vector<Outcome> outcomes_;
    std::size_t next_ = 0;
    bool running_ = false;
    std::uint32_t settledFrames_ = 0;
    double stepSeconds_ = 0.0;
    std::string run_;
    std::filesystem::path summaryPath_;
    std::string summary_;
};

} // namespace chmv::benchmark
