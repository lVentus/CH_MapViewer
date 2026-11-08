#include "benchmark/BenchmarkSuite.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <utility>

namespace chmv::benchmark {
namespace {

constexpr std::uint32_t kSettledFrames = 5;
// A step whose view is not complete by then (a pool too small for it, say) is skipped.
constexpr double kStepTimeoutSeconds = 30.0;

[[nodiscard]] std::string Milliseconds(double value) {
    return value < 0.0 ? std::string("-") : std::format("{:.3f}", value);
}

} // namespace

void BenchmarkSuite::Start(std::vector<Step> steps, std::string run,
                           std::filesystem::path summaryPath) {
    steps_ = std::move(steps);
    outcomes_.clear();
    next_ = 0;
    running_ = !steps_.empty();
    settledFrames_ = 0;
    stepSeconds_ = 0.0;
    run_ = std::move(run);
    summaryPath_ = std::move(summaryPath);
    summary_.clear();
}

void BenchmarkSuite::Cancel() {
    if (running_) {
        running_ = false;
        summary_ = "Cancelled after " + std::to_string(next_) + " of " +
                   std::to_string(steps_.size()) + " steps.";
    }
}

BenchmarkSuite::Wait BenchmarkSuite::Advance(bool settled, double seconds) {
    stepSeconds_ += seconds;
    settledFrames_ = settled ? settledFrames_ + 1u : 0u;
    if (settledFrames_ >= kSettledFrames) {
        return Wait::Ready;
    }
    return stepSeconds_ > kStepTimeoutSeconds ? Wait::TimedOut : Wait::Waiting;
}

void BenchmarkSuite::Finish(Outcome outcome) {
    outcomes_.push_back(std::move(outcome));
    settledFrames_ = 0;
    stepSeconds_ = 0.0;
    if (++next_ == steps_.size()) {
        running_ = false;
        WriteSummary();
    }
}

void BenchmarkSuite::WriteSummary() {
    // Expansions down, filters across, one table per culling setting and measure.
    std::vector<std::string> filters;
    std::vector<std::string> expansions;
    for (const auto& outcome : outcomes_) {
        if (std::find(filters.begin(), filters.end(), outcome.filter) == filters.end()) {
            filters.push_back(outcome.filter);
        }
        if (std::find(expansions.begin(), expansions.end(), outcome.expansion) ==
            expansions.end()) {
            expansions.push_back(outcome.expansion);
        }
    }
    std::string text = "Benchmark suite " + run_ +
                       ": mean ms per pass (filter + expansion) or draw; '-' not measured\n";
    const auto table = [&](const char* title, bool cull, auto value) {
        text += std::format("\n{} ({})\n{:<28}", title, cull ? "culled" : "not culled", "");
        for (const auto& filter : filters) {
            text += std::format("{:>18}", filter);
        }
        text += '\n';
        for (const auto& expansion : expansions) {
            text += std::format("{:<28}", expansion);
            for (const auto& filter : filters) {
                const auto found =
                    std::find_if(outcomes_.begin(), outcomes_.end(), [&](const Outcome& o) {
                        return o.filter == filter && o.expansion == expansion &&
                               o.cullSegments == cull;
                    });
                text += std::format("{:>18}",
                                    found == outcomes_.end() ? std::string("") : value(*found));
            }
            text += '\n';
        }
    };
    for (const bool cull : {false, true}) {
        table("Pass, wall time", cull, [](const Outcome& o) { return Milliseconds(o.passWallMs); });
        table("Pass, GPU timestamps", cull,
              [](const Outcome& o) { return Milliseconds(o.passGpuMs); });
        table("Draw, wall time", cull, [](const Outcome& o) { return Milliseconds(o.drawWallMs); });
        table("Segments drawn", cull, [](const Outcome& o) { return std::to_string(o.segments); });
        if (std::any_of(outcomes_.begin(), outcomes_.end(),
                        [](const Outcome& o) { return o.cpuTotalMs >= 0.0; })) {
            table("CPU pass, total", cull,
                  [](const Outcome& o) { return Milliseconds(o.cpuTotalMs); });
        }
    }
    bool notes = false;
    for (const auto& outcome : outcomes_) {
        if (!outcome.note.empty()) {
            text += std::format("{}{} + {} ({}): {}\n", notes ? "" : "\nNotes\n", outcome.filter,
                                outcome.expansion, outcome.cullSegments ? "culled" : "not culled",
                                outcome.note);
            notes = true;
        }
    }
    summary_ = text;
    std::filesystem::create_directories(summaryPath_.parent_path());
    std::ofstream(summaryPath_) << text;
}

} // namespace chmv::benchmark
