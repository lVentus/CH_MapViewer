#pragma once

#include "benchmark/TimingStats.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>

namespace chmv::benchmark {

// What a benchmark measured and under which conditions.
struct BenchmarkContext {
    // Groups the lines of one button press or one suite (ReportTimestamp).
    std::string run;
    std::string renderer;
    // "GPU", "CPU pages" or "CPU classic".
    std::string processor;
    unsigned cpuThreads = 0;
    std::string filter;
    std::string expansion;
    bool cullSegments = false;
    float tolerancePixels = 0.0f;
    int width = 0;
    int height = 0;
    double viewHeightMeters = 0.0;
    std::uint32_t cameraLevel = 0;
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
    std::uint64_t pages = 0;
    std::uint64_t candidates = 0;
    std::uint64_t segments = 0;
};

using NamedTiming = std::pair<const char*, TimingStats>;

// Appends to <directory>/benchmark.csv one line per measured quantity (filter, expansion, pass,
// draw, ...), with the context repeated on every line, so tables and plots can filter and pivot
// freely. The header is written with the first line.
void AppendBenchmark(const std::filesystem::path& directory, const BenchmarkContext& context,
                     std::span<const NamedTiming> timings);

} // namespace chmv::benchmark
