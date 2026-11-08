#include "benchmark/BenchmarkLog.h"

#include "streaming/lod/LodPageReport.h"

#include <format>
#include <fstream>
#include <stdexcept>

namespace chmv::benchmark {
namespace {

// A CSV field; GL renderer names contain commas.
[[nodiscard]] std::string Field(const std::string& text) {
    if (text.find_first_of(",\"\n") == std::string::npos) {
        return text;
    }
    std::string quoted = "\"";
    for (const auto c : text) {
        quoted += c == '"' ? std::string("\"\"") : std::string(1, c == '\n' ? ' ' : c);
    }
    return quoted + '"';
}

} // namespace

void AppendBenchmark(const std::filesystem::path& directory, const BenchmarkContext& context,
                     std::span<const NamedTiming> timings) {
    std::filesystem::create_directories(directory);
    const auto path = directory / "benchmark.csv";
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream csv(path, std::ios::app);
    if (!csv) {
        throw std::runtime_error("could not write " + path.string());
    }
    if (fresh) {
        csv << "run,time_utc,renderer,processor,cpu_threads,filter,expansion,cull,tolerance_px,"
               "width,height,view_m,camera_level,level,threshold,pages,candidates,segments,"
               "metric,samples,mean_ms,median_ms,p95_ms,p99_ms,min_ms,max_ms,stddev_ms\n";
    }
    const auto now = streaming::lod::ReportTimestamp();
    for (const auto& [metric, stats] : timings) {
        if (stats.samples == 0u) {
            continue;
        }
        csv << context.run << ',' << now << ',' << Field(context.renderer) << ','
            << context.processor << ',' << context.cpuThreads << ',' << Field(context.filter) << ','
            << Field(context.expansion) << ',' << (context.cullSegments ? 1 : 0) << ','
            << std::format("{:.3f},{},{},{:.1f}", context.tolerancePixels, context.width,
                           context.height, context.viewHeightMeters)
            << ',' << context.cameraLevel << ',' << context.level << ',' << context.threshold << ','
            << context.pages << ',' << context.candidates << ',' << context.segments << ','
            << metric << ',' << stats.samples << ','
            << std::format("{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}", stats.mean,
                           stats.median, stats.p95, stats.p99, stats.min, stats.max, stats.stddev)
            << '\n';
    }
}

} // namespace chmv::benchmark
