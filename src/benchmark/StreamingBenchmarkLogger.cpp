#include "benchmark/StreamingBenchmarkLogger.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace chmv::benchmark {
namespace {
constexpr double kMiB = 1024.0 * 1024.0;
constexpr std::size_t kFlushBytes = 32u * 1024u;

std::string TimestampForFilename() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return out.str();
}
}

StreamingBenchmarkLogger::~StreamingBenchmarkLogger() { Stop(); }

void StreamingBenchmarkLogger::Start(const std::filesystem::path& datasetPath,
                                     std::string runtimeMode) {
    Stop();
    runtimeMode_ = std::move(runtimeMode);
    std::error_code ec;
    auto directory = std::filesystem::current_path(ec) / "logs";
    if (ec) return;
    std::filesystem::create_directories(directory, ec);
    if (ec) return;
    path_ = directory / ("stream_benchmark_" + TimestampForFilename() + ".jsonl");
    {
        std::ofstream probe(path_, std::ios::trunc);
        if (!probe) return;
    }
    enabled_ = true;
    nextTransitionId_ = 1;
    lastFinishedSequence_ = 0;
    transition_.reset();
    lastCounters_.reset();
    writer_ = std::jthread([this](std::stop_token token) { WriterMain(token); });

    std::ostringstream line;
    line << "{\"type\":\"session_begin\",\"runtime_mode\":\""
         << JsonEscape(runtimeMode_) << "\",\"dataset\":\""
         << JsonEscape(datasetPath.string()) << "\"}";
    Queue(line.str());
}

void StreamingBenchmarkLogger::Stop() {
    if (enabled_) {
        if (transition_) {
            // No stats are available here, so the active transition will already have been logged
            // by the last Observe call or intentionally omitted on process shutdown.
            transition_.reset();
        }
        Queue("{\"type\":\"session_end\"}");
    }
    enabled_ = false;
    if (writer_.joinable()) {
        writer_.request_stop();
        condition_.notify_all();
        writer_.join();
    }
}

StreamingBenchmarkLogger::Counters StreamingBenchmarkLogger::Capture(
    const streaming::runtime::GraphPageStreamingStats& s) {
    return {
        s.totalRootPlannerMs,
        s.rootIoBytesRead,
        s.rootIoReadMilliseconds,
        s.estimatedSourceBytesRead,
        s.totalPageLoadMs,
        s.totalRefinementBlockLoadMs,
        s.gpuIncrementalRootBytesUploaded,
        s.gpuIncrementalBackingBytesUploaded,
        s.totalGpuRootUploadMs,
        s.totalGpuBackingUploadMs,
        s.totalPageLoads,
        s.totalRefinementBlockLoads,
        s.totalEvictions,
        s.totalRefinementBlockEvictions,
        s.gpuRootPageEvictions,
        s.gpuBackingBlockEvictions,
    };
}

std::uint64_t StreamingBenchmarkLogger::Delta(std::uint64_t value, std::uint64_t baseline) {
    return value >= baseline ? value - baseline : value;
}

double StreamingBenchmarkLogger::Delta(double value, double baseline) {
    return value >= baseline ? value - baseline : value;
}

std::string StreamingBenchmarkLogger::JsonEscape(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char c : value) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

std::string StreamingBenchmarkLogger::RuntimeSource(
    const streaming::runtime::GraphPageStreamingStats& stats) {
    if (stats.runtimeDataFullyResident) return "ram_backed";
    if (stats.runtimeDataPreloadActive) return "ram_preloading";
    return "ssd_backed";
}

void StreamingBenchmarkLogger::BeginTransition(
    std::uint64_t viewSequence,
    const streaming::runtime::GraphPageStreamingStats& stats) {
    Transition t;
    t.id = nextTransitionId_++;
    t.viewSequence = viewSequence;
    t.begin = std::chrono::steady_clock::now();
    t.lastSample = t.begin;
    t.baseline = lastCounters_.value_or(Capture(stats));
    transition_ = t;

    std::ostringstream line;
    line << "{\"type\":\"transition_begin\",\"id\":" << t.id
         << ",\"view_sequence\":" << viewSequence
         << ",\"runtime_source\":\"" << RuntimeSource(stats) << "\""
         << ",\"target_lod\":" << stats.targetLodLevelFloat
         << ",\"display_lod\":" << stats.displayLodLevelFloat
         << ",\"required_pages\":" << stats.requiredPageCount
         << ",\"required_resident\":" << stats.requiredResidentCount << "}";
    Queue(line.str());
}

void StreamingBenchmarkLogger::FinishTransition(
    std::string_view status,
    const streaming::runtime::GraphPageStreamingStats& stats) {
    if (!transition_) return;
    const auto t = *transition_;
    const auto now = std::chrono::steady_clock::now();
    const auto wallMs = std::chrono::duration<double, std::milli>(now - t.begin).count();
    const auto current = Capture(stats);
    const auto rootIoBytes = Delta(current.rootIoBytes, t.baseline.rootIoBytes);
    const auto sourceBytes = Delta(current.sourceBytes, t.baseline.sourceBytes);
    const auto rootUploadBytes = Delta(current.rootUploadBytes, t.baseline.rootUploadBytes);
    const auto backingUploadBytes = Delta(current.backingUploadBytes, t.baseline.backingUploadBytes);
    const auto plannerMs = Delta(current.plannerMs, t.baseline.plannerMs);
    const auto ioMs = Delta(current.pageLoadMs, t.baseline.pageLoadMs) +
                      Delta(current.refinementLoadMs, t.baseline.refinementLoadMs);
    const auto uploadMs = Delta(current.rootUploadMs, t.baseline.rootUploadMs) +
                          Delta(current.backingUploadMs, t.baseline.backingUploadMs);
    const auto syncCpuMs = t.readbackCpuMs;
    const auto gpuRefineMs = t.gpuRefineMs;
    const auto cpuPlannerRuntimeMs = t.updateCpuMs;
    std::string_view largestPhase = "planner";
    double largestMs = plannerMs;
    const auto consider = [&](std::string_view name, double value) {
        if (value > largestMs) {
            largestMs = value;
            largestPhase = name;
        }
    };
    consider("io_load", ioMs);
    consider("gpu_upload", uploadMs);
    consider("gpu_refine", gpuRefineMs);
    consider("gpu_readback_cpu", syncCpuMs);
    consider("app_stream_update_cpu", cpuPlannerRuntimeMs);

    std::ostringstream line;
    line << std::fixed << std::setprecision(4)
         << "{\"type\":\"transition_end\",\"id\":" << t.id
         << ",\"view_sequence\":" << t.viewSequence
         << ",\"status\":\"" << status << "\""
         << ",\"runtime_source\":\"" << RuntimeSource(stats) << "\""
         << ",\"wall_ms\":" << wallMs
         << ",\"largest_phase\":\"" << largestPhase << "\""
         << ",\"largest_phase_ms\":" << largestMs
         << ",\"frames\":" << t.frames
         << ",\"target_lod\":" << stats.targetLodLevelFloat
         << ",\"display_lod\":" << stats.displayLodLevelFloat
         << ",\"planner_ms\":" << Delta(current.plannerMs, t.baseline.plannerMs)
         << ",\"app_stream_update_cpu_ms\":" << t.updateCpuMs
         << ",\"gpu_process_submit_cpu_ms\":" << t.processSubmitCpuMs
         << ",\"gpu_fault_readback_cpu_ms\":" << t.readbackCpuMs
         << ",\"draw_submit_cpu_ms\":" << t.drawSubmitCpuMs
         << ",\"root_io_ms\":" << Delta(current.rootIoMs, t.baseline.rootIoMs)
         << ",\"root_io_mib\":" << static_cast<double>(rootIoBytes) / kMiB
         << ",\"page_load_ms\":" << Delta(current.pageLoadMs, t.baseline.pageLoadMs)
         << ",\"refinement_load_ms\":"
         << Delta(current.refinementLoadMs, t.baseline.refinementLoadMs)
         << ",\"source_read_mib\":" << static_cast<double>(sourceBytes) / kMiB
         << ",\"root_upload_ms\":" << Delta(current.rootUploadMs, t.baseline.rootUploadMs)
         << ",\"root_upload_mib\":" << static_cast<double>(rootUploadBytes) / kMiB
         << ",\"backing_upload_ms\":"
         << Delta(current.backingUploadMs, t.baseline.backingUploadMs)
         << ",\"backing_upload_mib\":" << static_cast<double>(backingUploadBytes) / kMiB
         << ",\"gpu_filter_ms_sum\":" << t.gpuFilterMs
         << ",\"gpu_refine_ms_sum\":" << t.gpuRefineMs
         << ",\"gpu_compose_ms_sum\":" << t.gpuComposeMs
         << ",\"page_loads\":" << Delta(current.pageLoads, t.baseline.pageLoads)
         << ",\"refinement_loads\":"
         << Delta(current.refinementLoads, t.baseline.refinementLoads)
         << ",\"page_evictions\":" << Delta(current.pageEvictions, t.baseline.pageEvictions)
         << ",\"refinement_evictions\":"
         << Delta(current.refinementEvictions, t.baseline.refinementEvictions)
         << ",\"gpu_root_evictions\":"
         << Delta(current.gpuRootEvictions, t.baseline.gpuRootEvictions)
         << ",\"gpu_backing_evictions\":"
         << Delta(current.gpuBackingEvictions, t.baseline.gpuBackingEvictions)
         << ",\"required_pages\":" << stats.requiredPageCount
         << ",\"required_resident\":" << stats.requiredResidentCount
         << ",\"refinement_required\":" << stats.desiredRefinementBlockCount
         << ",\"refinement_resident\":" << stats.residentRefinementBlockCount
         << ",\"gpu_missing_blocks\":" << stats.gpuMissingBlockRequests
         << ",\"gpu_pending_roots\":" << stats.gpuRefinementPendingRoots
         << ",\"resident_fast_path\":"
         << (stats.gpuResidentWorkingSetFastPath ? "true" : "false") << "}";
    Queue(line.str());
    lastFinishedSequence_ = t.viewSequence;
    transition_.reset();
}

void StreamingBenchmarkLogger::ObserveStreamingFrame(
    std::uint64_t viewSequence,
    const streaming::runtime::GraphPageStreamingStats& stats,
    const StreamingFrameTiming& timing) {
    if (!enabled_ || viewSequence == 0) return;

    if (transition_ && transition_->viewSequence != viewSequence) {
        FinishTransition("superseded", stats);
    }
    if (!transition_ && viewSequence != lastFinishedSequence_) {
        BeginTransition(viewSequence, stats);
    }
    if (!transition_) {
        lastCounters_ = Capture(stats);
        return;
    }

    auto& t = *transition_;
    ++t.frames;
    t.updateCpuMs += timing.updateStreamingCpuMs;
    t.processSubmitCpuMs += timing.gpuProcessSubmitCpuMs;
    t.readbackCpuMs += timing.gpuFaultReadbackCpuMs;
    t.drawSubmitCpuMs += timing.drawSubmitCpuMs;
    t.gpuFilterMs += stats.gpuFilterCullMs;
    t.gpuRefineMs += stats.gpuRefinementMs;
    t.gpuComposeMs += stats.gpuComposeMs;

    const bool rootReady = !stats.lodTransitionPending &&
                           stats.requiredPageCount == stats.requiredResidentCount &&
                           stats.queuedPageCount == 0 && stats.loadingPageCount == 0;
    const bool backingReady = stats.queuedRefinementBlockCount == 0 &&
                              stats.loadingRefinementBlockCount == 0 &&
                              stats.gpuMissingBlockRequests == 0;
    const bool refinementReady = stats.gpuRefinementPendingRoots == 0;
    if (rootReady && backingReady && refinementReady) {
        ++t.cleanFrames;
    } else {
        t.cleanFrames = 0;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - t.lastSample >= std::chrono::seconds(1)) {
        const auto current = Capture(stats);
        const auto wallMs = std::chrono::duration<double, std::milli>(now - t.begin).count();
        std::ostringstream sample;
        sample << std::fixed << std::setprecision(4)
               << "{\"type\":\"transition_sample\",\"id\":" << t.id
               << ",\"wall_ms\":" << wallMs
               << ",\"runtime_source\":\"" << RuntimeSource(stats) << "\""
               << ",\"target_lod\":" << stats.targetLodLevelFloat
               << ",\"display_lod\":" << stats.displayLodLevelFloat
               << ",\"required_pages\":" << stats.requiredPageCount
               << ",\"required_resident\":" << stats.requiredResidentCount
               << ",\"queued_pages\":" << stats.queuedPageCount
               << ",\"loading_pages\":" << stats.loadingPageCount
               << ",\"gpu_missing_blocks\":" << stats.gpuMissingBlockRequests
               << ",\"gpu_pending_roots\":" << stats.gpuRefinementPendingRoots
               << ",\"planner_ms\":" << Delta(current.plannerMs, t.baseline.plannerMs)
               << ",\"page_load_ms\":" << Delta(current.pageLoadMs, t.baseline.pageLoadMs)
               << ",\"refinement_load_ms\":"
               << Delta(current.refinementLoadMs, t.baseline.refinementLoadMs)
               << ",\"root_upload_ms\":" << Delta(current.rootUploadMs, t.baseline.rootUploadMs)
               << ",\"backing_upload_ms\":"
               << Delta(current.backingUploadMs, t.baseline.backingUploadMs)
               << ",\"gpu_refine_ms_sum\":" << t.gpuRefineMs
               << ",\"fault_readback_cpu_ms\":" << t.readbackCpuMs
               << "}";
        Queue(sample.str());
        t.lastSample = now;
    }

    lastCounters_ = Capture(stats);
    if (t.cleanFrames >= 2u) {
        FinishTransition("ready", stats);
    }
}

void StreamingBenchmarkLogger::RecordFullResidentInstall(
    std::uint64_t nodes, std::uint64_t edges, double gpuUploadMs, double totalInstallMs) {
    if (!enabled_) return;
    std::ostringstream line;
    line << std::fixed << std::setprecision(4)
         << "{\"type\":\"full_resident_install\",\"nodes\":" << nodes
         << ",\"edges\":" << edges
         << ",\"gpu_upload_ms\":" << gpuUploadMs
         << ",\"total_install_ms\":" << totalInstallMs << "}";
    Queue(line.str());
}

void StreamingBenchmarkLogger::RecordEvent(std::string eventName, std::string details) {
    if (!enabled_) return;
    std::ostringstream line;
    line << "{\"type\":\"event\",\"name\":\"" << JsonEscape(eventName) << "\"";
    if (!details.empty()) {
        line << ",\"details\":\"" << JsonEscape(details) << "\"";
    }
    line << "}";
    Queue(line.str());
}

void StreamingBenchmarkLogger::Queue(std::string line) {
    if (!enabled_ && !writer_.joinable()) return;
    bool wake = false;
    {
        std::lock_guard lock(mutex_);
        pendingBytes_ += line.size() + 1u;
        pending_.push_back(std::move(line));
        wake = pendingBytes_ >= kFlushBytes;
    }
    if (wake) condition_.notify_one();
}

void StreamingBenchmarkLogger::WriterMain(std::stop_token stopToken) {
    std::ofstream output(path_, std::ios::app);
    if (!output) return;
    for (;;) {
        std::deque<std::string> batch;
        {
            std::unique_lock lock(mutex_);
            condition_.wait_for(lock, std::chrono::seconds(2), [&] {
                return stopToken.stop_requested() || pendingBytes_ >= kFlushBytes;
            });
            batch.swap(pending_);
            pendingBytes_ = 0;
        }
        for (const auto& line : batch) output << line << '\n';
        if (!batch.empty()) output.flush();
        if (stopToken.stop_requested()) {
            std::lock_guard lock(mutex_);
            if (pending_.empty()) break;
        }
    }
    std::deque<std::string> tail;
    {
        std::lock_guard lock(mutex_);
        tail.swap(pending_);
        pendingBytes_ = 0;
    }
    for (const auto& line : tail) output << line << '\n';
    output.flush();
}

} // namespace chmv::benchmark
