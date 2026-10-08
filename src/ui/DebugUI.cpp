#include "ui/DebugUI.h"

#include "benchmark/CorrectnessValidator.h"
#include "core/Window.h"
#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHGraph.h"
#include "gpu/filtering/RangeFilterStrategyManager.h"
#include "gpu/unfolding/UnfoldingStrategyManager.h"
#include "pipeline/CPUReferencePipeline.h"
#include "pipeline/GPUDrivenPipeline.h"
#include "pipeline/ProcessingMode.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "streaming/index/CHIndex.h"
#include "streaming/runtime/GraphPageStreamer.h"
#include "streaming/analysis/TextSourceScanner.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace chmv::ui {
namespace {

const char* ProcessingModeName(pipeline::ProcessingMode mode) {
    switch (mode) {
    case pipeline::ProcessingMode::GPUDriven:
        return "GPU Driven";
    case pipeline::ProcessingMode::CPUReference:
        return "CPU Paper (Ordered)";
    case pipeline::ProcessingMode::Validation:
        return "Validation";
    }
    return "Unknown";
}

constexpr auto kOocTelemetryActiveSampleInterval = std::chrono::seconds(1);
constexpr auto kOocTelemetryIdleSampleInterval = std::chrono::seconds(5);
constexpr auto kOocTelemetryWriterFlushInterval = std::chrono::seconds(5);
constexpr auto kOocTelemetryBurstIdleGrace = std::chrono::milliseconds(250);
constexpr std::size_t kOocTelemetryWriterBatchBytes = 64u * 1024u;
constexpr std::uint32_t kOocTelemetryBacklogHighWatermark = 128u;
constexpr std::uint32_t kOocTelemetryBacklogLowWatermark = 32u;
constexpr double kBytesPerMiB = 1024.0 * 1024.0;

std::tm LocalTime(std::time_t value) {
    std::tm result{};
#if defined(_WIN32)
    localtime_s(&result, &value);
#else
    localtime_r(&value, &result);
#endif
    return result;
}

std::string WallClockStamp(std::chrono::system_clock::time_point now, bool fileName) {
    const auto wholeSeconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - wholeSeconds).count();
    const auto timeValue = std::chrono::system_clock::to_time_t(now);
    const auto local = LocalTime(timeValue);
    std::ostringstream out;
    out << std::put_time(&local, fileName ? "%Y-%m-%d_%H-%M-%S" : "%Y-%m-%d %H:%M:%S");
    if (!fileName) {
        out << '.' << std::setfill('0') << std::setw(3) << milliseconds;
    }
    return out.str();
}

template <class T>
T CounterDelta(T current, T previous) {
    return current >= previous ? current - previous : current;
}

const char* DatasetLoadPhaseName(data::DatasetLoadPhase phase) {
    switch (phase) {
    case data::DatasetLoadPhase::Idle:
        return "Idle";
    case data::DatasetLoadPhase::CheckingPreprocess:
        return "Checking preprocess cache";
    case data::DatasetLoadPhase::AnalyzingSourceLayout:
        return "Preprocess: scanning source layout";
    case data::DatasetLoadPhase::LoadingSourceForPreprocess:
        return "Preprocess: reading source / building compact tables";
    case data::DatasetLoadPhase::ReorderingNodes:
        return "Preprocess: reordering nodes";
    case data::DatasetLoadPhase::ResolvingGeometryBounds:
        return "Preprocess: resolving shortcut geometry bounds";
    case data::DatasetLoadPhase::ReorderingEdges:
        return "Preprocess: direct fixed-grid edge layout";
    case data::DatasetLoadPhase::WritingPreprocessedGraph:
        return "Preprocess: writing reordered graph";
    case data::DatasetLoadPhase::WritingPreprocessedRanges:
        return "Preprocess: writing reordered ranges";
    case data::DatasetLoadPhase::ValidatingPreprocessedLayout:
        return "Preprocess: validating output";
    case data::DatasetLoadPhase::WritingPreprocessedIndex:
        return "Preprocess: writing streaming index";
    case data::DatasetLoadPhase::ReadingNodes:
        return "Loading preprocessed nodes";
    case data::DatasetLoadPhase::ReadingEdges:
        return "Loading preprocessed edges";
    case data::DatasetLoadPhase::ReadingRanges:
        return "Loading preprocessed ranges";
    case data::DatasetLoadPhase::PreparingRangeIndex:
        return "Preparing range index";
    case data::DatasetLoadPhase::PreparingGeometryErrors:
        return "Preparing geometry errors";
    case data::DatasetLoadPhase::PreparingRoadStyleMetadata:
        return "Preparing road-type style metadata";
    case data::DatasetLoadPhase::PreprocessedForStreaming:
        return "Preprocess ready for out-of-core runtime";
    case data::DatasetLoadPhase::Ready:
        return "Ready";
    case data::DatasetLoadPhase::Failed:
        return "Load failed";
    }
    return "Loading";
}

} // namespace

DebugUI::DebugUI(const core::Window& window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    const float contentScale = window.ContentScale();
    ImGui::GetStyle().ScaleAllSizes(contentScale);

    ImFontConfig fontConfig;
    fontConfig.SizePixels = 16.0f * contentScale;
    ImGui::GetIO().Fonts->AddFontDefault(&fontConfig);

    ImGui_ImplGlfw_InitForOpenGL(window.Handle(), true);
    ImGui_ImplOpenGL3_Init("#version 430 core");
}

DebugUI::~DebugUI() {
    StopOocTelemetry();
    if (roadTypeDiagnosticWorker_.joinable()) {
        roadTypeDiagnosticWorker_.request_stop();
        roadTypeDiagnosticWorker_.join();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

DebugUI::OocTelemetryCounters DebugUI::CaptureOocTelemetryCounters(
    const streaming::runtime::GraphPageStreamingStats& stats) {
    OocTelemetryCounters counters;
    counters.pageCacheHits = stats.pageCacheHits;
    counters.pageCacheMisses = stats.pageCacheMisses;
    counters.uniquePageLoads = stats.uniquePageLoads;
    counters.pageReloadsAfterEviction = stats.pageReloadsAfterEviction;
    counters.pageInFlightReuses = stats.pageInFlightReuses;
    counters.pageQueueReprioritizations = stats.pageQueueReprioritizations;
    counters.stalePageQueueEntriesSkipped = stats.stalePageQueueEntriesSkipped;
    counters.stalePageLoadsDiscarded = stats.stalePageLoadsDiscarded;
    counters.rootIoReadOperations = stats.rootIoReadOperations;
    counters.rootIoBatchedReadOperations = stats.rootIoBatchedReadOperations;
    counters.rootIoPagesRead = stats.rootIoPagesRead;
    counters.rootIoBytesRead = stats.rootIoBytesRead;
    counters.rootIoReadMilliseconds = stats.rootIoReadMilliseconds;
    counters.totalEvictions = stats.totalEvictions;
    counters.totalRefinementBlockLoads = stats.totalRefinementBlockLoads;
    counters.totalRefinementBlockEvictions = stats.totalRefinementBlockEvictions;
    counters.nodeBlockCacheHits = stats.nodeBlockCacheHits;
    counters.nodeBlockCacheMisses = stats.nodeBlockCacheMisses;
    counters.nodeBlockCacheEvictions = stats.nodeBlockCacheEvictions;
    counters.rootPlannerRebuilds = stats.rootPlannerRebuilds;
    counters.rootPlannerCacheReuses = stats.rootPlannerCacheReuses;
    counters.gpuIncrementalRootBytesUploaded = stats.gpuIncrementalRootBytesUploaded;
    counters.gpuIncrementalBackingBytesUploaded = stats.gpuIncrementalBackingBytesUploaded;
    counters.gpuRootPageEvictions = stats.gpuRootPageEvictions;
    counters.gpuBackingBlockEvictions = stats.gpuBackingBlockEvictions;
    counters.gpuBackingBlockFirstUploads = stats.gpuBackingBlockFirstUploads;
    counters.gpuBackingBlockReuploadsAfterEviction =
        stats.gpuBackingBlockReuploadsAfterEviction;
    counters.gpuBackingGraceFallbackEvictions = stats.gpuBackingGraceFallbackEvictions;
    counters.gpuRootCacheAllocationFailures = stats.gpuRootCacheAllocationFailures;
    counters.gpuBackingCacheAllocationFailures = stats.gpuBackingCacheAllocationFailures;
    return counters;
}

void DebugUI::QueueOocTelemetryRecord(std::string payload,
                                      std::chrono::steady_clock::time_point now) {
    const auto elapsed = std::chrono::duration<double>(now - oocTelemetryStartTime_).count();
    std::ostringstream line;
    line << '[' << WallClockStamp(std::chrono::system_clock::now(), false) << ']'
         << "[t=" << std::fixed << std::setprecision(3) << elapsed << "s] " << payload;

    auto record = line.str();
    bool wakeWriter = false;
    {
        std::lock_guard lock(oocTelemetryWriterMutex_);
        oocTelemetryPendingBytes_ += record.size() + 1u;
        oocTelemetryPendingLines_.push_back(std::move(record));
        wakeWriter = oocTelemetryPendingBytes_ >= kOocTelemetryWriterBatchBytes;
    }
    if (wakeWriter) {
        oocTelemetryWriterCondition_.notify_one();
    }
}

void DebugUI::OocTelemetryWriterMain(std::stop_token stopToken, std::filesystem::path path) {
    std::ofstream output(path, std::ios::app);
    if (!output) {
        return;
    }

    for (;;) {
        std::deque<std::string> batch;
        {
            std::unique_lock lock(oocTelemetryWriterMutex_);
            oocTelemetryWriterCondition_.wait_for(
                lock, kOocTelemetryWriterFlushInterval, [&] {
                    return stopToken.stop_requested() ||
                           oocTelemetryPendingBytes_ >= kOocTelemetryWriterBatchBytes;
                });
            batch.swap(oocTelemetryPendingLines_);
            oocTelemetryPendingBytes_ = 0u;
        }

        for (const auto& line : batch) {
            output << line << '\n';
        }
        if (!batch.empty()) {
            output.flush();
        }

        if (stopToken.stop_requested()) {
            std::lock_guard lock(oocTelemetryWriterMutex_);
            if (oocTelemetryPendingLines_.empty()) {
                break;
            }
        }
    }

    std::deque<std::string> tail;
    {
        std::lock_guard lock(oocTelemetryWriterMutex_);
        tail.swap(oocTelemetryPendingLines_);
        oocTelemetryPendingBytes_ = 0u;
    }
    for (const auto& line : tail) {
        output << line << '\n';
    }
    output.flush();
}

bool DebugUI::StartOocTelemetry(
    const streaming::runtime::GraphPageStreamingStats& stats,
    const renderer::MapCamera2D* camera) {
    if (oocTelemetryEnabled_) {
        return true;
    }

    oocTelemetryError_.clear();
    oocTelemetryDeleteMessage_.clear();
    std::error_code ec;
    oocTelemetryDirectory_ = std::filesystem::current_path(ec) / "logs";
    if (ec) {
        oocTelemetryError_ = "Could not resolve current directory: " + ec.message();
        return false;
    }
    std::filesystem::create_directories(oocTelemetryDirectory_, ec);
    if (ec) {
        oocTelemetryError_ = "Could not create log directory: " + ec.message();
        return false;
    }

    const auto wallNow = std::chrono::system_clock::now();
    const auto baseName = std::string("ooc_") + WallClockStamp(wallNow, true);
    oocTelemetryCurrentPath_ = oocTelemetryDirectory_ / (baseName + ".log");
    for (std::uint32_t suffix = 1u; std::filesystem::exists(oocTelemetryCurrentPath_, ec); ++suffix) {
        ec.clear();
        oocTelemetryCurrentPath_ =
            oocTelemetryDirectory_ / (baseName + "_" + std::to_string(suffix) + ".log");
    }

    {
        std::ofstream probe(oocTelemetryCurrentPath_, std::ios::trunc);
        if (!probe) {
            oocTelemetryError_ = "Could not create telemetry log: " +
                                 oocTelemetryCurrentPath_.string();
            return false;
        }
        probe << "# CH_MapViewer Out-of-Core telemetry\n"
              << "# Active sampling: 1 s; idle heartbeat: 5 s; disk writes: background batches <= every 5 s\n"
              << "# Per-page events are aggregated into counters; logging is disabled by default.\n"
              << "# LOD changes are sampled, not emitted per level; backing first/re-upload counters diagnose GPU cache churn.\n";
    }

    {
        std::lock_guard lock(oocTelemetryWriterMutex_);
        oocTelemetryPendingLines_.clear();
        oocTelemetryPendingBytes_ = 0u;
    }

    const auto now = std::chrono::steady_clock::now();
    oocTelemetryStartTime_ = now;
    oocTelemetryLastSampleTime_ = now;
    oocTelemetryLastReloadEventTime_ = now - std::chrono::seconds(2);
    oocTelemetryBurstIdleSince_.reset();
    oocTelemetryLastSampleCounters_ = CaptureOocTelemetryCounters(stats);
    oocTelemetryBurstStartCounters_ = oocTelemetryLastSampleCounters_;
    oocTelemetryLastObservedReloads_ = stats.pageReloadsAfterEviction;
    oocTelemetryPendingReloadEvents_ = 0u;
    oocTelemetryBurstActive_ = false;
    oocTelemetryBacklogHigh_ = false;
    oocTelemetryEnabled_ = true;

    oocTelemetryWriter_ = std::jthread(
        [this, path = oocTelemetryCurrentPath_](std::stop_token stopToken) {
            OocTelemetryWriterMain(stopToken, path);
        });

    std::ostringstream start;
    start << "EVENT SESSION_BEGIN"
          << " targetLod=" << stats.targetLodLevel
          << " displayLod=" << stats.displayLodLevel
          << " residentPages=" << stats.residentPageCount
          << " cpuCacheMiB=" << std::fixed << std::setprecision(2)
          << static_cast<double>(stats.totalCpuCacheBytes) / kBytesPerMiB;
    if (camera) {
        start << " zoom=" << camera->ZoomFactor()
              << " centerX=" << camera->CenterX()
              << " centerY=" << camera->CenterY();
    }
    QueueOocTelemetryRecord(start.str(), now);
    return true;
}

void DebugUI::StopOocTelemetry() {
    if (!oocTelemetryEnabled_ && !oocTelemetryWriter_.joinable()) {
        return;
    }

    if (oocTelemetryEnabled_) {
        QueueOocTelemetryRecord("EVENT SESSION_END", std::chrono::steady_clock::now());
    }
    oocTelemetryEnabled_ = false;

    if (oocTelemetryWriter_.joinable()) {
        oocTelemetryWriter_.request_stop();
        oocTelemetryWriterCondition_.notify_all();
        oocTelemetryWriter_.join();
    }
}

void DebugUI::DeleteOocTelemetryLogs() {
    if (oocTelemetryEnabled_) {
        oocTelemetryDeleteMessage_ = "Disable telemetry before deleting logs.";
        return;
    }

    std::error_code ec;
    auto directory = oocTelemetryDirectory_;
    if (directory.empty()) {
        directory = std::filesystem::current_path(ec) / "logs";
    }
    if (ec || !std::filesystem::exists(directory, ec)) {
        oocTelemetryDeleteMessage_ = "No OOC telemetry logs found.";
        return;
    }

    std::uint32_t removed = 0u;
    for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end;
         it.increment(ec)) {
        const auto path = it->path();
        std::error_code entryEc;
        if (!it->is_regular_file(entryEc)) {
            continue;
        }
        const auto name = path.filename().string();
        if (name.starts_with("ooc_") && path.extension() == ".log") {
            std::filesystem::remove(path, ec);
            if (!ec) {
                ++removed;
            } else {
                break;
            }
        }
    }

    if (ec) {
        oocTelemetryDeleteMessage_ = "Failed to delete logs: " + ec.message();
    } else {
        oocTelemetryDeleteMessage_ = "Deleted " + std::to_string(removed) + " OOC log(s).";
        oocTelemetryCurrentPath_.clear();
    }
}

void DebugUI::UpdateOocTelemetry(
    const streaming::runtime::GraphPageStreamingStats& stats,
    const renderer::MapCamera2D* camera) {
    if (!oocTelemetryEnabled_) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto current = CaptureOocTelemetryCounters(stats);

    const bool countersReset =
        current.pageCacheHits < oocTelemetryLastSampleCounters_.pageCacheHits ||
        current.pageCacheMisses < oocTelemetryLastSampleCounters_.pageCacheMisses ||
        current.uniquePageLoads < oocTelemetryLastSampleCounters_.uniquePageLoads ||
        current.rootIoReadOperations < oocTelemetryLastSampleCounters_.rootIoReadOperations ||
        current.gpuIncrementalRootBytesUploaded <
            oocTelemetryLastSampleCounters_.gpuIncrementalRootBytesUploaded;
    if (countersReset) {
        QueueOocTelemetryRecord("EVENT COUNTERS_RESET streamer_or_dataset_replaced=1", now);
        oocTelemetryLastSampleCounters_ = current;
        oocTelemetryBurstStartCounters_ = current;
        oocTelemetryLastObservedReloads_ = current.pageReloadsAfterEviction;
        oocTelemetryPendingReloadEvents_ = 0u;
        oocTelemetryLastSampleTime_ = now;
        oocTelemetryBurstActive_ = false;
        oocTelemetryBurstIdleSince_.reset();
    }

    const auto queued = stats.queuedPageCount + stats.queuedRefinementBlockCount;
    const auto loading = stats.loadingPageCount + stats.loadingRefinementBlockCount;
    const bool streamingActive = queued != 0u || loading != 0u ||
                                 stats.lodTransitionPending || !stats.currentDemandReady;

    if (streamingActive) {
        oocTelemetryBurstIdleSince_.reset();
        if (!oocTelemetryBurstActive_) {
            oocTelemetryBurstActive_ = true;
            oocTelemetryBurstStartTime_ = now;
            oocTelemetryBurstStartCounters_ = current;
            std::ostringstream event;
            event << "EVENT STREAM_BURST_BEGIN queued=" << queued
                  << " loading=" << loading
                  << " targetLod=" << stats.targetLodLevel
                  << " displayLod=" << stats.displayLodLevel;
            QueueOocTelemetryRecord(event.str(), now);
        }
    } else if (oocTelemetryBurstActive_) {
        if (!oocTelemetryBurstIdleSince_) {
            oocTelemetryBurstIdleSince_ = now;
        } else if (now - *oocTelemetryBurstIdleSince_ >= kOocTelemetryBurstIdleGrace) {
            const auto duration =
                std::chrono::duration<double>(now - oocTelemetryBurstStartTime_).count();
            std::ostringstream event;
            event << "EVENT STREAM_BURST_END durationSec=" << std::fixed
                  << std::setprecision(3) << duration
                  << " uniqueLoads="
                  << CounterDelta(current.uniquePageLoads,
                                  oocTelemetryBurstStartCounters_.uniquePageLoads)
                  << " reads="
                  << CounterDelta(current.rootIoReadOperations,
                                  oocTelemetryBurstStartCounters_.rootIoReadOperations)
                  << " pagesRead="
                  << CounterDelta(current.rootIoPagesRead,
                                  oocTelemetryBurstStartCounters_.rootIoPagesRead)
                  << " reloads="
                  << CounterDelta(current.pageReloadsAfterEviction,
                                  oocTelemetryBurstStartCounters_.pageReloadsAfterEviction)
                  << " evictions="
                  << CounterDelta(current.totalEvictions,
                                  oocTelemetryBurstStartCounters_.totalEvictions);
            QueueOocTelemetryRecord(event.str(), now);
            oocTelemetryBurstActive_ = false;
            oocTelemetryBurstIdleSince_.reset();
        }
    }

    if (!oocTelemetryBacklogHigh_ && queued >= kOocTelemetryBacklogHighWatermark) {
        oocTelemetryBacklogHigh_ = true;
        std::ostringstream event;
        event << "EVENT QUEUE_BACKLOG_HIGH queued=" << queued
              << " loading=" << loading;
        QueueOocTelemetryRecord(event.str(), now);
    } else if (oocTelemetryBacklogHigh_ && queued <= kOocTelemetryBacklogLowWatermark) {
        oocTelemetryBacklogHigh_ = false;
        std::ostringstream event;
        event << "EVENT QUEUE_BACKLOG_RECOVER queued=" << queued
              << " loading=" << loading;
        QueueOocTelemetryRecord(event.str(), now);
    }

    if (current.pageReloadsAfterEviction < oocTelemetryLastObservedReloads_) {
        oocTelemetryLastObservedReloads_ = current.pageReloadsAfterEviction;
        oocTelemetryPendingReloadEvents_ = 0u;
    } else if (current.pageReloadsAfterEviction > oocTelemetryLastObservedReloads_) {
        oocTelemetryPendingReloadEvents_ +=
            current.pageReloadsAfterEviction - oocTelemetryLastObservedReloads_;
        oocTelemetryLastObservedReloads_ = current.pageReloadsAfterEviction;
    }
    if (oocTelemetryPendingReloadEvents_ != 0u &&
        now - oocTelemetryLastReloadEventTime_ >= std::chrono::seconds(1)) {
        std::ostringstream event;
        event << "EVENT RELOAD_AFTER_EVICT_ACTIVITY count="
              << oocTelemetryPendingReloadEvents_
              << " total=" << current.pageReloadsAfterEviction;
        QueueOocTelemetryRecord(event.str(), now);
        oocTelemetryPendingReloadEvents_ = 0u;
        oocTelemetryLastReloadEventTime_ = now;
    }

    const auto sampleInterval = streamingActive ? kOocTelemetryActiveSampleInterval
                                                : kOocTelemetryIdleSampleInterval;
    if (now - oocTelemetryLastSampleTime_ < sampleInterval) {
        return;
    }

    const auto intervalSeconds =
        std::max(std::chrono::duration<double>(now - oocTelemetryLastSampleTime_).count(), 1e-6);
    const auto dHits = CounterDelta(current.pageCacheHits,
                                    oocTelemetryLastSampleCounters_.pageCacheHits);
    const auto dMisses = CounterDelta(current.pageCacheMisses,
                                      oocTelemetryLastSampleCounters_.pageCacheMisses);
    const auto dReads = CounterDelta(current.rootIoReadOperations,
                                     oocTelemetryLastSampleCounters_.rootIoReadOperations);
    const auto dBatchedReads = CounterDelta(
        current.rootIoBatchedReadOperations,
        oocTelemetryLastSampleCounters_.rootIoBatchedReadOperations);
    const auto dPagesRead = CounterDelta(current.rootIoPagesRead,
                                         oocTelemetryLastSampleCounters_.rootIoPagesRead);
    const auto dBytesRead = CounterDelta(current.rootIoBytesRead,
                                         oocTelemetryLastSampleCounters_.rootIoBytesRead);
    const auto dIoMs = CounterDelta(current.rootIoReadMilliseconds,
                                    oocTelemetryLastSampleCounters_.rootIoReadMilliseconds);
    const auto dUploadRoot = CounterDelta(
        current.gpuIncrementalRootBytesUploaded,
        oocTelemetryLastSampleCounters_.gpuIncrementalRootBytesUploaded);
    const auto dUploadBacking = CounterDelta(
        current.gpuIncrementalBackingBytesUploaded,
        oocTelemetryLastSampleCounters_.gpuIncrementalBackingBytesUploaded);
    const auto readMiB = static_cast<double>(dBytesRead) / kBytesPerMiB;
    const auto wallReadMiBps = readMiB / intervalSeconds;
    const auto serviceReadMiBps = dIoMs > 0.0 ? readMiB * 1000.0 / dIoMs : 0.0;
    const auto pagesPerRead = dReads != 0u
                                  ? static_cast<double>(dPagesRead) /
                                        static_cast<double>(dReads)
                                  : 0.0;

    std::ostringstream sample;
    sample << "SAMPLE intervalSec=" << std::fixed << std::setprecision(3) << intervalSeconds
           << " targetLod=" << stats.targetLodLevel
           << " displayLod=" << stats.displayLodLevel
           << " targetFloat=" << std::setprecision(2) << stats.targetLodLevelFloat
           << " displayFloat=" << stats.displayLodLevelFloat;
    if (camera) {
        sample << " zoom=" << camera->ZoomFactor()
               << " targetZoom=" << camera->TargetZoomFactor()
               << " centerX=" << camera->CenterX()
               << " centerY=" << camera->CenterY();
    }
    sample << " resident=" << stats.residentPageCount
           << " desired=" << stats.desiredPageCount
           << " required=" << stats.requiredPageCount
           << " queued=" << stats.queuedPageCount
           << " loading=" << stats.loadingPageCount
           << " refinementResident=" << stats.residentRefinementBlockCount
           << " refinementQueued=" << stats.queuedRefinementBlockCount
           << " refinementLoading=" << stats.loadingRefinementBlockCount
           << " cpuCacheMiB=" << std::setprecision(2)
           << static_cast<double>(stats.totalCpuCacheBytes) / kBytesPerMiB
           << " gpuMiB=" << static_cast<double>(stats.gpuAllocatedBytes) / kBytesPerMiB
           << " hit=" << dHits
           << " miss=" << dMisses
           << " unique=" << CounterDelta(
                  current.uniquePageLoads, oocTelemetryLastSampleCounters_.uniquePageLoads)
           << " reuse=" << CounterDelta(
                  current.pageInFlightReuses, oocTelemetryLastSampleCounters_.pageInFlightReuses)
           << " reprio=" << CounterDelta(
                  current.pageQueueReprioritizations,
                  oocTelemetryLastSampleCounters_.pageQueueReprioritizations)
           << " reads=" << dReads
           << " batchedReads=" << dBatchedReads
           << " pagesRead=" << dPagesRead
           << " pagesPerRead=" << pagesPerRead
           << " readMiB=" << readMiB
           << " wallReadMiBps=" << wallReadMiBps
           << " serviceReadMiBps=" << serviceReadMiBps
           << " evict=" << CounterDelta(
                  current.totalEvictions, oocTelemetryLastSampleCounters_.totalEvictions)
           << " reload=" << CounterDelta(
                  current.pageReloadsAfterEviction,
                  oocTelemetryLastSampleCounters_.pageReloadsAfterEviction)
           << " staleQueue=" << CounterDelta(
                  current.stalePageQueueEntriesSkipped,
                  oocTelemetryLastSampleCounters_.stalePageQueueEntriesSkipped)
           << " staleIo=" << CounterDelta(
                  current.stalePageLoadsDiscarded,
                  oocTelemetryLastSampleCounters_.stalePageLoadsDiscarded)
           << " refinementLoads=" << CounterDelta(
                  current.totalRefinementBlockLoads,
                  oocTelemetryLastSampleCounters_.totalRefinementBlockLoads)
           << " refinementEvict=" << CounterDelta(
                  current.totalRefinementBlockEvictions,
                  oocTelemetryLastSampleCounters_.totalRefinementBlockEvictions)
           << " planner=" << CounterDelta(
                  current.rootPlannerRebuilds, oocTelemetryLastSampleCounters_.rootPlannerRebuilds)
           << " plannerReuse=" << CounterDelta(
                  current.rootPlannerCacheReuses,
                  oocTelemetryLastSampleCounters_.rootPlannerCacheReuses)
           << " gpuUploadMiB="
           << static_cast<double>(dUploadRoot + dUploadBacking) / kBytesPerMiB
           << " gpuRootEvict=" << CounterDelta(
                  current.gpuRootPageEvictions,
                  oocTelemetryLastSampleCounters_.gpuRootPageEvictions)
           << " gpuBackingEvict=" << CounterDelta(
                  current.gpuBackingBlockEvictions,
                  oocTelemetryLastSampleCounters_.gpuBackingBlockEvictions)
           << " gpuBackingFirst=" << CounterDelta(
                  current.gpuBackingBlockFirstUploads,
                  oocTelemetryLastSampleCounters_.gpuBackingBlockFirstUploads)
           << " gpuBackingReupload=" << CounterDelta(
                  current.gpuBackingBlockReuploadsAfterEviction,
                  oocTelemetryLastSampleCounters_.gpuBackingBlockReuploadsAfterEviction)
           << " gpuBackingGraceFallback=" << CounterDelta(
                  current.gpuBackingGraceFallbackEvictions,
                  oocTelemetryLastSampleCounters_.gpuBackingGraceFallbackEvictions)
           << " gpuRootAllocFail=" << CounterDelta(
                  current.gpuRootCacheAllocationFailures,
                  oocTelemetryLastSampleCounters_.gpuRootCacheAllocationFailures)
           << " gpuBackingAllocFail=" << CounterDelta(
                  current.gpuBackingCacheAllocationFailures,
                  oocTelemetryLastSampleCounters_.gpuBackingCacheAllocationFailures)
           << " gpuRootCached=" << stats.gpuPersistentRootPagesCached
           << " gpuRootActive=" << stats.gpuPersistentRootPagesActive
           << " gpuBackingResident=" << stats.gpuPersistentBackingBlocks
           << " gpuMissingBacking=" << stats.gpuMissingBlockRequests
           << " demandReady=" << (stats.currentDemandReady ? 1 : 0);

    QueueOocTelemetryRecord(sample.str(), now);
    oocTelemetryLastSampleCounters_ = current;
    oocTelemetryLastSampleTime_ = now;
}

void DebugUI::StartRoadTypeDiagnostic(std::filesystem::path graphPath,
                                      std::filesystem::path rangesPath) {
    if (roadTypeDiagnosticRunning_.exchange(true)) {
        return;
    }
    if (roadTypeDiagnosticWorker_.joinable()) {
        roadTypeDiagnosticWorker_.join();
    }

    roadTypeDiagnosticProcessed_.store(0u);
    roadTypeDiagnosticTotal_.store(0u);
    {
        std::lock_guard lock(roadTypeDiagnosticMutex_);
        roadTypeDiagnosticResult_.reset();
    }

    roadTypeDiagnosticWorker_ = std::jthread(
        [this, graphPath = std::move(graphPath), rangesPath = std::move(rangesPath)](
            std::stop_token stopToken) {
            RoadTypeDiagnosticResult result;
            result.graphPath = graphPath;
            try {
                using streaming::analysis::detail::TextSourceScanner;

                TextSourceScanner graphScanner(graphPath);
                const auto nodeCount = graphScanner.Read<std::uint64_t>();
                const auto edgeCount = graphScanner.Read<std::uint64_t>();

                const auto indexPath = streaming::index::CHIndex::DefaultPath(graphPath);
                const bool validIndex = streaming::index::CHIndex::IsValid(
                    indexPath, graphPath, rangesPath, {});

                std::optional<streaming::index::CHIndexData> index;
                if (validIndex) {
                    index = streaming::index::CHIndex::Load(indexPath);
                    if (index->nodeCount != nodeCount || index->edgeCount != edgeCount) {
                        throw std::runtime_error(
                            "CHIDX counts do not match the selected .sch file");
                    }
                    if (index->graphEdgeBlocks.empty()) {
                        throw std::runtime_error(
                            "CHIDX has no graph edge-block offsets for the selected .sch file");
                    }
                    graphScanner.Seek(index->graphEdgeBlocks.front().byteOffset);
                } else {
                    const auto totalWork = nodeCount + edgeCount;
                    roadTypeDiagnosticTotal_.store(totalWork);
                    constexpr std::uint64_t kProgressInterval = 1u << 16u;
                    for (std::uint64_t node = 0; node < nodeCount; ++node) {
                        if ((node & (kProgressInterval - 1u)) == 0u &&
                            stopToken.stop_requested()) {
                            roadTypeDiagnosticRunning_.store(false);
                            return;
                        }
                        graphScanner.Read<std::uint32_t>();
                        graphScanner.Read<std::uint64_t>();
                        graphScanner.Read<double>();
                        graphScanner.Read<double>();
                        graphScanner.Read<float>();
                        graphScanner.Read<std::uint32_t>();
                        if ((node + 1u) % kProgressInterval == 0u || node + 1u == nodeCount) {
                            roadTypeDiagnosticProcessed_.store(node + 1u);
                        }
                    }
                }

                std::uint64_t scanCount = edgeCount;
                if (index && index->HasRootPayload()) {
                    scanCount = std::min<std::uint64_t>(index->rootPayloadRecordCount, edgeCount);
                    result.rootSubset = true;
                    result.comparedIndex = true;
                }

                const auto progressBase = index ? 0u : nodeCount;
                roadTypeDiagnosticTotal_.store(progressBase + scanCount);

                std::ifstream indexStream;
                std::vector<streaming::index::CHIndexRootRecord> indexRecords;
                if (result.comparedIndex) {
                    indexStream.open(index->indexPath, std::ios::binary);
                    if (!indexStream) {
                        throw std::runtime_error("could not open CHIDX root payload for Type diagnostic");
                    }
                    if (index->rootPayloadSectionOffset >
                        static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
                        throw std::runtime_error("CHIDX root payload offset exceeds stream range");
                    }
                    indexStream.seekg(
                        static_cast<std::streamoff>(index->rootPayloadSectionOffset), std::ios::beg);
                    if (!indexStream) {
                        throw std::runtime_error("could not seek CHIDX root payload for Type diagnostic");
                    }
                    indexRecords.resize(1u << 16u);
                }

                constexpr std::uint64_t kChunkEdges = 1u << 16u;
                std::uint64_t processed = 0u;
                while (processed < scanCount) {
                    if (stopToken.stop_requested()) {
                        roadTypeDiagnosticRunning_.store(false);
                        return;
                    }
                    const auto chunkCount = static_cast<std::size_t>(
                        std::min<std::uint64_t>(kChunkEdges, scanCount - processed));

                    if (result.comparedIndex) {
                        indexStream.read(
                            reinterpret_cast<char*>(indexRecords.data()),
                            static_cast<std::streamsize>(
                                chunkCount * sizeof(streaming::index::CHIndexRootRecord)));
                        if (!indexStream) {
                            throw std::runtime_error("truncated CHIDX root payload during Type diagnostic");
                        }
                    }

                    for (std::size_t local = 0; local < chunkCount; ++local) {
                        graphScanner.Read<std::uint32_t>();
                        graphScanner.Read<std::uint32_t>();
                        graphScanner.Read<float>();
                        const auto rawType = graphScanner.Read<std::int32_t>();
                        graphScanner.Read<std::int32_t>();
                        graphScanner.Read<std::int64_t>();
                        graphScanner.Read<std::int64_t>();

                        const auto sourceStyle = data::RoadStyleIndexFromType(rawType);
                        ++result.sourceHistogram[sourceStyle];

                        if (result.comparedIndex) {
                            const auto& record = indexRecords[local];
                            const auto expectedEdgeId = processed + local;
                            if (record.globalEdgeId != expectedEdgeId) {
                                ++result.recordIdMismatches;
                            }
                            if (!streaming::index::HasRootRoadStyleMetadata(
                                    record.boundsLo, record.boundsHi)) {
                                ++result.missingIndexMetadata;
                            }
                            const auto indexStyle = std::min<std::uint32_t>(
                                streaming::index::DecodeRootRoadStyle(
                                    record.boundsLo, record.boundsHi),
                                data::RoadStyleTypeCount - 1u);
                            ++result.indexHistogram[indexStyle];
                            if (indexStyle != sourceStyle) {
                                ++result.styleMismatches;
                            }
                        }
                    }

                    processed += chunkCount;
                    result.scannedEdges = processed;
                    roadTypeDiagnosticProcessed_.store(progressBase + processed);
                }
            } catch (const std::exception& error) {
                result.error = error.what();
            } catch (...) {
                result.error = "unknown error while scanning road Types";
            }

            {
                std::lock_guard lock(roadTypeDiagnosticMutex_);
                roadTypeDiagnosticResult_ = std::move(result);
            }
            roadTypeDiagnosticRunning_.store(false);
        });
}

void DebugUI::BeginFrame() const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

DebugUIActions DebugUI::Draw(
    const data::CHGraph* graph,
    gpu::filtering::RangeFilterStrategyManager& rangeFilterStrategyManager,
    gpu::unfolding::UnfoldingStrategyManager& strategyManager,
    renderer::MapCamera2D* camera,
    renderer::LODController* lodController,
    data::DatasetCatalog& datasetCatalog,
    const data::DatasetLoadSnapshot& datasetLoad,
    pipeline::ProcessingMode& processingMode,
    geometry::RefinementMode& cpuGeometryRefinement,
    float& maxScreenErrorPixels,
    const pipeline::CPUProcessingStats* cpuStats,
    const pipeline::GPUProcessingStats* gpuStats,
    const pipeline::GPURangeBenchmarkResult* gpuRangeBenchmarkResult,
    const benchmark::PipelineValidationResult* validationResult,
    bool canValidate,
    const streaming::index::CHIndexData* streamingIndex,
    const streaming::runtime::GraphPageStreamingStats* streamingStats,
    renderer::RoadStyleConfig& roadStyles) {
    DebugUIActions actions;

    ImGui::Begin("CH_MapViewer");

    ImGui::TextUnformatted("Dataset");
    ImGui::TextDisabled("%s", datasetCatalog.Directory().string().c_str());

    const auto& datasets = datasetCatalog.Entries();
    if (selectedDataset_ >= datasets.size()) {
        selectedDataset_ = 0;
    }

    if (datasets.empty()) {
        ImGui::TextUnformatted("No datasets found.");
    } else {
        if (ImGui::BeginCombo("##dataset", datasets[selectedDataset_].name.c_str())) {
            for (std::size_t i = 0; i < datasets.size(); ++i) {
                const bool selected = i == selectedDataset_;
                if (ImGui::Selectable(datasets[i].name.c_str(), selected)) {
                    selectedDataset_ = i;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (datasetLoad.Active()) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Load dataset")) {
            actions.datasetRequest = selectedDataset_;
            actions.datasetStreaming = streamDataset_;
        }
        if (datasetLoad.Active()) {
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
    }

    if (ImGui::Button("Refresh datasets")) {
        datasetCatalog.Refresh();
    }

    ImGui::Checkbox("Out-of-core GPU runtime", &streamDataset_);
    ImGui::TextDisabled(
        "Enabled: use preprocessed .chidx pages. Disabled: whole-resident CHGraph for CPU/validation.");

    if (datasetLoad.Active()) {
        const auto* phaseName = DatasetLoadPhaseName(datasetLoad.phase);
        ImGui::ProgressBar(datasetLoad.progress, ImVec2(-1.0f, 0.0f), phaseName);
        ImGui::TextDisabled("%.1f%% | Dataset loading runs on a background thread.",
                            datasetLoad.progress * 100.0f);
    } else if (datasetLoad.phase == data::DatasetLoadPhase::Failed) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Load failed: %s",
                           datasetLoad.error.c_str());
    }

    ImGui::Separator();

    if (graph || streamingIndex) {
        if (streamingIndex) {
            ImGui::Text("Nodes: %llu", static_cast<unsigned long long>(streamingIndex->nodeCount));
            ImGui::Text("Edges: %llu", static_cast<unsigned long long>(streamingIndex->edgeCount));
            ImGui::Text("Runtime pages: %zu", streamingIndex->graphPages.size());
            ImGui::TextDisabled("Out-of-core runtime: CHGraph is not whole-resident.");
        }
        if (graph) {
            ImGui::Text("Nodes: %zu", graph->NodeCount());
            ImGui::Text("Edges: %zu", graph->EdgeCount());
            ImGui::Text("Shortcuts: %zu", graph->ShortcutCount());
            ImGui::Text("Drawable ranges: %zu", graph->DrawableEdgeCount());
        }
        if (camera && lodController) {
            ImGui::Text("Physical zoom: %.2fx", camera->ZoomFactor());
            ImGui::Text("Approx. vertical world span: %.2f km",
                        camera->ViewHalfHeightMeters() * 2.0 / 1000.0);
            ImGui::Text("LOD range: %d - %d", lodController->MinLevel(), lodController->MaxLevel());

            bool automaticLOD = lodController->Automatic();
            if (ImGui::Checkbox("Automatic LOD", &automaticLOD)) {
                if (!automaticLOD) {
                    lodController->SetManualLevel(lodController->Level(camera->ZoomFactor()));
                }
                lodController->SetAutomatic(automaticLOD);
            }

            if (lodController->Automatic()) {
                ImGui::Text("LOD level: %d (continuous %.2f)",
                            lodController->Level(camera->ZoomFactor()),
                            lodController->ContinuousLevel(camera->ZoomFactor()));
            } else {
                int manualLevel = lodController->ManualLevel();
                if (ImGui::SliderInt("LOD level", &manualLevel, lodController->MinLevel(),
                                     lodController->MaxLevel())) {
                    lodController->SetManualLevel(manualLevel);
                }
            }

            if (ImGui::Button("Reset view")) {
                camera->Reset();
            }
            ImGui::TextDisabled("Mouse wheel: zoom | Left mouse: pan");
        }
    } else {
        ImGui::TextUnformatted("No CH dataset loaded.");
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Road styling", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Use road type colors/widths", &roadStyles.enabled);
        ImGui::SliderFloat("Global road width scale", &roadStyles.globalWidthScale, 0.25f, 4.0f,
                           "%.2fx");
        ImGui::SliderInt("Road type", &selectedRoadStyleType_, 0,
                         static_cast<int>(data::RoadStyleTypeCount - 1u));
        const auto typeIndex = static_cast<std::size_t>(selectedRoadStyleType_);
        ImGui::ColorEdit4("Road color", roadStyles.colors[typeIndex].data());
        ImGui::SliderFloat("Road width (px)", &roadStyles.widthsPixels[typeIndex], 0.5f, 12.0f,
                           "%.2f");
        if (ImGui::Button("Reset this road type")) {
            roadStyles.ResetType(static_cast<std::uint32_t>(selectedRoadStyleType_));
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset all road styles")) {
            roadStyles.ResetDefaults();
        }
        ImGui::TextDisabled(
            "Source .sch exposes a numeric road type. Defaults treat lower type numbers as more "
            "important/thicker; type mapping, color and width remain configurable here.");

        ImGui::SeparatorText("Road Type diagnostic");
        ImGui::TextDisabled(
            "Manual/read-only diagnostic. It never runs during preprocess or rendering and does "
            "not write .sch, .ranges, .chidx or the preprocess manifest.");

        if (!datasets.empty()) {
            if (roadTypeDiagnosticRunning_.load()) {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("Scan selected dataset Types")) {
                StartRoadTypeDiagnostic(datasets[selectedDataset_].graphPath,
                                        datasets[selectedDataset_].rangesPath);
            }
            if (roadTypeDiagnosticRunning_.load()) {
                ImGui::EndDisabled();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("%s", datasets[selectedDataset_].name.c_str());
        }

        if (roadTypeDiagnosticRunning_.load()) {
            const auto current = roadTypeDiagnosticProcessed_.load();
            const auto total = roadTypeDiagnosticTotal_.load();
            const float fraction = total == 0u
                                       ? 0.0f
                                       : static_cast<float>(
                                             static_cast<double>(current) /
                                             static_cast<double>(total));
            ImGui::ProgressBar(fraction, ImVec2(-1.0f, 0.0f), "Scanning Types...");
            ImGui::TextDisabled("%llu / %llu records",
                                static_cast<unsigned long long>(current),
                                static_cast<unsigned long long>(total));
        }

        std::optional<RoadTypeDiagnosticResult> roadTypeResult;
        {
            std::lock_guard lock(roadTypeDiagnosticMutex_);
            roadTypeResult = roadTypeDiagnosticResult_;
        }
        if (roadTypeResult) {
            if (!roadTypeResult->error.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                                   "Type diagnostic failed: %s",
                                   roadTypeResult->error.c_str());
            } else {
                ImGui::Text("Scanned edges: %llu",
                            static_cast<unsigned long long>(roadTypeResult->scannedEdges));
                if (roadTypeResult->rootSubset) {
                    ImGui::TextDisabled(
                        "A valid CHIDX was found, so this scans the drawable root prefix used by "
                        "pre-refine rendering instead of all ~1B EUR edges.");
                } else {
                    ImGui::TextDisabled(
                        "No comparable RootPayload was available; source .sch edge Types were "
                        "scanned directly.");
                }

                if (roadTypeResult->comparedIndex) {
                    ImGui::Text(
                        "CHIDX metadata missing: %llu | Type mismatches: %llu | Edge-ID mismatches: %llu",
                        static_cast<unsigned long long>(roadTypeResult->missingIndexMetadata),
                        static_cast<unsigned long long>(roadTypeResult->styleMismatches),
                        static_cast<unsigned long long>(roadTypeResult->recordIdMismatches));
                    ImGui::TextDisabled(
                        "CHIDX counts are the effective pre-refine style values the renderer sees. "
                        "A large .sch/CHIDX mismatch directly explains a one-color coarse map.");
                }

                const int columns = roadTypeResult->comparedIndex ? 4 : 3;
                if (ImGui::BeginTable("RoadTypeDiagnosticTable", columns,
                                      ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_SizingStretchProp)) {
                    ImGui::TableSetupColumn("Type/style");
                    ImGui::TableSetupColumn(".sch count");
                    if (roadTypeResult->comparedIndex) {
                        ImGui::TableSetupColumn("CHIDX effective");
                    }
                    ImGui::TableSetupColumn(".sch share");
                    ImGui::TableHeadersRow();

                    for (std::uint32_t type = 0; type < data::RoadStyleTypeCount; ++type) {
                        const auto sourceCount = roadTypeResult->sourceHistogram[type];
                        const auto indexCount = roadTypeResult->indexHistogram[type];
                        if (sourceCount == 0u && indexCount == 0u) {
                            continue;
                        }
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        if (type == data::RoadStyleFallbackType) {
                            ImGui::Text("63 (fallback)");
                        } else {
                            ImGui::Text("%u", type);
                        }
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%llu", static_cast<unsigned long long>(sourceCount));
                        int shareColumn = 2;
                        if (roadTypeResult->comparedIndex) {
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%llu", static_cast<unsigned long long>(indexCount));
                            shareColumn = 3;
                        }
                        ImGui::TableSetColumnIndex(shareColumn);
                        const double share = roadTypeResult->scannedEdges == 0u
                                                 ? 0.0
                                                 : 100.0 * static_cast<double>(sourceCount) /
                                                       static_cast<double>(
                                                           roadTypeResult->scannedEdges);
                        ImGui::Text("%.3f%%", share);
                    }
                    ImGui::EndTable();
                }
            }
        }

    }

    ImGui::TextUnformatted("Processing pipeline");

    if (streamingIndex) {
        processingMode = pipeline::ProcessingMode::GPUDriven;
        ImGui::BeginDisabled();
    }
    if (ImGui::BeginCombo("##pipeline", ProcessingModeName(processingMode))) {
        constexpr pipeline::ProcessingMode modes[] = {
            pipeline::ProcessingMode::GPUDriven,
            pipeline::ProcessingMode::CPUReference,
            pipeline::ProcessingMode::Validation,
        };
        for (const auto mode : modes) {
            const bool selected = processingMode == mode;
            if (ImGui::Selectable(ProcessingModeName(mode), selected)) {
                processingMode = mode;
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    if (streamingIndex) {
        ImGui::EndDisabled();
        ImGui::TextDisabled("Streaming runtime currently uses the GPU Driven path only.");
    }

    if (processingMode != pipeline::ProcessingMode::CPUReference) {
        ImGui::TextUnformatted("GPU filtering strategy");
        const auto rangeFilterStrategies = rangeFilterStrategyManager.Strategies();
        if (!rangeFilterStrategies.empty()) {
            const auto currentName = rangeFilterStrategyManager.Current().Name();
            if (ImGui::BeginCombo("##range-filter-strategy", currentName.data())) {
                for (std::size_t i = 0; i < rangeFilterStrategies.size(); ++i) {
                    const bool supported =
                        !streamingIndex ||
                        rangeFilterStrategies[i]->PersistentStreamingKind() !=
                            gpu::filtering::PersistentStreamingRangeFilterKind::Unsupported;
                    const bool selected = i == rangeFilterStrategyManager.CurrentIndex();
                    if (!supported) {
                        ImGui::BeginDisabled();
                    }
                    if (ImGui::Selectable(rangeFilterStrategies[i]->Name().data(), selected) &&
                        supported) {
                        rangeFilterStrategyManager.Select(i);
                    }
                    if (!supported) {
                        ImGui::EndDisabled();
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }

        ImGui::TextUnformatted("GPU geometry refinement");
        const auto strategies = strategyManager.Strategies();
        if (!strategies.empty()) {
            const auto currentName = strategyManager.Current().Name();
            if (ImGui::BeginCombo("##strategy", currentName.data())) {
                for (std::size_t i = 0; i < strategies.size(); ++i) {
                    const bool adaptiveStreamingSupported =
                        !streamingIndex ||
                        (streamingIndex->HasRootPayload() &&
                         streamingIndex->edgeSpatialBoundsCount == streamingIndex->edgeCount);
                    const bool supported =
                        strategies[i]->Mode() != geometry::RefinementMode::Adaptive ||
                        adaptiveStreamingSupported;
                    const bool selected = i == strategyManager.CurrentIndex();
                    if (!supported) {
                        ImGui::BeginDisabled();
                    }
                    if (ImGui::Selectable(strategies[i]->Name().data(), selected) && supported) {
                        strategyManager.Select(i);
                    }
                    if (!supported) {
                        ImGui::EndDisabled();
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }
        if (streamingIndex) {
            ImGui::TextDisabled(
                "Adaptive is the normal large-map refinement path: viewport/guard roots stop once their projected geometry error is below the pixel threshold.");
            ImGui::TextDisabled(
                "Full DFS remains an exhaustive correctness/stress mode and can request a very large hierarchy working set.");
            if (!streamingIndex->HasRootPayload() ||
                streamingIndex->edgeSpatialBoundsCount != streamingIndex->edgeCount) {
                ImGui::TextDisabled(
                    "Adaptive requires the current preprocessed CHIDX root payload and per-edge spatial bounds.");
            }
        }
    } else {
        ImGui::TextUnformatted("CPU geometry refinement");
        int refinement = static_cast<int>(cpuGeometryRefinement);
        constexpr const char* refinementNames[] = {
            "None (paper ranges only)",
            "Full unfold",
            "Adaptive (screen-space)",
        };
        if (ImGui::BeginCombo("##cpu-geometry-refinement", refinementNames[refinement])) {
            for (int i = 0; i < 3; ++i) {
                const bool selected = refinement == i;
                if (ImGui::Selectable(refinementNames[i], selected)) {
                    refinement = i;
                    cpuGeometryRefinement = static_cast<geometry::RefinementMode>(refinement);
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (cpuGeometryRefinement == geometry::RefinementMode::Full) {
            ImGui::TextDisabled(
                "Birth-ordered lifetime retrieval first, then full shortcut geometry unfolding.");
        } else if (cpuGeometryRefinement == geometry::RefinementMode::Adaptive) {
            ImGui::TextDisabled(
                "Birth-ordered lifetime retrieval first, then screen-space geometry refinement.");
        } else {
            ImGui::TextDisabled(
                "Paper runtime path: birth-ordered lifetime retrieval, no geometry refinement.");
        }
    }

    const bool adaptiveGPU =
        processingMode != pipeline::ProcessingMode::CPUReference &&
        strategyManager.Current().Mode() == geometry::RefinementMode::Adaptive;
    const bool adaptiveCPU =
        processingMode == pipeline::ProcessingMode::CPUReference &&
        cpuGeometryRefinement == geometry::RefinementMode::Adaptive;
    if (adaptiveGPU || adaptiveCPU) {
        ImGui::SliderFloat("Max screen error", &maxScreenErrorPixels, 1.0f, 512.0f, "%.1f px",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::TextDisabled(
            "1 px is the finest setting; larger values keep coarser shortcut geometry.");
    }

    if (gpuStats && processingMode != pipeline::ProcessingMode::CPUReference) {
        if (gpuStats->rangeCandidateEdgeCount.has_value()) {
            ImGui::Text("GPU range candidates: %zu", *gpuStats->rangeCandidateEdgeCount);
        } else {
            ImGui::TextUnformatted("GPU range candidates: GPU-managed");
        }
    }

    if (!streamingStats && oocTelemetryEnabled_) {
        StopOocTelemetry();
    }

    if (streamingStats) {
        ImGui::Separator();
        ImGui::TextUnformatted("Runtime page streaming");

        ImGui::SeparatorText("OOC telemetry log");
        bool requestedTelemetry = oocTelemetryEnabled_;
        if (ImGui::Checkbox("Enable OOC telemetry", &requestedTelemetry)) {
            if (requestedTelemetry) {
                StartOocTelemetry(*streamingStats, camera);
            } else {
                StopOocTelemetry();
            }
        }
        if (oocTelemetryEnabled_) {
            UpdateOocTelemetry(*streamingStats, camera);
            ImGui::Text("Recording: %s", oocTelemetryCurrentPath_.filename().string().c_str());
            ImGui::TextDisabled("1 s while streaming, 5 s while idle; page events are aggregated; file writes happen on a background thread in <=5 s batches.");
            ImGui::TextDisabled("Disable telemetry to remove even the small sampling/formatting overhead.");
        } else {
            ImGui::TextDisabled("OFF by default: no telemetry sampling, formatting, or log file I/O.");
            if (ImGui::Button("Delete OOC telemetry logs")) {
                DeleteOocTelemetryLogs();
            }
        }
        if (!oocTelemetryError_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                               oocTelemetryError_.c_str());
        }
        if (!oocTelemetryDeleteMessage_.empty()) {
            ImGui::TextDisabled("%s", oocTelemetryDeleteMessage_.c_str());
        }

        ImGui::SeparatorText("Out-of-core cache budgets");
        ImGui::SliderInt("CPU cache budget (MiB, 0=Auto)", &streamingRamBudgetMiB_, 0, 15360);
        ImGui::SliderInt("GPU cache budget (MiB)", &streamingGpuBudgetMiB_, 128, 2048);
        if (ImGui::Button("Apply budgets")) {
            actions.applyStreamingBudgets = true;
            actions.streamingRamBudgetMiB = streamingRamBudgetMiB_ <= 0
                                                 ? 0u
                                                 : static_cast<std::uint32_t>(streamingRamBudgetMiB_);
            actions.streamingGpuBudgetMiB =
                static_cast<std::uint32_t>(std::max(streamingGpuBudgetMiB_, 128));
        }
        ImGui::SameLine();
        if (ImGui::Button("Stress preset")) {
            streamingRamBudgetMiB_ = 384;
            streamingGpuBudgetMiB_ = 192;
            actions.applyStreamingBudgets = true;
            actions.streamingRamBudgetMiB = 384;
            actions.streamingGpuBudgetMiB = 192;
        }
        ImGui::TextDisabled(
            "CPU 0 = automatic from currently available RAM (up to 15 GiB). Applying keeps .sch/.ranges/.chidx; only runtime caches are reset.");
        ImGui::Text("Required pages: %u / %u resident", streamingStats->requiredResidentCount,
                    streamingStats->requiredPageCount);
        ImGui::Text("RAM candidates: %u / %u resident | LOD-window pages: %u | GPU-warm: %u",
                    streamingStats->desiredResidentCount, streamingStats->desiredPageCount,
                    streamingStats->lodPrefetchPageCount,
                    streamingStats->lodGpuWarmPageCount);
        ImGui::TextDisabled("Spatial guard: current floor/ceil LOD use +%u fixed cells (base cell %.2f km); far prefetched LODs use viewport only",
                            streamingStats->spatialPrefetchRadius,
                            streamingStats->spatialCellSizeMeters / 1000.0);
        ImGui::Text("LOD target / displayed: %.2f / %.2f%s",
                    streamingStats->targetLodLevelFloat,
                    streamingStats->displayLodLevelFloat,
                    streamingStats->lodTransitionPending ? " (streaming/chasing)" : "");
        ImGui::Text("LOD preload window: %u .. %u | predictive focus %.2f",
                    streamingStats->lodPrefetchMinLevel,
                    streamingStats->lodPrefetchMaxLevel,
                    streamingStats->lodPrefetchFocusLevel);
        ImGui::TextDisabled("Overlapping LOD windows reuse resident/queued/loading page IDs; only newly entering pages are requested.");
        ImGui::TextDisabled("Range-only rendering blends edge lifetimes continuously between floor/ceil LOD; no whole-level snap is required.");
        ImGui::Text("Resident / queued / loading pages: %u / %u / %u",
                    streamingStats->residentPageCount, streamingStats->queuedPageCount,
                    streamingStats->loadingPageCount);
        ImGui::Text("Refinement tiles: %u / %u resident | %u queued | %u loading",
                    streamingStats->residentRefinementBlockCount,
                    streamingStats->desiredRefinementBlockCount,
                    streamingStats->queuedRefinementBlockCount,
                    streamingStats->loadingRefinementBlockCount);
        ImGui::Text("Refinement requests last: %u / %u accepted (%u outside 7x7)",
                    streamingStats->lastSpatiallyAcceptedRefinementRequestCount,
                    streamingStats->lastRefinementRequestCount,
                    streamingStats->lastRefinementRequestCount >=
                            streamingStats->lastSpatiallyAcceptedRefinementRequestCount
                        ? streamingStats->lastRefinementRequestCount -
                              streamingStats->lastSpatiallyAcceptedRefinementRequestCount
                        : 0u);
        ImGui::Text("GPU refinement requests / spatial rejects / tile loads: %llu / %llu / %llu",
                    static_cast<unsigned long long>(streamingStats->totalRefinementRequests),
                    static_cast<unsigned long long>(
                        streamingStats->totalSpatiallyRejectedRefinementRequests),
                    static_cast<unsigned long long>(streamingStats->totalRefinementBlockLoads));
        ImGui::Text("CPU cache total / peak / hard budget: %.2f / %.2f / %.2f MiB",
                    static_cast<double>(streamingStats->totalCpuCacheBytes) / (1024.0 * 1024.0),
                    static_cast<double>(streamingStats->peakCpuCacheBytes) / (1024.0 * 1024.0),
                    static_cast<double>(streamingStats->ramBudgetBytes) / (1024.0 * 1024.0));
        ImGui::Text("Page+refinement RAM: %.2f / %.2f MiB",
                    static_cast<double>(streamingStats->residentBytes) / (1024.0 * 1024.0),
                    static_cast<double>(streamingStats->dataCacheBudgetBytes) /
                        (1024.0 * 1024.0));
        ImGui::Text("Decoded node cache: %.2f / %.2f MiB",
                    static_cast<double>(streamingStats->nodeBlockCacheBytes) / (1024.0 * 1024.0),
                    static_cast<double>(streamingStats->nodeBlockCacheBudgetBytes) /
                        (1024.0 * 1024.0));
        ImGui::Text("Node-block hits / misses / evictions: %llu / %llu / %llu",
                    static_cast<unsigned long long>(streamingStats->nodeBlockCacheHits),
                    static_cast<unsigned long long>(streamingStats->nodeBlockCacheMisses),
                    static_cast<unsigned long long>(streamingStats->nodeBlockCacheEvictions));
        ImGui::Text("Page cache hit / miss | loads / evictions: %llu / %llu | %llu / %llu",
                    static_cast<unsigned long long>(streamingStats->pageCacheHits),
                    static_cast<unsigned long long>(streamingStats->pageCacheMisses),
                    static_cast<unsigned long long>(streamingStats->totalPageLoads),
                    static_cast<unsigned long long>(streamingStats->totalEvictions));
        ImGui::Text("Unique page loads / reloads after eviction: %llu / %llu",
                    static_cast<unsigned long long>(streamingStats->uniquePageLoads),
                    static_cast<unsigned long long>(streamingStats->pageReloadsAfterEviction));
        ImGui::Text("In-flight reuse / reprioritize: %llu / %llu | stale queue / completed I/O: %llu / %llu",
                    static_cast<unsigned long long>(streamingStats->pageInFlightReuses),
                    static_cast<unsigned long long>(streamingStats->pageQueueReprioritizations),
                    static_cast<unsigned long long>(streamingStats->stalePageQueueEntriesSkipped),
                    static_cast<unsigned long long>(streamingStats->stalePageLoadsDiscarded));
        ImGui::Text("Residency grace pages: %u | Root I/O reads / batched: %llu / %llu",
                    streamingStats->recentlyDesiredResidentPageCount,
                    static_cast<unsigned long long>(streamingStats->rootIoReadOperations),
                    static_cast<unsigned long long>(streamingStats->rootIoBatchedReadOperations));
        const auto rootIoPagesPerRead = streamingStats->rootIoReadOperations == 0u
                                            ? 0.0
                                            : static_cast<double>(streamingStats->rootIoPagesRead) /
                                                  static_cast<double>(streamingStats->rootIoReadOperations);
        const auto rootIoMiB = static_cast<double>(streamingStats->rootIoBytesRead) /
                               (1024.0 * 1024.0);
        const auto rootIoMiBPerSecond = streamingStats->rootIoReadMilliseconds > 0.0
                                            ? rootIoMiB * 1000.0 /
                                                  streamingStats->rootIoReadMilliseconds
                                            : 0.0;
        ImGui::Text("Root I/O: %.2f pages/read (max %u) | %.1f MiB | %.1f MiB/s service rate",
                    rootIoPagesPerRead, streamingStats->maxRootIoBatchPages, rootIoMiB,
                    rootIoMiBPerSecond);
        ImGui::Text("Backing RAM cache hit / miss | evictions: %llu / %llu | %llu",
                    static_cast<unsigned long long>(streamingStats->refinementBlockCacheHits),
                    static_cast<unsigned long long>(streamingStats->refinementBlockCacheMisses),
                    static_cast<unsigned long long>(
                        streamingStats->totalRefinementBlockEvictions));
        ImGui::Text("Last / mean page load: %.3f / %.3f ms", streamingStats->lastPageLoadMs,
                    streamingStats->meanPageLoadMs);
        ImGui::Text("Root planner last: %.3f ms | rebuilds / cache reuse: %llu / %llu",
                    streamingStats->lastRootPlannerMs,
                    static_cast<unsigned long long>(streamingStats->rootPlannerRebuilds),
                    static_cast<unsigned long long>(streamingStats->rootPlannerCacheReuses));
        ImGui::Text("Planner candidates alive/spatial: %u / %u | path: %s",
                    streamingStats->rootPlannerAliveCandidates,
                    streamingStats->rootPlannerSpatialCandidates,
                    streamingStats->rootPlannerUsedLodFirst ? "LOD-first" : "spatial-first");
        ImGui::Text("Last / mean refinement tile load: %.3f / %.3f ms",
                    streamingStats->lastRefinementBlockLoadMs,
                    streamingStats->meanRefinementBlockLoadMs);
        ImGui::Separator();
        ImGui::TextUnformatted("Persistent GPU runtime");
        ImGui::Text("GPU allocated / hard budget: %.2f / %.2f MiB",
                    static_cast<double>(streamingStats->gpuAllocatedBytes) / (1024.0 * 1024.0),
                    static_cast<double>(streamingStats->gpuBudgetBytes) / (1024.0 * 1024.0));
        ImGui::Text("Root records required / transition / capacity: %llu / %llu / %u",
                    static_cast<unsigned long long>(streamingStats->gpuRequiredRootRecords),
                    static_cast<unsigned long long>(streamingStats->gpuTransitionRootRecords),
                    streamingStats->gpuRootRecordCapacity);
        if (!streamingStats->gpuRootWorkingSetFits) {
            ImGui::TextColored(
                ImVec4(1.0f, 0.55f, 0.20f, 1.0f),
                "Root working set exceeds GPU cache: stable partial residency (no churn).");
        }
        ImGui::Text("Root pages cached / active: %u / %u",
                    streamingStats->gpuPersistentRootPagesCached,
                    streamingStats->gpuPersistentRootPagesActive);
        ImGui::Text("Backing tiles resident: %u / %u GPU slots",
                    streamingStats->gpuPersistentBackingBlocks,
                    streamingStats->gpuPersistentBackingBlockCapacity);
        ImGui::Text("Visible roots / draw requested / capacity: %u / %u / %u",
                    streamingStats->gpuVisibleRootCount,
                    streamingStats->gpuDrawEdgeCount,
                    streamingStats->gpuDrawCapacity);
        if (streamingStats->gpuDrawOverflowCount != 0u) {
            ImGui::TextColored(ImVec4(1.0f, 0.30f, 0.20f, 1.0f),
                               "DRAW OVERFLOW: %u segments dropped this frame",
                               streamingStats->gpuDrawOverflowCount);
        }
        ImGui::Text("GPU missing backing tiles: %u | Adaptive admitted this wave: %u",
                    streamingStats->gpuMissingBlockRequests,
                    streamingStats->gpuAdaptiveBlockRequestsAdmitted);
        ImGui::Text("Refinement cache hit / miss: %u / %u | cached roots: %u",
                    streamingStats->gpuRefinementCacheHits,
                    streamingStats->gpuRefinementCacheMisses,
                    streamingStats->gpuCachedRefinedRoots);
        ImGui::Text("Cached refinement geometry: %u / %u segments | overflows: %u",
                    streamingStats->gpuRefinementGeometryUsed,
                    streamingStats->gpuRefinementGeometryCapacity,
                    streamingStats->gpuRefinementGeometryOverflows);
        ImGui::Text("Refinement geometry banks: write %u | used %u / 8 | recycles: %llu",
                    streamingStats->gpuRefinementWriteBank,
                    streamingStats->gpuRefinementBanksUsed,
                    static_cast<unsigned long long>(streamingStats->gpuRefinementBankRecycles));
        ImGui::Text("GPU root/backing evictions: %llu / %llu",
                    static_cast<unsigned long long>(streamingStats->gpuRootPageEvictions),
                    static_cast<unsigned long long>(streamingStats->gpuBackingBlockEvictions));
        ImGui::Text("Backing uploads first / re-upload: %llu / %llu | grace fallback evict: %llu",
                    static_cast<unsigned long long>(
                        streamingStats->gpuBackingBlockFirstUploads),
                    static_cast<unsigned long long>(
                        streamingStats->gpuBackingBlockReuploadsAfterEviction),
                    static_cast<unsigned long long>(
                        streamingStats->gpuBackingGraceFallbackEvictions));
        ImGui::Text("GPU cache pressure root/backing: %llu / %llu | touched blocks: %u",
                    static_cast<unsigned long long>(
                        streamingStats->gpuRootCacheAllocationFailures),
                    static_cast<unsigned long long>(
                        streamingStats->gpuBackingCacheAllocationFailures),
                    streamingStats->gpuBackingBlocksTouchedLastReadback);
        ImGui::Text("GPU filter+cull / refine / compose: %.3f / %.3f / %.3f ms",
                    streamingStats->gpuFilterCullMs, streamingStats->gpuRefinementMs,
                    streamingStats->gpuComposeMs);
        ImGui::Text("Last incremental root/backing upload: %.3f / %.3f ms",
                    streamingStats->lastGpuRootUploadMs,
                    streamingStats->lastGpuBackingUploadMs);
        ImGui::Text("Cumulative GPU upload traffic root/backing: %.2f / %.2f MiB",
                    static_cast<double>(streamingStats->gpuIncrementalRootBytesUploaded) /
                        (1024.0 * 1024.0),
                    static_cast<double>(streamingStats->gpuIncrementalBackingBytesUploaded) /
                        (1024.0 * 1024.0));
        if (streamingStats->gpuRefinementStackOverflows != 0 ||
            streamingStats->gpuRefinementHashOverflows != 0 ||
            streamingStats->gpuRefinementGeometryOverflows != 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.20f, 1.0f),
                               "Refinement overflow stack/hash/geometry: %u / %u / %u",
                               streamingStats->gpuRefinementStackOverflows,
                               streamingStats->gpuRefinementHashOverflows,
                               streamingStats->gpuRefinementGeometryOverflows);
        }
        ImGui::TextDisabled(
            "7x7 controls residency only. GPU refinement uses viewport + a small guard band and caches Full-DFS results per root.");
        ImGui::TextDisabled(
            "Current view ready measures root-page demand only; it is not geometry-refinement time.");
        ImGui::Text("Estimated source read: %.2f MiB",
                    static_cast<double>(streamingStats->estimatedSourceBytesRead) /
                        (1024.0 * 1024.0));
        if (!streamingStats->currentDemandReady) {
            ImGui::TextDisabled("Current view waiting for pages: %.2f ms",
                                streamingStats->currentDemandWaitMs);
        } else {
            ImGui::TextDisabled("Current view ready: %.2f ms", streamingStats->lastDemandReadyMs);
        }
        if (!streamingStats->lastError.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Streaming error: %s",
                               streamingStats->lastError.c_str());
        }
    }

    if (gpuStats && processingMode != pipeline::ProcessingMode::CPUReference && !streamingIndex) {
        ImGui::Separator();
        ImGui::TextUnformatted("GPU range filter benchmark");
        if (ImGui::Button("Run GPU Range Benchmark")) {
            actions.runGpuRangeBenchmark = true;
        }
        ImGui::TextDisabled(
            "20 warm-up + 200 measured executions, synchronized in batches of 10.");
        ImGui::TextDisabled(
            "Measures range filtering only; geometry refinement and drawing are excluded.");

        if (gpuRangeBenchmarkResult) {
            ImGui::Text("Filter: %s | LOD: %d", gpuRangeBenchmarkResult->filterStrategy.c_str(),
                        gpuRangeBenchmarkResult->lodLevel);
            if (gpuRangeBenchmarkResult->candidateEdgeCount) {
                ImGui::Text("Candidate edges: %zu", *gpuRangeBenchmarkResult->candidateEdgeCount);
            } else {
                ImGui::TextDisabled("Candidate edges: unavailable");
            }
            ImGui::Text("Alive edges: %zu", gpuRangeBenchmarkResult->aliveEdgeCount);
            ImGui::Text("Mean: %.6f ms", gpuRangeBenchmarkResult->meanMs);
            ImGui::Text("Median: %.6f ms", gpuRangeBenchmarkResult->medianMs);
            ImGui::Text("Min / Max: %.6f / %.6f ms", gpuRangeBenchmarkResult->minMs,
                        gpuRangeBenchmarkResult->maxMs);
            ImGui::Text("P95 / P99: %.6f / %.6f ms", gpuRangeBenchmarkResult->p95Ms,
                        gpuRangeBenchmarkResult->p99Ms);
            ImGui::Text("StdDev: %.6f ms", gpuRangeBenchmarkResult->stdDevMs);
            ImGui::TextDisabled(
                "Statistics are computed from 20 synchronized batch-average samples.");
        }
    }

    if (cpuStats && processingMode != pipeline::ProcessingMode::GPUDriven) {
        ImGui::Text("CPU scanned edges: %zu", cpuStats->rangeScannedEdgeCount);
        ImGui::Text("CPU alive edges: %zu", cpuStats->aliveEdgeCount);
        ImGui::Text("CPU output edges: %zu", cpuStats->outputEdgeCount);
        ImGui::Text("CPU range filter (last): %.6f ms", cpuStats->rangeFilterMs);
        if (cpuStats->geometryRefinementMs > 0.0) {
            ImGui::Text("CPU geometry refinement (last): %.6f ms", cpuStats->geometryRefinementMs);
        } else {
            ImGui::TextDisabled("CPU geometry refinement: none (ranges only)");
        }
        if (cpuStats->cacheHit) {
            ImGui::TextDisabled("CPU result reused for unchanged LOD.");
        }
    }

    if (processingMode == pipeline::ProcessingMode::Validation) {
        ImGui::Separator();
        ImGui::TextUnformatted("Validation view");

        int viewMode = splitScreenValidation_ ? 0 : 1;
        ImGui::RadioButton("Split screen", &viewMode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Overlay", &viewMode, 1);
        splitScreenValidation_ = viewMode == 0;

        if (splitScreenValidation_) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.20f, 1.0f), "CPU = left");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.20f, 0.75f, 1.0f, 1.0f), "GPU = right");
            ImGui::TextDisabled("Both views use the same camera center and zoom.");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.20f, 1.0f), "CPU");
            ImGui::SameLine();
            ImGui::TextUnformatted("+");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.20f, 0.75f, 1.0f, 1.0f), "GPU overlay");
        }

        if (strategyManager.Current().Mode() == geometry::RefinementMode::Full) {
            ImGui::TextDisabled(
                "CPU full geometry unfolding mirrors the selected GPU refinement for validation.");
        } else if (strategyManager.Current().Mode() == geometry::RefinementMode::Adaptive) {
            ImGui::TextDisabled(
                "CPU adaptive refinement uses the same precomputed error and pixel threshold.");
        } else {
            ImGui::TextDisabled(
                "CPU and GPU both render the topology-safe lifetime edge set directly.");
        }

        if (!canValidate) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Validate current result")) {
            actions.validateCurrentResult = true;
        }
        if (!canValidate) {
            ImGui::EndDisabled();
        }

        if (validationResult) {
            ImGui::Text("Exact validation: %s", validationResult->Passed() ? "PASS" : "FAIL");
            ImGui::Text("Range filter: %s", validationResult->rangeFilter.Passed() ? "PASS" : "FAIL");
            ImGui::Text("  CPU/GPU edges: %zu / %zu", validationResult->rangeFilter.cpuEdgeCount,
                        validationResult->rangeFilter.gpuEdgeCount);
            ImGui::Text("  Missing / unexpected: %zu / %zu",
                        validationResult->rangeFilter.missingEdgeCount,
                        validationResult->rangeFilter.unexpectedEdgeCount);
            ImGui::Text("Final output: %s",
                        validationResult->finalOutput.Passed() ? "PASS" : "FAIL");
            ImGui::Text("  CPU/GPU edges: %zu / %zu", validationResult->finalOutput.cpuEdgeCount,
                        validationResult->finalOutput.gpuEdgeCount);
            ImGui::Text("  Missing / unexpected: %zu / %zu",
                        validationResult->finalOutput.missingEdgeCount,
                        validationResult->finalOutput.unexpectedEdgeCount);
            ImGui::Text("  CPU/GPU duplicate occurrences: %zu / %zu",
                        validationResult->finalOutput.cpuDuplicateCount,
                        validationResult->finalOutput.gpuDuplicateCount);
        } else {
            ImGui::TextDisabled("Exact comparison runs only when requested and reads GPU output back to CPU.");
        }
    }

    ImGui::End();
    return actions;
}

void DebugUI::EndFrame() const {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool DebugUI::WantsMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

} // namespace chmv::ui
