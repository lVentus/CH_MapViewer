#pragma once

#include "geometry/GeometryRefinement.h"
#include "renderer/RoadStyle.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace chmv::benchmark {
struct PipelineValidationResult;
}

namespace chmv::core {
class Window;
}

namespace chmv::data {
class CHGraph;
class DatasetCatalog;
struct DatasetLoadSnapshot;
}

namespace chmv::gpu::filtering {
class RangeFilterStrategyManager;
}

namespace chmv::gpu::unfolding {
class UnfoldingStrategyManager;
}

namespace chmv::pipeline {
enum class ProcessingMode;
struct CPUProcessingStats;
struct GPUProcessingStats;
struct GPURangeBenchmarkResult;
}

namespace chmv::renderer {
class LODController;
class MapCamera2D;
}

namespace chmv::streaming::index {
struct CHIndexData;
}

namespace chmv::streaming::runtime {
struct GraphPageStreamingStats;
}

namespace chmv::ui {

struct DebugUIActions {
    std::optional<std::size_t> datasetRequest;
    bool datasetStreaming = true;
    bool validateCurrentResult = false;
    bool runGpuRangeBenchmark = false;
    bool applyStreamingBudgets = false;
    std::uint32_t streamingRamBudgetMiB = 0;
    std::uint32_t streamingGpuBudgetMiB = 0;
};

class DebugUI {
public:
    explicit DebugUI(const core::Window& window);
    ~DebugUI();

    DebugUI(const DebugUI&) = delete;
    DebugUI& operator=(const DebugUI&) = delete;

    void BeginFrame() const;
    [[nodiscard]] DebugUIActions Draw(
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
        renderer::RoadStyleConfig& roadStyles);
    void EndFrame() const;
    [[nodiscard]] bool WantsMouse() const;
    [[nodiscard]] bool SplitScreenValidation() const { return splitScreenValidation_; }

private:
    struct RoadTypeDiagnosticResult {
        std::filesystem::path graphPath;
        std::array<std::uint64_t, data::RoadStyleTypeCount> sourceHistogram{};
        std::array<std::uint64_t, data::RoadStyleTypeCount> indexHistogram{};
        std::uint64_t scannedEdges = 0;
        std::uint64_t missingIndexMetadata = 0;
        std::uint64_t styleMismatches = 0;
        std::uint64_t recordIdMismatches = 0;
        bool rootSubset = false;
        bool comparedIndex = false;
        std::string error;
    };

    void StartRoadTypeDiagnostic(std::filesystem::path graphPath,
                                 std::filesystem::path rangesPath);


    struct OocTelemetryCounters {
        std::uint64_t pageCacheHits = 0;
        std::uint64_t pageCacheMisses = 0;
        std::uint64_t uniquePageLoads = 0;
        std::uint64_t pageReloadsAfterEviction = 0;
        std::uint64_t pageInFlightReuses = 0;
        std::uint64_t pageQueueReprioritizations = 0;
        std::uint64_t stalePageQueueEntriesSkipped = 0;
        std::uint64_t stalePageLoadsDiscarded = 0;
        std::uint64_t rootIoReadOperations = 0;
        std::uint64_t rootIoBatchedReadOperations = 0;
        std::uint64_t rootIoPagesRead = 0;
        std::uint64_t rootIoBytesRead = 0;
        double rootIoReadMilliseconds = 0.0;
        std::uint64_t totalEvictions = 0;
        std::uint64_t totalRefinementBlockLoads = 0;
        std::uint64_t totalRefinementBlockEvictions = 0;
        std::uint64_t nodeBlockCacheHits = 0;
        std::uint64_t nodeBlockCacheMisses = 0;
        std::uint64_t nodeBlockCacheEvictions = 0;
        std::uint64_t rootPlannerRebuilds = 0;
        std::uint64_t rootPlannerCacheReuses = 0;
        std::uint64_t gpuIncrementalRootBytesUploaded = 0;
        std::uint64_t gpuIncrementalBackingBytesUploaded = 0;
        std::uint64_t gpuRootPageEvictions = 0;
        std::uint64_t gpuBackingBlockEvictions = 0;
        std::uint64_t gpuBackingBlockFirstUploads = 0;
        std::uint64_t gpuBackingBlockReuploadsAfterEviction = 0;
        std::uint64_t gpuBackingGraceFallbackEvictions = 0;
        std::uint64_t gpuRootCacheAllocationFailures = 0;
        std::uint64_t gpuBackingCacheAllocationFailures = 0;
    };

    bool StartOocTelemetry(const streaming::runtime::GraphPageStreamingStats& stats,
                           const renderer::MapCamera2D* camera);
    void StopOocTelemetry();
    void UpdateOocTelemetry(const streaming::runtime::GraphPageStreamingStats& stats,
                            const renderer::MapCamera2D* camera);
    void QueueOocTelemetryRecord(std::string payload,
                                 std::chrono::steady_clock::time_point now);
    void OocTelemetryWriterMain(std::stop_token stopToken, std::filesystem::path path);
    void DeleteOocTelemetryLogs();
    [[nodiscard]] static OocTelemetryCounters CaptureOocTelemetryCounters(
        const streaming::runtime::GraphPageStreamingStats& stats);

    std::size_t selectedDataset_ = 0;
    bool streamDataset_ = true;
    bool splitScreenValidation_ = true;
    int streamingRamBudgetMiB_ = 0;
    int streamingGpuBudgetMiB_ = 384;
    int selectedRoadStyleType_ = 0;

    std::jthread roadTypeDiagnosticWorker_;
    std::atomic<bool> roadTypeDiagnosticRunning_{false};
    std::atomic<std::uint64_t> roadTypeDiagnosticProcessed_{0};
    std::atomic<std::uint64_t> roadTypeDiagnosticTotal_{0};
    std::mutex roadTypeDiagnosticMutex_;
    std::optional<RoadTypeDiagnosticResult> roadTypeDiagnosticResult_;

    bool oocTelemetryEnabled_ = false;
    std::filesystem::path oocTelemetryDirectory_;
    std::filesystem::path oocTelemetryCurrentPath_;
    std::string oocTelemetryError_;
    std::string oocTelemetryDeleteMessage_;
    std::jthread oocTelemetryWriter_;
    std::mutex oocTelemetryWriterMutex_;
    std::condition_variable oocTelemetryWriterCondition_;
    std::deque<std::string> oocTelemetryPendingLines_;
    std::size_t oocTelemetryPendingBytes_ = 0;
    std::chrono::steady_clock::time_point oocTelemetryStartTime_{};
    std::chrono::steady_clock::time_point oocTelemetryLastSampleTime_{};
    std::chrono::steady_clock::time_point oocTelemetryLastReloadEventTime_{};
    std::optional<std::chrono::steady_clock::time_point> oocTelemetryBurstIdleSince_;
    std::chrono::steady_clock::time_point oocTelemetryBurstStartTime_{};
    OocTelemetryCounters oocTelemetryLastSampleCounters_{};
    OocTelemetryCounters oocTelemetryBurstStartCounters_{};
    std::uint64_t oocTelemetryLastObservedReloads_ = 0;
    std::uint64_t oocTelemetryPendingReloadEvents_ = 0;
    bool oocTelemetryBurstActive_ = false;
    bool oocTelemetryBacklogHigh_ = false;

};

} // namespace chmv::ui
