#pragma once

#include "geometry/GeometryRefinement.h"
#include "renderer/RoadStyle.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
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
};

} // namespace chmv::ui
