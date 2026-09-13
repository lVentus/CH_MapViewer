#pragma once

#include "geometry/GeometryRefinement.h"

#include <cstddef>
#include <optional>

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

namespace chmv::ui {

struct DebugUIActions {
    std::optional<std::size_t> datasetRequest;
    bool validateCurrentResult = false;
    bool runGpuRangeBenchmark = false;
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
        bool canValidate);
    void EndFrame() const;
    [[nodiscard]] bool WantsMouse() const;
    [[nodiscard]] bool SplitScreenValidation() const { return splitScreenValidation_; }

private:
    std::size_t selectedDataset_ = 0;
    bool splitScreenValidation_ = true;
};

} // namespace chmv::ui
