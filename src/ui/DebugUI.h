#pragma once

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
}

namespace chmv::gpu::unfolding {
class UnfoldingStrategyManager;
}

namespace chmv::pipeline {
enum class ProcessingMode;
struct CPUProcessingStats;
struct GPUProcessingStats;
}

namespace chmv::renderer {
class LODController;
class MapCamera2D;
}

namespace chmv::ui {

struct DebugUIActions {
    std::optional<std::size_t> datasetRequest;
    bool validateCurrentResult = false;
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
        gpu::unfolding::UnfoldingStrategyManager& strategyManager,
        renderer::MapCamera2D* camera,
        renderer::LODController* lodController,
        data::DatasetCatalog& datasetCatalog,
        pipeline::ProcessingMode& processingMode,
        const pipeline::CPUProcessingStats* cpuStats,
        const pipeline::GPUProcessingStats* gpuStats,
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
