#pragma once

#include "benchmark/CorrectnessValidator.h"
#include "core/Window.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHGraph.h"
#include "gpu/OpenGLContext.h"
#include "gpu/unfolding/UnfoldingStrategyManager.h"
#include "pipeline/CPUReferencePipeline.h"
#include "pipeline/GPUDrivenPipeline.h"
#include "pipeline/ProcessingMode.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "renderer/Renderer.h"
#include "renderer/RoadRenderer.h"
#include "ui/DebugUI.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace chmv::core {

class Application {
public:
    explicit Application(const std::filesystem::path& assetDirectory);

    void LoadDataset(const std::filesystem::path& graphPath,
                     const std::filesystem::path& rangesPath);
    int Run();

private:
    void UpdateCameraInput();
    void ClearValidation();

    Window window_;
    gpu::OpenGLContext context_;
    renderer::Renderer renderer_;
    renderer::RoadRenderer roadRenderer_;
    pipeline::GPUDrivenPipeline gpuPipeline_;
    pipeline::CPUReferencePipeline cpuPipeline_;
    renderer::MapCamera2D camera_;
    renderer::LODController lodController_;
    gpu::unfolding::UnfoldingStrategyManager strategyManager_;
    ui::DebugUI debugUI_;
    data::DatasetCatalog datasetCatalog_;
    std::unique_ptr<data::CHGraph> graph_;
    pipeline::ProcessingMode processingMode_ = pipeline::ProcessingMode::GPUDriven;
    std::optional<benchmark::PipelineValidationResult> validationResult_;
    std::vector<std::uint32_t> gpuAliveReadback_;
    std::vector<std::uint32_t> gpuOutputReadback_;
    std::int32_t validationLod_ = 0;
    std::size_t validationStrategy_ = 0;
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
    bool wasPanning_ = false;
};

} // namespace chmv::core
