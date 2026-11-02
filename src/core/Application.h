#pragma once

#include "benchmark/LodPageZoomSweep.h"
#include "core/AdaptiveTolerance.h"
#include "core/Window.h"
#include "cpu/CpuPasses.h"
#include "cpu/CpuRuntime.h"
#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "gpu/OpenGLContext.h"
#include "gpu/filtering/RangeFilterStrategyManager.h"
#include "gpu/lod/LodPageRuntime.h"
#include "gpu/lod/SegmentRenderer.h"
#include "gpu/unfolding/UnfoldingStrategyManager.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "renderer/Renderer.h"
#include "renderer/RoadStyle.h"
#include "ui/DebugUI.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace chmv::core {

class Application {
public:
    explicit Application(const std::filesystem::path& assetDirectory);

    void LoadDataset(const data::DatasetEntry& dataset);
    int Run();

private:
    void InstallDataset(streaming::lod::LodPageDataset dataset);
    void UpdateCameraInput(float deltaSeconds);
    void ReloadRoadStyles();
    void CheckRoadStyles(float deltaSeconds);
    void ForgetResultsOnStrategyChange();
    [[nodiscard]] cpu::CpuStrategy CurrentCpuStrategy() const;
    // Runs the CPU method for a view of this size; true if its result is new. countSteps also
    // counts the segments of every tolerance step (page method).
    bool UpdateCpu(int width, int height, float tolerance, bool countSteps);
    // What the last frames cost and drew, for the automatic tolerance.
    [[nodiscard]] FrameCosts CurrentCosts(bool gpuOn, bool cpuOn) const;
    // GPU against CPU inside the view, once both show the same, complete, still view.
    void CompareGpuAndCpu();

    Window window_;
    gpu::OpenGLContext context_;
    renderer::Renderer renderer_;
    gpu::lod::SegmentRenderer segmentRenderer_;
    AdaptiveTolerance tolerance_;
    renderer::RoadStyleConfig roadStyleConfig_;
    gpu::filtering::RangeFilterStrategyManager rangeFilterStrategyManager_;
    gpu::unfolding::UnfoldingStrategyManager strategyManager_;
    renderer::MapCamera2D camera_;
    renderer::LODController lodController_;
    ui::DebugUI debugUI_;
    data::DatasetCatalog datasetCatalog_;
    data::AsyncDatasetLoader datasetLoader_;
    std::filesystem::path shaderDirectory_;
    // Road styles come from a config file next to the shaders, reloaded when it changes.
    std::filesystem::path roadStylePath_;
    std::filesystem::file_time_type roadStyleWriteTime_{};
    float roadStyleCheckSeconds_ = 0.0f;
    std::string roadStyleStatus_;
    bool roadStyleError_ = false;
    std::unique_ptr<gpu::lod::LodPageRuntime> lodPages_;
    std::unique_ptr<cpu::CpuRuntime> cpuRuntime_;
    // CPU counterparts of the registered GPU filters and expansions (by index).
    std::vector<std::optional<cpu::CpuFilter>> cpuFilters_;
    std::vector<std::optional<cpu::CpuExpansion>> cpuExpansions_;
    cpu::ProcessingMode processingMode_ = cpu::ProcessingMode::Gpu;
    cpu::CpuMethod cpuMethod_ = cpu::CpuMethod::Pages;
    int cpuThreads_ = 1;
    // View rectangle of the current CPU result, and the error code of one pixel there.
    std::array<double, 4> cpuView_{};
    std::uint32_t cpuPixelCode_ = 0;
    // Why the current CPU result is not drawn (empty when it is).
    std::string cpuDrawError_;
    std::optional<cpu::CpuGpuComparison> comparison_;
    // GPU passes and CPU result the comparison was made from.
    std::pair<std::uint64_t, std::uint64_t> comparedKey_{};
    std::optional<cpu::CpuBenchmarkResult> cpuBenchmark_;
    benchmark::LodPageZoomSweep lodPageSweep_;
    std::optional<gpu::lod::LodPageValidation> validation_;
    std::optional<gpu::lod::LodPageBenchmarkResult> benchmark_;
    // Strategies the validation and benchmark results above were made with.
    std::size_t resultFilter_ = 0;
    std::size_t resultUnfold_ = 0;
    float maxScreenErrorPixels_ = 1.0f;
    bool preloadAll_ = false;
    // GPU page pool budget; 0 = automatic.
    std::uint64_t gpuBudgetBytes_ = 0;
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
    bool wasPanning_ = false;
};

} // namespace chmv::core
