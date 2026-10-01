#pragma once

#include "benchmark/CorrectnessValidator.h"
#include "core/Window.h"
#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHGraph.h"
#include "gpu/OpenGLContext.h"
#include "gpu/streaming/PersistentStreamingPipeline.h"
#include "gpu/streaming/RefinementRequestCollector.h"
#include "gpu/filtering/RangeFilterStrategyManager.h"
#include "gpu/unfolding/UnfoldingStrategyManager.h"
#include "geometry/GeometryRefinement.h"
#include "pipeline/CPUReferencePipeline.h"
#include "pipeline/GPUDrivenPipeline.h"
#include "pipeline/ProcessingMode.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "renderer/Renderer.h"
#include "renderer/RoadRenderer.h"
#include "streaming/runtime/GraphPageStreamer.h"
#include "ui/DebugUI.h"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace chmv::core {

class Application {
public:
    explicit Application(const std::filesystem::path& assetDirectory);

    void LoadDataset(const std::filesystem::path& graphPath,
                     const std::filesystem::path& rangesPath,
                     bool streaming = true);
    int Run();

private:
    void InstallDataset(data::CHGraph graph);
    void InstallStreamingDataset(data::StreamingDataset dataset);
    void UpdateStreamingRuntime(int framebufferWidth, int framebufferHeight,
                                float targetLodLevel, float deltaSeconds);
    [[nodiscard]] streaming::runtime::StreamingViewRequest BuildStreamingView(
        int framebufferWidth, int framebufferHeight, float displayLodLevel,
        std::uint32_t prefetchMinLod, std::uint32_t prefetchMaxLod,
        float prefetchFocusLod) const;
    [[nodiscard]] streaming::runtime::StreamingSpatialWindow BuildRefinementGuardWindow() const;
    void ApplyStreamingBudgets(std::uint32_t ramBudgetMiB, std::uint32_t gpuBudgetMiB);
    void UpdateCameraInput(float deltaSeconds);
    void ClearValidation();

    Window window_;
    gpu::OpenGLContext context_;
    renderer::Renderer renderer_;
    renderer::RoadRenderer roadRenderer_;
    renderer::RoadStyleConfig roadStyleConfig_;
    pipeline::GPUDrivenPipeline gpuPipeline_;
    gpu::streaming::RefinementRequestCollector refinementRequestCollector_;
    gpu::streaming::PersistentStreamingPipeline persistentStreamingPipeline_;
    gpu::filtering::RangeFilterStrategyManager rangeFilterStrategyManager_;
    pipeline::CPUReferencePipeline cpuPipeline_;
    renderer::MapCamera2D camera_;
    renderer::LODController lodController_;
    gpu::unfolding::UnfoldingStrategyManager strategyManager_;
    ui::DebugUI debugUI_;
    data::DatasetCatalog datasetCatalog_;
    data::AsyncDatasetLoader datasetLoader_;
    std::unique_ptr<data::CHGraph> graph_;
    std::unique_ptr<streaming::runtime::GraphPageStreamer> pageStreamer_;
    renderer::RoadDrawData cpuDrawData_;
    pipeline::ProcessingMode processingMode_ = pipeline::ProcessingMode::GPUDriven;
    geometry::RefinementMode cpuGeometryRefinement_ = geometry::RefinementMode::None;
    float maxScreenErrorPixels_ = 1.0f;
    // CPU streaming cache is automatic by default. It is sized from currently available
    // physical memory and periodically adjusted with hysteresis; manual Apply budgets disables auto.
    bool streamingRamBudgetAuto_ = true;
    std::uint32_t streamingRamBudgetMiB_ = 0;
    float streamingMemoryBudgetPollSeconds_ = 0.0f;
    std::uint32_t streamingGpuBudgetMiB_ = 384;
    // The camera may cross an automatic LOD threshold before the next root-page set has arrived.
    // Keep drawing the last complete LOD and atomically commit the new one once its strict
    // viewport pages are GPU-resident; this prevents partial-road zoom transitions.
    std::int32_t streamingDisplayLod_ = -1;
    float streamingDisplayLodFloat_ = -1.0f;
    float streamingLastCameraLodFloat_ = -1.0f;
    float streamingLodVelocity_ = 0.0f;
    // Runtime-calibrated upload throughput. These are bytes/ms EMAs, not fixed page/block caps.
    double streamingRootUploadBytesPerMs_ = 0.0;
    double streamingBackingUploadBytesPerMs_ = 0.0;
    std::uint32_t streamingPrefetchMinLod_ = 0;
    std::uint32_t streamingPrefetchMaxLod_ = 0;
    float streamingPrefetchFocusLod_ = 0.0f;
    bool streamingLodTransitionPending_ = false;
    // Streaming demand is coalesced independently from the render loop. This prevents a smooth
    // inertial zoom from rebuilding/querying the root index at 60-240 Hz.
    float streamingPlanAccumulatorSeconds_ = 0.0f;
    bool streamingPlannerPrimed_ = false;
    // Planner work is fully asynchronous. The render thread submits latest-wins requests and
    // consumes only a completed plan; camera motion never calls QueryPages directly.
    std::uint64_t streamingLatestSubmittedViewSequence_ = 0;
    std::uint64_t streamingLastConsumedViewSequence_ = 0;
    std::uint64_t streamingCommittedViewSequence_ = 0;
    float streamingLatestSubmittedLodFloat_ = -1.0f;
    float streamingLatestSubmittedRefinementPixelScale_ = 0.0f;
    float streamingStagedLodFloat_ = -1.0f;
    float streamingStagedRefinementPixelScale_ = 0.0f;
    float streamingCommittedRefinementPixelScale_ = 0.0f;
    streaming::runtime::StreamingSpatialWindow streamingStagedViewportWindow_{};
    streaming::runtime::StreamingSpatialWindow streamingCommittedViewportWindow_{};
    bool streamingHasStagedViewportWindow_ = false;
    bool streamingHasCommittedViewportWindow_ = false;
    std::vector<std::uint32_t> streamingCommittedRequiredPages_;
    // Materialized only on the low-frequency root-planner tick. Keeping these vectors/sets avoids
    // copying + sorting four unordered page sets and rebuilding membership hashes every render
    // frame while workers are merely filling the same demand.
    std::vector<std::uint32_t> streamingPlannedDesiredPages_;
    std::vector<std::uint32_t> streamingPlannedRequiredPages_;
    std::vector<std::uint32_t> streamingPlannedLodPrefetchPages_;
    std::vector<std::uint32_t> streamingPlannedGpuWarmPages_;
    std::vector<std::uint32_t> streamingRootUploadOrder_;
    std::unordered_set<std::uint32_t> streamingRequiredPageSet_;
    std::unordered_set<std::uint32_t> streamingLodPrefetchPageSet_;
    // Last complete same-LOD root-page set actually dispatched to the GPU. During a zoom LOD
    // transition this remains active until the target strict viewport set is resident, so target
    // pages can warm silently without appearing a few pages at a time.
    std::vector<std::uint32_t> streamingCommittedActivePages_;
    geometry::RefinementMode lastStreamingRefinementMode_ = geometry::RefinementMode::None;
    std::optional<benchmark::PipelineValidationResult> validationResult_;
    std::optional<pipeline::GPURangeBenchmarkResult> gpuRangeBenchmarkResult_;
    std::vector<std::uint32_t> gpuAliveReadback_;
    std::vector<std::uint32_t> gpuOutputReadback_;
    std::int32_t validationLod_ = 0;
    std::size_t validationStrategy_ = 0;
    std::size_t validationRangeFilterStrategy_ = 0;
    geometry::RefinementParameters validationRefinement_{};
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
    bool wasPanning_ = false;
};

} // namespace chmv::core
