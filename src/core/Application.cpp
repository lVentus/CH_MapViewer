#include "core/Application.h"

#include "benchmark/CorrectnessValidator.h"
#include "gpu/filtering/BirthOrderedRangeFilterStrategy.h"
#include "gpu/filtering/FullScanRangeFilterStrategy.h"
#include "gpu/unfolding/AdaptiveDFSUnfoldingStrategy.h"
#include "gpu/unfolding/IterativeDFSUnfoldingStrategy.h"
#include "gpu/unfolding/NoUnfoldingStrategy.h"

#include <array>
#include <chrono>
#include <iostream>
#include <utility>

namespace chmv::core {
namespace {

constexpr std::array<float, 4> kDefaultRoadColor{0.82f, 0.84f, 0.87f, 1.0f};
constexpr std::array<float, 4> kCPURoadColor{1.0f, 0.55f, 0.20f, 1.0f};
constexpr std::array<float, 4> kGPURoadColor{0.20f, 0.75f, 1.0f, 1.0f};
constexpr std::array<float, 4> kCPUOverlayColor{1.0f, 0.55f, 0.20f, 0.70f};
constexpr std::array<float, 4> kGPUOverlayColor{0.20f, 0.75f, 1.0f, 0.70f};

renderer::RoadDrawData ToRoadDrawData(const pipeline::GPUProcessingResult& result) {
    return {
        result.edgeIdBuffer,
        result.drawCommandBuffer,
    };
}

} // namespace

Application::Application(const std::filesystem::path& assetDirectory)
    : window_("CH_MapViewer"),
      context_(window_),
      roadRenderer_(assetDirectory / "shaders"),
      debugUI_(window_),
      datasetCatalog_(assetDirectory / "data") {
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::BirthOrderedRangeFilterStrategy>(
            assetDirectory / "shaders"));
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::FullScanRangeFilterStrategy>(assetDirectory / "shaders"));

    strategyManager_.Register(std::make_unique<gpu::unfolding::NoUnfoldingStrategy>());
    strategyManager_.Register(
        std::make_unique<gpu::unfolding::IterativeDFSUnfoldingStrategy>(assetDirectory / "shaders"));
    strategyManager_.Register(
        std::make_unique<gpu::unfolding::AdaptiveDFSUnfoldingStrategy>(assetDirectory / "shaders"));
}

void Application::LoadDataset(const std::filesystem::path& graphPath,
                              const std::filesystem::path& rangesPath) {
    if (datasetLoader_.Start(graphPath, rangesPath)) {
        std::cout << "Loading " << graphPath << " in background\n";
    }
}

void Application::InstallDataset(data::CHGraph graph) {
    std::cout << "Loaded " << graph.NodeCount() << " nodes and " << graph.EdgeCount()
              << " edges\n";
    graph_ = std::make_unique<data::CHGraph>(std::move(graph));
    roadRenderer_.SetGraph(*graph_);
    gpuPipeline_.SetGraph(roadRenderer_.EdgeCount(),
                          static_cast<std::uint32_t>(graph_->DrawableEdgeCount()));
    rangeFilterStrategyManager_.SetGraph(*graph_);
    cpuPipeline_.SetGraph(*graph_);
    cpuDrawData_ = {};
    lodController_.SetLevelRange(roadRenderer_.MinLevel(), roadRenderer_.MaxLevel());
    lodController_.SetManualLevel(roadRenderer_.MaxLevel());
    camera_.Reset();
    ClearValidation();
}

int Application::Run() {
    auto previousFrameTime = std::chrono::steady_clock::now();

    while (!window_.ShouldClose()) {
        const auto currentFrameTime = std::chrono::steady_clock::now();
        const auto deltaSeconds =
            std::chrono::duration<float>(currentFrameTime - previousFrameTime).count();
        previousFrameTime = currentFrameTime;
        window_.PollEvents();
        debugUI_.BeginFrame();
        UpdateCameraInput(deltaSeconds);

        int width = 0;
        int height = 0;
        window_.FramebufferSize(width, height);

        renderer_.BeginFrame(width, height);

        std::optional<pipeline::GPUProcessingResult> gpuResult;
        std::optional<pipeline::CPUProcessingResult> cpuResult;
        std::int32_t lodLevel = 0;
        geometry::RefinementParameters gpuRefinement;

        if (graph_) {
            lodLevel = lodController_.Level(camera_.ZoomFactor());
            const auto screenPixelScale =
                camera_.ViewScale() * static_cast<float>(height) * 0.5f;
            gpuRefinement = {
                strategyManager_.Current().Mode(),
                screenPixelScale,
                maxScreenErrorPixels_,
            };
            const geometry::RefinementParameters cpuRefinement{
                cpuGeometryRefinement_,
                screenPixelScale,
                maxScreenErrorPixels_,
            };
            const auto strategyIndex = strategyManager_.CurrentIndex();
            const auto rangeFilterStrategyIndex = rangeFilterStrategyManager_.CurrentIndex();
            if (validationResult_ &&
                (validationLod_ != lodLevel || validationStrategy_ != strategyIndex ||
                 validationRangeFilterStrategy_ != rangeFilterStrategyIndex ||
                 validationRefinement_ != gpuRefinement)) {
                ClearValidation();
            }

            switch (processingMode_) {
            case pipeline::ProcessingMode::GPUDriven:
                gpuResult = gpuPipeline_.Process(roadRenderer_.EdgeBufferId(),
                                                 roadRenderer_.EdgeCount(), lodLevel,
                                                 rangeFilterStrategyManager_.Current(),
                                                 strategyManager_.Current(), gpuRefinement);
                roadRenderer_.Draw(camera_, width, height, ToRoadDrawData(*gpuResult), 1.0f, 0.0f,
                                   kDefaultRoadColor);
                break;

            case pipeline::ProcessingMode::CPUReference:
                cpuResult = cpuPipeline_.Process(*graph_, lodLevel, cpuRefinement);
                if (!cpuResult->stats.cacheHit || cpuDrawData_.edgeIdBuffer == 0) {
                    cpuDrawData_ = roadRenderer_.UploadEdgeIds(cpuResult->edgeIds);
                }
                roadRenderer_.Draw(camera_, width, height, cpuDrawData_, 1.0f, 0.0f,
                                   kDefaultRoadColor);
                break;

            case pipeline::ProcessingMode::Validation:
                cpuResult = cpuPipeline_.Process(*graph_, lodLevel, gpuRefinement);
                gpuResult = gpuPipeline_.Process(roadRenderer_.EdgeBufferId(),
                                                 roadRenderer_.EdgeCount(), lodLevel,
                                                 rangeFilterStrategyManager_.Current(),
                                                 strategyManager_.Current(), gpuRefinement);
                if (!cpuResult->stats.cacheHit || cpuDrawData_.edgeIdBuffer == 0) {
                    cpuDrawData_ = roadRenderer_.UploadEdgeIds(cpuResult->edgeIds);
                }

                if (debugUI_.SplitScreenValidation()) {
                    const int leftWidth = width / 2;
                    const int rightWidth = width - leftWidth;
                    roadRenderer_.DrawViewport(camera_, 0, 0, leftWidth, height, cpuDrawData_,
                                               kCPURoadColor);
                    roadRenderer_.DrawViewport(camera_, leftWidth, 0, rightWidth, height,
                                               ToRoadDrawData(*gpuResult), kGPURoadColor);
                } else {
                    roadRenderer_.Draw(camera_, width, height, cpuDrawData_, 1.0f, 0.0f,
                                       kCPUOverlayColor);
                    roadRenderer_.Draw(camera_, width, height, ToRoadDrawData(*gpuResult), 1.0f,
                                       0.0f, kGPUOverlayColor);
                }
                break;
            }
        }

        const auto gpuStats = gpuPipeline_.Stats();
        const auto datasetLoad = datasetLoader_.Snapshot();
        const auto actions = debugUI_.Draw(
            graph_.get(), rangeFilterStrategyManager_, strategyManager_,
            graph_ ? &camera_ : nullptr,
            graph_ ? &lodController_ : nullptr, datasetCatalog_, datasetLoad, processingMode_,
            cpuGeometryRefinement_, maxScreenErrorPixels_,
            cpuResult ? &cpuResult->stats : nullptr, graph_ ? &gpuStats : nullptr,
            validationResult_ ? &*validationResult_ : nullptr,
            processingMode_ == pipeline::ProcessingMode::Validation && cpuResult && gpuResult);
        debugUI_.EndFrame();

        if (actions.validateCurrentResult && cpuResult && gpuResult) {
            gpuPipeline_.ReadBackEdgeIds(gpuResult->aliveEdgeIdBuffer,
                                         gpuResult->aliveDrawCommandBuffer, gpuAliveReadback_);
            gpuPipeline_.ReadBackEdgeIds(gpuResult->edgeIdBuffer, gpuResult->drawCommandBuffer,
                                         gpuOutputReadback_);

            benchmark::PipelineValidationResult result;
            result.rangeFilter = benchmark::CorrectnessValidator::Compare(
                cpuResult->aliveEdgeIds, gpuAliveReadback_);
            result.finalOutput =
                benchmark::CorrectnessValidator::Compare(cpuResult->edgeIds, gpuOutputReadback_);
            validationResult_ = result;
            validationLod_ = lodLevel;
            validationStrategy_ = strategyManager_.CurrentIndex();
            validationRangeFilterStrategy_ = rangeFilterStrategyManager_.CurrentIndex();
            validationRefinement_ = gpuRefinement;
        }

        if (actions.datasetRequest) {
            const auto& dataset = datasetCatalog_.Entries()[*actions.datasetRequest];
            LoadDataset(dataset.graphPath, dataset.rangesPath);
        }

        window_.SwapBuffers();

        if (auto loadedGraph = datasetLoader_.TakeCompleted()) {
            InstallDataset(std::move(*loadedGraph));
        }
    }

    return 0;
}

void Application::UpdateCameraInput(float deltaSeconds) {
    const auto scroll = window_.ConsumeScrollY();
    if (debugUI_.WantsMouse()) {
        wasPanning_ = false;
        camera_.Update(deltaSeconds);
        return;
    }

    if (scroll != 0.0) {
        camera_.Zoom(static_cast<float>(scroll));
    }

    double cursorX = 0.0;
    double cursorY = 0.0;
    window_.CursorPosition(cursorX, cursorY);

    if (window_.LeftMouseDown()) {
        if (wasPanning_) {
            int width = 0;
            int height = 0;
            window_.FramebufferSize(width, height);
            camera_.PanPixels(cursorX - lastCursorX_, cursorY - lastCursorY_, height);
        }
        wasPanning_ = true;
    } else {
        wasPanning_ = false;
    }

    lastCursorX_ = cursorX;
    lastCursorY_ = cursorY;
    camera_.Update(deltaSeconds);
}

void Application::ClearValidation() {
    validationResult_.reset();
    gpuAliveReadback_.clear();
    gpuOutputReadback_.clear();
    validationRefinement_ = {};
}

} // namespace chmv::core
