#include "core/Application.h"

#include "gpu/filtering/BirthOrderedRangeFilterStrategy.h"
#include "gpu/filtering/FullScanRangeFilterStrategy.h"
#include "gpu/filtering/PageCullingRangeFilterStrategy.h"
#include "gpu/unfolding/AdaptiveDFSUnfoldingStrategy.h"
#include "gpu/unfolding/BFSUnfoldingStrategy.h"
#include "gpu/unfolding/DualIntervalUnfoldingStrategy.h"
#include "gpu/unfolding/NoUnfoldingStrategy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

namespace chmv::core {
namespace {

constexpr std::uint32_t kBenchmarkPasses = 100;
// How often the road style file is checked for changes.
constexpr float kRoadStyleCheckSeconds = 1.0f;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;

} // namespace

Application::Application(const std::filesystem::path& assetDirectory)
    : window_("CH_MapViewer"), context_(window_), debugUI_(window_),
      datasetCatalog_(assetDirectory / "data"), shaderDirectory_(assetDirectory / "shaders"),
      roadStylePath_(assetDirectory / "road_styles.cfg") {
    ReloadRoadStyles();
    // Filters find the records of the current level, expansions decide which of them (or of
    // their descendants) are drawn. Both lists are open for more strategies.
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::FullScanRangeFilterStrategy>(shaderDirectory_));
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::BirthOrderedRangeFilterStrategy>(shaderDirectory_));
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::PageCullingRangeFilterStrategy>(shaderDirectory_));

    strategyManager_.Register(
        std::make_unique<gpu::unfolding::NoUnfoldingStrategy>(shaderDirectory_));
    strategyManager_.Register(std::make_unique<gpu::unfolding::BFSUnfoldingStrategy>(
        shaderDirectory_, geometry::RefinementMode::Full));
    strategyManager_.Register(
        std::make_unique<gpu::unfolding::AdaptiveDFSUnfoldingStrategy>(shaderDirectory_));
    strategyManager_.Register(std::make_unique<gpu::unfolding::BFSUnfoldingStrategy>(
        shaderDirectory_, geometry::RefinementMode::Adaptive));
    strategyManager_.Register(
        std::make_unique<gpu::unfolding::DualIntervalUnfoldingStrategy>(shaderDirectory_));
    // Dual interval works with every page file, so it is the default.
    strategyManager_.Select(strategyManager_.Strategies().size() - 1u);
}

void Application::ReloadRoadStyles() {
    std::error_code error;
    roadStyleWriteTime_ = std::filesystem::last_write_time(roadStylePath_, error);
    try {
        roadStyleConfig_ = renderer::LoadRoadStyles(roadStylePath_);
        roadStyleStatus_ = "Road styles: " + roadStylePath_.filename().string() + ", " +
                           std::to_string(roadStyleConfig_.classes.size()) +
                           " classes (reloaded when the file is saved)";
        roadStyleError_ = false;
    } catch (const std::exception& failure) {
        // Keep drawing with what was there; a fresh start has no styles yet.
        if (roadStyleConfig_.classes.empty()) {
            roadStyleConfig_ = renderer::UniformRoadStyle();
        }
        roadStyleStatus_ = std::string("Road styles: ") + failure.what();
        roadStyleError_ = true;
    }
    std::cout << roadStyleStatus_ << '\n';
}

void Application::CheckRoadStyles(float deltaSeconds) {
    roadStyleCheckSeconds_ += deltaSeconds;
    if (roadStyleCheckSeconds_ < kRoadStyleCheckSeconds) {
        return;
    }
    roadStyleCheckSeconds_ = 0.0f;
    std::error_code error;
    const auto writeTime = std::filesystem::last_write_time(roadStylePath_, error);
    if (!error && writeTime != roadStyleWriteTime_) {
        ReloadRoadStyles();
    }
}

void Application::LoadDataset(const data::DatasetEntry& dataset) {
    if (datasetLoader_.Snapshot().Active()) {
        return;
    }
    // The current dataset goes first: its page reader keeps the file open (an outdated file is
    // rebuilt in place), and its RAM image and GPU pool must not count against the new one.
    lodPageSweep_.Cancel();
    lodPages_.reset();
    validation_.reset();
    benchmark_.reset();
    const bool started = dataset.HasSources()
                             ? datasetLoader_.Start(dataset.graphPath, dataset.rangesPath)
                             : datasetLoader_.StartPageFile(dataset.pageFilePath);
    if (started) {
        std::cout << "Loading " << (dataset.HasSources() ? dataset.graphPath : dataset.pageFilePath)
                  << " in background\n";
    }
}

void Application::InstallDataset(streaming::lod::LodPageDataset dataset) {
    lodPageSweep_.Cancel();
    lodPages_.reset();
    validation_.reset();
    benchmark_.reset();

    const auto& header = dataset.file->Header();
    lodController_.SetLevelRange(0, header.maxLevel);
    lodController_.SetManualLevel(header.maxLevel);
    camera_.SetMetersPerNormalizedUnit(header.MetersPerNormalizedUnit());
    camera_.Reset();
    lodController_.SetOverviewZoom(camera_.ZoomFactor());
    lodPages_ = std::make_unique<gpu::lod::LodPageRuntime>(shaderDirectory_, std::move(dataset),
                                                           gpuBudgetBytes_);
}

void Application::ForgetResultsOnStrategyChange() {
    if (resultFilter_ != rangeFilterStrategyManager_.CurrentIndex() ||
        resultUnfold_ != strategyManager_.CurrentIndex()) {
        validation_.reset();
        benchmark_.reset();
        resultFilter_ = rangeFilterStrategyManager_.CurrentIndex();
        resultUnfold_ = strategyManager_.CurrentIndex();
    }
}

int Application::Run() {
    auto previousFrameTime = std::chrono::steady_clock::now();

    while (!window_.ShouldClose()) {
        const auto currentFrameTime = std::chrono::steady_clock::now();
        const auto deltaSeconds =
            std::chrono::duration<float>(currentFrameTime - previousFrameTime).count();
        previousFrameTime = currentFrameTime;
        window_.PollEvents();
        CheckRoadStyles(deltaSeconds);
        debugUI_.BeginFrame();
        if (lodPageSweep_.Running()) {
            lodPageSweep_.Advance(camera_, deltaSeconds);
        } else {
            UpdateCameraInput(deltaSeconds);
        }

        int width = 0;
        int height = 0;
        window_.FramebufferSize(width, height);
        renderer_.BeginFrame(width, height);

        auto& filter = rangeFilterStrategyManager_.Current();
        auto& unfold = strategyManager_.Current();
        if (lodPages_) {
            lodPages_->SetPreloadAll(preloadAll_);
            lodPages_->Update(camera_, lodController_, width, height, maxScreenErrorPixels_, filter,
                              unfold);
            lodPages_->Draw(camera_, width, height, roadStyleConfig_);
            lodPageSweep_.Record(lodPages_->Stats(), camera_,
                                 static_cast<double>(deltaSeconds) * 1000.0);
        }

        const auto datasetLoad = datasetLoader_.Snapshot();
        const ui::DebugUIState state{
            .catalog = datasetCatalog_,
            .load = datasetLoad,
            .camera = lodPages_ ? &camera_ : nullptr,
            .lod = lodPages_ ? &lodController_ : nullptr,
            .filters = rangeFilterStrategyManager_,
            .unfolds = strategyManager_,
            .pixelTolerance = maxScreenErrorPixels_,
            .preloadAll = preloadAll_,
            .roadStyleStatus = roadStyleStatus_,
            .roadStyleError = roadStyleError_,
            .stats = lodPages_ ? &lodPages_->Stats() : nullptr,
            .validation = validation_ ? &*validation_ : nullptr,
            .benchmark = benchmark_ ? &*benchmark_ : nullptr,
            .sweep = &lodPageSweep_,
        };
        const auto actions = debugUI_.Draw(state);
        debugUI_.EndFrame();
        ForgetResultsOnStrategyChange();

        if (actions.validate && lodPages_) {
            try {
                validation_ = lodPages_->Validate();
            } catch (const std::exception& error) {
                validation_.emplace();
                validation_->error = error.what();
            }
            const auto& check = *validation_;
            if (!check.error.empty()) {
                std::cout << "[Validation] could not read the page file: " << check.error << '\n';
            } else {
                std::cout << "[Validation] " << filter.Name() << " + " << unfold.Name()
                          << ", level " << check.level << ", error code " << check.threshold
                          << ": " << (check.Passed() ? "identical" : "MISMATCH") << " ("
                          << check.cpuSegmentsInView << " roads in view on the CPU, "
                          << check.gpuSegmentsInView << " on the GPU; " << check.missingOnGpu
                          << " missing, " << check.extraOnGpu << " extra; candidates "
                          << check.gpuCandidates << " / " << check.cpuCandidates << ", "
                          << check.candidateMismatches << " differ)\n";
            }
        }
        if (actions.runBenchmark && lodPages_) {
            benchmark_ = lodPages_->Benchmark(filter, unfold, kBenchmarkPasses);
            const auto& result = *benchmark_;
            std::cout << "[Benchmark] " << filter.Name() << " + " << unfold.Name() << ", "
                      << result.iterations << " passes over " << result.pages << " pages: filter "
                      << result.filterAverageMs << " ms (" << result.filterMinMs << " - "
                      << result.filterMaxMs << "), expansion " << result.unfoldAverageMs << " ms ("
                      << result.unfoldMinMs << " - " << result.unfoldMaxMs << "); "
                      << result.candidates << " candidates, " << result.segments << " segments\n";
        }
        if (actions.datasetRequest) {
            LoadDataset(datasetCatalog_.Entries()[*actions.datasetRequest]);
        }
        if (actions.startZoomSweep && lodPages_) {
            auto csvPath = lodPages_->File().Path();
            csvPath.replace_extension(".lod_zoom_sweep.csv");
            lodPageSweep_.Start(csvPath, camera_);
        }
        if (actions.cancelZoomSweep) {
            lodPageSweep_.Cancel();
        }
        if (actions.gpuBudgetMiB) {
            gpuBudgetBytes_ = std::uint64_t{*actions.gpuBudgetMiB} * kMiB;
            if (lodPages_) {
                lodPages_->SetPoolBudget(gpuBudgetBytes_);
            }
        }

        window_.SwapBuffers();

        if (auto dataset = datasetLoader_.TakeCompleted()) {
            InstallDataset(std::move(*dataset));
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
            const auto dx = cursorX - lastCursorX_;
            const auto dy = cursorY - lastCursorY_;
            if (dx != 0.0 || dy != 0.0) {
                camera_.PanPixels(dx, dy, height);
            }
        }
        wasPanning_ = true;
    } else {
        wasPanning_ = false;
    }
    lastCursorX_ = cursorX;
    lastCursorY_ = cursorY;
    camera_.Update(deltaSeconds);
}

} // namespace chmv::core
