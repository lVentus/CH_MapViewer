#include "core/Application.h"

#include "gpu/filtering/BirthOrderedRangeFilterStrategy.h"
#include "gpu/filtering/FullScanRangeFilterStrategy.h"
#include "gpu/filtering/PageCullingRangeFilterStrategy.h"
#include "gpu/unfolding/AdaptiveDFSUnfoldingStrategy.h"
#include "gpu/unfolding/BFSUnfoldingStrategy.h"
#include "gpu/unfolding/DualIntervalUnfoldingStrategy.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"
#include "gpu/unfolding/NoUnfoldingStrategy.h"
#include "streaming/lod/LodPagePlanner.h"

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
// CPU passes can be slow on large views: the benchmark stops after this long (3 passes at least).
constexpr double kCpuBenchmarkSeconds = 10.0;
// How often the road style file is checked for changes.
constexpr float kRoadStyleCheckSeconds = 1.0f;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;

} // namespace

Application::Application(const std::filesystem::path& assetDirectory)
    : window_("CH_MapViewer"), context_(window_), segmentRenderer_(assetDirectory / "shaders"),
      debugUI_(window_),
      datasetCatalog_(assetDirectory / "data"), shaderDirectory_(assetDirectory / "shaders"),
      roadStylePath_(assetDirectory / "road_styles.cfg") {
    ReloadRoadStyles();
    // Filters find the records of the current level, expansions decide which of them (or of
    // their descendants) are drawn. Both lists are open for more strategies; each one names
    // its CPU counterpart, if it has one.
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::FullScanRangeFilterStrategy>(shaderDirectory_));
    cpuFilters_.push_back(cpu::CpuFilter::FullScan);
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::BirthOrderedRangeFilterStrategy>(shaderDirectory_));
    cpuFilters_.push_back(cpu::CpuFilter::BirthOrdered);
    rangeFilterStrategyManager_.Register(
        std::make_unique<gpu::filtering::PageCullingRangeFilterStrategy>(shaderDirectory_));
    cpuFilters_.push_back(cpu::CpuFilter::Culling);

    strategyManager_.Register(
        std::make_unique<gpu::unfolding::NoUnfoldingStrategy>(shaderDirectory_));
    cpuExpansions_.push_back(cpu::CpuExpansion::Roots);
    strategyManager_.Register(std::make_unique<gpu::unfolding::BFSUnfoldingStrategy>(
        shaderDirectory_, geometry::RefinementMode::Full));
    cpuExpansions_.push_back(cpu::CpuExpansion::Bfs);
    strategyManager_.Register(
        std::make_unique<gpu::unfolding::AdaptiveDFSUnfoldingStrategy>(shaderDirectory_));
    cpuExpansions_.push_back(cpu::CpuExpansion::Dfs);
    strategyManager_.Register(std::make_unique<gpu::unfolding::BFSUnfoldingStrategy>(
        shaderDirectory_, geometry::RefinementMode::Adaptive));
    cpuExpansions_.push_back(cpu::CpuExpansion::Bfs);
    strategyManager_.Register(
        std::make_unique<gpu::unfolding::DualIntervalUnfoldingStrategy>(shaderDirectory_));
    cpuExpansions_.push_back(cpu::CpuExpansion::Intervals);
    cpuThreads_ = static_cast<int>(cpu::CpuWorkers::HardwareThreads());
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
    cpuRuntime_.reset();
    validation_.reset();
    benchmark_.reset();
    comparison_.reset();
    cpuBenchmark_.reset();
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
    cpuRuntime_.reset();
    validation_.reset();
    benchmark_.reset();
    comparison_.reset();
    cpuBenchmark_.reset();
    cpuDrawError_ = segmentRenderer_.Upload({});

    const auto& header = dataset.file->Header();
    lodController_.SetLevelRange(0, header.maxLevel);
    lodController_.SetManualLevel(header.maxLevel);
    camera_.SetMetersPerNormalizedUnit(header.MetersPerNormalizedUnit());
    camera_.Reset();
    lodController_.SetOverviewZoom(camera_.ZoomFactor());
    cpuRuntime_ = std::make_unique<cpu::CpuRuntime>(dataset.file, dataset.pageImage,
                                                    dataset.childrenImage);
    lodPages_ = std::make_unique<gpu::lod::LodPageRuntime>(shaderDirectory_, std::move(dataset),
                                                           gpuBudgetBytes_);
}

void Application::ForgetResultsOnStrategyChange() {
    if (resultFilter_ != rangeFilterStrategyManager_.CurrentIndex() ||
        resultUnfold_ != strategyManager_.CurrentIndex()) {
        validation_.reset();
        benchmark_.reset();
        cpuBenchmark_.reset();
        resultFilter_ = rangeFilterStrategyManager_.CurrentIndex();
        resultUnfold_ = strategyManager_.CurrentIndex();
    }
}

cpu::CpuStrategy Application::CurrentCpuStrategy() const {
    cpu::CpuStrategy strategy;
    strategy.filter = cpuFilters_[rangeFilterStrategyManager_.CurrentIndex()];
    strategy.expansion = cpuExpansions_[strategyManager_.CurrentIndex()];
    strategy.mode = strategyManager_.Current().Mode();
    return strategy;
}

bool Application::UpdateCpu(int width, int height, float tolerance, bool countSteps) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    cpuRuntime_->SetThreads(static_cast<unsigned>(std::max(cpuThreads_, 1)));
    const auto view = streaming::lod::PlanView(camera_, lodController_, width, height, tolerance,
                                               lodPages_->File().Header().maxLevel);
    auto query = view.query;
    query.threshold =
        gpu::unfolding::ThresholdFor(strategyManager_.Current().Mode(), view.toleranceCode);
    cpuView_ = view.rectangle;
    cpuPixelCode_ = view.pixelCode;
    const bool computed =
        cpuRuntime_->Update(query, cpuMethod_, CurrentCpuStrategy(),
                            countSteps ? std::optional{view.pixelCode} : std::nullopt);
    if (computed) {
        cpuDrawError_ = segmentRenderer_.Upload(cpuRuntime_->Result().segments);
    }
    return computed;
}

FrameCosts Application::CurrentCosts(bool gpuOn, bool cpuOn) const {
    FrameCosts costs;
    if (gpuOn && lodPages_) {
        const auto& stats = lodPages_->Stats();
        costs.drawnStep = renderer::StepOfThreshold(stats.drawThreshold, stats.pixelCode);
        costs.passFixedMs = stats.gpuFilterMs.value_or(0.0);
        costs.passScaledMs = stats.gpuUnfoldMs.value_or(0.0);
        costs.drawScaledMs = stats.gpuDrawMs.value_or(0.0);
        if (stats.stepCounts) {
            costs.passFixedMs += stats.gpuCountMs.value_or(0.0);
            costs.counts.emplace();
            std::copy(stats.stepCounts->begin(), stats.stepCounts->end(), costs.counts->begin());
        }
    }
    if (cpuOn && cpuRuntime_ && cpuRuntime_->Result().unavailable.empty()) {
        // The CPU pass runs inside the frame.
        const auto& result = cpuRuntime_->Result();
        costs.passFixedMs += result.prepareMs + result.filterMs + result.countMs;
        costs.passScaledMs += result.expandMs + result.outputMs;
        costs.drawScaledMs += segmentRenderer_.DrawMilliseconds().value_or(0.0);
        if (!gpuOn) {
            costs.drawnStep = renderer::StepOfThreshold(result.threshold, cpuPixelCode_);
            costs.counts = result.stepCounts;
        }
    }
    return costs;
}

void Application::CompareGpuAndCpu() {
    const auto& stats = lodPages_->Stats();
    const auto& result = cpuRuntime_->Result();
    if (!stats.complete || stats.cameraMoving || !result.unavailable.empty() ||
        !(cpuRuntime_->ResultQuery() == lodPages_->DrawQuery())) {
        return;
    }
    const std::pair key{stats.filterDispatches, cpuRuntime_->ResultSerial()};
    if (comparison_ && key == comparedKey_) {
        return;
    }
    cpu::CpuGpuComparison comparison;
    comparison.level = lodPages_->DrawQuery().level;
    comparison.threshold = lodPages_->DrawQuery().threshold;
    // A pass that ran out of room is run again with larger lists while it can (a new pass, so a
    // new comparison); until then, and for good once it cannot, there is nothing to compare.
    const auto status = lodPages_->CurrentStatus();
    const auto overflow = [&](gpu::lod::LodPageStatus kind) {
        return status[static_cast<std::size_t>(kind)] != 0u;
    };
    const char* full = overflow(gpu::lod::LodPageStatus::CandidateOverflow) ? "candidate list"
                       : overflow(gpu::lod::LodPageStatus::SegmentOverflow) ? "segment list"
                       : overflow(gpu::lod::LodPageStatus::FrontierOverflow)
                           ? "breadth-first frontier"
                       : overflow(gpu::lod::LodPageStatus::StackOverflow) ? "depth-first stack"
                                                                           : nullptr;
    if (full != nullptr) {
        comparison.notCompared = std::string("GPU result truncated (") + full + " full)";
        comparison_ = comparison;
        comparedKey_ = key;
        return;
    }
    try {
        const auto gpuEdges = lodPages_->DrawnEdgesInView();
        const auto cpuEdges = cpu::EdgesInView(result.segments, result.edges, cpuView_);
        const auto difference = cpu::CompareEdges(gpuEdges, cpuEdges);
        comparison.gpuInView = gpuEdges.size();
        comparison.cpuInView = cpuEdges.size();
        comparison.onlyGpu = difference.missing;
        comparison.onlyCpu = difference.extra;
        comparison_ = comparison;
        comparedKey_ = key;
        std::cout << "[Compare] " << rangeFilterStrategyManager_.Current().Name() << " + "
                  << strategyManager_.Current().Name() << ", CPU "
                  << (cpuMethod_ == cpu::CpuMethod::Pages ? "pages" : "classic") << ", level "
                  << comparison.level << ", error code " << comparison.threshold << ": "
                  << (comparison.Identical() ? "identical" : "DIFFERENT") << " ("
                  << comparison.gpuInView << " roads in view on the GPU, " << comparison.cpuInView
                  << " on the CPU; " << comparison.onlyGpu << " only on the GPU, "
                  << comparison.onlyCpu << " only on the CPU)\n";
    } catch (const std::exception& error) {
        std::cout << "[Compare] could not read the page file: " << error.what() << '\n';
        comparison.notCompared = std::string("could not read the page file: ") + error.what();
        comparison_ = comparison;
        comparedKey_ = key;
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
        const bool gpuOn = processingMode_ != cpu::ProcessingMode::Cpu;
        const bool cpuOn = processingMode_ != cpu::ProcessingMode::Gpu;
        const bool sideBySide = processingMode_ == cpu::ProcessingMode::Compare;
        // Side by side, both halves show the same view, half as wide.
        const int viewWidth = sideBySide ? std::max(width / 2, 1) : width;
        // The tolerance of this frame, from the manual setting or from the last frames' costs.
        const bool adaptive = unfold.Mode() == geometry::RefinementMode::Adaptive;
        const auto tolerance =
            tolerance_.BeginFrame(camera_, viewWidth, height, maxScreenErrorPixels_, adaptive,
                                  CurrentCosts(gpuOn, cpuOn), static_cast<double>(deltaSeconds));
        const bool countSteps = tolerance_.Settings().mode == renderer::ToleranceMode::Predictive;
        bool passRan = false;
        if (lodPages_) {
            if (gpuOn) {
                lodPages_->SetPreloadAll(preloadAll_);
                lodPages_->Update(camera_, lodController_, viewWidth, height, tolerance, filter,
                                  unfold, countSteps);
                passRan = lodPages_->Stats().filtered;
            }
            // Side by side, the GPU counts the steps for both.
            const bool cpuComputed =
                cpuOn && UpdateCpu(viewWidth, height, tolerance, countSteps && !gpuOn);
            passRan = passRan || cpuComputed;
            if (!sideBySide) {
                if (gpuOn) {
                    lodPages_->Draw(camera_, width, height, roadStyleConfig_);
                } else {
                    segmentRenderer_.Draw(camera_, width, height, roadStyleConfig_);
                }
            } else {
                renderer_.SetViewport(0, 0, viewWidth, height);
                lodPages_->Draw(camera_, viewWidth, height, roadStyleConfig_);
                renderer_.SetViewport(width - viewWidth, 0, viewWidth, height);
                segmentRenderer_.Draw(camera_, viewWidth, height, roadStyleConfig_);
                renderer_.SetViewport(0, 0, width, height);
                CompareGpuAndCpu();
            }
            lodPageSweep_.Record({gpuOn ? &lodPages_->Stats() : nullptr,
                                  cpuOn ? &cpuRuntime_->Result() : nullptr, cpuComputed, tolerance,
                                  tolerance_.LastFrameMs()},
                                 camera_, static_cast<double>(deltaSeconds) * 1000.0);
        }
        // Only a comparison of what is shown now counts.
        const bool comparisonCurrent =
            sideBySide && lodPages_ && comparison_ &&
            comparedKey_ == std::pair{lodPages_->Stats().filterDispatches,
                                      cpuRuntime_->ResultSerial()};
        const auto classic = cpuRuntime_ ? cpuRuntime_->Classic() : cpu::ClassicStatus{};

        const auto datasetLoad = datasetLoader_.Snapshot();
        const ui::DebugUIState state{
            .catalog = datasetCatalog_,
            .load = datasetLoad,
            .camera = lodPages_ ? &camera_ : nullptr,
            .lod = lodPages_ ? &lodController_ : nullptr,
            .filters = rangeFilterStrategyManager_,
            .unfolds = strategyManager_,
            .pixelTolerance = maxScreenErrorPixels_,
            .tolerance = tolerance_.Settings(),
            .toleranceStatus = tolerance_.Status(),
            .tolerancePixels = tolerance,
            .toleranceCoarsened = tolerance_.Coarsened(),
            .preloadAll = preloadAll_,
            .roadStyleStatus = roadStyleStatus_,
            .roadStyleError = roadStyleError_,
            .stats = lodPages_ ? &lodPages_->Stats() : nullptr,
            .validation = validation_ ? &*validation_ : nullptr,
            .benchmark = benchmark_ ? &*benchmark_ : nullptr,
            .sweep = &lodPageSweep_,
            .mode = processingMode_,
            .cpuMethod = cpuMethod_,
            .cpuThreads = cpuThreads_,
            .cpu = cpuOn && cpuRuntime_ ? &cpuRuntime_->Result() : nullptr,
            .cpuDrawError = cpuDrawError_,
            .classic = classic,
            .comparison = comparisonCurrent ? &*comparison_ : nullptr,
            .cpuBenchmark = cpuBenchmark_ ? &*cpuBenchmark_ : nullptr,
        };
        const auto actions = debugUI_.Draw(state);
        debugUI_.EndFrame();
        tolerance_.EndFrame(passRan, CurrentCosts(gpuOn, cpuOn).drawnStep);
        ForgetResultsOnStrategyChange();

        if (actions.validate && lodPages_ && gpuOn) {
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
        if (actions.runBenchmark && lodPages_ && gpuOn) {
            benchmark_ = lodPages_->Benchmark(filter, unfold, kBenchmarkPasses);
            const auto& result = *benchmark_;
            std::cout << "[Benchmark] " << filter.Name() << " + " << unfold.Name() << ", "
                      << result.iterations << " passes over " << result.pages << " pages: filter "
                      << result.filterAverageMs << " ms (" << result.filterMinMs << " - "
                      << result.filterMaxMs << "), expansion " << result.unfoldAverageMs << " ms ("
                      << result.unfoldMinMs << " - " << result.unfoldMaxMs << "); "
                      << result.candidates << " candidates, " << result.segments << " segments\n";
        }
        if (actions.runBenchmark && cpuRuntime_ && cpuOn) {
            cpuBenchmark_ = cpuRuntime_->Benchmark(cpuRuntime_->ResultQuery(), cpuMethod_,
                                                   CurrentCpuStrategy(), kBenchmarkPasses,
                                                   kCpuBenchmarkSeconds);
            const auto& result = *cpuBenchmark_;
            std::cout << "[Benchmark] CPU "
                      << (cpuMethod_ == cpu::CpuMethod::Pages ? "pages" : "classic") << ", "
                      << filter.Name() << " + " << unfold.Name() << ", " << result.threads
                      << " threads: ";
            if (!result.unavailable.empty()) {
                std::cout << result.unavailable << '\n';
            } else {
                std::cout << result.iterations << " passes: filter " << result.filterAverageMs
                          << " ms (" << result.filterMinMs << " - " << result.filterMaxMs
                          << "), expansion " << result.expandAverageMs << " ms ("
                          << result.expandMinMs << " - " << result.expandMaxMs
                          << "), preparing " << result.prepareAverageMs << " ms; "
                          << result.candidates << " candidates, " << result.segments
                          << " segments\n";
            }
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
