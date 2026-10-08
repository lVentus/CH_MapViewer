#include "core/Application.h"

#include "benchmark/CorrectnessValidator.h"
#include "gpu/filtering/BirthOrderedRangeFilterStrategy.h"
#include "gpu/filtering/FullScanRangeFilterStrategy.h"
#include "gpu/unfolding/AdaptiveDFSUnfoldingStrategy.h"
#include "gpu/unfolding/IterativeDFSUnfoldingStrategy.h"
#include "gpu/unfolding/NoUnfoldingStrategy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <utility>

namespace chmv::core {
namespace {

constexpr std::array<float, 4> kDefaultRoadColor{0.82f, 0.84f, 0.87f, 1.0f};
constexpr std::array<float, 4> kRoadStyleNeutralTint{1.0f, 1.0f, 1.0f, 1.0f};
constexpr std::array<float, 4> kCPURoadColor{1.0f, 0.55f, 0.20f, 1.0f};
constexpr std::array<float, 4> kGPURoadColor{0.20f, 0.75f, 1.0f, 1.0f};
constexpr std::array<float, 4> kCPUOverlayColor{1.0f, 0.55f, 0.20f, 0.70f};
constexpr std::array<float, 4> kGPUOverlayColor{0.20f, 0.75f, 1.0f, 0.70f};

// LOD candidates are offered across the hierarchy; GraphPageStreamer admits as many as the
// measured page cost fits in its current RAM budget. Only the predictive focus is fixed in time.
constexpr float kLodPredictionSeconds = 0.18f;
constexpr float kLodVelocityResponse = 8.0f;
constexpr float kLodPrefetchVelocityDeadZone = 0.50f;
constexpr float kMaxDisplayLodStepPerFrame = 4.0f;
constexpr float kStreamingPlanIntervalSeconds = 0.05f;
constexpr float kStreamingMemoryBudgetPollSeconds = 2.0f;
constexpr std::uint32_t kStreamingMemoryBudgetHysteresisMiB = 512u;
constexpr std::size_t kAdaptiveCpuFaultWaveLimit = 128u;
constexpr double kMetersPerProjectedDegree = 111319.49079327357;

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
      refinementRequestCollector_(assetDirectory / "shaders"),
      persistentStreamingPipeline_(assetDirectory / "shaders"),
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
                              const std::filesystem::path& rangesPath,
                              bool streaming) {
    const bool started = streaming ? datasetLoader_.StartStreaming(graphPath, rangesPath)
                                   : datasetLoader_.Start(graphPath, rangesPath);
    if (started) {
        std::cout << "Loading " << graphPath << " in background\n";
    }
}

void Application::InstallDataset(data::CHGraph graph) {
    std::cout << "Loaded " << graph.NodeCount() << " nodes and " << graph.EdgeCount()
              << " edges\n";
    streamingDisplayLod_ = -1;
    streamingDisplayLodFloat_ = -1.0f;
    streamingLastCameraLodFloat_ = -1.0f;
    streamingLodVelocity_ = 0.0f;
    streamingRootUploadBytesPerMs_ = 0.0;
    streamingBackingUploadBytesPerMs_ = 0.0;
    streamingPrefetchMinLod_ = 0u;
    streamingPrefetchMaxLod_ = 0u;
    streamingPrefetchFocusLod_ = 0.0f;
    streamingLodTransitionPending_ = false;
    streamingPlanAccumulatorSeconds_ = kStreamingPlanIntervalSeconds;
    streamingPlannerPrimed_ = false;
    streamingLatestSubmittedViewSequence_ = 0u;
    streamingLastConsumedViewSequence_ = 0u;
    streamingCommittedViewSequence_ = 0u;
    streamingLatestSubmittedLodFloat_ = -1.0f;
    streamingLatestSubmittedRefinementPixelScale_ = 0.0f;
    streamingStagedLodFloat_ = -1.0f;
    streamingStagedRefinementPixelScale_ = 0.0f;
    streamingCommittedRefinementPixelScale_ = 0.0f;
    streamingHasStagedViewportWindow_ = false;
    streamingHasCommittedViewportWindow_ = false;
    streamingMemoryBudgetPollSeconds_ = 0.0f;
    streamingCommittedRequiredPages_.clear();
    streamingPlannedDesiredPages_.clear();
    streamingPlannedRequiredPages_.clear();
    streamingPlannedLodPrefetchPages_.clear();
    streamingPlannedGpuWarmPages_.clear();
    streamingRootUploadOrder_.clear();
    streamingRequiredPageSet_.clear();
    streamingLodPrefetchPageSet_.clear();
    streamingCommittedActivePages_.clear();
    lastStreamingRefinementMode_ = geometry::RefinementMode::None;
    graph_ = std::make_unique<data::CHGraph>(std::move(graph));
    roadRenderer_.SetGraph(*graph_);
    gpuPipeline_.SetGraph(roadRenderer_.EdgeCount(),
                          static_cast<std::uint32_t>(graph_->DrawableEdgeCount()));
    rangeFilterStrategyManager_.SetGraph(*graph_);
    cpuPipeline_.SetGraph(*graph_);
    cpuDrawData_ = {};
    lodController_.SetLevelRange(roadRenderer_.MinLevel(), roadRenderer_.MaxLevel());
    lodController_.SetManualLevel(roadRenderer_.MaxLevel());
    const auto& projection = graph_->Projection();
    camera_.SetMetersPerNormalizedUnit(
        kMetersPerProjectedDegree / std::max(projection.normalization, 1e-12));
    camera_.Reset();
    lodController_.SetOverviewZoom(camera_.ZoomFactor());
    ClearValidation();
    gpuRangeBenchmarkResult_.reset();
}

void Application::InstallStreamingDataset(data::StreamingDataset dataset) {
    graph_.reset();
    cpuDrawData_ = {};
    ClearValidation();
    gpuRangeBenchmarkResult_.reset();
    roadRenderer_.ClearGraph();
    gpuPipeline_.SetGraph(0, 0);

    const auto maxLevel = static_cast<std::int32_t>(dataset.index.maxLevel);
    lodController_.SetLevelRange(0, maxLevel);
    lodController_.SetManualLevel(maxLevel);

    // Preprocessing normalizes every dataset to roughly [-0.95, 0.95]. Without compensating for
    // that normalization, the same numeric zoom means a completely different metres-per-pixel on
    // BW and EUR. Calibrate the camera before Reset(): reset still fits the dataset, but equal
    // ZoomFactor values now have equal physical scale across datasets.
    constexpr double kPi = 3.14159265358979323846;
    const auto centerLatitude = (dataset.index.minLatitude + dataset.index.maxLatitude) * 0.5;
    const auto longitudeScale = std::cos(centerLatitude * kPi / 180.0);
    const auto halfDatasetWidth = std::max(
        (dataset.index.maxLongitude - dataset.index.minLongitude) * 0.5 * longitudeScale, 1e-12);
    const auto halfDatasetHeight = std::max(
        (dataset.index.maxLatitude - dataset.index.minLatitude) * 0.5, 1e-12);
    const auto normalization = 0.95 / std::max(halfDatasetWidth, halfDatasetHeight);
    camera_.SetMetersPerNormalizedUnit(
        kMetersPerProjectedDegree / std::max(normalization, 1e-12));
    camera_.Reset();
    lodController_.SetOverviewZoom(camera_.ZoomFactor());

    const auto filters = rangeFilterStrategyManager_.Strategies();
    if (!filters.empty() &&
        rangeFilterStrategyManager_.Current().PersistentStreamingKind() ==
            gpu::filtering::PersistentStreamingRangeFilterKind::Unsupported) {
        for (std::size_t i = 0; i < filters.size(); ++i) {
            if (filters[i]->PersistentStreamingKind() !=
                gpu::filtering::PersistentStreamingRangeFilterKind::Unsupported) {
                rangeFilterStrategyManager_.Select(i);
                break;
            }
        }
    }

    // Streaming is a data-supply mode, not a reason to force exhaustive geometry unfolding.
    // Start with ranges only; Full DFS remains available as an explicit exhaustive benchmark.
    const auto strategies = strategyManager_.Strategies();
    for (std::size_t i = 0; i < strategies.size(); ++i) {
        if (strategies[i]->Mode() == geometry::RefinementMode::None) {
            strategyManager_.Select(i);
            break;
        }
    }
    processingMode_ = pipeline::ProcessingMode::GPUDriven;

    std::cout << "Starting out-of-core runtime with " << dataset.index.graphPages.size()
              << " indexed graph pages\n";
    streaming::runtime::GraphPageStreamerConfig streamerConfig;
    if (streamingRamBudgetAuto_ || streamingRamBudgetMiB_ == 0u) {
        const auto memory = streaming::runtime::QuerySystemMemoryInfo();
        const auto budget = streaming::runtime::RecommendGraphPageCacheBudget(memory);
        streamingRamBudgetMiB_ = static_cast<std::uint32_t>(budget / (1024ull * 1024ull));
        streamingRamBudgetAuto_ = true;
        std::cout << "Automatic out-of-core RAM cache budget: " << streamingRamBudgetMiB_
                  << " MiB (available " << (memory.availablePhysicalBytes / (1024ull * 1024ull))
                  << " MiB)\n";
    }
    streamerConfig.ramBudgetBytes =
        static_cast<std::uint64_t>(streamingRamBudgetMiB_) * 1024ull * 1024ull;
    pageStreamer_ = std::make_unique<streaming::runtime::GraphPageStreamer>(
        std::move(dataset.index), std::move(dataset.graphPath), std::move(dataset.rangesPath),
        streamerConfig);

    gpu::streaming::PersistentStreamingConfig gpuStreamingConfig;
    gpuStreamingConfig.gpuBudgetBytes =
        static_cast<std::uint64_t>(streamingGpuBudgetMiB_) * 1024ull * 1024ull;
    persistentStreamingPipeline_.Initialize(pageStreamer_->Index(), gpuStreamingConfig);
    // Let the first rendered frame initialize directly from the physical camera scale. BW and EUR
    // now reset to different numeric zoom values because they cover different metre extents.
    streamingDisplayLod_ = -1;
    streamingDisplayLodFloat_ = -1.0f;
    streamingLastCameraLodFloat_ = -1.0f;
    streamingLodVelocity_ = 0.0f;
    streamingRootUploadBytesPerMs_ = 0.0;
    streamingBackingUploadBytesPerMs_ = 0.0;
    streamingPrefetchMinLod_ = static_cast<std::uint32_t>(maxLevel);
    streamingPrefetchMaxLod_ = static_cast<std::uint32_t>(maxLevel);
    streamingPrefetchFocusLod_ = static_cast<float>(maxLevel);
    streamingLodTransitionPending_ = false;
    streamingPlanAccumulatorSeconds_ = kStreamingPlanIntervalSeconds;
    streamingPlannerPrimed_ = false;
    streamingLatestSubmittedViewSequence_ = 0u;
    streamingLastConsumedViewSequence_ = 0u;
    streamingCommittedViewSequence_ = 0u;
    streamingLatestSubmittedLodFloat_ = -1.0f;
    streamingLatestSubmittedRefinementPixelScale_ = 0.0f;
    streamingStagedLodFloat_ = -1.0f;
    streamingStagedRefinementPixelScale_ = 0.0f;
    streamingCommittedRefinementPixelScale_ = 0.0f;
    streamingHasStagedViewportWindow_ = false;
    streamingHasCommittedViewportWindow_ = false;
    streamingMemoryBudgetPollSeconds_ = 0.0f;
    streamingCommittedRequiredPages_.clear();
    streamingPlannedDesiredPages_.clear();
    streamingPlannedRequiredPages_.clear();
    streamingPlannedLodPrefetchPages_.clear();
    streamingPlannedGpuWarmPages_.clear();
    streamingRootUploadOrder_.clear();
    streamingRequiredPageSet_.clear();
    streamingLodPrefetchPageSet_.clear();
    streamingCommittedActivePages_.clear();
    lastStreamingRefinementMode_ = geometry::RefinementMode::None;
}

streaming::runtime::StreamingViewRequest Application::BuildStreamingView(
    int framebufferWidth, int framebufferHeight, float displayLodLevel,
    std::uint32_t prefetchMinLod, std::uint32_t prefetchMaxLod,
    float prefetchFocusLod) const {
    streaming::runtime::StreamingViewRequest view;
    if (!pageStreamer_) {
        return view;
    }

    const auto& index = pageStreamer_->Index();
    const auto centerLatitude = (index.minLatitude + index.maxLatitude) * 0.5;
    const auto centerLongitude = (index.minLongitude + index.maxLongitude) * 0.5;
    constexpr double kPi = 3.14159265358979323846;
    const auto longitudeScale = std::cos(centerLatitude * kPi / 180.0);
    const auto halfDatasetWidth = std::max(
        (index.maxLongitude - index.minLongitude) * 0.5 * longitudeScale, 1e-12);
    const auto halfDatasetHeight =
        std::max((index.maxLatitude - index.minLatitude) * 0.5, 1e-12);
    const auto normalization = 0.95 / std::max(halfDatasetWidth, halfDatasetHeight);

    const auto viewScale = std::max<double>(camera_.ViewScale(), 1e-12);
    const auto aspect = framebufferHeight > 0
                            ? static_cast<double>(framebufferWidth) /
                                  static_cast<double>(framebufferHeight)
                            : 1.0;
    const auto halfY = 1.0 / viewScale;
    const auto halfX = aspect * halfY;
    const auto minX = static_cast<double>(camera_.CenterX()) - halfX;
    const auto maxX = static_cast<double>(camera_.CenterX()) + halfX;
    const auto minY = static_cast<double>(camera_.CenterY()) - halfY;
    const auto maxY = static_cast<double>(camera_.CenterY()) + halfY;

    const auto clampedLod = std::clamp(displayLodLevel, 0.0f,
                                       static_cast<float>(index.maxLevel));
    view.lodLevelFloat = clampedLod;
    view.lodLevel = static_cast<std::uint32_t>(std::lround(clampedLod));
    view.prefetchMinLod = std::min(prefetchMinLod, index.maxLevel);
    view.prefetchMaxLod = std::min(prefetchMaxLod, index.maxLevel);
    view.prefetchFocusLod = std::clamp(prefetchFocusLod, 0.0f,
                                       static_cast<float>(index.maxLevel));
    view.minLatitude = centerLatitude + minY / normalization;
    view.maxLatitude = centerLatitude + maxY / normalization;
    if (std::abs(longitudeScale) > 1e-12) {
        view.minLongitude = centerLongitude + minX / (longitudeScale * normalization);
        view.maxLongitude = centerLongitude + maxX / (longitudeScale * normalization);
    } else {
        view.minLongitude = index.minLongitude;
        view.maxLongitude = index.maxLongitude;
    }
    return view;
}

streaming::runtime::StreamingSpatialWindow Application::BuildRefinementGuardWindow() const {
    if (!pageStreamer_) {
        return {};
    }
    auto window = streamingHasCommittedViewportWindow_
                      ? streamingCommittedViewportWindow_
                      : pageStreamer_->CurrentViewportWindow();
    const auto gridSize = std::max(pageStreamer_->Index().spatialGridSize, 1u);
    const auto width = window.maxX >= window.minX ? window.maxX - window.minX + 1u : 1u;
    const auto height = window.maxY >= window.minY ? window.maxY - window.minY + 1u : 1u;
    // Refinement is deliberately tighter than the 7x7 residency neighborhood. A 20% guard
    // keeps newly entering roads warm without spending DFS work on the entire prefetch area.
    const auto guardX = std::max(width / 5u, 1u);
    const auto guardY = std::max(height / 5u, 1u);
    window.minX = window.minX > guardX ? window.minX - guardX : 0u;
    window.minY = window.minY > guardY ? window.minY - guardY : 0u;
    window.maxX = std::min(gridSize - 1u, window.maxX + guardX);
    window.maxY = std::min(gridSize - 1u, window.maxY + guardY);
    return window;
}


void Application::UpdateStreamingRuntime(int framebufferWidth, int framebufferHeight,
                                         float targetLodLevel, float deltaSeconds) {
    if (!pageStreamer_ || framebufferWidth <= 0 || framebufferHeight <= 0) {
        return;
    }

    const auto maxLod = static_cast<float>(pageStreamer_->Index().maxLevel);
    targetLodLevel = std::clamp(targetLodLevel, 0.0f, maxLod);
    const auto dt = std::clamp(deltaSeconds, 0.0f, 0.1f);

    // Auto RAM cache follows real system headroom, not a fixed 1 GiB constant. Add the current
    // cache budget back to MemAvailable before recomputing so our own resident cache does not cause
    // a feedback loop that immediately asks us to shrink it again. Hysteresis keeps this off the
    // render hot path and avoids churn from small OS-memory fluctuations.
    if (streamingRamBudgetAuto_) {
        streamingMemoryBudgetPollSeconds_ += dt;
        if (streamingMemoryBudgetPollSeconds_ >= kStreamingMemoryBudgetPollSeconds) {
            streamingMemoryBudgetPollSeconds_ = 0.0f;
            auto memory = streaming::runtime::QuerySystemMemoryInfo();
            memory.availablePhysicalBytes = std::min<std::uint64_t>(
                memory.totalPhysicalBytes,
                memory.availablePhysicalBytes +
                    static_cast<std::uint64_t>(streamingRamBudgetMiB_) * 1024ull * 1024ull);
            const auto recommendedBytes =
                streaming::runtime::RecommendGraphPageCacheBudget(memory);
            const auto recommendedMiB = static_cast<std::uint32_t>(
                recommendedBytes / (1024ull * 1024ull));
            const auto difference = recommendedMiB > streamingRamBudgetMiB_
                                        ? recommendedMiB - streamingRamBudgetMiB_
                                        : streamingRamBudgetMiB_ - recommendedMiB;
            if (difference >= kStreamingMemoryBudgetHysteresisMiB) {
                streamingRamBudgetMiB_ = recommendedMiB;
                pageStreamer_->ReconfigureBudgets(recommendedBytes);
                std::cout << "Adjusted automatic out-of-core RAM cache budget to "
                          << streamingRamBudgetMiB_ << " MiB\n";
            }
        }
    }

    if (streamingLastCameraLodFloat_ < 0.0f) {
        streamingLastCameraLodFloat_ = targetLodLevel;
        streamingLodVelocity_ = 0.0f;
    }

    // Camera state is live and never waits for data. The committed render LOD is a separate
    // snapshot: while a new root working set is planned/loaded/uploaded in the background, the
    // previous complete snapshot keeps drawing under the new camera transform.
    if (streamingLastCameraLodFloat_ >= 0.0f && dt > 1e-5f) {
        const auto sampleVelocity = (targetLodLevel - streamingLastCameraLodFloat_) / dt;
        const auto response = 1.0f - std::exp(-kLodVelocityResponse * dt);
        streamingLodVelocity_ += (sampleVelocity - streamingLodVelocity_) * response;
    }
    streamingLastCameraLodFloat_ = targetLodLevel;

    const auto planningBaseLod = streamingDisplayLodFloat_ >= 0.0f
                                     ? streamingDisplayLodFloat_
                                     : targetLodLevel;
    const auto delta = std::clamp(targetLodLevel - planningBaseLod,
                                  -kMaxDisplayLodStepPerFrame, kMaxDisplayLodStepPerFrame);
    const auto proposedDisplayLod = std::clamp(planningBaseLod + delta, 0.0f, maxLod);

    const auto movingLod = std::abs(streamingLodVelocity_) >= kLodPrefetchVelocityDeadZone;
    streamingPrefetchFocusLod_ = movingLod
                                     ? std::clamp(proposedDisplayLod +
                                                      streamingLodVelocity_ * kLodPredictionSeconds,
                                                  0.0f, maxLod)
                                     : proposedDisplayLod;
    const auto temporalMinLod = std::min(proposedDisplayLod, streamingPrefetchFocusLod_);
    const auto temporalMaxLod = std::max(proposedDisplayLod, streamingPrefetchFocusLod_);
    streamingPrefetchMinLod_ = static_cast<std::uint32_t>(
        std::floor(std::clamp(temporalMinLod, 0.0f, maxLod)));
    streamingPrefetchMaxLod_ = static_cast<std::uint32_t>(
        std::ceil(std::clamp(temporalMaxLod, 0.0f, maxLod)));

    // Submit only the newest view to a dedicated planner thread. QueryPages never runs on the
    // render/UI thread anymore, so even a pathological spatial query cannot hitch the mouse wheel.
    streamingPlanAccumulatorSeconds_ += dt;
    const bool plannerDue = !streamingPlannerPrimed_ ||
                            streamingPlanAccumulatorSeconds_ >= kStreamingPlanIntervalSeconds;
    if (plannerDue) {
        const auto view = BuildStreamingView(framebufferWidth, framebufferHeight,
                                             proposedDisplayLod,
                                             streamingPrefetchMinLod_, streamingPrefetchMaxLod_,
                                             streamingPrefetchFocusLod_);
        streamingLatestSubmittedViewSequence_ = pageStreamer_->SubmitView(view);
        streamingLatestSubmittedLodFloat_ = proposedDisplayLod;
        streamingLatestSubmittedRefinementPixelScale_ =
            camera_.ViewScale() * static_cast<float>(framebufferHeight) * 0.5f;
        streamingPlanAccumulatorSeconds_ = 0.0f;
        streamingPlannerPrimed_ = true;
    }

    // A planner result is staging data, not immediately visible state. Ignore stale completed plans
    // if a newer camera sample has already been submitted; latest-wins keeps violent zoom/pan input
    // from walking through obsolete intermediate working sets.
    const auto appliedViewSequence = pageStreamer_->LatestAppliedViewSequence();
    if (appliedViewSequence != 0u &&
        appliedViewSequence == streamingLatestSubmittedViewSequence_ &&
        appliedViewSequence != streamingLastConsumedViewSequence_) {
        streamingPlannedDesiredPages_ = pageStreamer_->DesiredPageIds();
        streamingPlannedRequiredPages_ = pageStreamer_->RequiredPageIds();
        streamingPlannedLodPrefetchPages_ = pageStreamer_->LodPrefetchPageIds();
        streamingPlannedGpuWarmPages_ = pageStreamer_->GpuWarmPageIds();
        streamingStagedLodFloat_ = streamingLatestSubmittedLodFloat_;
        streamingStagedRefinementPixelScale_ = streamingLatestSubmittedRefinementPixelScale_;
        streamingStagedViewportWindow_ = pageStreamer_->CurrentViewportWindow();
        streamingHasStagedViewportWindow_ = true;
        streamingLastConsumedViewSequence_ = appliedViewSequence;

        streamingRequiredPageSet_.clear();
        streamingRequiredPageSet_.reserve(streamingPlannedRequiredPages_.size());
        streamingRequiredPageSet_.insert(streamingPlannedRequiredPages_.begin(),
                                         streamingPlannedRequiredPages_.end());
        streamingLodPrefetchPageSet_.clear();
        streamingLodPrefetchPageSet_.reserve(streamingPlannedLodPrefetchPages_.size());
        streamingLodPrefetchPageSet_.insert(streamingPlannedLodPrefetchPages_.begin(),
                                            streamingPlannedLodPrefetchPages_.end());

        streamingRootUploadOrder_.clear();
        streamingRootUploadOrder_.reserve(streamingPlannedRequiredPages_.size() +
                                          streamingPlannedDesiredPages_.size() +
                                          streamingPlannedGpuWarmPages_.size());
        std::unordered_set<std::uint32_t> orderedSet;
        orderedSet.reserve(streamingRootUploadOrder_.capacity());
        for (const auto pageId : streamingPlannedRequiredPages_) {
            if (orderedSet.insert(pageId).second) {
                streamingRootUploadOrder_.push_back(pageId);
            }
        }
        for (const auto pageId : streamingPlannedDesiredPages_) {
            if (!streamingLodPrefetchPageSet_.contains(pageId) &&
                orderedSet.insert(pageId).second) {
                streamingRootUploadOrder_.push_back(pageId);
            }
        }
        for (const auto pageId : streamingPlannedGpuWarmPages_) {
            if (orderedSet.insert(pageId).second) {
                streamingRootUploadOrder_.push_back(pageId);
            }
        }
    }

    const auto plannerStats = pageStreamer_->Snapshot();
    streamingPrefetchMinLod_ = plannerStats.lodPrefetchMinLevel;
    streamingPrefetchMaxLod_ = plannerStats.lodPrefetchMaxLevel;

    const auto& desiredPageIds = streamingPlannedDesiredPages_;
    const auto& requiredPageIds = streamingPlannedRequiredPages_;
    const auto& requiredPageSet = streamingRequiredPageSet_;
    const auto& lodPrefetchPageSet = streamingLodPrefetchPageSet_;
    const auto& orderedPages = streamingRootUploadOrder_;

    const auto rootCapacity = persistentStreamingPipeline_.Stats().gpuRootRecordCapacity;
    const auto rootRecordsForPages = [this](std::span<const std::uint32_t> pageIds) {
        std::uint64_t total = 0;
        const auto& pages = pageStreamer_->Index().graphPages;
        for (const auto pageId : pageIds) {
            if (pageId < pages.size()) {
                total += pages[pageId].edgeCount;
            }
        }
        return total;
    };
    const auto requiredRootRecords = rootRecordsForPages(requiredPageIds);
    const bool requiredFitsRootCache =
        rootCapacity != 0u && requiredRootRecords <= rootCapacity;

    // Keep the currently drawn snapshot protected while warming the next one. If the union fits,
    // protect both sets and the handoff is literally atomic. If it does not fit, never sacrifice
    // currently visible roads merely to accelerate a future plan; the old snapshot remains stable.
    std::vector<std::uint32_t> protectedRootPages = streamingCommittedRequiredPages_;
    protectedRootPages.insert(protectedRootPages.end(), requiredPageIds.begin(),
                              requiredPageIds.end());
    std::sort(protectedRootPages.begin(), protectedRootPages.end());
    protectedRootPages.erase(
        std::unique(protectedRootPages.begin(), protectedRootPages.end()),
        protectedRootPages.end());
    // Protecting the union also prevents a too-large transition from thrashing staged pages in
    // and out. If an extreme old+new union exceeds capacity, the new snapshot simply remains
    // pending while the camera stays responsive; normal nearby zoom/pan transitions overlap heavily
    // and fit the cache.
    if (protectedRootPages.empty()) {
        protectedRootPages = requiredPageIds;
    }
    persistentStreamingPipeline_.SetProtectedRootPages(protectedRootPages);

    const bool requiredMissingAtFrameStart = std::any_of(
        requiredPageIds.begin(), requiredPageIds.end(), [this](std::uint32_t pageId) {
            return !persistentStreamingPipeline_.HasRootPage(pageId);
        });

    constexpr double kTargetFrameMs = 1000.0 / 60.0;
    const auto previousFrameMs = std::max(0.0, static_cast<double>(dt) * 1000.0);
    const auto frameHeadroomMs = std::clamp(kTargetFrameMs - previousFrameMs, 0.0, 4.0);
    const auto rootUploadBudgetMs = requiredMissingAtFrameStart
                                        ? std::clamp(0.35 + frameHeadroomMs * 0.75, 0.35, 3.0)
                                        : std::clamp(frameHeadroomMs * 0.5, 0.0, 1.5);
    double rootUploadSpentMs = 0.0;
    bool uploadedAnyRequiredRoot = false;

    for (const auto pageId : orderedPages) {
        if (pageId >= pageStreamer_->Index().graphPages.size() ||
            persistentStreamingPipeline_.HasRootPage(pageId)) {
            continue;
        }
        const bool required = requiredPageSet.contains(pageId);
        if (requiredMissingAtFrameStart && !required) {
            continue;
        }

        auto page = pageStreamer_->FindResidentPage(pageId);
        if (!page) {
            continue;
        }
        const auto rootCount = !page->rootRecords.empty() ? page->rootRecords.size()
                                                          : page->edges.size();
        const auto uploadBytes = static_cast<double>(rootCount) *
                                 sizeof(streaming::index::CHIndexRootRecord);
        const auto predictedMs = streamingRootUploadBytesPerMs_ > 0.0
                                     ? uploadBytes / streamingRootUploadBytesPerMs_
                                     : 0.0;
        const bool forceFirstRequired = required && !uploadedAnyRequiredRoot;
        if (!forceFirstRequired &&
            (rootUploadBudgetMs <= 0.0 ||
             rootUploadSpentMs + predictedMs > rootUploadBudgetMs)) {
            continue;
        }

        const auto before = persistentStreamingPipeline_.Stats();
        if (!persistentStreamingPipeline_.UploadRootPage(
                *page, pageStreamer_->Index().edgeSpatialBounds)) {
            continue;
        }
        const auto after = persistentStreamingPipeline_.Stats();
        const auto uploadedBytes = after.incrementalRootBytesUploaded -
                                   before.incrementalRootBytesUploaded;
        const auto elapsedMs = std::max(after.lastRootUploadMs, 1e-6);
        const auto sampleBytesPerMs = static_cast<double>(uploadedBytes) / elapsedMs;
        streamingRootUploadBytesPerMs_ = streamingRootUploadBytesPerMs_ <= 0.0
                                             ? sampleBytesPerMs
                                             : streamingRootUploadBytesPerMs_ * 0.8 +
                                                   sampleBytesPerMs * 0.2;
        rootUploadSpentMs += after.lastRootUploadMs;
        uploadedAnyRequiredRoot = uploadedAnyRequiredRoot || required;
    }

    const bool stagedPlanAvailable = streamingLastConsumedViewSequence_ != 0u;
    const bool requiredReady = stagedPlanAvailable && requiredFitsRootCache && std::all_of(
        requiredPageIds.begin(), requiredPageIds.end(), [this](std::uint32_t pageId) {
            return persistentStreamingPipeline_.HasRootPage(pageId);
        });

    if (requiredReady) {
        std::vector<std::uint32_t> activePagesToRender;
        activePagesToRender.reserve(desiredPageIds.size());
        for (const auto pageId : desiredPageIds) {
            if (!lodPrefetchPageSet.contains(pageId) &&
                persistentStreamingPipeline_.HasRootPage(pageId)) {
                activePagesToRender.push_back(pageId);
            }
        }
        std::sort(activePagesToRender.begin(), activePagesToRender.end());
        activePagesToRender.erase(
            std::unique(activePagesToRender.begin(), activePagesToRender.end()),
            activePagesToRender.end());

        // Atomic render-state commit: only now do the new range-filter LOD and page descriptors
        // become visible. The camera has been moving the whole time using the previous complete
        // snapshot, so there is no "zoom waits for disk" path and no half-loaded street set.
        streamingCommittedRequiredPages_ = requiredPageIds;
        streamingCommittedActivePages_ = std::move(activePagesToRender);
        streamingDisplayLodFloat_ = streamingStagedLodFloat_ >= 0.0f
                                        ? streamingStagedLodFloat_
                                        : targetLodLevel;
        streamingDisplayLod_ =
            static_cast<std::int32_t>(std::lround(streamingDisplayLodFloat_));
        streamingCommittedRefinementPixelScale_ = streamingStagedRefinementPixelScale_;
        if (streamingHasStagedViewportWindow_) {
            streamingCommittedViewportWindow_ = streamingStagedViewportWindow_;
            streamingHasCommittedViewportWindow_ = true;
        }
        streamingCommittedViewSequence_ = streamingLastConsumedViewSequence_;
        persistentStreamingPipeline_.SetActiveRootPages(streamingCommittedActivePages_);
    }

    streamingLodTransitionPending_ =
        streamingCommittedViewSequence_ != streamingLatestSubmittedViewSequence_ ||
        streamingDisplayLodFloat_ < 0.0f ||
        std::abs(streamingDisplayLodFloat_ - targetLodLevel) > 0.01f;

    auto refinementMode = strategyManager_.Current().Mode();
    const bool adaptiveSupported =
        pageStreamer_->Index().HasRootPayload() &&
        pageStreamer_->Index().edgeSpatialBoundsCount == pageStreamer_->Index().edgeCount;
    if (refinementMode == geometry::RefinementMode::Adaptive && !adaptiveSupported) {
        refinementMode = geometry::RefinementMode::None;
    }
    if (refinementMode != lastStreamingRefinementMode_) {
        // Full and Adaptive cache different leaf sets. Drop pending CPU backing demand when the
        // mode changes; the persistent GPU pass will immediately fault only the blocks needed by
        // the new mode/current camera instead of inheriting an obsolete queue.
        pageStreamer_->ClearRefinement();
        constexpr std::array<std::uint32_t, 0> none{};
        persistentStreamingPipeline_.SetPinnedBackingBlocks(none);
        lastStreamingRefinementMode_ = refinementMode;
    }

    if (refinementMode == geometry::RefinementMode::None) {
        // Range-only navigation needs no backing data. A root page contains its straight segment
        // endpoints, LOD lifetime and spatial bounds, so one page upload makes it drawable.
        return;
    }

    // Do not pre-request every source block referenced by the visible root pages. Root records now
    // carry childA/childB directly, so Full/Adaptive DFS can start from the root without its source
    // block. The GPU requests only hierarchy blocks actually reached by the traversal. This is
    // essential for Adaptive and also prevents Full DFS from front-loading the whole visible map.
    std::unordered_set<std::uint32_t> pinnedBackingSet;

    // RAM -> GPU hierarchy upload uses the same measured-throughput rule as roots. There is no
    // fixed "32 blocks/frame" knob: the selected count follows actual block size, measured GPU
    // upload bandwidth and whatever frame headroom remains after mandatory root uploads.
    const auto residentBlockIds = pageStreamer_->ResidentRefinementBlockIds();
    std::unordered_set<std::uint32_t> uploadThisFrame;
    const auto backingHeadroomMs = std::max(0.0, frameHeadroomMs - rootUploadSpentMs);
    const auto backingUploadBudgetMs = std::clamp(backingHeadroomMs * 0.75, 0.25, 2.5);
    double predictedBackingMs = 0.0;
    for (const auto blockId : residentBlockIds) {
        if (persistentStreamingPipeline_.HasBackingBlock(blockId)) {
            continue;
        }
        const auto block = pageStreamer_->FindResidentRefinementBlock(blockId);
        if (!block) {
            continue;
        }
        // GPU stores one 8-byte children record, one 16-byte endpoint and one 4-byte
        // conservative geometry-error value per edge.
        const auto uploadBytes = static_cast<double>(block->edges.size()) * 28.0;
        // Bootstrap bandwidth with exactly one real block. Once that upload is measured, later
        // frames use the EMA bytes/ms budget; never interpret "no measurement yet" as zero cost.
        if (streamingBackingUploadBytesPerMs_ <= 0.0 && !uploadThisFrame.empty()) {
            break;
        }
        const auto estimateMs = streamingBackingUploadBytesPerMs_ > 0.0
                                    ? uploadBytes / streamingBackingUploadBytesPerMs_
                                    : backingUploadBudgetMs;
        if (!uploadThisFrame.empty() &&
            predictedBackingMs + estimateMs > backingUploadBudgetMs) {
            break;
        }
        uploadThisFrame.insert(blockId);
        predictedBackingMs += estimateMs;
    }

    for (const auto blockId : uploadThisFrame) {
        pinnedBackingSet.insert(blockId);
    }
    std::vector<std::uint32_t> pinnedBackingBlocks(pinnedBackingSet.begin(),
                                                   pinnedBackingSet.end());
    std::sort(pinnedBackingBlocks.begin(), pinnedBackingBlocks.end());
    persistentStreamingPipeline_.SetPinnedBackingBlocks(pinnedBackingBlocks);

    std::vector<std::uint32_t> consumedBlocks;
    consumedBlocks.reserve(residentBlockIds.size());
    for (const auto blockId : residentBlockIds) {
        if (!persistentStreamingPipeline_.HasBackingBlock(blockId)) {
            if (!uploadThisFrame.contains(blockId)) {
                continue;
            }
            auto block = pageStreamer_->FindResidentRefinementBlock(blockId);
            if (!block) {
                continue;
            }
            const auto before = persistentStreamingPipeline_.Stats();
            if (!persistentStreamingPipeline_.UploadBackingBlock(*block)) {
                continue;
            }
            const auto after = persistentStreamingPipeline_.Stats();
            const auto uploadedBytes = after.incrementalBackingBytesUploaded -
                                       before.incrementalBackingBytesUploaded;
            const auto elapsedMs = std::max(after.lastBackingUploadMs, 1e-6);
            const auto sampleBytesPerMs = static_cast<double>(uploadedBytes) / elapsedMs;
            streamingBackingUploadBytesPerMs_ = streamingBackingUploadBytesPerMs_ <= 0.0
                                                    ? sampleBytesPerMs
                                                    : streamingBackingUploadBytesPerMs_ * 0.8 +
                                                          sampleBytesPerMs * 0.2;
        }
        consumedBlocks.push_back(blockId);
    }
    if (!consumedBlocks.empty()) {
        pageStreamer_->ReleaseBackingBlocks(consumedBlocks);
    }
}

void Application::ApplyStreamingBudgets(std::uint32_t ramBudgetMiB,
                                        std::uint32_t gpuBudgetMiB) {
    streamingRamBudgetAuto_ = ramBudgetMiB == 0u;
    if (streamingRamBudgetAuto_) {
        const auto memory = streaming::runtime::QuerySystemMemoryInfo();
        const auto budget = streaming::runtime::RecommendGraphPageCacheBudget(memory);
        streamingRamBudgetMiB_ = static_cast<std::uint32_t>(budget / (1024ull * 1024ull));
    } else {
        streamingRamBudgetMiB_ = std::clamp<std::uint32_t>(ramBudgetMiB, 64u, 15360u);
    }
    streamingGpuBudgetMiB_ = std::clamp<std::uint32_t>(gpuBudgetMiB, 128u, 2048u);
    streamingMemoryBudgetPollSeconds_ = 0.0f;
    if (!pageStreamer_) {
        return;
    }

    pageStreamer_->ReconfigureBudgets(
        static_cast<std::uint64_t>(streamingRamBudgetMiB_) * 1024ull * 1024ull);

    gpu::streaming::PersistentStreamingConfig gpuStreamingConfig;
    gpuStreamingConfig.gpuBudgetBytes =
        static_cast<std::uint64_t>(streamingGpuBudgetMiB_) * 1024ull * 1024ull;
    persistentStreamingPipeline_.Initialize(pageStreamer_->Index(), gpuStreamingConfig);
    streamingDisplayLod_ = -1;
    streamingDisplayLodFloat_ = -1.0f;
    streamingLastCameraLodFloat_ = -1.0f;
    streamingLodVelocity_ = 0.0f;
    streamingRootUploadBytesPerMs_ = 0.0;
    streamingBackingUploadBytesPerMs_ = 0.0;
    streamingPrefetchMinLod_ = 0u;
    streamingPrefetchMaxLod_ = 0u;
    streamingPrefetchFocusLod_ = 0.0f;
    streamingLodTransitionPending_ = false;
    streamingPlanAccumulatorSeconds_ = kStreamingPlanIntervalSeconds;
    streamingPlannerPrimed_ = false;
    streamingLatestSubmittedViewSequence_ = 0u;
    streamingLastConsumedViewSequence_ = 0u;
    streamingCommittedViewSequence_ = 0u;
    streamingLatestSubmittedLodFloat_ = -1.0f;
    streamingLatestSubmittedRefinementPixelScale_ = 0.0f;
    streamingStagedLodFloat_ = -1.0f;
    streamingStagedRefinementPixelScale_ = 0.0f;
    streamingCommittedRefinementPixelScale_ = 0.0f;
    streamingHasStagedViewportWindow_ = false;
    streamingHasCommittedViewportWindow_ = false;
    streamingMemoryBudgetPollSeconds_ = 0.0f;
    streamingCommittedRequiredPages_.clear();
    streamingPlannedDesiredPages_.clear();
    streamingPlannedRequiredPages_.clear();
    streamingPlannedLodPrefetchPages_.clear();
    streamingPlannedGpuWarmPages_.clear();
    streamingRootUploadOrder_.clear();
    streamingRequiredPageSet_.clear();
    streamingLodPrefetchPageSet_.clear();
    streamingCommittedActivePages_.clear();
    lastStreamingRefinementMode_ = geometry::RefinementMode::None;
    std::cout << "Applied out-of-core budgets: RAM " << streamingRamBudgetMiB_
              << " MiB" << (streamingRamBudgetAuto_ ? " (auto)" : " (manual)")
              << ", GPU " << streamingGpuBudgetMiB_ << " MiB\n";
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

        const bool streamingRuntime = pageStreamer_ != nullptr;
        if (graph_ || streamingRuntime) {
            const auto continuousLod = lodController_.ContinuousLevel(camera_.ZoomFactor());
            lodLevel = static_cast<std::int32_t>(std::lround(continuousLod));
            const auto& normalRoadTint =
                roadStyleConfig_.enabled ? kRoadStyleNeutralTint : kDefaultRoadColor;

            if (streamingRuntime) {
                UpdateStreamingRuntime(width, height, continuousLod, deltaSeconds);
            }
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

            if (streamingRuntime) {
                const auto persistentKind =
                    rangeFilterStrategyManager_.Current().PersistentStreamingKind();
                auto filterKind = gpu::streaming::PersistentRangeFilterKind::FullScan;
                if (persistentKind ==
                    gpu::filtering::PersistentStreamingRangeFilterKind::BirthOrdered) {
                    filterKind = gpu::streaming::PersistentRangeFilterKind::BirthOrdered;
                }

                auto streamingRefinement = gpuRefinement;
                // Refinement decisions are committed with the root snapshot, not driven directly
                // by every intermediate camera wheel frame. The live camera remains perfectly
                // smooth while Adaptive continues from the last committed frontier in background.
                if (streamingCommittedRefinementPixelScale_ > 0.0f) {
                    streamingRefinement.screenPixelScale = streamingCommittedRefinementPixelScale_;
                }
                if (streamingRefinement.mode == geometry::RefinementMode::Adaptive &&
                    (!pageStreamer_->Index().HasRootPayload() ||
                     pageStreamer_->Index().edgeSpatialBoundsCount !=
                         pageStreamer_->Index().edgeCount)) {
                    streamingRefinement.mode = geometry::RefinementMode::None;
                }
                const auto displayLod = streamingDisplayLodFloat_ >= 0.0f
                                            ? streamingDisplayLodFloat_
                                            : static_cast<float>(lodLevel);
                persistentStreamingPipeline_.Process(
                    displayLod, filterKind, streamingRefinement, BuildRefinementGuardWindow());
                // Poll the asynchronous GPU readback ring without stalling. A fresh result
                // replaces the CPU backing working set, including when it is empty; a not-yet-ready
                // readback leaves the previous work alone. This prevents old camera generations
                // from accumulating tens of thousands of queued refinement blocks.
                auto missingBlocks = persistentStreamingPipeline_.ReadMissingBackingBlocks();
                if (persistentStreamingPipeline_.LastBackingReadbackWasFresh()) {
                    // GPU backing capacity is the natural admission window: queuing more blocks
                    // than can physically reside cannot make traversal complete sooner and was the
                    // main reason Full DFS could accumulate 30K+ obsolete requests. As resident
                    // blocks satisfy faults, later readbacks automatically admit the next batch.
                    const auto backingCapacity =
                        persistentStreamingPipeline_.Stats().gpuBackingBlockCapacity;
                    if (backingCapacity != 0u && missingBlocks.size() > backingCapacity) {
                        missingBlocks.resize(backingCapacity);
                    }
                    if (streamingRefinement.mode == geometry::RefinementMode::Adaptive &&
                        missingBlocks.size() > kAdaptiveCpuFaultWaveLimit) {
                        missingBlocks.resize(kAdaptiveCpuFaultWaveLimit);
                    }
                    pageStreamer_->SetBackingBlockDemand(missingBlocks);
                }
                persistentStreamingPipeline_.Draw(camera_, width, height, normalRoadTint, roadStyleConfig_);
            } else {
                switch (processingMode_) {
                case pipeline::ProcessingMode::GPUDriven:
                gpuResult = gpuPipeline_.Process(roadRenderer_.EdgeBufferId(),
                                                 roadRenderer_.EdgeCount(), lodLevel,
                                                 rangeFilterStrategyManager_.Current(),
                                                 strategyManager_.Current(), gpuRefinement);
                roadRenderer_.Draw(camera_, width, height, ToRoadDrawData(*gpuResult), 1.0f, 0.0f,
                                   normalRoadTint, roadStyleConfig_);
                break;

                case pipeline::ProcessingMode::CPUReference:
                cpuResult = cpuPipeline_.Process(*graph_, lodLevel, cpuRefinement);
                if (!cpuResult->stats.cacheHit || cpuDrawData_.edgeIdBuffer == 0) {
                    cpuDrawData_ = roadRenderer_.UploadEdgeIds(cpuResult->edgeIds);
                }
                roadRenderer_.Draw(camera_, width, height, cpuDrawData_, 1.0f, 0.0f,
                                   normalRoadTint, roadStyleConfig_);
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
                                               kCPURoadColor, roadStyleConfig_);
                    roadRenderer_.DrawViewport(camera_, leftWidth, 0, rightWidth, height,
                                               ToRoadDrawData(*gpuResult), kGPURoadColor, roadStyleConfig_);
                } else {
                    roadRenderer_.Draw(camera_, width, height, cpuDrawData_, 1.0f, 0.0f,
                                       kCPUOverlayColor, roadStyleConfig_);
                    roadRenderer_.Draw(camera_, width, height, ToRoadDrawData(*gpuResult), 1.0f,
                                       0.0f, kGPUOverlayColor, roadStyleConfig_);
                }
                break;
                }
            }
        }

        const auto gpuStats = gpuPipeline_.Stats();
        auto streamingStats = pageStreamer_ ? std::optional(pageStreamer_->Snapshot())
                                             : std::nullopt;
        if (streamingStats) {
            const auto persistent = persistentStreamingPipeline_.Stats();
            streamingStats->targetLodLevel = lodLevel;
            streamingStats->displayLodLevel = streamingDisplayLod_;
            streamingStats->targetLodLevelFloat = lodController_.ContinuousLevel(camera_.ZoomFactor());
            streamingStats->displayLodLevelFloat = streamingDisplayLodFloat_;
            streamingStats->lodPrefetchMinLevel = streamingPrefetchMinLod_;
            streamingStats->lodPrefetchMaxLevel = streamingPrefetchMaxLod_;
            streamingStats->lodPrefetchFocusLevel = streamingPrefetchFocusLod_;
            streamingStats->lodTransitionPending = streamingLodTransitionPending_;
            streamingStats->gpuBudgetBytes = persistent.gpuBudgetBytes;
            streamingStats->gpuAllocatedBytes = persistent.gpuAllocatedBytes;
            streamingStats->gpuRootRecordCapacity = persistent.gpuRootRecordCapacity;
            {
                const auto required = pageStreamer_->RequiredPageIds();
                std::vector<std::uint32_t> transition = required;
                transition.insert(transition.end(), streamingCommittedRequiredPages_.begin(),
                                  streamingCommittedRequiredPages_.end());
                std::sort(transition.begin(), transition.end());
                transition.erase(std::unique(transition.begin(), transition.end()),
                                 transition.end());
                const auto countRoots = [this](std::span<const std::uint32_t> ids) {
                    std::uint64_t count = 0;
                    const auto& pages = pageStreamer_->Index().graphPages;
                    for (const auto id : ids) {
                        if (id < pages.size()) {
                            count += pages[id].edgeCount;
                        }
                    }
                    return count;
                };
                streamingStats->gpuRequiredRootRecords = countRoots(required);
                streamingStats->gpuTransitionRootRecords = countRoots(transition);
                streamingStats->gpuRootWorkingSetFits =
                    persistent.gpuRootRecordCapacity != 0u &&
                    streamingStats->gpuRequiredRootRecords <= persistent.gpuRootRecordCapacity;
            }
            streamingStats->gpuPersistentRootPagesCached = persistent.gpuRootPagesCached;
            streamingStats->gpuPersistentRootPagesActive = persistent.gpuRootPagesActive;
            streamingStats->gpuPersistentBackingBlocks = persistent.gpuBackingBlocksResident;
            streamingStats->gpuPersistentBackingBlockCapacity =
                persistent.gpuBackingBlockCapacity;
            streamingStats->gpuVisibleRootCount = persistent.visibleRootCount;
            streamingStats->gpuDrawEdgeCount = persistent.drawEdgeCount;
            streamingStats->gpuDrawCapacity = persistent.drawCapacity;
            streamingStats->gpuDrawOverflowCount = persistent.drawOverflowCount;
            streamingStats->gpuMissingBlockRequests = persistent.missingBlockRequestCount;
            streamingStats->gpuRefinementCacheHits = persistent.refinementCacheHits;
            streamingStats->gpuRefinementCacheMisses = persistent.refinementCacheMisses;
            streamingStats->gpuCachedRefinedRoots = persistent.cachedRefinedRoots;
            streamingStats->gpuRefinementStackOverflows =
                persistent.refinementStackOverflows;
            streamingStats->gpuRefinementHashOverflows = persistent.refinementHashOverflows;
            streamingStats->gpuAdaptiveBlockRequestsAdmitted =
                persistent.adaptiveBlockRequestsAdmitted;
            streamingStats->gpuRefinementGeometryUsed = persistent.refinementGeometryUsed;
            streamingStats->gpuRefinementGeometryCapacity = persistent.refinementGeometryCapacity;
            streamingStats->gpuRefinementGeometryOverflows =
                persistent.refinementGeometryOverflows;
            streamingStats->gpuRefinementWriteBank = persistent.refinementWriteBank;
            streamingStats->gpuRefinementBanksUsed = persistent.refinementBanksUsed;
            streamingStats->gpuRefinementBankRecycles = persistent.refinementBankRecycles;
            streamingStats->gpuRootPageEvictions = persistent.rootPageEvictions;
            streamingStats->gpuRootCacheAllocationFailures =
                persistent.rootCacheAllocationFailures;
            streamingStats->gpuBackingBlockEvictions = persistent.backingBlockEvictions;
            streamingStats->gpuBackingBlockFirstUploads = persistent.backingBlockFirstUploads;
            streamingStats->gpuBackingBlockReuploadsAfterEviction =
                persistent.backingBlockReuploadsAfterEviction;
            streamingStats->gpuBackingGraceFallbackEvictions =
                persistent.backingGraceFallbackEvictions;
            streamingStats->gpuBackingCacheAllocationFailures =
                persistent.backingCacheAllocationFailures;
            streamingStats->gpuBackingBlocksTouchedLastReadback =
                persistent.backingBlocksTouchedLastReadback;
            streamingStats->gpuIncrementalRootBytesUploaded =
                persistent.incrementalRootBytesUploaded;
            streamingStats->gpuIncrementalBackingBytesUploaded =
                persistent.incrementalBackingBytesUploaded;
            streamingStats->lastGpuRootUploadMs = persistent.lastRootUploadMs;
            streamingStats->lastGpuBackingUploadMs = persistent.lastBackingUploadMs;
            streamingStats->gpuFilterCullMs = persistent.gpuFilterCullMs;
            streamingStats->gpuRefinementMs = persistent.gpuRefinementMs;
            streamingStats->gpuComposeMs = persistent.gpuComposeMs;
        }
        const auto datasetLoad = datasetLoader_.Snapshot();
        const auto actions = debugUI_.Draw(
            graph_.get(), rangeFilterStrategyManager_, strategyManager_,
            (graph_ || pageStreamer_) ? &camera_ : nullptr,
            (graph_ || pageStreamer_) ? &lodController_ : nullptr, datasetCatalog_, datasetLoad,
            processingMode_,
            cpuGeometryRefinement_, maxScreenErrorPixels_,
            cpuResult ? &cpuResult->stats : nullptr,
            (graph_ || pageStreamer_) ? &gpuStats : nullptr,
            gpuRangeBenchmarkResult_ ? &*gpuRangeBenchmarkResult_ : nullptr,
            validationResult_ ? &*validationResult_ : nullptr,
            processingMode_ == pipeline::ProcessingMode::Validation && cpuResult && gpuResult,
            pageStreamer_ ? &pageStreamer_->Index() : nullptr,
            streamingStats ? &*streamingStats : nullptr, roadStyleConfig_);
        debugUI_.EndFrame();

        if (actions.runGpuRangeBenchmark && graph_) {
            gpuRangeBenchmarkResult_ = gpuPipeline_.RunRangeFilterBenchmark(
                roadRenderer_.EdgeBufferId(), roadRenderer_.EdgeCount(), lodLevel,
                rangeFilterStrategyManager_.Current());

            const auto& benchmark = *gpuRangeBenchmarkResult_;
            std::cout << "\n[GPU Range Filter Benchmark]\n"
                      << "  Filter: " << benchmark.filterStrategy << '\n'
                      << "  LOD: " << benchmark.lodLevel << '\n';
            if (benchmark.candidateEdgeCount) {
                std::cout << "  Candidate edges: " << *benchmark.candidateEdgeCount << '\n';
            }
            std::cout << "  Alive edges: " << benchmark.aliveEdgeCount << '\n'
                      << "  Warm-up iterations: " << benchmark.warmupIterations << '\n'
                      << "  Measured iterations: " << benchmark.measuredIterations << '\n'
                      << "  Batch size: " << benchmark.batchSize << '\n'
                      << std::fixed << std::setprecision(6)
                      << "  Mean: " << benchmark.meanMs << " ms\n"
                      << "  Median: " << benchmark.medianMs << " ms\n"
                      << "  Min: " << benchmark.minMs << " ms\n"
                      << "  Max: " << benchmark.maxMs << " ms\n"
                      << "  P95: " << benchmark.p95Ms << " ms\n"
                      << "  P99: " << benchmark.p99Ms << " ms\n"
                      << "  StdDev: " << benchmark.stdDevMs << " ms\n\n"
                      << std::defaultfloat;
        }

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
            LoadDataset(dataset.graphPath, dataset.rangesPath, actions.datasetStreaming);
        }

        if (actions.applyStreamingBudgets && pageStreamer_) {
            ApplyStreamingBudgets(actions.streamingRamBudgetMiB,
                                  actions.streamingGpuBudgetMiB);
        }

        window_.SwapBuffers();

        if (auto streamingDataset = datasetLoader_.TakeStreamingCompleted()) {
            InstallStreamingDataset(std::move(*streamingDataset));
        }
        if (auto loadedGraph = datasetLoader_.TakeCompleted()) {
            pageStreamer_.reset();
            persistentStreamingPipeline_.Reset();
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
