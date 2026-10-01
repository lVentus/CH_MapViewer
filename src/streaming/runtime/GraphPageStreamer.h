#pragma once

#include "streaming/index/CHIndex.h"
#include "streaming/analysis/TextSourceScanner.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace chmv::streaming::runtime {

// Refinement backing uses a finer logical residency tile than the 16K text-source blocks.
// The physical .sch block layout stays unchanged; this only controls runtime fault/cache granularity.
inline constexpr std::uint32_t kRuntimeRefinementTileTargetRecords = 1024u;

[[nodiscard]] inline constexpr std::uint32_t RuntimeRefinementTileRecordCount(
    std::uint32_t sourceBlockRecordCount) {
    if (sourceBlockRecordCount == 0u) {
        return 0u;
    }
    std::uint32_t tile = sourceBlockRecordCount < kRuntimeRefinementTileTargetRecords
                             ? sourceBlockRecordCount
                             : kRuntimeRefinementTileTargetRecords;
    // Keep logical tiles aligned to physical source blocks even for non-default datasets.
    while (tile > 1u && sourceBlockRecordCount % tile != 0u) {
        tile >>= 1u;
    }
    return tile;
}

[[nodiscard]] inline constexpr std::uint32_t RuntimeRefinementTileCount(
    std::uint64_t edgeCount, std::uint32_t sourceBlockRecordCount) {
    const auto tile = RuntimeRefinementTileRecordCount(sourceBlockRecordCount);
    if (tile == 0u) {
        return 0u;
    }
    return static_cast<std::uint32_t>((edgeCount + tile - 1u) / tile);
}

struct StreamedNode {
    std::uint32_t globalNodeId = 0;
    float x = 0.0f;
    float y = 0.0f;
};

struct StreamedEdge {
    std::uint32_t globalEdgeId = 0;
    std::uint32_t sourceLocal = 0;
    std::uint32_t targetLocal = 0;
    std::uint32_t childA = 0;
    std::uint32_t childB = 0;
    std::uint32_t roadStyleType = 0;
    // Conservative world/projection-space error for this edge's entire shortcut geometry.
    // Zero for original/non-shortcut edges. Used by out-of-core Adaptive DFS on the GPU.
    float geometryError = 0.0f;
    std::int32_t birthLevel = -1;
    std::int32_t deathLevel = -1;
};

struct ResidentGraphPage {
    std::uint32_t pageId = 0;
    // CHIDX v6 fast path: GPU-ready roots are one contiguous binary read and need neither
    // .sch/.ranges parsing nor decoded endpoint-node blocks. Legacy indices still use nodes/edges.
    std::vector<index::CHIndexRootRecord> rootRecords;
    std::vector<StreamedNode> nodes;
    std::vector<StreamedEdge> edges;
    std::uint64_t memoryBytes = 0;
    std::uint64_t estimatedSourceBytesRead = 0;
    std::uint32_t edgeSourceBlocksRead = 0;
    std::uint32_t nodeSourceBlocksRead = 0;
    double loadMilliseconds = 0.0;
};

// Geometry refinement does not follow drawable-page membership. A shortcut child can be a
// non-drawable backing edge. Runtime refinement residency is keyed by 1K-ish logical tiles;
// physical 16K source blocks remain only an on-disk text indexing unit.
struct ResidentRefinementBlock {
    std::uint32_t blockId = 0;
    std::vector<StreamedNode> nodes;
    std::vector<StreamedEdge> edges;
    std::uint64_t memoryBytes = 0;
    std::uint64_t estimatedSourceBytesRead = 0;
    std::uint32_t nodeSourceBlocksRead = 0;
    double loadMilliseconds = 0.0;
};

struct SystemMemoryInfo {
    std::uint64_t totalPhysicalBytes = 0;
    std::uint64_t availablePhysicalBytes = 0;
};

[[nodiscard]] SystemMemoryInfo QuerySystemMemoryInfo();
[[nodiscard]] std::uint64_t RecommendGraphPageCacheBudget(const SystemMemoryInfo& memory);

struct GraphPageStreamerConfig {
    std::uint64_t ramBudgetBytes = 0;
    // Decoded spatial node blocks are shared across pages. Zero chooses an automatic
    // bounded cache (about 1/5 of the total RAM budget, up to 4 GiB).
    std::uint64_t nodeBlockCacheBudgetBytes = 0;
    // Spatial prefetch is measured in fixed base-grid cells. Radius 3 means the viewport
    // plus a three-cell border, i.e. a 7x7 neighborhood when the viewport occupies one cell.
    std::uint32_t spatialPrefetchRadius = 3;
    // LOD prefetch is independent from the 7x7 spatial guard. The application supplies a
    // predicted LOD interval; nearby future levels get a one-cell spatial guard while farther
    // levels preload only the strict viewport. This keeps a wide LOD window affordable.
    std::uint32_t lodPrefetchNearSpatialRadius = 1;
    std::uint32_t lodPrefetchNearDistance = 2;
    // Wide LOD prediction is primarily a RAM cache. Only levels close to either the visible
    // LOD or the predicted focus are proactively uploaded into the bounded GPU root cache.
    std::uint32_t lodPrefetchGpuWarmDistance = 2;
    // Lower-LOD drawable pages are not guessed for geometry refinement. Actual shortcut
    // misses request preprocessed edge source blocks instead.
    bool prefetchLodChildren = false;
    // Zero selects an automatic count that leaves two hardware threads for the UI/GPU driver.
    std::uint32_t workerCount = 0;
    double softBudgetRatio = 0.93;
};

struct StreamingViewRequest {
    // Integer level is retained for compatibility/diagnostics. lodLevelFloat is authoritative
    // for smooth rendering; floor/ceil are kept resident simultaneously during transitions.
    std::uint32_t lodLevel = 0;
    float lodLevelFloat = -1.0f;
    std::uint32_t prefetchMinLod = 0xffffffffu;
    std::uint32_t prefetchMaxLod = 0xffffffffu;
    float prefetchFocusLod = -1.0f;
    double minLatitude = 0.0;
    double minLongitude = 0.0;
    double maxLatitude = 0.0;
    double maxLongitude = 0.0;
};

struct StreamingSpatialWindow {
    std::uint32_t minX = 0;
    std::uint32_t minY = 0;
    std::uint32_t maxX = 0;
    std::uint32_t maxY = 0;
    bool operator==(const StreamingSpatialWindow&) const = default;
};

struct RefinementEdgeRequest {
    std::uint32_t parentEdgeId = 0;
    std::uint32_t childEdgeId = 0;
    bool operator==(const RefinementEdgeRequest&) const = default;
};

struct StreamingViewToken {
    std::uint64_t generation = 0;
    std::uint32_t requiredPageCount = 0;
    std::uint32_t requiredCacheHits = 0;
    std::uint32_t requiredCacheMisses = 0;
    std::uint32_t prefetchedPageCount = 0;
};

struct RefinementRequestToken {
    std::uint64_t generation = 0;
    std::uint32_t requestedEdgeCount = 0;
    std::uint32_t uniqueBlockCount = 0;
    std::uint32_t residentBlockHits = 0;
    std::uint32_t queuedBlockCount = 0;
    std::uint32_t spatiallyRejectedRequestCount = 0;
};

struct GraphPageStreamingStats {
    // Total CPU-side streaming cache budget. This includes graph/refinement page data
    // plus the decoded-node cache; the two pools are partitioned so the sum is hard-bounded.
    std::uint64_t ramBudgetBytes = 0;
    std::uint64_t dataCacheBudgetBytes = 0;
    std::uint64_t softBudgetBytes = 0;
    std::uint64_t residentBytes = 0;
    std::uint64_t totalCpuCacheBytes = 0;
    std::uint64_t peakCpuCacheBytes = 0;
    std::uint64_t reservedLoadBytes = 0;
    std::uint64_t pinnedBytes = 0;
    std::uint32_t residentPageCount = 0;
    std::uint32_t queuedPageCount = 0;
    std::uint32_t loadingPageCount = 0;
    std::uint32_t desiredPageCount = 0;
    std::uint32_t desiredResidentCount = 0;
    std::uint32_t requiredPageCount = 0;
    std::uint32_t requiredResidentCount = 0;
    std::uint32_t spatialPrefetchRadius = 0;
    std::uint32_t prefetchReferenceSpatialLevel = 0;
    double spatialCellSizeMeters = 0.0;
    std::uint32_t lodPrefetchPageCount = 0;
    std::uint32_t lodGpuWarmPageCount = 0;
    std::uint32_t lodPrefetchMinLevel = 0;
    std::uint32_t lodPrefetchMaxLevel = 0;
    float lodPrefetchFocusLevel = 0.0f;
    std::int32_t targetLodLevel = -1;
    std::int32_t displayLodLevel = -1;
    float targetLodLevelFloat = -1.0f;
    float displayLodLevelFloat = -1.0f;
    bool lodTransitionPending = false;
    std::uint32_t desiredRefinementBlockCount = 0;
    std::uint32_t residentRefinementBlockCount = 0;
    std::uint32_t queuedRefinementBlockCount = 0;
    std::uint32_t loadingRefinementBlockCount = 0;
    std::uint64_t totalRefinementRequests = 0;
    std::uint64_t totalSpatiallyRejectedRefinementRequests = 0;
    std::uint32_t lastRefinementRequestCount = 0;
    std::uint32_t lastSpatiallyAcceptedRefinementRequestCount = 0;
    std::uint64_t totalRefinementBlockLoads = 0;
    std::uint64_t totalRefinementBlockEvictions = 0;
    std::uint64_t totalPageLoads = 0;
    std::uint64_t totalEvictions = 0;
    std::uint64_t pageCacheHits = 0;
    std::uint64_t pageCacheMisses = 0;
    std::uint64_t refinementBlockCacheHits = 0;
    std::uint64_t refinementBlockCacheMisses = 0;
    std::uint64_t budgetRejectedLoads = 0;
    std::uint64_t estimatedSourceBytesRead = 0;
    std::uint64_t edgeSourceBlocksRead = 0;
    std::uint64_t nodeSourceBlocksRead = 0;
    std::uint64_t nodeBlockCacheHits = 0;
    std::uint64_t nodeBlockCacheMisses = 0;
    std::uint64_t nodeBlockCacheEvictions = 0;
    std::uint64_t nodeBlockCacheBytes = 0;
    std::uint64_t nodeBlockCacheBudgetBytes = 0;
    double lastPageLoadMs = 0.0;
    double meanPageLoadMs = 0.0;
    double maxPageLoadMs = 0.0;
    double lastRootPlannerMs = 0.0;
    std::uint64_t rootPlannerRebuilds = 0;
    std::uint64_t rootPlannerCacheReuses = 0;
    std::uint32_t rootPlannerAliveCandidates = 0;
    std::uint32_t rootPlannerSpatialCandidates = 0;
    bool rootPlannerUsedLodFirst = false;
    double lastRefinementBlockLoadMs = 0.0;
    double meanRefinementBlockLoadMs = 0.0;
    double lastGpuWorkingSetBuildMs = 0.0;
    std::uint32_t gpuWorkingSetEdgeCount = 0;
    std::uint32_t gpuWorkingSetRootPageCount = 0;
    std::uint32_t gpuWorkingSetRefinementBlockCount = 0;
    std::uint64_t gpuBudgetBytes = 0;
    std::uint64_t gpuAllocatedBytes = 0;
    std::uint32_t gpuRootRecordCapacity = 0;
    std::uint64_t gpuRequiredRootRecords = 0;
    std::uint64_t gpuTransitionRootRecords = 0;
    bool gpuRootWorkingSetFits = true;
    std::uint32_t gpuPersistentRootPagesCached = 0;
    std::uint32_t gpuPersistentRootPagesActive = 0;
    std::uint32_t gpuPersistentBackingBlocks = 0;
    std::uint32_t gpuPersistentBackingBlockCapacity = 0;
    std::uint32_t gpuVisibleRootCount = 0;
    std::uint32_t gpuDrawEdgeCount = 0;
    std::uint32_t gpuDrawCapacity = 0;
    std::uint32_t gpuDrawOverflowCount = 0;
    std::uint32_t gpuMissingBlockRequests = 0;
    std::uint32_t gpuRefinementCacheHits = 0;
    std::uint32_t gpuRefinementCacheMisses = 0;
    std::uint32_t gpuCachedRefinedRoots = 0;
    std::uint32_t gpuRefinementStackOverflows = 0;
    std::uint32_t gpuRefinementHashOverflows = 0;
    std::uint32_t gpuAdaptiveBlockRequestsAdmitted = 0;
    std::uint32_t gpuRefinementGeometryUsed = 0;
    std::uint32_t gpuRefinementGeometryCapacity = 0;
    std::uint32_t gpuRefinementGeometryOverflows = 0;
    std::uint32_t gpuRefinementWriteBank = 0;
    std::uint32_t gpuRefinementBanksUsed = 0;
    std::uint64_t gpuRefinementBankRecycles = 0;
    std::uint64_t gpuRootPageEvictions = 0;
    std::uint64_t gpuRootCacheAllocationFailures = 0;
    std::uint64_t gpuBackingBlockEvictions = 0;
    std::uint64_t gpuBackingCacheAllocationFailures = 0;
    std::uint32_t gpuBackingBlocksTouchedLastReadback = 0;
    std::uint64_t gpuIncrementalRootBytesUploaded = 0;
    std::uint64_t gpuIncrementalBackingBytesUploaded = 0;
    double lastGpuRootUploadMs = 0.0;
    double lastGpuBackingUploadMs = 0.0;
    double gpuFilterCullMs = 0.0;
    double gpuRefinementMs = 0.0;
    double gpuComposeMs = 0.0;
    double currentDemandWaitMs = 0.0;
    double lastDemandReadyMs = 0.0;
    bool currentDemandReady = true;
    std::string lastError;
};

class GraphPageStreamer {
public:
    GraphPageStreamer(index::CHIndexData index,
                      std::filesystem::path graphPath,
                      std::filesystem::path rangesPath,
                      GraphPageStreamerConfig config = {});
    ~GraphPageStreamer();

    GraphPageStreamer(const GraphPageStreamer&) = delete;
    GraphPageStreamer& operator=(const GraphPageStreamer&) = delete;

    [[nodiscard]] StreamingViewToken RequestView(const StreamingViewRequest& view);
    // Non-blocking latest-wins planner submission. Expensive spatial/LOD selection runs on a
    // dedicated planner thread; the render thread only posts the newest camera demand.
    [[nodiscard]] std::uint64_t SubmitView(const StreamingViewRequest& view);
    [[nodiscard]] std::uint64_t LatestAppliedViewSequence() const {
        return latestAppliedViewSequence_.load(std::memory_order_acquire);
    }
    [[nodiscard]] float LatestAppliedViewLodFloat() const {
        return latestAppliedViewLodFloat_.load(std::memory_order_acquire);
    }
    [[nodiscard]] RefinementRequestToken RequestRefinementEdges(
        std::span<const RefinementEdgeRequest> requests);
    void RequestBackingBlocks(std::span<const std::uint32_t> blockIds);
    // Replaces the current GPU page-fault working set instead of accumulating requests forever.
    // Use this for per-frame persistent-GPU refinement demand; stale queued requests are abandoned
    // automatically while already-running I/O is allowed to finish and is discarded if no longer wanted.
    void SetBackingBlockDemand(std::span<const std::uint32_t> blockIds);
    void ReleaseBackingBlocks(std::span<const std::uint32_t> blockIds);
    [[nodiscard]] bool WaitForView(std::uint64_t generation,
                                   std::chrono::milliseconds timeout);
    [[nodiscard]] GraphPageStreamingStats Snapshot() const;
    [[nodiscard]] std::vector<std::uint32_t> RequiredPageIds() const;
    [[nodiscard]] std::vector<std::uint32_t> DesiredPageIds() const;
    [[nodiscard]] std::vector<std::uint32_t> LodPrefetchPageIds() const;
    [[nodiscard]] std::vector<std::uint32_t> GpuWarmPageIds() const;
    [[nodiscard]] std::vector<std::uint32_t> ResidentRefinementBlockIds() const;
    [[nodiscard]] std::shared_ptr<const ResidentGraphPage> FindResidentPage(
        std::uint32_t pageId) const;
    [[nodiscard]] std::shared_ptr<const ResidentRefinementBlock> FindResidentRefinementBlock(
        std::uint32_t blockId) const;
    void ClearRefinement();
    void ClearUnpinned();

    // Repartitions the existing bounded caches without touching the preprocessed dataset.
    // ramBudgetBytes is the hard total for page/refinement data + decoded-node blocks.
    // nodeBlockCacheBudgetBytes == 0 selects the normal automatic fraction.
    void ReconfigureBudgets(std::uint64_t ramBudgetBytes,
                            std::uint64_t nodeBlockCacheBudgetBytes = 0);

    [[nodiscard]] const index::CHIndexData& Index() const { return index_; }
    [[nodiscard]] StreamingSpatialWindow CurrentViewportWindow() const;
    [[nodiscard]] StreamingSpatialWindow CurrentRefinementWindow() const;

private:
    enum class LoadKind : std::uint8_t {
        GraphPage,
        RefinementBlock,
    };

    struct DecodedNodeBlock {
        std::uint32_t firstRecord = 0;
        std::vector<std::array<float, 2>> positions;
        std::uint64_t memoryBytes = 0;
        std::uint64_t sourceBytesRead = 0;
    };

    struct NodeBlockAccess {
        std::shared_ptr<const DecodedNodeBlock> block;
        bool loadedFromSource = false;
    };

    struct NodeBlockCacheEntry {
        std::shared_ptr<const DecodedNodeBlock> block;
        std::uint64_t lastUse = 0;
    };


    struct PendingViewRequest {
        StreamingViewRequest view{};
        std::uint64_t sequence = 0;
    };

    struct LoadRequest {
        LoadKind kind = LoadKind::GraphPage;
        std::uint32_t id = 0;
        std::uint32_t priority = 0;
        std::uint64_t generation = 0;
        std::uint64_t sequence = 0;
    };

    struct LoadRequestCompare {
        bool operator()(const LoadRequest& a, const LoadRequest& b) const {
            if (a.priority != b.priority) {
                return a.priority > b.priority;
            }
            return a.sequence > b.sequence;
        }
    };

    struct QueuedPageState {
        std::uint32_t priority = 0;
        std::uint64_t generation = 0;
        std::uint64_t sequence = 0;
    };

    struct CacheEntry {
        std::shared_ptr<ResidentGraphPage> page;
        std::uint64_t lastUse = 0;
        bool pinned = false;
    };

    struct RefinementCacheEntry {
        std::shared_ptr<ResidentRefinementBlock> block;
        std::uint64_t lastUse = 0;
    };


    struct PageSpatialBounds {
        std::uint32_t minX = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t minY = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxX = 0;
        std::uint32_t maxY = 0;
        bool valid = false;
    };

    struct PageSelection {
        std::uint32_t pageId = 0;
        std::uint32_t spatialRadius = 0;
        std::uint32_t lodDistance = 0;
        std::uint32_t priority = 0;
        bool required = false;
        bool renderCandidate = false;
        bool gpuWarm = false;
    };

    struct PageQueryResult {
        std::vector<PageSelection> pages;
        std::uint32_t referenceSpatialLevel = 0;
        std::uint32_t selectedPrefetchMinLod = 0;
        std::uint32_t selectedPrefetchMaxLod = 0;
        StreamingSpatialWindow viewportWindow{};
        StreamingSpatialWindow refinementWindow{};
    };

    struct PageQueryCacheKey {
        std::uint32_t minViewX = 0;
        std::uint32_t minViewY = 0;
        std::uint32_t maxViewX = 0;
        std::uint32_t maxViewY = 0;
        std::uint32_t renderLow = 0;
        std::uint32_t renderHigh = 0;
        std::uint32_t prefetchMin = 0;
        std::uint32_t prefetchMax = 0;
        std::int32_t focusHalfLevel = 0;
        std::uint64_t softBudgetBytes = 0;
        bool operator==(const PageQueryCacheKey&) const = default;
    };

    [[nodiscard]] const PageQueryResult& QueryPages(const StreamingViewRequest& view) const;
    void QueuePageLocked(std::uint32_t pageId, std::uint32_t priority,
                         std::uint64_t generation);
    void PumpPageQueueLocked(std::span<const PageSelection> selectedPages,
                             std::uint64_t generation);
    void QueueRefinementBlockLocked(std::uint32_t blockId, std::uint64_t generation);
    void PlannerWorker(std::stop_token stopToken);
    void Worker(std::stop_token stopToken);
    [[nodiscard]] ResidentGraphPage LoadPage(
        std::uint32_t pageId, std::ifstream& indexStream,
        analysis::detail::TextSourceScanner& graphScanner,
        analysis::detail::TextSourceScanner& rangeScanner);
    [[nodiscard]] ResidentRefinementBlock LoadRefinementBlock(
        std::uint32_t blockId, analysis::detail::TextSourceScanner& graphScanner);
    [[nodiscard]] std::uint64_t RefinementTileByteOffset(
        std::uint32_t tileId, analysis::detail::TextSourceScanner& graphScanner,
        bool& builtPhysicalOffsetTable);
    [[nodiscard]] NodeBlockAccess AcquireNodeBlock(
        std::uint32_t blockId, analysis::detail::TextSourceScanner& graphScanner);
    void EvictNodeBlocksLocked(std::uint64_t targetBytes);
    [[nodiscard]] bool ReserveForLoadLocked(std::uint64_t bytes, bool required);
    void ReleaseReservationLocked(std::uint64_t bytes);
    void EvictToLocked(std::uint64_t targetBytes, bool allowCurrentDemand = false);
    void RecomputeBudgetPartitionLocked();
    void UpdatePeakCpuCacheBytes();
    void UpdateDemandReadyLocked();
    [[nodiscard]] std::uint64_t EstimatePageBytes(std::uint32_t pageId) const;
    [[nodiscard]] std::uint64_t EstimateRefinementBlockBytes(std::uint32_t blockId) const;

    index::CHIndexData index_;
    // Root planning has two exact indices and chooses the cheaper one per query. At overview
    // scales only a handful of high-LOD tiles are alive, so iterating those tiles and testing their
    // precomputed spatial bounds is O(alive-pages), not O(all 10M+ spatial refs). At local/detail
    // scales the fixed-grid lookup remains cheaper and is evaluated once then masked by LOD.
    std::vector<PageSpatialBounds> pageSpatialBounds_;
    std::vector<std::uint64_t> lodAlivePageBits_;
    std::vector<std::uint32_t> lodAlivePageCounts_;
    std::size_t lodAliveWordsPerLevel_ = 0;
    mutable std::vector<std::uint32_t> spatialCandidateStamp_;
    mutable std::vector<std::uint8_t> spatialCandidateRadius_;
    mutable std::vector<std::uint32_t> selectionStamp_;
    mutable std::vector<PageSelection> selectionScratch_;
    mutable std::uint32_t spatialQueryEpoch_ = 0;
    mutable std::uint32_t selectionEpoch_ = 0;
    mutable std::optional<PageQueryCacheKey> queryCacheKey_;
    mutable PageQueryResult queryCacheResult_;
    mutable std::uint64_t queryCacheRevision_ = 0;
    mutable std::atomic<std::uint32_t> lastPlannerAliveCandidates_{0};
    mutable std::atomic<std::uint32_t> lastPlannerSpatialCandidates_{0};
    mutable std::atomic<bool> lastPlannerUsedLodFirst_{false};
    std::uint64_t appliedQueryCacheRevision_ = std::numeric_limits<std::uint64_t>::max();
    std::filesystem::path graphPath_;
    std::filesystem::path rangesPath_;
    GraphPageStreamerConfig config_;
    // The decoded-node cache is a sub-budget of ramBudgetBytes. residentBytes_ and
    // reservedLoadBytes_ are constrained by dataCacheBudgetBytes_ so the combined
    // runtime cache cannot silently exceed the configured RAM budget.
    std::uint64_t dataCacheBudgetBytes_ = 0;
    std::uint64_t softBudgetBytes_ = 0;

    mutable std::mutex mutex_;
    std::condition_variable_any condition_;
    std::priority_queue<LoadRequest, std::vector<LoadRequest>, LoadRequestCompare> queue_;
    std::unordered_map<std::uint32_t, QueuedPageState> queuedPages_;
    std::unordered_set<std::uint32_t> loadingPages_;
    std::unordered_map<std::uint32_t, CacheEntry> cache_;
    std::unordered_set<std::uint32_t> desiredPages_;
    std::unordered_set<std::uint32_t> requiredPages_;
    std::unordered_set<std::uint32_t> lodPrefetchPages_;
    std::unordered_set<std::uint32_t> gpuWarmPages_;
    std::unordered_set<std::uint32_t> queuedRefinementBlocks_;
    std::unordered_set<std::uint32_t> loadingRefinementBlocks_;
    std::unordered_map<std::uint32_t, RefinementCacheEntry> refinementCache_;
    std::unordered_set<std::uint32_t> desiredRefinementBlocks_;
    std::uint64_t backingDemandGeneration_ = 0;

    // Byte offsets for logical refinement tiles are derived lazily from each physical 16K text
    // block once, then reused by every worker. This avoids reparsing 0..15K preceding records
    // every time a 1K tile is requested.
    mutable std::mutex refinementOffsetMutex_;
    std::condition_variable refinementOffsetCondition_;
    std::vector<std::uint64_t> refinementTileByteOffsets_;
    std::unordered_set<std::uint32_t> loadingRefinementOffsetBlocks_;

    // Fixed-grid boundary coordinates are cached once; per-edge Adaptive error evaluation then
    // becomes four point-to-segment distances instead of repeated WebMercator exp/atan work.
    std::vector<double> geometryBoundaryX_;
    std::vector<double> geometryBoundaryY_;

    StreamingSpatialWindow currentViewportWindow_{};
    StreamingSpatialWindow currentRefinementWindow_{};
    std::uint32_t currentPrefetchMinLod_ = 0;
    std::uint32_t currentPrefetchMaxLod_ = 0;
    float currentPrefetchFocusLod_ = 0.0f;

    mutable std::mutex nodeCacheMutex_;
    std::condition_variable nodeCacheCondition_;
    std::unordered_map<std::uint32_t, NodeBlockCacheEntry> nodeBlockCache_;
    std::unordered_set<std::uint32_t> loadingNodeBlocks_;
    std::uint64_t nodeBlockCacheBudgetBytes_ = 0;
    std::uint64_t nodeBlockCacheBytes_ = 0;
    std::uint64_t nodeBlockUseCounter_ = 0;
    std::uint64_t nodeBlockCacheHits_ = 0;
    std::uint64_t nodeBlockCacheMisses_ = 0;
    std::uint64_t nodeBlockCacheEvictions_ = 0;

    // Planner is intentionally separate from disk workers. QueryPages can touch millions of
    // index entries in a worst-case view; it must never run on the UI/render thread. Pending
    // requests are coalesced: while one plan is being built, newer camera samples overwrite the
    // pending slot and only the newest one is processed next.
    mutable std::mutex plannerMutex_;
    std::condition_variable_any plannerCondition_;
    std::optional<PendingViewRequest> pendingViewRequest_;
    std::jthread plannerWorker_;
    std::uint64_t submittedViewSequence_ = 0;
    std::atomic<std::uint64_t> latestAppliedViewSequence_{0};
    std::atomic<float> latestAppliedViewLodFloat_{-1.0f};

    std::vector<std::jthread> workers_;

    std::uint32_t prefetchReferenceSpatialLevel_ = 0;
    std::uint64_t generation_ = 0;
    std::uint64_t sequence_ = 0;
    std::uint64_t useCounter_ = 0;
    std::uint64_t residentBytes_ = 0;
    std::uint64_t reservedLoadBytes_ = 0;
    std::uint64_t pinnedBytes_ = 0;
    std::uint64_t totalRefinementRequests_ = 0;
    std::uint64_t totalSpatiallyRejectedRefinementRequests_ = 0;
    std::uint32_t lastRefinementRequestCount_ = 0;
    std::uint32_t lastSpatiallyAcceptedRefinementRequestCount_ = 0;
    std::uint64_t totalRefinementBlockLoads_ = 0;
    std::uint64_t totalRefinementBlockEvictions_ = 0;
    std::uint64_t totalPageLoads_ = 0;
    std::uint64_t totalEvictions_ = 0;
    std::uint64_t pageCacheHits_ = 0;
    std::uint64_t pageCacheMisses_ = 0;
    std::uint64_t refinementBlockCacheHits_ = 0;
    std::uint64_t refinementBlockCacheMisses_ = 0;
    std::uint64_t budgetRejectedLoads_ = 0;
    mutable std::atomic<std::uint64_t> peakCpuCacheBytes_{0};
    std::uint64_t estimatedSourceBytesRead_ = 0;
    std::uint64_t edgeSourceBlocksRead_ = 0;
    std::uint64_t nodeSourceBlocksRead_ = 0;
    double totalPageLoadMs_ = 0.0;
    double lastPageLoadMs_ = 0.0;
    double maxPageLoadMs_ = 0.0;
    mutable std::atomic<double> lastRootPlannerMs_{0.0};
    mutable std::atomic<std::uint64_t> rootPlannerRebuilds_{0};
    mutable std::atomic<std::uint64_t> rootPlannerCacheReuses_{0};
    double totalRefinementBlockLoadMs_ = 0.0;
    double lastRefinementBlockLoadMs_ = 0.0;
    double lastDemandReadyMs_ = 0.0;
    bool currentDemandReady_ = true;
    std::chrono::steady_clock::time_point demandStart_{};
    std::string lastError_;
};

} // namespace chmv::streaming::runtime
