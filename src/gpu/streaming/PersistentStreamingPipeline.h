#pragma once

#include "benchmark/GPUTimer.h"
#include "geometry/GeometryRefinement.h"
#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/GraphicsProgram.h"
#include "streaming/index/CHIndex.h"
#include "streaming/runtime/GraphPageStreamer.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace chmv::renderer {
class MapCamera2D;
struct RoadStyleConfig;
}

namespace chmv::gpu::streaming {

enum class PersistentRangeFilterKind : std::uint8_t {
    FullScan,
    BirthOrdered,
};

struct PersistentStreamingConfig {
    // Hard budget for persistent GPU streaming buffers. Capacities for root records,
    // backing blocks, refinement geometry, hash state and draw lists are derived from it.
    std::uint64_t gpuBudgetBytes = 384ull << 20;
    // Zero selects automatic partitioning within gpuBudgetBytes.
    std::uint64_t rootCacheBudgetBytes = 0;
    std::uint64_t refinementGeometryBudgetBytes = 0;
    std::uint64_t refinementHashBudgetBytes = 0;
};

struct PersistentStreamingStats {
    std::uint64_t gpuBudgetBytes = 0;
    std::uint64_t gpuAllocatedBytes = 0;
    std::uint32_t gpuRootRecordCapacity = 0;
    std::uint32_t gpuRootPagesCached = 0;
    std::uint32_t gpuRootPagesActive = 0;
    std::uint32_t gpuBackingBlocksResident = 0;
    std::uint32_t gpuBackingBlockCapacity = 0;
    std::uint32_t visibleRootCount = 0;
    std::uint32_t drawEdgeCount = 0;
    std::uint32_t drawCapacity = 0;
    std::uint32_t drawOverflowCount = 0;
    std::uint32_t missingBlockRequestCount = 0;
    std::uint32_t refinementCacheHits = 0;
    std::uint32_t refinementCacheMisses = 0;
    std::uint32_t cachedRefinedRoots = 0;
    std::uint32_t refinementStackOverflows = 0;
    std::uint32_t refinementHashOverflows = 0;
    std::uint32_t adaptiveBlockRequestsAdmitted = 0;
    std::uint32_t refinementGeometryUsed = 0;
    std::uint32_t refinementGeometryCapacity = 0;
    std::uint32_t refinementGeometryOverflows = 0;
    std::uint32_t refinementWriteBank = 0;
    std::uint32_t refinementBanksUsed = 0;
    std::uint64_t refinementBankRecycles = 0;
    std::uint64_t rootPageEvictions = 0;
    std::uint64_t rootCacheAllocationFailures = 0;
    std::uint64_t backingBlockEvictions = 0;
    std::uint64_t backingCacheAllocationFailures = 0;
    std::uint32_t backingBlocksTouchedLastReadback = 0;
    std::uint64_t incrementalRootBytesUploaded = 0;
    std::uint64_t incrementalBackingBytesUploaded = 0;
    double lastRootUploadMs = 0.0;
    double lastBackingUploadMs = 0.0;
    double gpuFilterCullMs = 0.0;
    double gpuRefinementMs = 0.0;
    double gpuComposeMs = 0.0;
};

// GPU-oriented out-of-core runtime.
//
// Root-page records and preprocessed edge blocks are uploaded once into persistent GPU caches.
// Camera movement only changes a small active-page descriptor list. LOD filtering, exact
// viewport culling, Full-DFS cache lookup/refinement and draw-list composition all stay on GPU.
// Missing backing blocks are reduced to one GPU bit per physical preprocessed edge block; CPU only
// consumes the compact block list to schedule disk I/O.
class PersistentStreamingPipeline {
public:
    explicit PersistentStreamingPipeline(const std::filesystem::path& shaderDirectory);
    ~PersistentStreamingPipeline();

    PersistentStreamingPipeline(const PersistentStreamingPipeline&) = delete;
    PersistentStreamingPipeline& operator=(const PersistentStreamingPipeline&) = delete;

    void Initialize(const chmv::streaming::index::CHIndexData& index,
                    PersistentStreamingConfig config = {});
    void Reset();

    [[nodiscard]] bool HasRootPage(std::uint32_t pageId) const;
    [[nodiscard]] bool UploadRootPage(
        const chmv::streaming::runtime::ResidentGraphPage& page,
        std::span<const std::uint64_t> edgeSpatialBounds);
    // Only strict viewport pages are protected from GPU root-cache eviction. The wider 7x7
    // residency set remains active for culling/prefetch but is replaceable under pressure.
    void SetProtectedRootPages(std::span<const std::uint32_t> pageIds);
    void SetActiveRootPages(std::span<const std::uint32_t> pageIds);

    [[nodiscard]] bool HasBackingBlock(std::uint32_t blockId) const;
    [[nodiscard]] bool UploadBackingBlock(
        const chmv::streaming::runtime::ResidentRefinementBlock& block);
    void SetPinnedBackingBlocks(std::span<const std::uint32_t> blockIds);

    // Runs the GPU-only per-frame work. refinementWindow is intentionally much smaller than
    // the 7x7 residency window (normally viewport + a small guard band).
    void Process(float lodLevel,
                 PersistentRangeFilterKind filterKind,
                 const geometry::RefinementParameters& refinement,
                 const chmv::streaming::runtime::StreamingSpatialWindow& refinementWindow);

    // Compact GPU page-fault output. Each block appears at most once because the shader writes
    // a block bitset rather than one request per shortcut edge.
    [[nodiscard]] std::vector<std::uint32_t> ReadMissingBackingBlocks();
    // True only when the most recent ReadMissingBackingBlocks() consumed at least one completed
    // asynchronous GPU readback. This lets the streamer replace stale backing demand on an empty
    // *fresh* fault set without cancelling work merely because the readback ring is not ready yet.
    [[nodiscard]] bool LastBackingReadbackWasFresh() const { return lastBackingReadbackWasFresh_; }

    void Draw(const chmv::renderer::MapCamera2D& camera,
              int framebufferWidth,
              int framebufferHeight,
              const std::array<float, 4>& color,
              const chmv::renderer::RoadStyleConfig& styles) const;

    [[nodiscard]] PersistentStreamingStats Stats() const { return stats_; }
    [[nodiscard]] std::uint32_t DrawEdgeBufferId() const { return drawEdgeBuffer_.Id(); }
    [[nodiscard]] std::uint32_t DrawCommandBufferId() const { return drawCommandBuffer_.Id(); }

private:
    struct RootPageAllocation {
        std::uint32_t offset = 0;
        std::uint32_t count = 0;
        std::uint64_t lastUse = 0;
    };

    struct FreeRange {
        std::uint32_t offset = 0;
        std::uint32_t count = 0;
    };

    struct BackingSlot {
        std::uint32_t blockId = 0xffffffffu;
        std::uint64_t lastUse = 0;
    };

    [[nodiscard]] std::uint32_t AllocateRootRange(std::uint32_t count);
    void FreeRootRange(std::uint32_t offset, std::uint32_t count);
    bool EvictOneInactiveRootPage();
    bool EvictOneUnpinnedBackingBlock();
    void SetPageTableEntry(std::uint32_t blockId, std::uint32_t slotPlusOne);
    void EnsureActiveDescriptorCapacity(std::size_t count);
    void ResetRefinementCache();
    void RecycleRefinementBank(std::uint32_t bank);
    void RotateRefinementBankIfNeeded();
    void FinalizeDrawCommand();
    void ScheduleReadback();
    [[nodiscard]] std::uint32_t RootRecordUpperBound() const;

    static constexpr std::uint32_t InvalidId = 0xffffffffu;
    static constexpr std::uint32_t LocalSize = 256u;
    static constexpr std::uint32_t RefineLocalSize = 64u;
    static constexpr std::uint32_t MaxRootRecordCapacity = 1u << 21u;
    static constexpr std::uint32_t MaxBackingBlockCapacity = 8192u;
    static constexpr std::uint32_t MaxAdaptiveBlockRequestsPerFrame = 128u;
    static constexpr std::uint32_t MaxRefinementHashCapacity = 1u << 20u;
    // Final Full-DFS geometry is cached directly as vec4 line segments. 4M segments = 64 MiB.
    // Saturation is graceful: existing cached roots remain valid and new roots fall back to the
    // shortcut instead of clearing the entire cache and causing visible refine/unrefine oscillation.
    static constexpr std::uint32_t MaxLeafArenaCapacity = 1u << 22u;
    static constexpr std::uint32_t RefinementBankCount = 8u;
    static constexpr std::uint32_t RefinementBankReuseDelayFrames = 8u;
    static constexpr std::size_t ReadbackRingSize = 3u;

    ComputeProgram fullScanFilterProgram_;
    ComputeProgram birthOrderedFilterProgram_;
    ComputeProgram prepareDispatchProgram_;
    ComputeProgram refineProgram_;
    ComputeProgram composeProgram_;
    ComputeProgram recycleBankProgram_;
    ComputeProgram finalizeDrawProgram_;
    GraphicsProgram roadProgram_;

    GPUBuffer rootRecordBuffer_;
    GPUBuffer activePageDescriptorBuffer_;
    GPUBuffer visibleRootBuffer_;
    GPUBuffer visibleRootCounterBuffer_;

    GPUBuffer blockEndpointBuffer_;
    GPUBuffer blockChildBuffer_;
    GPUBuffer blockGeometryErrorBuffer_;
    GPUBuffer blockRoadTypeBuffer_;
    GPUBuffer blockPageTableBuffer_;

    GPUBuffer refinementHashBuffer_;
    GPUBuffer leafArenaBuffer_;
    GPUBuffer leafRoadTypeBuffer_;
    GPUBuffer leafArenaCounterBuffer_;
    GPUBuffer refinementBankLastUseBuffer_;
    GPUBuffer refinementStatsBuffer_;
    GPUBuffer blockRequestBitsetBuffer_;
    GPUBuffer blockUseBitsetBuffer_;
    GPUBuffer dispatchBuffer_;

    GPUBuffer drawEdgeBuffer_;
    GPUBuffer drawAlphaBuffer_;
    GPUBuffer drawCounterBuffer_;
    GPUBuffer drawCommandBuffer_;

    std::array<GPUBuffer, ReadbackRingSize> readbackBuffers_;
    std::array<void*, ReadbackRingSize> readbackFences_{};
    std::array<bool, ReadbackRingSize> readbackPending_{};
    std::array<std::uint64_t, ReadbackRingSize> readbackSequence_{};
    std::size_t readbackSizeBytes_ = 0;
    std::size_t nextReadbackSlot_ = 0;
    std::uint64_t nextReadbackSequence_ = 1;

    benchmark::GPUTimer filterTimer_;
    benchmark::GPUTimer refinementTimer_;
    benchmark::GPUTimer composeTimer_;

    const chmv::streaming::index::CHIndexData* index_ = nullptr;
    PersistentStreamingConfig config_{};
    std::uint64_t allocatedGpuBytes_ = 0;
    std::uint32_t rootRecordCapacity_ = 0;
    std::uint32_t activeDescriptorCapacity_ = 0;
    std::uint32_t activePageCount_ = 0;
    std::uint32_t backingBlockCapacity_ = 0;
    std::uint32_t backingBlockRecordCount_ = 0;
    std::uint32_t backingBlockCount_ = 0;
    std::uint32_t refinementHashCapacity_ = 0;
    std::uint32_t leafArenaCapacity_ = 0;
    std::uint32_t drawCapacity_ = 0;
    std::uint32_t requestWordCount_ = 0;
    std::uint32_t vertexArray_ = 0;
    std::uint32_t refinementWriteBank_ = 0;
    std::uint32_t frameId_ = 1;
    geometry::RefinementMode lastProcessedRefinementMode_ = geometry::RefinementMode::None;
    std::int32_t lastAdaptiveScaleBucket_ = std::numeric_limits<std::int32_t>::min();
    float lastAdaptiveDecisionScale_ = 0.0f;
    bool lastBackingReadbackWasFresh_ = false;
    std::uint64_t refinementBankRecycleCount_ = 0;
    std::array<std::uint32_t, RefinementBankCount> latestRefinementBankUsed_{};
    std::array<std::uint32_t, RefinementBankCount> latestRefinementBankOverflow_{};
    std::array<std::uint32_t, RefinementBankCount> latestRefinementBankLastUse_{};
    std::uint64_t useCounter_ = 0;

    std::vector<FreeRange> rootFreeRanges_;
    std::unordered_map<std::uint32_t, RootPageAllocation> rootPages_;
    std::vector<std::uint32_t> activePageIds_;
    std::unordered_set<std::uint32_t> protectedRootPageIds_;

    std::vector<BackingSlot> backingSlots_;
    std::vector<std::uint32_t> freeBackingSlots_;
    std::unordered_map<std::uint32_t, std::uint32_t> backingBlockToSlot_;
    std::unordered_set<std::uint32_t> pinnedBackingBlocks_;

    PersistentStreamingStats stats_{};
};

} // namespace chmv::gpu::streaming
