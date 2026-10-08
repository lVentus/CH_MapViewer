#include "gpu/streaming/PersistentStreamingPipeline.h"

#include "data/ch/CHTypes.h"
#include "renderer/MapCamera2D.h"
#include "renderer/RoadStyle.h"

#include <glad/gl.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace chmv::gpu::streaming {
namespace {

struct alignas(16) GPURootRecord {
    std::uint32_t globalEdgeId = 0;
    std::int32_t birthLevel = -1;
    std::int32_t deathLevel = -1;
    std::uint32_t boundsLo = 0;
    std::uint32_t boundsHi = 0;
    std::uint32_t childA = data::InvalidEdgeId;
    std::uint32_t childB = data::InvalidEdgeId;
    float geometryError = 0.0f;
    float sourceX = 0.0f;
    float sourceY = 0.0f;
    float targetX = 0.0f;
    float targetY = 0.0f;
};

struct alignas(16) GPUActivePageDescriptor {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
    std::uint32_t pageId = 0;
    std::uint32_t padding = 0;
};

struct alignas(16) GPUBackingEndpoint {
    float sourceX = 0.0f;
    float sourceY = 0.0f;
    float targetX = 0.0f;
    float targetY = 0.0f;
};

struct GPUBackingChildren {
    std::uint32_t childA = data::InvalidEdgeId;
    std::uint32_t childB = data::InvalidEdgeId;
};

struct DrawArraysIndirectCommand {
    std::uint32_t count = 0;
    std::uint32_t instanceCount = 1;
    std::uint32_t first = 0;
    std::uint32_t baseInstance = 0;
};

struct DispatchIndirectCommand {
    std::uint32_t groupsX = 0;
    std::uint32_t groupsY = 1;
    std::uint32_t groupsZ = 1;
};

static_assert(sizeof(GPURootRecord) == 48);
static_assert(sizeof(GPUActivePageDescriptor) == 16);
static_assert(sizeof(GPUBackingEndpoint) == 16);
static_assert(sizeof(GPUBackingChildren) == 8);
static_assert(sizeof(DrawArraysIndirectCommand) == 16);
static_assert(sizeof(DispatchIndirectCommand) == 12);

void CheckBufferSize(std::size_t sizeBytes, const char* what) {
    GLint64 maxBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
    if (sizeBytes > static_cast<std::size_t>(maxBlockSize)) {
        throw std::runtime_error(std::string(what) + " exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE");
    }
}

std::uint32_t FloorPowerOfTwo(std::uint64_t value) {
    if (value == 0u) {
        return 0u;
    }
    return static_cast<std::uint32_t>(std::bit_floor(value));
}

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

} // namespace

PersistentStreamingPipeline::PersistentStreamingPipeline(
    const std::filesystem::path& shaderDirectory)
    : fullScanFilterProgram_(shaderDirectory / "streaming_root_filter_full_scan.comp"),
      birthOrderedFilterProgram_(shaderDirectory / "streaming_root_filter_birth_ordered.comp"),
      prepareDispatchProgram_(shaderDirectory / "streaming_prepare_dispatch.comp"),
      refineProgram_(shaderDirectory / "streaming_persistent_refine.comp"),
      composeProgram_(shaderDirectory / "streaming_compose_draw.comp"),
      recycleBankProgram_(shaderDirectory / "streaming_recycle_refinement_bank.comp"),
      finalizeDrawProgram_(shaderDirectory / "streaming_finalize_draw.comp"),
      roadProgram_(shaderDirectory / "streaming_road.vert",
                   shaderDirectory / "streaming_road.geom",
                   shaderDirectory / "streaming_road.frag") {
    glGenVertexArrays(1, &vertexArray_);
}

PersistentStreamingPipeline::~PersistentStreamingPipeline() {
    Reset();
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
    }
}

void PersistentStreamingPipeline::Initialize(
    const chmv::streaming::index::CHIndexData& index,
    PersistentStreamingConfig config) {
    Reset();
    index_ = &index;
    config_ = config;
    config_.gpuBudgetBytes = std::max<std::uint64_t>(config_.gpuBudgetBytes, 128ull * kMiB);
    if (index.edgeBlockRecordCount == 0 || index.graphEdgeBlocks.empty()) {
        throw std::runtime_error("persistent streaming requires a non-empty preprocessed edge-block index");
    }
    backingBlockRecordCount_ =
        chmv::streaming::runtime::RuntimeRefinementTileRecordCount(index.edgeBlockRecordCount);
    backingBlockCount_ =
        chmv::streaming::runtime::RuntimeRefinementTileCount(index.edgeCount, index.edgeBlockRecordCount);
    if (backingBlockRecordCount_ == 0u || backingBlockCount_ == 0u) {
        throw std::runtime_error("persistent streaming refinement tile layout is invalid");
    }

    GLint64 maxBlockSizeSigned = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSizeSigned);
    const auto maxBlockSize = static_cast<std::uint64_t>(
        std::max<GLint64>(maxBlockSizeSigned, 1));

    // Root residency is correctness-critical: if the strict current-view set almost fits, it
    // must not churn simply because the automatic partition stopped a few MiB short. At the
    // default 384 MiB budget, 104 MiB reaches the hard 2^21-record root cap exactly while
    // preserving most of the previous backing-cache budget. Smaller budgets scale down.
    const auto rootBudget = config_.rootCacheBudgetBytes != 0
                                ? config_.rootCacheBudgetBytes
                                : std::clamp<std::uint64_t>(
                                      (config_.gpuBudgetBytes * 13u) / 48u,
                                      32ull * kMiB, 104ull * kMiB);
    const auto geometryBudget = config_.refinementGeometryBudgetBytes != 0
                                    ? config_.refinementGeometryBudgetBytes
                                    : std::clamp<std::uint64_t>(
                                          config_.gpuBudgetBytes / 6u, 16ull * kMiB, 64ull * kMiB);
    const auto hashBudget = config_.refinementHashBudgetBytes != 0
                                ? config_.refinementHashBudgetBytes
                                : std::clamp<std::uint64_t>(
                                      config_.gpuBudgetBytes / 24u, 4ull * kMiB, 16ull * kMiB);
    // Final draw items are correctness output, not a cache. Give them enough room for the
    // observed multi-million-segment refined views without consuming the much larger theoretical
    // root+leaf worst case from the backing cache. Overflow remains explicitly reported below.
    const auto drawBudget = std::clamp<std::uint64_t>(
        config_.gpuBudgetBytes / 12u, 8ull * kMiB, 32ull * kMiB);
    const auto rootByBudget = rootBudget / (sizeof(GPURootRecord) + sizeof(std::uint32_t));
    const auto rootBySsbo = std::min<std::uint64_t>(
        maxBlockSize / sizeof(GPURootRecord), maxBlockSize / sizeof(std::uint32_t));
    rootRecordCapacity_ = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
        std::min<std::uint64_t>({MaxRootRecordCapacity, rootByBudget, rootBySsbo}),
        65536u, MaxRootRecordCapacity));

    const auto hashEntriesByBudget = hashBudget / (4u * sizeof(std::uint32_t));
    const auto hashEntriesBySsbo = maxBlockSize / (4u * sizeof(std::uint32_t));
    refinementHashCapacity_ = FloorPowerOfTwo(std::min<std::uint64_t>(
        {MaxRefinementHashCapacity, hashEntriesByBudget, hashEntriesBySsbo}));
    refinementHashCapacity_ = std::max<std::uint32_t>(refinementHashCapacity_, 65536u);

    const auto leafByBudget = geometryBudget / sizeof(GPUBackingEndpoint);
    const auto leafBySsbo = maxBlockSize / sizeof(GPUBackingEndpoint);
    leafArenaCapacity_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        {MaxLeafArenaCapacity, leafByBudget, leafBySsbo}));
    leafArenaCapacity_ = std::max<std::uint32_t>(
        (leafArenaCapacity_ / RefinementBankCount) * RefinementBankCount,
        RefinementBankCount * 32768u);

    // The old gpuBudget/20 policy capped the default draw list near 2.5M items, below real
    // refined views (>3.5M) and therefore dropped a different tail after atomic append each frame.
    // Keep the allocation bounded so Full DFS does not lose most of its backing slots; if a view
    // still exceeds this larger list, the explicit overflow counter makes that visible.
    const auto drawByBudget = drawBudget / (sizeof(std::uint32_t) + sizeof(float));
    const auto drawBySsbo = maxBlockSize / sizeof(std::uint32_t);
    drawCapacity_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        {static_cast<std::uint64_t>(MaxLeafArenaCapacity), drawByBudget, drawBySsbo}));
    drawCapacity_ = std::max<std::uint32_t>(drawCapacity_, rootRecordCapacity_);

    requestWordCount_ = (backingBlockCount_ + 31u) / 32u;

    const auto rootBytes = static_cast<std::size_t>(rootRecordCapacity_) * sizeof(GPURootRecord);
    const auto visibleRootBytes =
        static_cast<std::size_t>(rootRecordCapacity_) * sizeof(std::uint32_t);
    const auto activeDescriptorBytes =
        std::max<std::size_t>(index.graphPages.size(), 1u) * sizeof(GPUActivePageDescriptor);
    activeDescriptorCapacity_ = static_cast<std::uint32_t>(
        std::max<std::size_t>(index.graphPages.size(), 1u));
    const auto pageTableBytes =
        static_cast<std::size_t>(backingBlockCount_) * sizeof(std::uint32_t);
    const auto hashBytes =
        static_cast<std::size_t>(refinementHashCapacity_) * 4u * sizeof(std::uint32_t);
    const auto leafBytes =
        static_cast<std::size_t>(leafArenaCapacity_) * sizeof(GPUBackingEndpoint);
    const auto leafRoadTypeBytes =
        ((static_cast<std::size_t>(leafArenaCapacity_) + 3u) / 4u) * sizeof(std::uint32_t);
    const auto drawBytes = static_cast<std::size_t>(drawCapacity_) * sizeof(std::uint32_t);
    const auto drawAlphaBytes = static_cast<std::size_t>(drawCapacity_) * sizeof(float);
    const auto requestBytes =
        std::max<std::size_t>(requestWordCount_, 1u) * sizeof(std::uint32_t);

    readbackSizeBytes_ =
        (static_cast<std::size_t>(requestWordCount_) * 2u + 5u +
         (RefinementBankCount * 2u + 1u) + RefinementBankCount + 1u + 1u) *
        sizeof(std::uint32_t);

    const std::size_t smallFixedBytes =
        activeDescriptorBytes +
        sizeof(std::uint32_t) + // visible root counter
        pageTableBytes +
        (RefinementBankCount * 2u + 1u) * sizeof(std::uint32_t) +
        RefinementBankCount * sizeof(std::uint32_t) +
        5u * sizeof(std::uint32_t) +
        requestBytes * 2u +
        sizeof(DispatchIndirectCommand) +
        sizeof(std::uint32_t) +
        sizeof(DrawArraysIndirectCommand) +
        readbackSizeBytes_ * ReadbackRingSize;

    const std::uint64_t fixedWithoutBacking =
        static_cast<std::uint64_t>(rootBytes + visibleRootBytes + hashBytes +
                                   leafBytes + leafRoadTypeBytes + drawBytes + drawAlphaBytes +
                                   smallFixedBytes);
    if (fixedWithoutBacking >= config_.gpuBudgetBytes) {
        throw std::runtime_error(
            "persistent GPU budget is too small for the minimum streaming buffers");
    }

    const auto packedRoadTypeBytesPerSlot =
        ((static_cast<std::uint64_t>(backingBlockRecordCount_) + 3u) / 4u) *
        sizeof(std::uint32_t);
    const auto perBackingSlotBytes =
        static_cast<std::uint64_t>(backingBlockRecordCount_) *
            (sizeof(GPUBackingEndpoint) + sizeof(GPUBackingChildren) + sizeof(float)) +
        packedRoadTypeBytesPerSlot;
    const auto remainingForBacking = config_.gpuBudgetBytes - fixedWithoutBacking;
    const auto slotsByBudget =
        perBackingSlotBytes == 0 ? 0u : remainingForBacking / perBackingSlotBytes;
    const auto slotsByEndpointSsbo =
        maxBlockSize / (static_cast<std::uint64_t>(backingBlockRecordCount_) *
                        sizeof(GPUBackingEndpoint));
    const auto slotsByChildSsbo =
        maxBlockSize / (static_cast<std::uint64_t>(backingBlockRecordCount_) *
                        sizeof(GPUBackingChildren));
    const auto slotsByErrorSsbo =
        maxBlockSize / (static_cast<std::uint64_t>(backingBlockRecordCount_) * sizeof(float));
    const auto slotsByRoadTypeSsbo =
        packedRoadTypeBytesPerSlot == 0u ? 0u : maxBlockSize / packedRoadTypeBytesPerSlot;
    backingBlockCapacity_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        {MaxBackingBlockCapacity, static_cast<std::uint64_t>(backingBlockCount_),
         slotsByBudget, slotsByEndpointSsbo, slotsByChildSsbo, slotsByErrorSsbo,
         slotsByRoadTypeSsbo}));
    if (backingBlockCapacity_ < 16u) {
        throw std::runtime_error(
            "persistent GPU budget leaves fewer than 16 backing-block slots");
    }

    const auto endpointRecordCapacity =
        static_cast<std::size_t>(backingBlockCapacity_) * backingBlockRecordCount_;
    const auto endpointBytes = endpointRecordCapacity * sizeof(GPUBackingEndpoint);
    const auto childBytes = endpointRecordCapacity * sizeof(GPUBackingChildren);
    const auto geometryErrorBytes = endpointRecordCapacity * sizeof(float);
    const auto packedRoadTypeWordsPerSlot =
        (static_cast<std::size_t>(backingBlockRecordCount_) + 3u) / 4u;
    const auto roadTypeBytes =
        static_cast<std::size_t>(backingBlockCapacity_) * packedRoadTypeWordsPerSlot *
        sizeof(std::uint32_t);

    CheckBufferSize(rootBytes, "persistent root-record cache");
    CheckBufferSize(visibleRootBytes, "persistent visible-root list");
    CheckBufferSize(endpointBytes, "persistent backing endpoint cache");
    CheckBufferSize(childBytes, "persistent backing child cache");
    CheckBufferSize(geometryErrorBytes, "persistent backing geometry-error cache");
    CheckBufferSize(roadTypeBytes, "persistent backing road-type cache");
    CheckBufferSize(hashBytes, "persistent refinement hash");
    CheckBufferSize(leafBytes, "persistent refinement leaf arena");
    CheckBufferSize(leafRoadTypeBytes, "persistent refinement road-type arena");
    CheckBufferSize(drawBytes, "persistent draw edge list");
    CheckBufferSize(drawAlphaBytes, "persistent draw alpha list");

    rootRecordBuffer_.Allocate(rootBytes, nullptr, GL_DYNAMIC_DRAW);
    visibleRootBuffer_.Allocate(visibleRootBytes, nullptr, GL_DYNAMIC_DRAW);
    activePageDescriptorBuffer_.Allocate(activeDescriptorBytes, nullptr, GL_DYNAMIC_DRAW);

    const std::uint32_t zero = 0;
    visibleRootCounterBuffer_.Allocate(sizeof(zero), &zero, GL_DYNAMIC_DRAW);

    blockEndpointBuffer_.Allocate(endpointBytes, nullptr, GL_DYNAMIC_DRAW);
    blockChildBuffer_.Allocate(childBytes, nullptr, GL_DYNAMIC_DRAW);
    blockGeometryErrorBuffer_.Allocate(geometryErrorBytes, nullptr, GL_DYNAMIC_DRAW);
    blockRoadTypeBuffer_.Allocate(roadTypeBytes, nullptr, GL_DYNAMIC_DRAW);
    std::vector<std::uint32_t> emptyPageTable(backingBlockCount_, 0u);
    blockPageTableBuffer_.Allocate(pageTableBytes, emptyPageTable.data(), GL_DYNAMIC_DRAW);

    refinementHashBuffer_.Allocate(hashBytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, refinementHashBuffer_.Id());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

    leafArenaBuffer_.Allocate(leafBytes, nullptr, GL_DYNAMIC_DRAW);
    leafRoadTypeBuffer_.Allocate(leafRoadTypeBytes, nullptr, GL_DYNAMIC_DRAW);
    std::array<std::uint32_t, RefinementBankCount * 2u + 1u> arenaMeta{};
    leafArenaCounterBuffer_.Allocate(sizeof(arenaMeta), arenaMeta.data(), GL_DYNAMIC_DRAW);
    std::array<std::uint32_t, RefinementBankCount> bankLastUse{};
    refinementBankLastUseBuffer_.Allocate(sizeof(bankLastUse), bankLastUse.data(), GL_DYNAMIC_DRAW);
    const std::uint32_t refinementStats[5] = {0u, 0u, 0u, 0u, 0u};
    refinementStatsBuffer_.Allocate(sizeof(refinementStats), refinementStats, GL_DYNAMIC_DRAW);

    blockRequestBitsetBuffer_.Allocate(requestBytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, blockRequestBitsetBuffer_.Id());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    blockUseBitsetBuffer_.Allocate(requestBytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, blockUseBitsetBuffer_.Id());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

    const DispatchIndirectCommand dispatch{};
    dispatchBuffer_.Allocate(sizeof(dispatch), &dispatch, GL_DYNAMIC_DRAW);

    drawEdgeBuffer_.Allocate(drawBytes, nullptr, GL_DYNAMIC_DRAW);
    drawAlphaBuffer_.Allocate(drawAlphaBytes, nullptr, GL_DYNAMIC_DRAW);
    drawCounterBuffer_.Allocate(sizeof(zero), &zero, GL_DYNAMIC_DRAW);
    const DrawArraysIndirectCommand draw{};
    drawCommandBuffer_.Allocate(sizeof(draw), &draw, GL_DYNAMIC_DRAW);

    for (auto& buffer : readbackBuffers_) {
        buffer.Allocate(std::max<std::size_t>(readbackSizeBytes_, sizeof(std::uint32_t)), nullptr,
                        GL_STREAM_READ);
    }

    rootFreeRanges_.push_back({0u, rootRecordCapacity_});
    backingSlots_.resize(backingBlockCapacity_);
    backingBlockEverUploaded_.assign(backingBlockCount_, 0u);
    freeBackingSlots_.reserve(backingBlockCapacity_);
    for (std::uint32_t slot = backingBlockCapacity_; slot-- > 0;) {
        freeBackingSlots_.push_back(slot);
    }

    allocatedGpuBytes_ = fixedWithoutBacking + endpointBytes + childBytes + geometryErrorBytes +
                         roadTypeBytes;
    if (allocatedGpuBytes_ > config_.gpuBudgetBytes) {
        throw std::runtime_error("persistent GPU cache allocation exceeded its hard budget");
    }

    stats_.gpuBudgetBytes = config_.gpuBudgetBytes;
    stats_.gpuAllocatedBytes = allocatedGpuBytes_;
    stats_.gpuRootRecordCapacity = rootRecordCapacity_;
    stats_.drawCapacity = drawCapacity_;
    stats_.gpuBackingBlockCapacity = backingBlockCapacity_;
    stats_.refinementGeometryCapacity = leafArenaCapacity_;
    stats_.refinementWriteBank = refinementWriteBank_;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void PersistentStreamingPipeline::Reset() {
    for (std::size_t i = 0; i < ReadbackRingSize; ++i) {
        if (readbackFences_[i] != nullptr) {
            glDeleteSync(reinterpret_cast<GLsync>(readbackFences_[i]));
            readbackFences_[i] = nullptr;
        }
        readbackPending_[i] = false;
        readbackSequence_[i] = 0;
    }
    readbackSizeBytes_ = 0;
    nextReadbackSlot_ = 0;
    nextReadbackSequence_ = 1;
    index_ = nullptr;
    config_ = {};
    allocatedGpuBytes_ = 0;
    rootRecordCapacity_ = 0;
    activeDescriptorCapacity_ = 0;
    activePageCount_ = 0;
    backingBlockCapacity_ = 0;
    backingBlockRecordCount_ = 0;
    backingBlockCount_ = 0;
    refinementHashCapacity_ = 0;
    leafArenaCapacity_ = 0;
    drawCapacity_ = 0;
    requestWordCount_ = 0;
    refinementWriteBank_ = 0;
    frameId_ = 1;
    lastProcessedRefinementMode_ = geometry::RefinementMode::None;
    lastAdaptiveScaleBucket_ = std::numeric_limits<std::int32_t>::min();
    lastAdaptiveDecisionScale_ = 0.0f;
    lastBackingReadbackWasFresh_ = false;
    refinementBankRecycleCount_ = 0;
    latestRefinementBankUsed_.fill(0u);
    latestRefinementBankOverflow_.fill(0u);
    latestRefinementBankLastUse_.fill(0u);
    useCounter_ = 0;
    backingFrameSerial_ = 1;
    rootFreeRanges_.clear();
    rootPages_.clear();
    activePageIds_.clear();
    protectedRootPageIds_.clear();
    backingSlots_.clear();
    freeBackingSlots_.clear();
    backingBlockToSlot_.clear();
    pinnedBackingBlocks_.clear();
    backingBlockEverUploaded_.clear();
    stats_ = {};
}

bool PersistentStreamingPipeline::HasRootPage(std::uint32_t pageId) const {
    return rootPages_.contains(pageId);
}

std::uint32_t PersistentStreamingPipeline::AllocateRootRange(std::uint32_t count) {
    if (count == 0 || count > rootRecordCapacity_) {
        return InvalidId;
    }
    for (;;) {
        for (std::size_t i = 0; i < rootFreeRanges_.size(); ++i) {
            auto& range = rootFreeRanges_[i];
            if (range.count < count) {
                continue;
            }
            const auto offset = range.offset;
            range.offset += count;
            range.count -= count;
            if (range.count == 0) {
                rootFreeRanges_.erase(rootFreeRanges_.begin() + static_cast<std::ptrdiff_t>(i));
            }
            return offset;
        }
        if (!EvictOneInactiveRootPage()) {
            return InvalidId;
        }
    }
}

void PersistentStreamingPipeline::FreeRootRange(std::uint32_t offset, std::uint32_t count) {
    if (count == 0) {
        return;
    }
    rootFreeRanges_.push_back({offset, count});
    std::sort(rootFreeRanges_.begin(), rootFreeRanges_.end(), [](const auto& a, const auto& b) {
        return a.offset < b.offset;
    });
    std::vector<FreeRange> merged;
    merged.reserve(rootFreeRanges_.size());
    for (const auto& range : rootFreeRanges_) {
        if (!merged.empty() && merged.back().offset + merged.back().count == range.offset) {
            merged.back().count += range.count;
        } else {
            merged.push_back(range);
        }
    }
    rootFreeRanges_ = std::move(merged);
}

bool PersistentStreamingPipeline::EvictOneInactiveRootPage() {
    auto candidate = rootPages_.end();
    for (auto it = rootPages_.begin(); it != rootPages_.end(); ++it) {
        if (protectedRootPageIds_.contains(it->first)) {
            continue;
        }
        if (candidate == rootPages_.end() || it->second.lastUse < candidate->second.lastUse) {
            candidate = it;
        }
    }
    if (candidate == rootPages_.end()) {
        return false;
    }
    FreeRootRange(candidate->second.offset, candidate->second.count);
    rootPages_.erase(candidate);
    ++stats_.rootPageEvictions;
    stats_.gpuRootPagesCached = static_cast<std::uint32_t>(rootPages_.size());
    return true;
}

bool PersistentStreamingPipeline::UploadRootPage(
    const chmv::streaming::runtime::ResidentGraphPage& page,
    std::span<const std::uint64_t> edgeSpatialBounds) {
    if (!index_ || (page.rootRecords.empty() && page.edges.empty())) {
        return false;
    }
    if (auto found = rootPages_.find(page.pageId); found != rootPages_.end()) {
        found->second.lastUse = ++useCounter_;
        return true;
    }
    const auto recordCount = !page.rootRecords.empty() ? page.rootRecords.size() : page.edges.size();
    if (recordCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("root page edge count exceeds uint32");
    }
    const auto count = static_cast<std::uint32_t>(recordCount);
    const auto offset = AllocateRootRange(count);
    if (offset == InvalidId) {
        ++stats_.rootCacheAllocationFailures;
        return false;
    }

    const auto begin = std::chrono::steady_clock::now();
    std::vector<GPURootRecord> records;
    records.reserve(recordCount);
    if (!page.rootRecords.empty()) {
        for (const auto& root : page.rootRecords) {
            records.push_back({
                root.globalEdgeId, root.birthLevel, root.deathLevel,
                root.boundsLo, root.boundsHi, root.childA, root.childB, root.geometryError,
                root.sourceX, root.sourceY, root.targetX, root.targetY,
            });
        }
    } else {
        for (const auto& edge : page.edges) {
            if (edge.globalEdgeId >= edgeSpatialBounds.size()) {
                FreeRootRange(offset, count);
                throw std::runtime_error("root page edge is missing preprocessed spatial bounds");
            }
            if (edge.sourceLocal >= page.nodes.size() || edge.targetLocal >= page.nodes.size()) {
                FreeRootRange(offset, count);
                throw std::runtime_error(
                    "root page edge endpoint is outside the resident page node table");
            }
            const auto packed = edgeSpatialBounds[edge.globalEdgeId];
            auto boundsLo = static_cast<std::uint32_t>(packed & 0xffffffffull);
            auto boundsHi = static_cast<std::uint32_t>(packed >> 32u);
            chmv::streaming::index::EncodeRootRoadStyle(
                boundsLo, boundsHi, edge.roadStyleType);
            const auto& source = page.nodes[edge.sourceLocal];
            const auto& target = page.nodes[edge.targetLocal];
            records.push_back({
                edge.globalEdgeId, edge.birthLevel, edge.deathLevel,
                boundsLo, boundsHi, edge.childA, edge.childB, 0.0f,
                source.x, source.y, target.x, target.y,
            });
        }
    }
    rootRecordBuffer_.Update(static_cast<std::size_t>(offset) * sizeof(GPURootRecord),
                             records.size() * sizeof(GPURootRecord), records.data());
    rootPages_[page.pageId] = {offset, count, ++useCounter_};

    stats_.incrementalRootBytesUploaded += records.size() * sizeof(GPURootRecord);
    stats_.lastRootUploadMs = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - begin)
                                  .count();
    stats_.gpuRootPagesCached = static_cast<std::uint32_t>(rootPages_.size());
    return true;
}

void PersistentStreamingPipeline::SetProtectedRootPages(
    std::span<const std::uint32_t> pageIds) {
    protectedRootPageIds_.clear();
    protectedRootPageIds_.reserve(pageIds.size());
    for (const auto pageId : pageIds) {
        protectedRootPageIds_.insert(pageId);
        if (const auto found = rootPages_.find(pageId); found != rootPages_.end()) {
            found->second.lastUse = ++useCounter_;
        }
    }
}

void PersistentStreamingPipeline::EnsureActiveDescriptorCapacity(std::size_t count) {
    if (count > activeDescriptorCapacity_) {
        throw std::runtime_error(
            "active root-page descriptor count exceeds the bounded GPU descriptor cache");
    }
}

void PersistentStreamingPipeline::SetActiveRootPages(
    std::span<const std::uint32_t> pageIds) {
    std::vector<std::uint32_t> sorted(pageIds.begin(), pageIds.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    std::vector<GPUActivePageDescriptor> descriptors;
    descriptors.reserve(sorted.size());
    std::vector<std::uint32_t> installed;
    installed.reserve(sorted.size());
    for (const auto pageId : sorted) {
        const auto found = rootPages_.find(pageId);
        if (found == rootPages_.end()) {
            continue;
        }
        found->second.lastUse = ++useCounter_;
        descriptors.push_back({found->second.offset, found->second.count, pageId, 0u});
        installed.push_back(pageId);
    }

    if (installed == activePageIds_) {
        activePageCount_ = static_cast<std::uint32_t>(installed.size());
        stats_.gpuRootPagesActive = activePageCount_;
        return;
    }
    EnsureActiveDescriptorCapacity(descriptors.size());
    if (!descriptors.empty()) {
        activePageDescriptorBuffer_.Update(0, descriptors.size() * sizeof(GPUActivePageDescriptor),
                                           descriptors.data());
    }
    activePageIds_ = std::move(installed);
    activePageCount_ = static_cast<std::uint32_t>(activePageIds_.size());
    stats_.gpuRootPagesActive = activePageCount_;
}

bool PersistentStreamingPipeline::HasBackingBlock(std::uint32_t blockId) const {
    return backingBlockToSlot_.contains(blockId);
}

void PersistentStreamingPipeline::SetPageTableEntry(std::uint32_t blockId,
                                                     std::uint32_t slotPlusOne) {
    if (!index_ || blockId >= backingBlockCount_) {
        return;
    }
    blockPageTableBuffer_.Update(static_cast<std::size_t>(blockId) * sizeof(std::uint32_t),
                                 sizeof(slotPlusOne), &slotPlusOne);
}

bool PersistentStreamingPipeline::EvictOneUnpinnedBackingBlock() {
    auto matureCandidate = backingBlockToSlot_.end();
    auto fallbackCandidate = backingBlockToSlot_.end();
    for (auto it = backingBlockToSlot_.begin(); it != backingBlockToSlot_.end(); ++it) {
        if (pinnedBackingBlocks_.contains(it->first)) {
            continue;
        }
        const auto& slot = backingSlots_[it->second];
        if (fallbackCandidate == backingBlockToSlot_.end() ||
            slot.lastUse < backingSlots_[fallbackCandidate->second].lastUse) {
            fallbackCandidate = it;
        }

        // GPU block-use feedback is asynchronous. Avoid throwing out a block during the few
        // Process() calls between upload and its first readback-visible use.
        const auto ageFrames = backingFrameSerial_ - slot.uploadedFrame;
        if (ageFrames < BackingUploadGraceFrames) {
            continue;
        }
        if (matureCandidate == backingBlockToSlot_.end() ||
            slot.lastUse < backingSlots_[matureCandidate->second].lastUse) {
            matureCandidate = it;
        }
    }

    auto candidate = matureCandidate;
    if (candidate == backingBlockToSlot_.end()) {
        candidate = fallbackCandidate;
        if (candidate != backingBlockToSlot_.end()) {
            // Soft grace must never turn cache pressure into an allocation failure. Count the
            // fallback so telemetry can tell whether the configured GPU budget is too tight for
            // even the readback-latency window.
            ++stats_.backingGraceFallbackEvictions;
        }
    }
    if (candidate == backingBlockToSlot_.end()) {
        return false;
    }
    const auto blockId = candidate->first;
    const auto slotId = candidate->second;
    SetPageTableEntry(blockId, 0u);
    backingSlots_[slotId] = {};
    freeBackingSlots_.push_back(slotId);
    backingBlockToSlot_.erase(candidate);
    ++stats_.backingBlockEvictions;
    stats_.gpuBackingBlocksResident = static_cast<std::uint32_t>(backingBlockToSlot_.size());
    return true;
}

bool PersistentStreamingPipeline::UploadBackingBlock(
    const chmv::streaming::runtime::ResidentRefinementBlock& block) {
    if (!index_ || block.blockId >= backingBlockCount_) {
        return false;
    }
    if (auto found = backingBlockToSlot_.find(block.blockId); found != backingBlockToSlot_.end()) {
        backingSlots_[found->second].lastUse = ++useCounter_;
        return true;
    }
    while (freeBackingSlots_.empty()) {
        if (!EvictOneUnpinnedBackingBlock()) {
            ++stats_.backingCacheAllocationFailures;
            return false;
        }
    }

    const auto slot = freeBackingSlots_.back();
    freeBackingSlots_.pop_back();
    const auto begin = std::chrono::steady_clock::now();

    std::vector<GPUBackingEndpoint> endpoints;
    std::vector<GPUBackingChildren> children;
    std::vector<float> geometryErrors;
    const auto packedRoadTypeWordsPerSlot =
        (static_cast<std::size_t>(backingBlockRecordCount_) + 3u) / 4u;
    std::vector<std::uint32_t> roadTypes((block.edges.size() + 3u) / 4u, 0u);
    endpoints.reserve(block.edges.size());
    children.reserve(block.edges.size());
    geometryErrors.reserve(block.edges.size());
    std::size_t edgeIndex = 0u;
    for (const auto& edge : block.edges) {
        if (edge.sourceLocal >= block.nodes.size() || edge.targetLocal >= block.nodes.size()) {
            freeBackingSlots_.push_back(slot);
            throw std::runtime_error("refinement block contains invalid local node indices");
        }
        const auto& source = block.nodes[edge.sourceLocal];
        const auto& target = block.nodes[edge.targetLocal];
        endpoints.push_back({source.x, source.y, target.x, target.y});
        children.push_back({edge.childA, edge.childB});
        geometryErrors.push_back(edge.geometryError);
        const auto shift = static_cast<std::uint32_t>((edgeIndex & 3u) * 8u);
        roadTypes[edgeIndex >> 2u] |= (edge.roadStyleType & 0x3fu) << shift;
        ++edgeIndex;
    }

    if (block.edges.size() > backingBlockRecordCount_) {
        freeBackingSlots_.push_back(slot);
        throw std::runtime_error("refinement runtime tile exceeds GPU backing-slot capacity");
    }
    const auto baseRecord =
        static_cast<std::size_t>(slot) * backingBlockRecordCount_;
    if (!endpoints.empty()) {
        blockEndpointBuffer_.Update(baseRecord * sizeof(GPUBackingEndpoint),
                                    endpoints.size() * sizeof(GPUBackingEndpoint),
                                    endpoints.data());
        blockChildBuffer_.Update(baseRecord * sizeof(GPUBackingChildren),
                                 children.size() * sizeof(GPUBackingChildren), children.data());
        blockGeometryErrorBuffer_.Update(baseRecord * sizeof(float),
                                         geometryErrors.size() * sizeof(float),
                                         geometryErrors.data());
        const auto baseRoadTypeWord =
            static_cast<std::size_t>(slot) * packedRoadTypeWordsPerSlot;
        blockRoadTypeBuffer_.Update(baseRoadTypeWord * sizeof(std::uint32_t),
                                    roadTypes.size() * sizeof(std::uint32_t), roadTypes.data());
    }

    backingSlots_[slot] = {block.blockId, ++useCounter_, backingFrameSerial_};
    backingBlockToSlot_[block.blockId] = slot;
    SetPageTableEntry(block.blockId, slot + 1u);

    if (block.blockId < backingBlockEverUploaded_.size()) {
        if (backingBlockEverUploaded_[block.blockId] != 0u) {
            ++stats_.backingBlockReuploadsAfterEviction;
        } else {
            backingBlockEverUploaded_[block.blockId] = 1u;
            ++stats_.backingBlockFirstUploads;
        }
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    stats_.incrementalBackingBytesUploaded +=
        endpoints.size() * sizeof(GPUBackingEndpoint) +
        children.size() * sizeof(GPUBackingChildren) +
        geometryErrors.size() * sizeof(float) +
        roadTypes.size() * sizeof(std::uint32_t);
    stats_.lastBackingUploadMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - begin)
                                     .count();
    stats_.gpuBackingBlocksResident = static_cast<std::uint32_t>(backingBlockToSlot_.size());
    return true;
}

void PersistentStreamingPipeline::SetPinnedBackingBlocks(
    std::span<const std::uint32_t> blockIds) {
    pinnedBackingBlocks_.clear();
    pinnedBackingBlocks_.reserve(blockIds.size());
    for (const auto id : blockIds) {
        pinnedBackingBlocks_.insert(id);
        if (const auto found = backingBlockToSlot_.find(id); found != backingBlockToSlot_.end()) {
            backingSlots_[found->second].lastUse = ++useCounter_;
        }
    }
}

std::uint32_t PersistentStreamingPipeline::RootRecordUpperBound() const {
    std::uint64_t total = 0;
    for (const auto pageId : activePageIds_) {
        const auto found = rootPages_.find(pageId);
        if (found != rootPages_.end()) {
            total += found->second.count;
        }
    }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(total, rootRecordCapacity_));
}

void PersistentStreamingPipeline::ResetRefinementCache() {
    const std::uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, refinementHashBuffer_.Id());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    std::array<std::uint32_t, RefinementBankCount * 2u + 1u> arenaMeta{};
    leafArenaCounterBuffer_.Update(0, sizeof(arenaMeta), arenaMeta.data());
    std::array<std::uint32_t, RefinementBankCount> bankLastUse{};
    refinementBankLastUseBuffer_.Update(0, sizeof(bankLastUse), bankLastUse.data());
    latestRefinementBankUsed_.fill(0u);
    latestRefinementBankOverflow_.fill(0u);
    latestRefinementBankLastUse_.fill(0u);
    refinementWriteBank_ = 0u;
    frameId_ = 1u;
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    stats_.cachedRefinedRoots = 0;
    stats_.refinementGeometryUsed = 0;
    stats_.refinementGeometryOverflows = 0;
    stats_.refinementWriteBank = 0;
    stats_.refinementBanksUsed = 0;
}

void PersistentStreamingPipeline::RecycleRefinementBank(std::uint32_t bank) {
    if (bank >= RefinementBankCount || refinementHashCapacity_ == 0) {
        return;
    }

    recycleBankProgram_.Bind();
    refinementHashBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    leafArenaCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glUniform1ui(glGetUniformLocation(recycleBankProgram_.Id(), "uHashCapacity"),
                 refinementHashCapacity_);
    glUniform1ui(glGetUniformLocation(recycleBankProgram_.Id(), "uTargetBank"), bank);
    recycleBankProgram_.Dispatch((refinementHashCapacity_ + 255u) / 256u);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    const std::uint32_t zero = 0;
    leafArenaCounterBuffer_.Update(static_cast<std::size_t>(bank) * sizeof(std::uint32_t),
                                   sizeof(zero), &zero);
    leafArenaCounterBuffer_.Update(
        static_cast<std::size_t>(RefinementBankCount + bank) * sizeof(std::uint32_t),
        sizeof(zero), &zero);
    refinementBankLastUseBuffer_.Update(static_cast<std::size_t>(bank) * sizeof(std::uint32_t),
                                        sizeof(zero), &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    latestRefinementBankUsed_[bank] = 0u;
    latestRefinementBankOverflow_[bank] = 0u;
    latestRefinementBankLastUse_[bank] = 0u;
    refinementWriteBank_ = bank;
    ++refinementBankRecycleCount_;
    stats_.refinementWriteBank = bank;
    stats_.refinementBankRecycles = refinementBankRecycleCount_;
}

void PersistentStreamingPipeline::RotateRefinementBankIfNeeded() {
    if (refinementWriteBank_ >= RefinementBankCount ||
        latestRefinementBankOverflow_[refinementWriteBank_] == 0u) {
        return;
    }

    // Prefer a never-used bank first. This is free and avoids touching the hash table.
    for (std::uint32_t bank = 0; bank < RefinementBankCount; ++bank) {
        if (bank == refinementWriteBank_) {
            continue;
        }
        if (latestRefinementBankUsed_[bank] == 0u &&
            latestRefinementBankOverflow_[bank] == 0u) {
            refinementWriteBank_ = bank;
            stats_.refinementWriteBank = bank;
            return;
        }
    }

    // Recycle only a bank the GPU has not drawn from recently. This keeps roots still inside the
    // viewport/guard band stable while allowing old camera regions/old LODs to make room for new
    // refinement geometry. If every bank is still hot, keep the already-refined geometry and let
    // new roots temporarily fall back to shortcuts instead of creating a refine/unrefine loop.
    std::uint32_t candidate = InvalidId;
    std::uint32_t oldestUse = std::numeric_limits<std::uint32_t>::max();
    const auto safeFrame = frameId_ > RefinementBankReuseDelayFrames
                               ? frameId_ - RefinementBankReuseDelayFrames
                               : 0u;
    for (std::uint32_t bank = 0; bank < RefinementBankCount; ++bank) {
        if (bank == refinementWriteBank_) {
            continue;
        }
        const auto lastUse = latestRefinementBankLastUse_[bank];
        if (lastUse >= safeFrame) {
            continue;
        }
        if (candidate == InvalidId || lastUse < oldestUse) {
            candidate = bank;
            oldestUse = lastUse;
        }
    }
    if (candidate != InvalidId) {
        RecycleRefinementBank(candidate);
    }
}

void PersistentStreamingPipeline::Process(
    float lodLevel,
    PersistentRangeFilterKind filterKind,
    const geometry::RefinementParameters& refinement,
    const chmv::streaming::runtime::StreamingSpatialWindow& refinementWindow) {
    const auto refinementMode = refinement.mode;
    if (!index_ || activePageCount_ == 0) {
        const std::uint32_t zero = 0;
        drawCounterBuffer_.Update(0, sizeof(zero), &zero);
        FinalizeDrawCommand();
        stats_.visibleRootCount = 0;
        stats_.drawEdgeCount = 0;
        return;
    }

    // Adaptive geometry depends continuously on screen scale. Cache it in conservative quarter-
    // octave buckets: each bucket uses the *upper* scale bound, so reuse can only over-refine,
    // never violate the requested max screen-space error. Crossing a bucket or switching Full/
    // Adaptive invalidates the geometry hash because the cached leaf set has different semantics.
    float adaptiveDecisionScale = 0.0f;
    std::int32_t adaptiveScaleBucket = std::numeric_limits<std::int32_t>::min();
    if (refinementMode == geometry::RefinementMode::Adaptive) {
        const auto maxError = std::max(refinement.maxScreenErrorPixels, 1e-6f);
        const auto ratio = std::max(refinement.screenPixelScale / maxError, 1e-12f);
        const auto bucketFloat = std::ceil(std::log2(ratio) * 4.0f);
        adaptiveScaleBucket = static_cast<std::int32_t>(std::clamp(
            bucketFloat,
            static_cast<float>(std::numeric_limits<std::int16_t>::min()),
            static_cast<float>(std::numeric_limits<std::int16_t>::max())));
        adaptiveDecisionScale = std::exp2(static_cast<float>(adaptiveScaleBucket) * 0.25f);
    }
    // Adaptive refinement is monotonic across camera-scale changes. Frontier nodes retain their
    // direct children and geometry error, so zoom-in continues from the current approximation
    // instead of clearing the hash and unfolding from the root again. Zoom-out keeps the already
    // finer geometry; only changing refinement algorithms changes cache semantics.
    const bool refinementSemanticChanged = refinementMode != lastProcessedRefinementMode_;
    if (refinementSemanticChanged &&
        (refinementMode != geometry::RefinementMode::None ||
         lastProcessedRefinementMode_ != geometry::RefinementMode::None)) {
        ResetRefinementCache();
        lastAdaptiveDecisionScale_ = 0.0f;
    }
    const bool adaptiveForceProgress =
        refinementMode == geometry::RefinementMode::Adaptive &&
        adaptiveDecisionScale > lastAdaptiveDecisionScale_ * 1.0001f;
    if (refinementMode == geometry::RefinementMode::Adaptive) {
        lastAdaptiveDecisionScale_ = adaptiveDecisionScale;
    }
    lastProcessedRefinementMode_ = refinementMode;
    lastAdaptiveScaleBucket_ = adaptiveScaleBucket;

    const std::uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visibleRootCounterBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, drawCounterBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, blockRequestBitsetBuffer_.Id());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, blockUseBitsetBuffer_.Id());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, refinementStatsBuffer_.Id());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

    filterTimer_.Begin();
    auto& filterProgram = filterKind == PersistentRangeFilterKind::BirthOrdered
                              ? birthOrderedFilterProgram_
                              : fullScanFilterProgram_;
    filterProgram.Bind();
    rootRecordBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    activePageDescriptorBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    visibleRootBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    visibleRootCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    glUniform1f(glGetUniformLocation(filterProgram.Id(), "uLevelFloat"), lodLevel);
    glUniform1ui(glGetUniformLocation(filterProgram.Id(), "uOutputCapacity"),
                 rootRecordCapacity_);
    glUniform1ui(glGetUniformLocation(filterProgram.Id(), "uCullMinX"), refinementWindow.minX);
    glUniform1ui(glGetUniformLocation(filterProgram.Id(), "uCullMinY"), refinementWindow.minY);
    glUniform1ui(glGetUniformLocation(filterProgram.Id(), "uCullMaxX"), refinementWindow.maxX);
    glUniform1ui(glGetUniformLocation(filterProgram.Id(), "uCullMaxY"), refinementWindow.maxY);
    filterProgram.Dispatch(activePageCount_);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    filterTimer_.End();

    prepareDispatchProgram_.Bind();
    visibleRootCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    dispatchBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glUniform1ui(glGetUniformLocation(prepareDispatchProgram_.Id(), "uLocalSize"),
                 RefineLocalSize);
    prepareDispatchProgram_.Dispatch(1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    if (refinementMode != geometry::RefinementMode::None) {
        refinementTimer_.Begin();
        refineProgram_.Bind();
        blockPageTableBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
        blockChildBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
        blockEndpointBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
        visibleRootBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
        visibleRootCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 4);
        refinementHashBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
        leafArenaBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
        leafArenaCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 7);
        blockRequestBitsetBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 8);
        refinementStatsBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 9);
        blockUseBitsetBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 10);
        rootRecordBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 11);
        blockGeometryErrorBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 12);
        blockRoadTypeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 13);
        leafRoadTypeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 14);
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uEdgeCount"),
                     static_cast<std::uint32_t>(index_->edgeCount));
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uBlockRecordCount"),
                     backingBlockRecordCount_);
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uBlockCount"),
                     backingBlockCount_);
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uHashMask"),
                     refinementHashCapacity_ - 1u);
        const auto activeArenaCapacity =
            refinementMode == geometry::RefinementMode::Adaptive
                ? leafArenaCapacity_ / 2u
                : leafArenaCapacity_;
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uBankCapacity"),
                     activeArenaCapacity / RefinementBankCount);
        stats_.refinementGeometryCapacity = activeArenaCapacity;
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uWriteBank"),
                     refinementWriteBank_);
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uRefinementMode"),
                     refinementMode == geometry::RefinementMode::Adaptive ? 2u : 1u);
        const auto adaptiveRequestBudget = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
            std::max<std::uint64_t>(32u, backingBlockCapacity_ / 32u),
            32u, MaxAdaptiveBlockRequestsPerFrame));
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uMaxAdaptiveBlockRequests"),
                     refinementMode == geometry::RefinementMode::Adaptive
                         ? adaptiveRequestBudget
                         : std::numeric_limits<std::uint32_t>::max());
        glUniform1ui(glGetUniformLocation(refineProgram_.Id(), "uAdaptiveForceProgress"),
                     adaptiveForceProgress ? 1u : 0u);
        glUniform1f(glGetUniformLocation(refineProgram_.Id(), "uScreenPixelScale"),
                    refinementMode == geometry::RefinementMode::Adaptive
                        ? adaptiveDecisionScale
                        : 0.0f);
        glUniform1f(glGetUniformLocation(refineProgram_.Id(), "uMaxScreenErrorPixels"),
                    refinementMode == geometry::RefinementMode::Adaptive ? 1.0f : 0.0f);
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, dispatchBuffer_.Id());
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        refinementTimer_.End();
    }

    composeTimer_.Begin();
    composeProgram_.Bind();
    blockPageTableBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    visibleRootBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    visibleRootCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    refinementHashBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    leafArenaBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 4);
    drawEdgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    drawCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    blockRequestBitsetBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 7);
    refinementBankLastUseBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 8);
    blockUseBitsetBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 9);
    rootRecordBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 10);
    drawAlphaBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 11);
    glUniform1ui(glGetUniformLocation(composeProgram_.Id(), "uBlockRecordCount"),
                 backingBlockRecordCount_);
    glUniform1ui(glGetUniformLocation(composeProgram_.Id(), "uBlockCount"),
                 backingBlockCount_);
    glUniform1ui(glGetUniformLocation(composeProgram_.Id(), "uHashMask"),
                 refinementHashCapacity_ - 1u);
    glUniform1ui(glGetUniformLocation(composeProgram_.Id(), "uDrawCapacity"), drawCapacity_);
    glUniform1ui(glGetUniformLocation(composeProgram_.Id(), "uRefinementEnabled"),
                 refinementMode != geometry::RefinementMode::None ? 1u : 0u);
    glUniform1ui(glGetUniformLocation(composeProgram_.Id(), "uFrameId"), frameId_);
    glUniform1f(glGetUniformLocation(composeProgram_.Id(), "uLevelFloat"), lodLevel);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, dispatchBuffer_.Id());
    glDispatchComputeIndirect(0);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    composeTimer_.End();

    FinalizeDrawCommand();
    ScheduleReadback();
    ++frameId_;
    if (frameId_ == 0u) {
        frameId_ = 1u;
    }
    ++backingFrameSerial_;

    if (const auto value = filterTimer_.LastMilliseconds()) {
        stats_.gpuFilterCullMs = *value;
    }
    if (const auto value = refinementTimer_.LastMilliseconds()) {
        stats_.gpuRefinementMs = *value;
    }
    if (const auto value = composeTimer_.LastMilliseconds()) {
        stats_.gpuComposeMs = *value;
    }
}

void PersistentStreamingPipeline::FinalizeDrawCommand() {
    finalizeDrawProgram_.Bind();
    drawCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    drawCommandBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glUniform1ui(glGetUniformLocation(finalizeDrawProgram_.Id(), "uDrawCapacity"),
                 drawCapacity_);
    finalizeDrawProgram_.Dispatch(1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
}

void PersistentStreamingPipeline::ScheduleReadback() {
    if (!index_ || readbackSizeBytes_ == 0) {
        return;
    }

    std::size_t slot = ReadbackRingSize;
    for (std::size_t attempt = 0; attempt < ReadbackRingSize; ++attempt) {
        const auto candidate = (nextReadbackSlot_ + attempt) % ReadbackRingSize;
        if (!readbackPending_[candidate]) {
            slot = candidate;
            break;
        }
    }
    if (slot == ReadbackRingSize) {
        // Never stall the renderer just to collect diagnostics/page faults. With three staging
        // slots this only happens when the GPU is more than three frames behind.
        return;
    }

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT |
                    GL_COMMAND_BARRIER_BIT);
    glBindBuffer(GL_COPY_WRITE_BUFFER, readbackBuffers_[slot].Id());
    std::size_t offset = 0;
    auto copyFrom = [&](std::uint32_t sourceBuffer, std::size_t bytes) {
        glBindBuffer(GL_COPY_READ_BUFFER, sourceBuffer);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0,
                            static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(bytes));
        offset += bytes;
    };

    copyFrom(blockRequestBitsetBuffer_.Id(),
             static_cast<std::size_t>(requestWordCount_) * sizeof(std::uint32_t));
    copyFrom(blockUseBitsetBuffer_.Id(),
             static_cast<std::size_t>(requestWordCount_) * sizeof(std::uint32_t));
    copyFrom(refinementStatsBuffer_.Id(), 5u * sizeof(std::uint32_t));
    copyFrom(leafArenaCounterBuffer_.Id(),
             (RefinementBankCount * 2u + 1u) * sizeof(std::uint32_t));
    copyFrom(refinementBankLastUseBuffer_.Id(),
             RefinementBankCount * sizeof(std::uint32_t));
    copyFrom(visibleRootCounterBuffer_.Id(), sizeof(std::uint32_t));
    copyFrom(drawCounterBuffer_.Id(), sizeof(std::uint32_t));
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    readbackFences_[slot] = reinterpret_cast<void*>(
        glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0));
    readbackPending_[slot] = true;
    readbackSequence_[slot] = nextReadbackSequence_++;
    nextReadbackSlot_ = (slot + 1u) % ReadbackRingSize;
}

std::vector<std::uint32_t> PersistentStreamingPipeline::ReadMissingBackingBlocks() {
    std::vector<std::uint32_t> result;
    lastBackingReadbackWasFresh_ = false;
    if (!index_ || readbackSizeBytes_ == 0) {
        return result;
    }

    std::vector<std::uint32_t> mergedWords(requestWordCount_, 0u);
    std::vector<std::uint32_t> mergedUseWords(requestWordCount_, 0u);
    std::uint64_t latestSequence = 0;
    std::array<std::uint32_t, 5> latestRefinementStats{};
    std::array<std::uint32_t, RefinementBankCount * 2u + 1u> latestArenaMeta{};
    std::array<std::uint32_t, RefinementBankCount> latestBankLastUse{};
    std::uint32_t latestVisibleCount = stats_.visibleRootCount;
    std::uint32_t latestDrawCount = stats_.drawEdgeCount;

    for (std::size_t slot = 0; slot < ReadbackRingSize; ++slot) {
        if (!readbackPending_[slot] || readbackFences_[slot] == nullptr) {
            continue;
        }
        const auto status = glClientWaitSync(
            reinterpret_cast<GLsync>(readbackFences_[slot]), 0, 0);
        if (status != GL_ALREADY_SIGNALED && status != GL_CONDITION_SATISFIED) {
            continue;
        }

        std::vector<std::uint32_t> data(readbackSizeBytes_ / sizeof(std::uint32_t), 0u);
        glBindBuffer(GL_COPY_READ_BUFFER, readbackBuffers_[slot].Id());
        glGetBufferSubData(GL_COPY_READ_BUFFER, 0,
                           static_cast<GLsizeiptr>(readbackSizeBytes_), data.data());
        glBindBuffer(GL_COPY_READ_BUFFER, 0);

        for (std::uint32_t word = 0; word < requestWordCount_; ++word) {
            mergedWords[word] |= data[word];
            mergedUseWords[word] |= data[requestWordCount_ + word];
        }

        if (readbackSequence_[slot] >= latestSequence) {
            latestSequence = readbackSequence_[slot];
            std::size_t cursor = static_cast<std::size_t>(requestWordCount_) * 2u;
            for (auto& value : latestRefinementStats) {
                value = data[cursor++];
            }
            for (auto& value : latestArenaMeta) {
                value = data[cursor++];
            }
            for (auto& value : latestBankLastUse) {
                value = data[cursor++];
            }
            latestVisibleCount = data[cursor++];
            latestDrawCount = data[cursor++];
        }

        glDeleteSync(reinterpret_cast<GLsync>(readbackFences_[slot]));
        readbackFences_[slot] = nullptr;
        readbackPending_[slot] = false;
        readbackSequence_[slot] = 0;
    }

    for (std::uint32_t word = 0; word < mergedWords.size(); ++word) {
        auto bits = mergedWords[word];
        while (bits != 0u) {
            const auto bitIndex = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto blockId = word * 32u + bitIndex;
            if (blockId < backingBlockCount_ && !HasBackingBlock(blockId)) {
                result.push_back(blockId);
            }
            bits &= bits - 1u;
        }
    }

    std::uint32_t touchedBlocks = 0;
    for (std::uint32_t word = 0; word < mergedUseWords.size(); ++word) {
        auto bits = mergedUseWords[word];
        while (bits != 0u) {
            const auto bitIndex = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto blockId = word * 32u + bitIndex;
            if (const auto found = backingBlockToSlot_.find(blockId);
                found != backingBlockToSlot_.end()) {
                backingSlots_[found->second].lastUse = ++useCounter_;
                ++touchedBlocks;
            }
            bits &= bits - 1u;
        }
    }
    stats_.backingBlocksTouchedLastReadback = touchedBlocks;

    if (latestSequence != 0) {
        lastBackingReadbackWasFresh_ = true;
        stats_.visibleRootCount = latestVisibleCount;
        stats_.drawEdgeCount = latestDrawCount;
        stats_.drawOverflowCount = latestDrawCount > drawCapacity_
                                       ? latestDrawCount - drawCapacity_
                                       : 0u;
        stats_.refinementCacheHits = latestRefinementStats[0];
        stats_.refinementCacheMisses = latestRefinementStats[1];
        stats_.refinementStackOverflows = latestRefinementStats[2];
        stats_.refinementHashOverflows = latestRefinementStats[3];
        stats_.adaptiveBlockRequestsAdmitted = latestRefinementStats[4];
        std::uint64_t geometryUsed = 0;
        std::uint64_t geometryOverflows = 0;
        std::uint32_t banksUsed = 0;
        for (std::uint32_t bank = 0; bank < RefinementBankCount; ++bank) {
            latestRefinementBankUsed_[bank] = latestArenaMeta[bank];
            latestRefinementBankOverflow_[bank] =
                latestArenaMeta[RefinementBankCount + bank];
            latestRefinementBankLastUse_[bank] = latestBankLastUse[bank];
            const auto activeArenaCapacity =
                lastProcessedRefinementMode_ == geometry::RefinementMode::Adaptive
                    ? leafArenaCapacity_ / 2u
                    : leafArenaCapacity_;
            geometryUsed += std::min(latestRefinementBankUsed_[bank],
                                     activeArenaCapacity / RefinementBankCount);
            geometryOverflows += latestRefinementBankOverflow_[bank];
            if (latestRefinementBankUsed_[bank] != 0u) {
                ++banksUsed;
            }
        }
        stats_.cachedRefinedRoots = latestArenaMeta[RefinementBankCount * 2u];
        const auto activeArenaCapacity =
            lastProcessedRefinementMode_ == geometry::RefinementMode::Adaptive
                ? leafArenaCapacity_ / 2u
                : leafArenaCapacity_;
        stats_.refinementGeometryUsed = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(geometryUsed, activeArenaCapacity));
        stats_.refinementGeometryCapacity = activeArenaCapacity;
        stats_.refinementGeometryOverflows = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(geometryOverflows, std::numeric_limits<std::uint32_t>::max()));
        stats_.refinementWriteBank = refinementWriteBank_;
        stats_.refinementBanksUsed = banksUsed;
        stats_.refinementBankRecycles = refinementBankRecycleCount_;
        RotateRefinementBankIfNeeded();
    }
    stats_.missingBlockRequestCount = static_cast<std::uint32_t>(result.size());
    return result;
}

void PersistentStreamingPipeline::Draw(
    const chmv::renderer::MapCamera2D& camera,
    int framebufferWidth,
    int framebufferHeight,
    const std::array<float, 4>& color,
    const chmv::renderer::RoadStyleConfig& styles) const {
    if (!index_ || framebufferWidth <= 0 || framebufferHeight <= 0) {
        return;
    }

    roadProgram_.Bind();
    drawEdgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    blockPageTableBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    blockEndpointBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    leafArenaBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    rootRecordBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 4);
    drawAlphaBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    leafRoadTypeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    glUniform1ui(glGetUniformLocation(roadProgram_.Id(), "uBlockRecordCount"),
                 backingBlockRecordCount_);
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uCameraCenterX"), camera.CenterX());
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uCameraCenterY"), camera.CenterY());
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uZoom"), camera.ViewScale());
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uAspectScale"),
                static_cast<float>(framebufferHeight) / static_cast<float>(framebufferWidth));
    glUniform4f(glGetUniformLocation(roadProgram_.Id(), "uColor"), color[0], color[1], color[2],
                color[3]);
    glUniform2f(glGetUniformLocation(roadProgram_.Id(), "uViewportSize"),
                static_cast<float>(framebufferWidth), static_cast<float>(framebufferHeight));
    const auto roadStyleDisabledLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadStyleDisabled");
    const auto roadStyleConfigValidLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadStyleConfigValid");
    const auto roadColorsLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadColors[0]");
    const auto roadWidthsLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadWidths[0]");
    const bool roadStyleConfigValid =
        roadColorsLocation >= 0 && roadWidthsLocation >= 0;
    glUniform1ui(roadStyleDisabledLocation, styles.enabled ? 0u : 1u);
    glUniform1ui(roadStyleConfigValidLocation, roadStyleConfigValid ? 1u : 0u);
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uRoadWidthScale"),
                styles.globalWidthScale);
    if (roadStyleConfigValid) {
        glUniform4fv(roadColorsLocation,
                     static_cast<GLsizei>(styles.colors.size()), styles.colors.front().data());
        glUniform1fv(roadWidthsLocation,
                     static_cast<GLsizei>(styles.widthsPixels.size()), styles.widthsPixels.data());
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, drawCommandBuffer_.Id());
    glDrawArraysIndirect(GL_LINES, nullptr);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
}

} // namespace chmv::gpu::streaming
