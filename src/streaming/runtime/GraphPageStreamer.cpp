#include "streaming/runtime/GraphPageStreamer.h"

#include "data/ch/CHProjection.h"
#include "data/ch/CHTypes.h"
#include "streaming/analysis/TextSourceScanner.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace chmv::streaming::runtime {
namespace {

using analysis::detail::TextSourceScanner;

constexpr double kPi = 3.14159265358979323846;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

std::uint32_t DecodeChild(std::int64_t value) {
    if (value < 0) {
        return data::InvalidEdgeId;
    }
    if (value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("edge child id does not fit in uint32");
    }
    return static_cast<std::uint32_t>(value);
}

constexpr double kWebMercatorRadius = 6378137.0;
constexpr double kWebMercatorMaxLatitude = 85.0511287798066;
constexpr double kWorldHalfExtentMeters = kPi * kWebMercatorRadius;

data::CHProjection RuntimeProjection(const index::CHIndexData& index) {
    data::CHProjection projection;
    projection.centerLatitude = (index.minLatitude + index.maxLatitude) * 0.5;
    projection.centerLongitude = (index.minLongitude + index.maxLongitude) * 0.5;
    projection.longitudeScale = std::cos(projection.centerLatitude * kPi / 180.0);
    const auto halfWidth = std::max(
        (index.maxLongitude - index.minLongitude) * 0.5 * projection.longitudeScale, 1e-12);
    const auto halfHeight =
        std::max((index.maxLatitude - index.minLatitude) * 0.5, 1e-12);
    projection.normalization = 0.95 / std::max(halfWidth, halfHeight);
    return projection;
}

data::ProjectedPoint WorldCellBoundaryPoint(std::uint32_t x, std::uint32_t y,
                                            double cellSizeMeters,
                                            const data::CHProjection& projection) {
    const auto xMeters = std::clamp(
        static_cast<double>(x) * cellSizeMeters - kWorldHalfExtentMeters,
        -kWorldHalfExtentMeters, kWorldHalfExtentMeters);
    const auto yMeters = std::clamp(
        static_cast<double>(y) * cellSizeMeters - kWorldHalfExtentMeters,
        -kWorldHalfExtentMeters, kWorldHalfExtentMeters);
    const auto longitude = xMeters / kWebMercatorRadius * 180.0 / kPi;
    const auto latitude =
        (2.0 * std::atan(std::exp(yMeters / kWebMercatorRadius)) - kPi * 0.5) *
        180.0 / kPi;
    return projection.Project(latitude, longitude);
}

double PointSegmentDistance(data::ProjectedPoint point, data::ProjectedPoint segmentStart,
                            data::ProjectedPoint segmentEnd) {
    const auto dx = segmentEnd.x - segmentStart.x;
    const auto dy = segmentEnd.y - segmentStart.y;
    const auto lengthSquared = dx * dx + dy * dy;
    if (lengthSquared <= 1e-30) {
        return std::hypot(point.x - segmentStart.x, point.y - segmentStart.y);
    }
    const auto px = point.x - segmentStart.x;
    const auto py = point.y - segmentStart.y;
    const auto t = std::clamp((px * dx + py * dy) / lengthSquared, 0.0, 1.0);
    const auto closestX = segmentStart.x + t * dx;
    const auto closestY = segmentStart.y + t * dy;
    return std::hypot(point.x - closestX, point.y - closestY);
}

float ConservativeGeometryError(std::uint64_t packedBounds,
                                data::ProjectedPoint source,
                                data::ProjectedPoint target,
                                double cellSizeMeters,
                                const data::CHProjection& projection) {
    const auto minX = static_cast<std::uint32_t>(packedBounds & 0xffffu);
    const auto minY = static_cast<std::uint32_t>((packedBounds >> 16u) & 0xffffu);
    const auto maxX = static_cast<std::uint32_t>((packedBounds >> 32u) & 0xffffu);
    const auto maxY = static_cast<std::uint32_t>((packedBounds >> 48u) & 0xffffu);
    const std::uint32_t xs[] = {minX, maxX + 1u};
    const std::uint32_t ys[] = {minY, maxY + 1u};
    double error = 0.0;
    for (const auto x : xs) {
        for (const auto y : ys) {
            error = std::max(
                error,
                PointSegmentDistance(
                    WorldCellBoundaryPoint(x, y, cellSizeMeters, projection), source, target));
        }
    }
    // The resident node cache stores projected endpoints as floats. Inflate the bound very
    // slightly so that float round-off cannot turn a conservative preprocess bound into an
    // under-estimate at the Adaptive DFS stopping test.
    return static_cast<float>(error * 1.0001 + 1e-7);
}

float ConservativeGeometryErrorCached(std::uint64_t packedBounds,
                                      data::ProjectedPoint source,
                                      data::ProjectedPoint target,
                                      std::span<const double> boundaryX,
                                      std::span<const double> boundaryY) {
    const auto minX = static_cast<std::uint32_t>(packedBounds & 0xffffu);
    const auto minY = static_cast<std::uint32_t>((packedBounds >> 16u) & 0xffffu);
    const auto maxX = static_cast<std::uint32_t>((packedBounds >> 32u) & 0xffffu);
    const auto maxY = static_cast<std::uint32_t>((packedBounds >> 48u) & 0xffffu);
    if (maxX + 1u >= boundaryX.size() || maxY + 1u >= boundaryY.size()) {
        throw std::runtime_error("refinement spatial bounds exceed cached fixed-grid boundaries");
    }
    const std::uint32_t xs[] = {minX, maxX + 1u};
    const std::uint32_t ys[] = {minY, maxY + 1u};
    double error = 0.0;
    for (const auto x : xs) {
        for (const auto y : ys) {
            error = std::max(
                error, PointSegmentDistance({boundaryX[x], boundaryY[y]}, source, target));
        }
    }
    return static_cast<float>(error * 1.0001 + 1e-7);
}

std::pair<std::uint32_t, std::uint32_t> WorldCell(
    double longitude, double latitude, double cellSizeMeters,
    std::uint32_t gridSize) {
    if (!(cellSizeMeters > 0.0) || gridSize == 0) {
        return {0, 0};
    }
    const auto lon = std::clamp(longitude, -180.0, 180.0);
    const auto lat = std::clamp(latitude, -kWebMercatorMaxLatitude, kWebMercatorMaxLatitude);
    const auto xMeters = kWebMercatorRadius * lon * kPi / 180.0 + kWorldHalfExtentMeters;
    const auto latRadians = lat * kPi / 180.0;
    const auto yMeters = kWebMercatorRadius *
                             std::log(std::tan(kPi * 0.25 + latRadians * 0.5)) +
                         kWorldHalfExtentMeters;
    const auto toCell = [gridSize, cellSizeMeters](double meters) {
        const auto raw = meters <= 0.0 ? std::uint64_t{0}
                                        : static_cast<std::uint64_t>(meters / cellSizeMeters);
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(raw, static_cast<std::uint64_t>(gridSize) - 1u));
    };
    return {toCell(xMeters), toCell(yMeters)};
}

std::uint32_t RangeDistance(std::uint32_t aMinimum, std::uint32_t aMaximum,
                            std::uint32_t bMinimum, std::uint32_t bMaximum) {
    if (aMaximum < bMinimum) {
        return bMinimum - aMaximum;
    }
    if (bMaximum < aMinimum) {
        return aMinimum - bMaximum;
    }
    return 0;
}

std::uint64_t SpatialBucketKey(std::uint32_t level, std::uint32_t x,
                               std::uint32_t y) {
    return static_cast<std::uint64_t>(x) |
           (static_cast<std::uint64_t>(y) << 16u) |
           (static_cast<std::uint64_t>(level) << 32u);
}

bool PackedBoundsIntersects(std::uint64_t packed, const StreamingSpatialWindow& window) {
    const auto minX = static_cast<std::uint32_t>(packed & 0xffffu);
    const auto minY = static_cast<std::uint32_t>((packed >> 16u) & 0xffffu);
    const auto maxX = static_cast<std::uint32_t>((packed >> 32u) & 0xffffu);
    const auto maxY = static_cast<std::uint32_t>((packed >> 48u) & 0xffffu);
    return maxX >= window.minX && minX <= window.maxX &&
           maxY >= window.minY && minY <= window.maxY;
}

std::uint64_t BlockByteSize(const std::vector<index::CHIndexSourceBlock>& blocks,
                            std::uint32_t blockId, std::uint64_t finalOffset) {
    if (blockId >= blocks.size()) {
        throw std::runtime_error("source block id out of range");
    }
    const auto begin = blocks[blockId].byteOffset;
    const auto end = blockId + 1u < blocks.size() ? blocks[blockId + 1u].byteOffset : finalOffset;
    return end >= begin ? end - begin : 0;
}

struct TemporaryEdge {
    std::uint32_t globalEdgeId = 0;
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    std::uint32_t childA = data::InvalidEdgeId;
    std::uint32_t childB = data::InvalidEdgeId;
    std::uint32_t roadStyleType = 0;
    std::int32_t birthLevel = -1;
    std::int32_t deathLevel = -1;
    bool graphLoaded = false;
    bool rangeLoaded = false;
};

} // namespace

SystemMemoryInfo QuerySystemMemoryInfo() {
    SystemMemoryInfo result;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        result.totalPhysicalBytes = status.ullTotalPhys;
        result.availablePhysicalBytes = status.ullAvailPhys;
    }
#elif defined(__linux__)
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        std::uint64_t valueKiB = 0;
        std::string unit;
        while (meminfo >> key >> valueKiB >> unit) {
            if (key == "MemTotal:") {
                result.totalPhysicalBytes = valueKiB * 1024ull;
            } else if (key == "MemAvailable:") {
                result.availablePhysicalBytes = valueKiB * 1024ull;
            }
        }
    }
    if (result.totalPhysicalBytes == 0 || result.availablePhysicalBytes == 0) {
        struct sysinfo info {};
        if (sysinfo(&info) == 0) {
            result.totalPhysicalBytes =
                static_cast<std::uint64_t>(info.totalram) * info.mem_unit;
            result.availablePhysicalBytes =
                static_cast<std::uint64_t>(info.freeram + info.bufferram) * info.mem_unit;
        }
    }
#endif
    return result;
}

std::uint64_t RecommendGraphPageCacheBudget(const SystemMemoryInfo& memory) {
    // Runtime streaming is deliberately allowed to use a large fraction of otherwise idle RAM.
    // The old 4 GiB ceiling (and the application's 1 GiB explicit default) forced a 50+ GiB EUR
    // dataset to continuously evict decoded node/refinement data even on 48 GiB machines. Keep a
    // generous OS/application reserve, then use 60% of what remains, capped at 15 GiB.
    if (memory.availablePhysicalBytes == 0) {
        return 4ull * kGiB;
    }

    const auto systemReserve = std::max<std::uint64_t>(6ull * kGiB,
                                                        memory.totalPhysicalBytes / 5u);
    if (memory.availablePhysicalBytes <= systemReserve + 512ull * kMiB) {
        return 512ull * kMiB;
    }
    const auto afterReserve = memory.availablePhysicalBytes - systemReserve;
    return std::clamp<std::uint64_t>(afterReserve * 3u / 5u, 512ull * kMiB, 15ull * kGiB);
}

GraphPageStreamer::GraphPageStreamer(index::CHIndexData index,
                                     std::filesystem::path graphPath,
                                     std::filesystem::path rangesPath,
                                     GraphPageStreamerConfig config)
    : index_(std::move(index)),
      graphPath_(std::move(graphPath)),
      rangesPath_(std::move(rangesPath)),
      config_(config) {
    if (config_.ramBudgetBytes == 0) {
        config_.ramBudgetBytes = RecommendGraphPageCacheBudget(QuerySystemMemoryInfo());
    }
    if (config_.softBudgetRatio <= 0.0 || config_.softBudgetRatio > 1.0) {
        throw std::runtime_error("streaming soft-budget ratio must be in (0, 1]");
    }
    RecomputeBudgetPartitionLocked();

    const auto refinementTileRecords = RuntimeRefinementTileRecordCount(index_.edgeBlockRecordCount);
    const auto refinementTileCount = RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount);
    if (refinementTileRecords == 0u || refinementTileCount == 0u) {
        throw std::runtime_error("streaming refinement tile layout is invalid");
    }
    refinementTileByteOffsets_.assign(
        refinementTileCount, std::numeric_limits<std::uint64_t>::max());
    for (std::uint32_t sourceBlockId = 0; sourceBlockId < index_.graphEdgeBlocks.size();
         ++sourceBlockId) {
        const auto& sourceBlock = index_.graphEdgeBlocks[sourceBlockId];
        if (sourceBlock.recordCount == 0u || sourceBlock.firstRecord >= index_.edgeCount) {
            continue;
        }
        const auto tileId = static_cast<std::uint32_t>(sourceBlock.firstRecord / refinementTileRecords);
        if (tileId < refinementTileByteOffsets_.size()) {
            refinementTileByteOffsets_[tileId] = sourceBlock.byteOffset;
        }
    }

    // Cache projected fixed-grid boundaries once. Adaptive backing loads can then compute the same
    // conservative error as before without doing exp/atan for every shortcut edge.
    const auto projection = RuntimeProjection(index_);
    geometryBoundaryX_.resize(static_cast<std::size_t>(index_.spatialGridSize) + 1u);
    geometryBoundaryY_.resize(static_cast<std::size_t>(index_.spatialGridSize) + 1u);
    for (std::uint32_t i = 0; i <= index_.spatialGridSize; ++i) {
        const auto pointX = WorldCellBoundaryPoint(i, 0u, index_.spatialCellSizeMeters, projection);
        const auto pointY = WorldCellBoundaryPoint(0u, i, index_.spatialCellSizeMeters, projection);
        geometryBoundaryX_[i] = pointX.x;
        geometryBoundaryY_[i] = pointY.y;
    }

    // Build compact runtime acceleration once. The source CHIDX keeps the exact cell->tile refs,
    // but an overview query must not walk all of those refs just to discover that only a dozen
    // high-LOD tiles are alive. Page bounds + level-major alive bitsets let QueryPages choose the
    // more selective dimension (LOD-first or spatial-first) without changing CHIDX/preprocess.
    const auto pageCount = index_.graphPages.size();
    pageSpatialBounds_.resize(pageCount);
    const auto gridMax = index_.spatialGridSize == 0
                             ? std::uint64_t{0}
                             : static_cast<std::uint64_t>(index_.spatialGridSize) - 1u;
    for (const auto& ref : index_.spatialPageRefs) {
        if (ref.pageId >= pageCount) {
            throw std::runtime_error("CH index spatial page reference is invalid");
        }
        const auto level = static_cast<std::uint32_t>((ref.key >> 32u) & 0xffu);
        if (level >= 31u) {
            continue;
        }
        const auto tileX = static_cast<std::uint32_t>(ref.key & 0xffffu);
        const auto tileY = static_cast<std::uint32_t>((ref.key >> 16u) & 0xffffu);
        const auto scale = std::uint64_t{1} << level;
        const auto reach = scale / 2u;
        const auto centerStartX = static_cast<std::uint64_t>(tileX) * scale;
        const auto centerStartY = static_cast<std::uint64_t>(tileY) * scale;
        const auto centerEndX = std::min<std::uint64_t>(
            gridMax, (static_cast<std::uint64_t>(tileX) + 1u) * scale - 1u);
        const auto centerEndY = std::min<std::uint64_t>(
            gridMax, (static_cast<std::uint64_t>(tileY) + 1u) * scale - 1u);
        const auto minX = static_cast<std::uint32_t>(
            centerStartX > reach ? centerStartX - reach : 0u);
        const auto minY = static_cast<std::uint32_t>(
            centerStartY > reach ? centerStartY - reach : 0u);
        const auto maxX = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(gridMax, centerEndX + reach));
        const auto maxY = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(gridMax, centerEndY + reach));
        auto& bounds = pageSpatialBounds_[ref.pageId];
        if (!bounds.valid) {
            bounds = {minX, minY, maxX, maxY, true};
        } else {
            bounds.minX = std::min(bounds.minX, minX);
            bounds.minY = std::min(bounds.minY, minY);
            bounds.maxX = std::max(bounds.maxX, maxX);
            bounds.maxY = std::max(bounds.maxY, maxY);
        }
    }

    lodAliveWordsPerLevel_ = (pageCount + 63u) / 64u;
    lodAlivePageCounts_.assign(static_cast<std::size_t>(index_.maxLevel) + 1u, 0u);
    lodAlivePageBits_.assign((static_cast<std::size_t>(index_.maxLevel) + 1u) *
                                 lodAliveWordsPerLevel_,
                             0ull);
    if (index_.lodMaskWordsPerPage != 0u) {
        for (std::uint32_t pageId = 0; pageId < pageCount; ++pageId) {
            const auto& page = index_.graphPages[pageId];
            for (std::uint32_t wordIndex = 0; wordIndex < index_.lodMaskWordsPerPage;
                 ++wordIndex) {
                const auto sourceIndex = static_cast<std::size_t>(page.lodMaskOffset) + wordIndex;
                if (sourceIndex >= index_.pageAliveMasks.size()) {
                    throw std::runtime_error("CH index page alive-mask range is invalid");
                }
                auto bits = index_.pageAliveMasks[sourceIndex];
                while (bits != 0u) {
                    const auto bit = static_cast<std::uint32_t>(std::countr_zero(bits));
                    const auto level = wordIndex * 64u + bit;
                    if (level <= index_.maxLevel) {
                        const auto destination =
                            static_cast<std::size_t>(level) * lodAliveWordsPerLevel_ +
                            pageId / 64u;
                        lodAlivePageBits_[destination] |=
                            std::uint64_t{1} << (pageId & 63u);
                        ++lodAlivePageCounts_[level];
                    }
                    bits &= bits - 1u;
                }
            }
        }
    } else {
        // Legacy safety path. v7 always has masks, but never make an absent optional accelerator
        // change correctness. QueryPages will choose the spatial index if these counts are zero.
        lodAlivePageBits_.clear();
        lodAlivePageCounts_.clear();
        lodAliveWordsPerLevel_ = 0;
    }

    spatialCandidateStamp_.resize(pageCount, 0u);
    spatialCandidateRadius_.resize(index_.graphPages.size(), 0xffu);
    selectionStamp_.resize(index_.graphPages.size(), 0u);
    selectionScratch_.resize(index_.graphPages.size());

    const auto hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    const auto automaticWorkers = std::clamp<std::uint32_t>(
        hardwareThreads > 2u ? hardwareThreads - 2u : 1u, 1u, 12u);
    const auto workerCount = config_.workerCount == 0
                                 ? automaticWorkers
                                 : std::clamp<std::uint32_t>(config_.workerCount, 1u, 12u);
    workers_.reserve(workerCount);
    for (std::uint32_t i = 0; i < workerCount; ++i) {
        workers_.emplace_back([this](std::stop_token stopToken) { Worker(stopToken); });
    }
    plannerWorker_ = std::jthread([this](std::stop_token stopToken) { PlannerWorker(stopToken); });
}

GraphPageStreamer::~GraphPageStreamer() {
    if (plannerWorker_.joinable()) {
        plannerWorker_.request_stop();
    }
    plannerCondition_.notify_all();
    if (plannerWorker_.joinable()) {
        plannerWorker_.join();
    }
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.request_stop();
        }
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

const GraphPageStreamer::PageQueryResult& GraphPageStreamer::QueryPages(
    const StreamingViewRequest& view) const {
    if (view.maxLatitude < index_.minLatitude || view.minLatitude > index_.maxLatitude ||
        view.maxLongitude < index_.minLongitude || view.minLongitude > index_.maxLongitude ||
        index_.spatialGridSize == 0 || !(index_.spatialCellSizeMeters > 0.0)) {
        queryCacheKey_.reset();
        queryCacheResult_ = {};
        ++queryCacheRevision_;
        lastRootPlannerMs_ = 0.0;
        lastPlannerAliveCandidates_ = 0;
        lastPlannerSpatialCandidates_ = 0;
        lastPlannerUsedLodFirst_ = false;
        return queryCacheResult_;
    }

    const auto lodFloat = std::clamp(
        view.lodLevelFloat >= 0.0f ? view.lodLevelFloat : static_cast<float>(view.lodLevel),
        0.0f, static_cast<float>(index_.maxLevel));
    const auto renderLow = static_cast<std::uint32_t>(std::floor(lodFloat));
    const auto renderHigh = static_cast<std::uint32_t>(std::ceil(lodFloat));
    const bool hasExplicitPrefetchWindow =
        view.prefetchMinLod != 0xffffffffu && view.prefetchMaxLod != 0xffffffffu;
    const auto requestedMin = hasExplicitPrefetchWindow
                                  ? std::min(view.prefetchMinLod, view.prefetchMaxLod)
                                  : renderLow;
    const auto requestedMax = hasExplicitPrefetchWindow
                                  ? std::max(view.prefetchMinLod, view.prefetchMaxLod)
                                  : renderHigh;
    const auto prefetchMin = std::min(index_.maxLevel, std::min(requestedMin, renderLow));
    const auto prefetchMax = std::min(index_.maxLevel, std::max(requestedMax, renderHigh));
    const auto focusLod = std::clamp(
        view.prefetchFocusLod >= 0.0f ? view.prefetchFocusLod : lodFloat,
        0.0f, static_cast<float>(index_.maxLevel));

    const auto [aX, aY] = WorldCell(view.minLongitude, view.minLatitude,
                                    index_.spatialCellSizeMeters, index_.spatialGridSize);
    const auto [bX, bY] = WorldCell(view.maxLongitude, view.maxLatitude,
                                    index_.spatialCellSizeMeters, index_.spatialGridSize);
    const auto minViewX = std::min(aX, bX);
    const auto maxViewX = std::max(aX, bX);
    const auto minViewY = std::min(aY, bY);
    const auto maxViewY = std::max(aY, bY);

    const PageQueryCacheKey cacheKey{
        minViewX,
        minViewY,
        maxViewX,
        maxViewY,
        renderLow,
        renderHigh,
        prefetchMin,
        prefetchMax,
        static_cast<std::int32_t>(std::lround(focusLod * 2.0f)),
        softBudgetBytes_,
    };
    if (queryCacheKey_ && *queryCacheKey_ == cacheKey) {
        ++rootPlannerCacheReuses_;
        lastRootPlannerMs_ = 0.0;
        return queryCacheResult_;
    }

    const auto plannerBegin = std::chrono::steady_clock::now();
    ++rootPlannerRebuilds_;
    lastPlannerAliveCandidates_ = 0;
    lastPlannerSpatialCandidates_ = 0;
    lastPlannerUsedLodFirst_ = false;

    PageQueryResult result;
    result.referenceSpatialLevel = 0;
    result.selectedPrefetchMinLod = renderLow;
    result.selectedPrefetchMaxLod = renderHigh;
    result.viewportWindow = {minViewX, minViewY, maxViewX, maxViewY};

    const auto residencyRadius = config_.spatialPrefetchRadius;
    const auto expandedMinX = minViewX > residencyRadius ? minViewX - residencyRadius : 0u;
    const auto expandedMinY = minViewY > residencyRadius ? minViewY - residencyRadius : 0u;
    const auto expandedMaxX =
        std::min(index_.spatialGridSize - 1u, maxViewX + residencyRadius);
    const auto expandedMaxY =
        std::min(index_.spatialGridSize - 1u, maxViewY + residencyRadius);
    result.refinementWindow = {expandedMinX, expandedMinY, expandedMaxX, expandedMaxY};

    if (index_.spatialPageRefs.empty() || index_.graphPages.empty()) {
        queryCacheKey_ = cacheKey;
        queryCacheResult_ = std::move(result);
        ++queryCacheRevision_;
        lastRootPlannerMs_ = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - plannerBegin)
                                 .count();
        return queryCacheResult_;
    }

    struct CandidatePage {
        std::uint32_t pageId = 0;
        std::uint32_t radius = 0;
    };

    // Spatial lookup is lazy. A Europe-wide overview at high CH LOD has very few alive tiles; in
    // that case touching the 10M+ cell refs is pure waste. Only local/detail queries pay for the
    // spatial index, and even then it is built once for all requested LODs in this plan.
    std::vector<std::uint32_t> spatialCandidates;
    bool spatialCandidatesBuilt = false;
    const auto gridMax = static_cast<std::uint64_t>(index_.spatialGridSize) - 1u;
    const auto ensureSpatialCandidates = [&]() {
        if (spatialCandidatesBuilt) {
            return;
        }
        spatialCandidatesBuilt = true;
        ++spatialQueryEpoch_;
        if (spatialQueryEpoch_ == 0u) {
            std::fill(spatialCandidateStamp_.begin(), spatialCandidateStamp_.end(), 0u);
            spatialQueryEpoch_ = 1u;
        }
        const auto spatialEpoch = spatialQueryEpoch_;
        spatialCandidates.reserve(std::min<std::size_t>(index_.graphPages.size(), 8192u));

        const auto maxSpatialLevel = static_cast<std::uint32_t>(
            (index_.spatialPageRefs.back().key >> 32u) & 0xffu);
        for (std::uint32_t spatialLevel = 0;
             spatialLevel <= maxSpatialLevel && spatialLevel < 31u; ++spatialLevel) {
            const auto scale = std::uint64_t{1} << spatialLevel;
            const auto reach = scale / 2u;
            const auto centerMinX = expandedMinX > reach ? expandedMinX - reach : 0u;
            const auto centerMinY = expandedMinY > reach ? expandedMinY - reach : 0u;
            const auto centerMaxX = std::min<std::uint64_t>(
                gridMax, static_cast<std::uint64_t>(expandedMaxX) + reach);
            const auto centerMaxY = std::min<std::uint64_t>(
                gridMax, static_cast<std::uint64_t>(expandedMaxY) + reach);
            const auto minTileX = static_cast<std::uint32_t>(centerMinX / scale);
            const auto maxTileX = static_cast<std::uint32_t>(centerMaxX / scale);
            const auto minTileY = static_cast<std::uint32_t>(centerMinY / scale);
            const auto maxTileY = static_cast<std::uint32_t>(centerMaxY / scale);

            for (std::uint32_t tileY = minTileY; tileY <= maxTileY; ++tileY) {
                const auto firstKey = SpatialBucketKey(spatialLevel, minTileX, tileY);
                const auto lastKey = SpatialBucketKey(spatialLevel, maxTileX, tileY);
                const auto first = std::lower_bound(
                    index_.spatialPageRefs.begin(), index_.spatialPageRefs.end(), firstKey,
                    [](const index::CHIndexSpatialPageRef& entry, std::uint64_t key) {
                        return entry.key < key;
                    });
                const auto last = std::upper_bound(
                    first, index_.spatialPageRefs.end(), lastKey,
                    [](std::uint64_t key, const index::CHIndexSpatialPageRef& entry) {
                        return key < entry.key;
                    });

                for (auto it = first; it != last; ++it) {
                    if (it->pageId >= index_.graphPages.size()) {
                        continue;
                    }
                    const auto refTileX = static_cast<std::uint32_t>(it->key & 0xffffu);
                    const auto refTileY = static_cast<std::uint32_t>((it->key >> 16u) & 0xffffu);
                    const auto centerStartX = static_cast<std::uint64_t>(refTileX) * scale;
                    const auto centerStartY = static_cast<std::uint64_t>(refTileY) * scale;
                    const auto centerEndX = std::min<std::uint64_t>(
                        gridMax, (static_cast<std::uint64_t>(refTileX) + 1u) * scale - 1u);
                    const auto centerEndY = std::min<std::uint64_t>(
                        gridMax, (static_cast<std::uint64_t>(refTileY) + 1u) * scale - 1u);
                    const auto pageMinX = static_cast<std::uint32_t>(
                        centerStartX > reach ? centerStartX - reach : 0u);
                    const auto pageMinY = static_cast<std::uint32_t>(
                        centerStartY > reach ? centerStartY - reach : 0u);
                    const auto pageMaxX = static_cast<std::uint32_t>(
                        std::min<std::uint64_t>(gridMax, centerEndX + reach));
                    const auto pageMaxY = static_cast<std::uint32_t>(
                        std::min<std::uint64_t>(gridMax, centerEndY + reach));
                    const auto dx = RangeDistance(pageMinX, pageMaxX, minViewX, maxViewX);
                    const auto dy = RangeDistance(pageMinY, pageMaxY, minViewY, maxViewY);
                    const auto pageRadius = std::max(dx, dy);
                    if (pageRadius > residencyRadius) {
                        continue;
                    }

                    if (spatialCandidateStamp_[it->pageId] != spatialEpoch) {
                        spatialCandidateStamp_[it->pageId] = spatialEpoch;
                        spatialCandidateRadius_[it->pageId] = static_cast<std::uint8_t>(
                            std::min<std::uint32_t>(pageRadius, 0xffu));
                        spatialCandidates.push_back(it->pageId);
                    } else {
                        spatialCandidateRadius_[it->pageId] = static_cast<std::uint8_t>(
                            std::min<std::uint32_t>(spatialCandidateRadius_[it->pageId],
                                                    pageRadius));
                    }
                }

                if (tileY == std::numeric_limits<std::uint32_t>::max()) {
                    break;
                }
            }
        }
        lastPlannerSpatialCandidates_ = static_cast<std::uint32_t>(
            std::min<std::size_t>(spatialCandidates.size(),
                                  std::numeric_limits<std::uint32_t>::max()));
    };

    const auto pageRadiusFromBounds = [&](std::uint32_t pageId) {
        if (pageId >= pageSpatialBounds_.size() || !pageSpatialBounds_[pageId].valid) {
            return residencyRadius + 1u;
        }
        const auto& bounds = pageSpatialBounds_[pageId];
        const auto dx = RangeDistance(bounds.minX, bounds.maxX, minViewX, maxViewX);
        const auto dy = RangeDistance(bounds.minY, bounds.maxY, minViewY, maxViewY);
        return std::max(dx, dy);
    };

    const auto viewWidth = static_cast<std::uint64_t>(maxViewX) - minViewX + 1u;
    const auto viewHeight = static_cast<std::uint64_t>(maxViewY) - minViewY + 1u;
    const auto viewArea = viewWidth * viewHeight;
    const auto worldArea = static_cast<std::uint64_t>(index_.spatialGridSize) *
                           index_.spatialGridSize;
    const auto estimatedSpatialPages = static_cast<std::uint64_t>(std::ceil(
        static_cast<double>(index_.graphPages.size()) *
        std::min(1.0, 4.0 * static_cast<double>(viewArea) /
                          std::max<double>(static_cast<double>(worldArea), 1.0))));

    const auto collectLevelCandidates = [&](std::uint32_t queryLod,
                                            std::uint32_t queryRadius,
                                            std::vector<CandidatePage>& output) {
        output.clear();
        if (queryLod > index_.maxLevel) {
            return;
        }

        const auto aliveCount = queryLod < lodAlivePageCounts_.size()
                                    ? lodAlivePageCounts_[queryLod]
                                    : 0u;
        const auto lodFirstThreshold = std::max<std::uint64_t>(4096u, estimatedSpatialPages);
        const bool canUseLodFirst = lodAliveWordsPerLevel_ != 0u && aliveCount != 0u;
        const bool useLodFirst = canUseLodFirst && aliveCount <= lodFirstThreshold;

        if (useLodFirst) {
            lastPlannerUsedLodFirst_ = true;
            const auto accumulatedAliveCandidates = std::min<std::uint64_t>(
                static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()),
                static_cast<std::uint64_t>(
                    lastPlannerAliveCandidates_.load(std::memory_order_relaxed)) + aliveCount);
            lastPlannerAliveCandidates_.store(
                static_cast<std::uint32_t>(accumulatedAliveCandidates),
                std::memory_order_relaxed);
            const auto base = static_cast<std::size_t>(queryLod) * lodAliveWordsPerLevel_;
            for (std::size_t wordIndex = 0; wordIndex < lodAliveWordsPerLevel_; ++wordIndex) {
                auto bits = lodAlivePageBits_[base + wordIndex];
                while (bits != 0u) {
                    const auto bit = static_cast<std::uint32_t>(std::countr_zero(bits));
                    const auto pageId = static_cast<std::uint32_t>(wordIndex * 64u + bit);
                    if (pageId < index_.graphPages.size()) {
                        const auto radius = pageRadiusFromBounds(pageId);
                        if (radius <= queryRadius) {
                            output.push_back({pageId, radius});
                        }
                    }
                    bits &= bits - 1u;
                }
            }
            return;
        }

        ensureSpatialCandidates();
        output.reserve(std::min<std::size_t>(spatialCandidates.size(), 4096u));
        for (const auto pageId : spatialCandidates) {
            const auto radius = static_cast<std::uint32_t>(spatialCandidateRadius_[pageId]);
            if (radius <= queryRadius && index_.GraphPageHasAliveEdges(pageId, queryLod)) {
                output.push_back({pageId, radius});
            }
        }
    };

    ++selectionEpoch_;
    if (selectionEpoch_ == 0u) {
        std::fill(selectionStamp_.begin(), selectionStamp_.end(), 0u);
        selectionEpoch_ = 1u;
    }
    const auto selectionEpoch = selectionEpoch_;
    std::vector<std::uint32_t> selectedPageIds;
    selectedPageIds.reserve(8192u);
    std::uint64_t selectedBytes = 0;

    const auto mergeSelection = [&](const PageSelection& candidate) {
        const auto pageId = candidate.pageId;
        if (selectionStamp_[pageId] != selectionEpoch) {
            selectionStamp_[pageId] = selectionEpoch;
            selectionScratch_[pageId] = candidate;
            selectedPageIds.push_back(pageId);
            selectedBytes += EstimatePageBytes(pageId);
            return;
        }
        auto& current = selectionScratch_[pageId];
        current.required = current.required || candidate.required;
        current.renderCandidate = current.renderCandidate || candidate.renderCandidate;
        current.gpuWarm = current.gpuWarm || candidate.gpuWarm;
        current.priority = std::min(current.priority, candidate.priority);
        current.spatialRadius = std::min(current.spatialRadius, candidate.spatialRadius);
        current.lodDistance = current.renderCandidate
                                  ? 0u
                                  : std::min(current.lodDistance, candidate.lodDistance);
        if (current.required) {
            current.priority = 0u;
            current.lodDistance = 0u;
            current.spatialRadius = 0u;
        }
    };

    std::vector<CandidatePage> levelCandidates;
    levelCandidates.reserve(4096u);
    const auto addRenderLod = [&](std::uint32_t queryLod) {
        collectLevelCandidates(queryLod, residencyRadius, levelCandidates);
        for (const auto& candidate : levelCandidates) {
            const bool required = candidate.radius == 0u;
            const auto priority = required ? 0u : 8u + candidate.radius;
            mergeSelection({candidate.pageId, candidate.radius, 0u, priority,
                            required, true, true});
        }
    };

    addRenderLod(renderLow);
    if (renderHigh != renderLow) {
        addRenderLod(renderHigh);
    }

    // Speculative LOD data is a near-future working set, not a request to fill spare RAM.
    const auto predictedSpan = std::abs(focusLod - lodFloat);
    const auto predictionWeight = std::clamp(predictedSpan / 4.0f, 0.0f, 1.0f);
    const auto maxSpeculativeBytes = softBudgetBytes_ > selectedBytes
                                         ? softBudgetBytes_ - selectedBytes
                                         : 0ull;
    const auto scaledCurrentBytes = std::max<std::uint64_t>(selectedBytes, 8ull * kMiB);
    const auto speculativeCeiling = std::min<std::uint64_t>(
        scaledCurrentBytes, std::max<std::uint64_t>(8ull * kMiB, softBudgetBytes_ / 8u));
    const auto speculativeBudget = static_cast<std::uint64_t>(
        static_cast<double>(speculativeCeiling) * static_cast<double>(predictionWeight));
    const auto prefetchBudget = selectedBytes +
        std::min<std::uint64_t>(maxSpeculativeBytes, speculativeBudget);

    std::vector<std::uint32_t> candidateLevels;
    candidateLevels.reserve(prefetchMax >= prefetchMin ? prefetchMax - prefetchMin + 1u : 0u);
    for (std::uint32_t queryLod = prefetchMin; queryLod <= prefetchMax; ++queryLod) {
        if (queryLod != renderLow && queryLod != renderHigh) {
            candidateLevels.push_back(queryLod);
        }
        if (queryLod == prefetchMax) {
            break;
        }
    }

    const auto predictedDirection = focusLod > lodFloat + 0.05f
                                        ? 1
                                        : (focusLod < lodFloat - 0.05f ? -1 : 0);
    std::sort(candidateLevels.begin(), candidateLevels.end(),
              [&](std::uint32_t a, std::uint32_t b) {
                  const auto aCurrent = std::abs(static_cast<float>(a) - lodFloat);
                  const auto bCurrent = std::abs(static_cast<float>(b) - lodFloat);
                  const auto aFocus = std::abs(static_cast<float>(a) - focusLod);
                  const auto bFocus = std::abs(static_cast<float>(b) - focusLod);
                  const auto aScore = std::min(aCurrent, aFocus);
                  const auto bScore = std::min(bCurrent, bFocus);
                  if (aScore != bScore) {
                      return aScore < bScore;
                  }
                  if (predictedDirection != 0) {
                      const bool aAhead = predictedDirection > 0
                                              ? static_cast<float>(a) >= lodFloat
                                              : static_cast<float>(a) <= lodFloat;
                      const bool bAhead = predictedDirection > 0
                                              ? static_cast<float>(b) >= lodFloat
                                              : static_cast<float>(b) <= lodFloat;
                      if (aAhead != bAhead) {
                          return aAhead;
                      }
                  }
                  return aCurrent != bCurrent ? aCurrent < bCurrent : a < b;
              });

    for (const auto queryLod : candidateLevels) {
        if (selectedBytes >= prefetchBudget) {
            break;
        }
        const auto lodDistance = static_cast<std::uint32_t>(
            std::ceil(std::abs(static_cast<float>(queryLod) - lodFloat)));
        const auto focusDistance = static_cast<std::uint32_t>(
            std::ceil(std::abs(static_cast<float>(queryLod) - focusLod)));
        const auto priorityDistance = std::min(lodDistance, focusDistance);
        const auto queryRadius = lodDistance <= config_.lodPrefetchNearDistance
                                     ? std::min(config_.spatialPrefetchRadius,
                                                config_.lodPrefetchNearSpatialRadius)
                                     : 0u;
        const bool gpuWarm = lodDistance <= config_.lodPrefetchGpuWarmDistance ||
                             focusDistance <= config_.lodPrefetchGpuWarmDistance;

        collectLevelCandidates(queryLod, queryRadius, levelCandidates);
        std::uint64_t newBytes = 0;
        for (const auto& candidate : levelCandidates) {
            if (selectionStamp_[candidate.pageId] != selectionEpoch) {
                newBytes += EstimatePageBytes(candidate.pageId);
                if (newBytes > prefetchBudget - selectedBytes) {
                    break;
                }
            }
        }

        if (newBytes > prefetchBudget - selectedBytes) {
            break;
        }
        if (newBytes == 0) {
            continue;
        }

        for (const auto& candidate : levelCandidates) {
            const auto priority = 16u + priorityDistance * 2u + candidate.radius;
            mergeSelection({candidate.pageId, candidate.radius, lodDistance, priority,
                            false, false, gpuWarm});
        }
        result.selectedPrefetchMinLod = std::min(result.selectedPrefetchMinLod, queryLod);
        result.selectedPrefetchMaxLod = std::max(result.selectedPrefetchMaxLod, queryLod);
    }

    result.pages.reserve(selectedPageIds.size());
    for (const auto pageId : selectedPageIds) {
        result.pages.push_back(selectionScratch_[pageId]);
    }
    std::sort(result.pages.begin(), result.pages.end(),
              [](const PageSelection& a, const PageSelection& b) {
                  if (a.priority != b.priority) {
                      return a.priority < b.priority;
                  }
                  return a.pageId < b.pageId;
              });

    queryCacheKey_ = cacheKey;
    queryCacheResult_ = std::move(result);
    ++queryCacheRevision_;
    lastRootPlannerMs_ = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - plannerBegin)
                             .count();
    return queryCacheResult_;
}

std::uint64_t GraphPageStreamer::SubmitView(const StreamingViewRequest& view) {
    std::lock_guard lock(plannerMutex_);
    const auto sequence = ++submittedViewSequence_;
    pendingViewRequest_ = PendingViewRequest{view, sequence};
    plannerCondition_.notify_all();
    return sequence;
}

void GraphPageStreamer::PlannerWorker(std::stop_token stopToken) {
    for (;;) {
        PendingViewRequest request;
        {
            std::unique_lock lock(plannerMutex_);
            plannerCondition_.wait(lock, [&]() {
                return stopToken.stop_requested() || pendingViewRequest_.has_value();
            });
            if (stopToken.stop_requested()) {
                return;
            }
            request = *pendingViewRequest_;
            pendingViewRequest_.reset();
        }

        try {
            static_cast<void>(RequestView(request.view));
            latestAppliedViewLodFloat_.store(
                request.view.lodLevelFloat >= 0.0f
                    ? request.view.lodLevelFloat
                    : static_cast<float>(request.view.lodLevel),
                std::memory_order_release);
            latestAppliedViewSequence_.store(request.sequence, std::memory_order_release);
        } catch (const std::exception& e) {
            std::lock_guard lock(mutex_);
            lastError_ = e.what();
            latestAppliedViewSequence_.store(request.sequence, std::memory_order_release);
        }
    }
}

StreamingViewToken GraphPageStreamer::RequestView(const StreamingViewRequest& view) {
    const auto& query = QueryPages(view);
    const auto& selectedPages = query.pages;

    // Even a cached plan is revisited on the low-frequency planner tick so another bounded wave
    // can be admitted after workers have consumed the previous one. This is cheap and keeps disk
    // work progressive without rebuilding the spatial/LOD selection.
    {
        std::lock_guard lock(mutex_);
        if (appliedQueryCacheRevision_ == queryCacheRevision_) {
            PumpPageQueueLocked(selectedPages, generation_);
            UpdateDemandReadyLocked();
            condition_.notify_all();
            StreamingViewToken token;
            token.generation = generation_;
            token.requiredPageCount = static_cast<std::uint32_t>(requiredPages_.size());
            token.prefetchedPageCount = static_cast<std::uint32_t>(
                desiredPages_.size() >= requiredPages_.size()
                    ? desiredPages_.size() - requiredPages_.size()
                    : 0u);
            return token;
        }
    }

    std::unordered_set<std::uint32_t> newDesiredPages;
    std::unordered_set<std::uint32_t> newRequiredPages;
    std::unordered_set<std::uint32_t> newLodPrefetchPages;
    std::unordered_set<std::uint32_t> newGpuWarmPages;
    newDesiredPages.reserve(selectedPages.size());
    newRequiredPages.reserve(selectedPages.size());
    newLodPrefetchPages.reserve(selectedPages.size());
    newGpuWarmPages.reserve(selectedPages.size());
    for (const auto& selection : selectedPages) {
        newDesiredPages.insert(selection.pageId);
        if (selection.required) {
            newRequiredPages.insert(selection.pageId);
        }
        if (!selection.renderCandidate) {
            newLodPrefetchPages.insert(selection.pageId);
        }
        if (selection.gpuWarm) {
            newGpuWarmPages.insert(selection.pageId);
        }
    }

    std::unique_lock lock(mutex_);
    prefetchReferenceSpatialLevel_ = query.referenceSpatialLevel;
    const bool hasExplicitPrefetchWindow =
        view.prefetchMinLod != 0xffffffffu && view.prefetchMaxLod != 0xffffffffu;
    const auto displayLow = static_cast<std::uint32_t>(std::floor(
        view.lodLevelFloat >= 0.0f ? view.lodLevelFloat : static_cast<float>(view.lodLevel)));
    const auto displayHigh = static_cast<std::uint32_t>(std::ceil(
        view.lodLevelFloat >= 0.0f ? view.lodLevelFloat : static_cast<float>(view.lodLevel)));
    const auto requestedPrefetchMin = hasExplicitPrefetchWindow
                                          ? std::min(view.prefetchMinLod, view.prefetchMaxLod)
                                          : displayLow;
    const auto requestedPrefetchMax = hasExplicitPrefetchWindow
                                          ? std::max(view.prefetchMinLod, view.prefetchMaxLod)
                                          : displayHigh;
    static_cast<void>(requestedPrefetchMin);
    static_cast<void>(requestedPrefetchMax);
    currentPrefetchMinLod_ = query.selectedPrefetchMinLod;
    currentPrefetchMaxLod_ = query.selectedPrefetchMaxLod;
    currentPrefetchFocusLod_ = std::clamp(
        view.prefetchFocusLod >= 0.0f ? view.prefetchFocusLod
                                      : (view.lodLevelFloat >= 0.0f
                                             ? view.lodLevelFloat
                                             : static_cast<float>(view.lodLevel)),
        0.0f, static_cast<float>(index_.maxLevel));

    const bool changed = newRequiredPages != requiredPages_ ||
                         newDesiredPages != desiredPages_ ||
                         newLodPrefetchPages != lodPrefetchPages_ ||
                         newGpuWarmPages != gpuWarmPages_ ||
                         query.viewportWindow != currentViewportWindow_ ||
                         query.refinementWindow != currentRefinementWindow_;

    if (changed) {
        ++generation_;
        desiredPages_ = std::move(newDesiredPages);
        requiredPages_ = std::move(newRequiredPages);
        lodPrefetchPages_ = std::move(newLodPrefetchPages);
        gpuWarmPages_ = std::move(newGpuWarmPages);
        currentViewportWindow_ = query.viewportWindow;
        currentRefinementWindow_ = query.refinementWindow;

        // Cancel obsolete prediction work immediately at the logical queue level. priority_queue
        // entries cannot be erased cheaply, so their sequence/state entry is removed here and the
        // worker skips the stale heap record without touching disk. At most the already-running
        // request on each worker can finish after a violent zoom reversal.
        for (auto it = queuedPages_.begin(); it != queuedPages_.end();) {
            if (!desiredPages_.contains(it->first)) {
                it = queuedPages_.erase(it);
            } else {
                ++it;
            }
        }

        demandStart_ = std::chrono::steady_clock::now();
        currentDemandReady_ = false;
        lastDemandReadyMs_ = 0.0;
    }

    const auto generation = generation_;
    StreamingViewToken token;
    token.generation = generation;

    for (const auto& selection : selectedPages) {
        const auto pageId = selection.pageId;
        if (selection.required) {
            ++token.requiredPageCount;
        } else {
            ++token.prefetchedPageCount;
        }

        const auto found = cache_.find(pageId);
        if (found != cache_.end()) {
            found->second.lastUse = ++useCounter_;
            if (changed) {
                ++pageCacheHits_;
            }
            if (selection.required) {
                ++token.requiredCacheHits;
            }
            continue;
        }

        const bool alreadyInFlight = queuedPages_.contains(pageId) || loadingPages_.contains(pageId);
        if (changed && !alreadyInFlight) {
            ++pageCacheMisses_;
        }
        if (selection.required) {
            ++token.requiredCacheMisses;
            currentDemandReady_ = false;
        }
    }

    // Admit only a bounded priority-ordered wave. The rest remains desired and is admitted on
    // subsequent planner ticks as workers finish, so a 2k-page view never becomes a 2k-request
    // render-thread spike.
    PumpPageQueueLocked(selectedPages, generation);

    if (config_.prefetchLodChildren) {
        std::vector<std::uint32_t> lodChildren;
        for (const auto pageId : requiredPages_) {
            const auto& page = index_.graphPages[pageId];
            for (const auto childId : index_.LodChildPageRefs(page)) {
                if (desiredPages_.insert(childId).second) {
                    lodChildren.push_back(childId);
                }
            }
        }
        for (const auto childId : lodChildren) {
            if (!cache_.contains(childId) && !loadingPages_.contains(childId)) {
                QueuePageLocked(childId, 2, generation);
                ++token.prefetchedPageCount;
            }
        }
    }

    UpdateDemandReadyLocked();
    if (residentBytes_ > softBudgetBytes_) {
        EvictToLocked(softBudgetBytes_);
    }
    appliedQueryCacheRevision_ = queryCacheRevision_;
    condition_.notify_all();
    return token;
}

RefinementRequestToken GraphPageStreamer::RequestRefinementEdges(
    std::span<const RefinementEdgeRequest> requests) {
    RefinementRequestToken token;
    std::unique_lock lock(mutex_);
    token.generation = generation_;
    token.requestedEdgeCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(requests.size(), std::numeric_limits<std::uint32_t>::max()));
    totalRefinementRequests_ += requests.size();
    lastRefinementRequestCount_ = token.requestedEdgeCount;

    std::unordered_set<std::uint32_t> blocks;
    blocks.reserve(requests.size());
    std::uint32_t accepted = 0;
    for (const auto& request : requests) {
        if (request.parentEdgeId >= index_.edgeSpatialBoundsCount ||
            request.childEdgeId >= index_.edgeCount || index_.edgeBlockRecordCount == 0) {
            continue;
        }
        const auto parentBounds = !index_.edgeSpatialBounds.empty()
                                      ? index_.edgeSpatialBounds[request.parentEdgeId]
                                      : index::CHIndex::ReadEdgeSpatialBounds(
                                            index_, request.parentEdgeId, 1u)
                                            .front();
        if (!PackedBoundsIntersects(parentBounds, currentRefinementWindow_)) {
            ++token.spatiallyRejectedRequestCount;
            continue;
        }
        ++accepted;
        const auto blockId = request.childEdgeId / RuntimeRefinementTileRecordCount(index_.edgeBlockRecordCount);
        if (blockId < RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount)) {
            blocks.insert(blockId);
        }
    }
    totalSpatiallyRejectedRefinementRequests_ += token.spatiallyRejectedRequestCount;
    lastSpatiallyAcceptedRefinementRequestCount_ = accepted;
    token.uniqueBlockCount = static_cast<std::uint32_t>(blocks.size());

    for (const auto blockId : blocks) {
        desiredRefinementBlocks_.insert(blockId);
        const auto found = refinementCache_.find(blockId);
        if (found != refinementCache_.end()) {
            found->second.lastUse = ++useCounter_;
            ++refinementBlockCacheHits_;
            ++token.residentBlockHits;
            continue;
        }
        if (!queuedRefinementBlocks_.contains(blockId) &&
            !loadingRefinementBlocks_.contains(blockId)) {
            ++refinementBlockCacheMisses_;
        }
        const auto before = queuedRefinementBlocks_.size();
        QueueRefinementBlockLocked(blockId, generation_);
        if (queuedRefinementBlocks_.size() != before) {
            ++token.queuedBlockCount;
        }
    }

    condition_.notify_all();
    return token;
}

void GraphPageStreamer::RequestBackingBlocks(std::span<const std::uint32_t> blockIds) {
    std::unique_lock lock(mutex_);
    const auto refinementTileCount =
        RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount);
    for (const auto blockId : blockIds) {
        if (blockId >= refinementTileCount) {
            continue;
        }
        desiredRefinementBlocks_.insert(blockId);
        if (auto found = refinementCache_.find(blockId); found != refinementCache_.end()) {
            found->second.lastUse = ++useCounter_;
            ++refinementBlockCacheHits_;
            continue;
        }
        if (!queuedRefinementBlocks_.contains(blockId) &&
            !loadingRefinementBlocks_.contains(blockId)) {
            ++refinementBlockCacheMisses_;
        }
        QueueRefinementBlockLocked(blockId, generation_);
    }
    condition_.notify_all();
}

void GraphPageStreamer::SetBackingBlockDemand(std::span<const std::uint32_t> blockIds) {
    std::unique_lock lock(mutex_);

    // Keep one bounded GPU-fault wave stable until it has reached GPU residency. Replacing a
    // 128-tile wave with a different arbitrary 128-tile subset every asynchronous readback can
    // otherwise cancel useful I/O forever. A camera/root-plan generation change is allowed to
    // abandon the old wave immediately so navigation never waits on stale refinement.
    if (!desiredRefinementBlocks_.empty() && backingDemandGeneration_ == generation_) {
        return;
    }

    std::unordered_set<std::uint32_t> nextDemand;
    nextDemand.reserve(blockIds.size());
    const auto refinementTileCount =
        RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount);
    for (const auto blockId : blockIds) {
        if (blockId < refinementTileCount) {
            nextDemand.insert(blockId);
        }
    }

    desiredRefinementBlocks_ = std::move(nextDemand);
    backingDemandGeneration_ = generation_;
    for (auto it = queuedRefinementBlocks_.begin(); it != queuedRefinementBlocks_.end();) {
        if (!desiredRefinementBlocks_.contains(*it)) {
            it = queuedRefinementBlocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (const auto blockId : desiredRefinementBlocks_) {
        if (auto found = refinementCache_.find(blockId); found != refinementCache_.end()) {
            found->second.lastUse = ++useCounter_;
            ++refinementBlockCacheHits_;
            continue;
        }
        if (!queuedRefinementBlocks_.contains(blockId) &&
            !loadingRefinementBlocks_.contains(blockId)) {
            ++refinementBlockCacheMisses_;
        }
        QueueRefinementBlockLocked(blockId, generation_);
    }
    condition_.notify_all();
}

void GraphPageStreamer::ReleaseBackingBlocks(std::span<const std::uint32_t> blockIds) {
    std::unique_lock lock(mutex_);
    for (const auto blockId : blockIds) {
        desiredRefinementBlocks_.erase(blockId);
    }
    if (desiredRefinementBlocks_.empty()) {
        backingDemandGeneration_ = generation_;
    }
    condition_.notify_all();
}

bool GraphPageStreamer::WaitForView(std::uint64_t generation,
                                    std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, timeout, [&]() {
        return generation != generation_ || currentDemandReady_ || !lastError_.empty();
    }) && generation == generation_ && currentDemandReady_;
}

GraphPageStreamingStats GraphPageStreamer::Snapshot() const {
    std::lock_guard lock(mutex_);
    GraphPageStreamingStats stats;
    stats.ramBudgetBytes = config_.ramBudgetBytes;
    stats.dataCacheBudgetBytes = dataCacheBudgetBytes_;
    stats.softBudgetBytes = softBudgetBytes_;
    stats.residentBytes = residentBytes_;
    stats.reservedLoadBytes = reservedLoadBytes_;
    stats.pinnedBytes = pinnedBytes_;
    stats.residentPageCount = static_cast<std::uint32_t>(cache_.size());
    stats.queuedPageCount = static_cast<std::uint32_t>(queuedPages_.size());
    stats.loadingPageCount = static_cast<std::uint32_t>(loadingPages_.size());
    stats.desiredRefinementBlockCount =
        static_cast<std::uint32_t>(desiredRefinementBlocks_.size());
    for (const auto blockId : desiredRefinementBlocks_) {
        stats.residentRefinementBlockCount += refinementCache_.contains(blockId) ? 1u : 0u;
    }
    stats.queuedRefinementBlockCount =
        static_cast<std::uint32_t>(queuedRefinementBlocks_.size());
    stats.loadingRefinementBlockCount =
        static_cast<std::uint32_t>(loadingRefinementBlocks_.size());
    stats.totalRefinementRequests = totalRefinementRequests_;
    stats.totalSpatiallyRejectedRefinementRequests = totalSpatiallyRejectedRefinementRequests_;
    stats.lastRefinementRequestCount = lastRefinementRequestCount_;
    stats.lastSpatiallyAcceptedRefinementRequestCount =
        lastSpatiallyAcceptedRefinementRequestCount_;
    stats.totalRefinementBlockLoads = totalRefinementBlockLoads_;
    stats.totalRefinementBlockEvictions = totalRefinementBlockEvictions_;
    stats.desiredPageCount = static_cast<std::uint32_t>(desiredPages_.size());
    for (const auto pageId : desiredPages_) {
        stats.desiredResidentCount += cache_.contains(pageId) ? 1u : 0u;
    }
    stats.requiredPageCount = static_cast<std::uint32_t>(requiredPages_.size());
    stats.spatialPrefetchRadius = config_.spatialPrefetchRadius;
    stats.prefetchReferenceSpatialLevel = prefetchReferenceSpatialLevel_;
    stats.spatialCellSizeMeters = index_.spatialCellSizeMeters;
    stats.lodPrefetchPageCount = static_cast<std::uint32_t>(lodPrefetchPages_.size());
    stats.lodGpuWarmPageCount = static_cast<std::uint32_t>(gpuWarmPages_.size());
    stats.lodPrefetchMinLevel = currentPrefetchMinLod_;
    stats.lodPrefetchMaxLevel = currentPrefetchMaxLod_;
    stats.lodPrefetchFocusLevel = currentPrefetchFocusLod_;
    for (const auto pageId : requiredPages_) {
        stats.requiredResidentCount += cache_.contains(pageId) ? 1u : 0u;
    }
    stats.totalPageLoads = totalPageLoads_;
    stats.totalEvictions = totalEvictions_;
    stats.pageCacheHits = pageCacheHits_;
    stats.pageCacheMisses = pageCacheMisses_;
    stats.refinementBlockCacheHits = refinementBlockCacheHits_;
    stats.refinementBlockCacheMisses = refinementBlockCacheMisses_;
    stats.budgetRejectedLoads = budgetRejectedLoads_;
    stats.estimatedSourceBytesRead = estimatedSourceBytesRead_;
    stats.edgeSourceBlocksRead = edgeSourceBlocksRead_;
    stats.nodeSourceBlocksRead = nodeSourceBlocksRead_;
    {
        std::lock_guard nodeLock(nodeCacheMutex_);
        stats.nodeBlockCacheHits = nodeBlockCacheHits_;
        stats.nodeBlockCacheMisses = nodeBlockCacheMisses_;
        stats.nodeBlockCacheEvictions = nodeBlockCacheEvictions_;
        stats.nodeBlockCacheBytes = nodeBlockCacheBytes_;
        stats.nodeBlockCacheBudgetBytes = nodeBlockCacheBudgetBytes_;
    }
    stats.totalCpuCacheBytes = stats.residentBytes + stats.nodeBlockCacheBytes;
    auto peak = peakCpuCacheBytes_.load(std::memory_order_relaxed);
    while (peak < stats.totalCpuCacheBytes &&
           !peakCpuCacheBytes_.compare_exchange_weak(
               peak, stats.totalCpuCacheBytes, std::memory_order_relaxed)) {
    }
    stats.peakCpuCacheBytes = peakCpuCacheBytes_.load(std::memory_order_relaxed);
    stats.lastPageLoadMs = lastPageLoadMs_;
    stats.meanPageLoadMs = totalPageLoads_ == 0
                               ? 0.0
                               : totalPageLoadMs_ / static_cast<double>(totalPageLoads_);
    stats.maxPageLoadMs = maxPageLoadMs_;
    stats.lastRootPlannerMs = lastRootPlannerMs_;
    stats.rootPlannerRebuilds = rootPlannerRebuilds_;
    stats.rootPlannerCacheReuses = rootPlannerCacheReuses_;
    stats.rootPlannerAliveCandidates = lastPlannerAliveCandidates_;
    stats.rootPlannerSpatialCandidates = lastPlannerSpatialCandidates_;
    stats.rootPlannerUsedLodFirst = lastPlannerUsedLodFirst_;
    stats.lastRefinementBlockLoadMs = lastRefinementBlockLoadMs_;
    stats.meanRefinementBlockLoadMs = totalRefinementBlockLoads_ == 0
                                          ? 0.0
                                          : totalRefinementBlockLoadMs_ /
                                                static_cast<double>(totalRefinementBlockLoads_);
    stats.currentDemandReady = currentDemandReady_;
    stats.lastDemandReadyMs = lastDemandReadyMs_;
    if (!currentDemandReady_) {
        stats.currentDemandWaitMs = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - demandStart_)
                                        .count();
    }
    stats.lastError = lastError_;
    return stats;
}

std::vector<std::uint32_t> GraphPageStreamer::RequiredPageIds() const {
    std::lock_guard lock(mutex_);
    std::vector<std::uint32_t> pageIds(requiredPages_.begin(), requiredPages_.end());
    std::sort(pageIds.begin(), pageIds.end());
    return pageIds;
}

std::vector<std::uint32_t> GraphPageStreamer::DesiredPageIds() const {
    std::lock_guard lock(mutex_);
    std::vector<std::uint32_t> pageIds(desiredPages_.begin(), desiredPages_.end());
    std::sort(pageIds.begin(), pageIds.end());
    return pageIds;
}

std::vector<std::uint32_t> GraphPageStreamer::LodPrefetchPageIds() const {
    std::lock_guard lock(mutex_);
    std::vector<std::uint32_t> pageIds(lodPrefetchPages_.begin(), lodPrefetchPages_.end());
    std::sort(pageIds.begin(), pageIds.end());
    return pageIds;
}

std::vector<std::uint32_t> GraphPageStreamer::GpuWarmPageIds() const {
    std::lock_guard lock(mutex_);
    std::vector<std::uint32_t> pageIds(gpuWarmPages_.begin(), gpuWarmPages_.end());
    std::sort(pageIds.begin(), pageIds.end());
    return pageIds;
}

std::vector<std::uint32_t> GraphPageStreamer::ResidentRefinementBlockIds() const {
    std::lock_guard lock(mutex_);
    std::vector<std::uint32_t> blockIds;
    blockIds.reserve(desiredRefinementBlocks_.size());
    for (const auto blockId : desiredRefinementBlocks_) {
        if (refinementCache_.contains(blockId)) {
            blockIds.push_back(blockId);
        }
    }
    std::sort(blockIds.begin(), blockIds.end());
    return blockIds;
}

std::shared_ptr<const ResidentGraphPage> GraphPageStreamer::FindResidentPage(
    std::uint32_t pageId) const {
    std::lock_guard lock(mutex_);
    const auto found = cache_.find(pageId);
    return found == cache_.end() ? nullptr : found->second.page;
}

std::shared_ptr<const ResidentRefinementBlock> GraphPageStreamer::FindResidentRefinementBlock(
    std::uint32_t blockId) const {
    std::lock_guard lock(mutex_);
    const auto found = refinementCache_.find(blockId);
    return found == refinementCache_.end() ? nullptr : found->second.block;
}

StreamingSpatialWindow GraphPageStreamer::CurrentViewportWindow() const {
    std::lock_guard lock(mutex_);
    return currentViewportWindow_;
}

StreamingSpatialWindow GraphPageStreamer::CurrentRefinementWindow() const {
    std::lock_guard lock(mutex_);
    return currentRefinementWindow_;
}

void GraphPageStreamer::ClearRefinement() {
    std::unique_lock lock(mutex_);
    desiredRefinementBlocks_.clear();
    backingDemandGeneration_ = generation_;
    queuedRefinementBlocks_.clear();

    std::priority_queue<LoadRequest, std::vector<LoadRequest>, LoadRequestCompare> retained;
    while (!queue_.empty()) {
        auto request = queue_.top();
        queue_.pop();
        if (request.kind == LoadKind::GraphPage) {
            retained.push(request);
        }
    }
    queue_ = std::move(retained);

    for (const auto& [blockId, entry] : refinementCache_) {
        static_cast<void>(blockId);
        residentBytes_ -= entry.block->memoryBytes;
    }
    refinementCache_.clear();
    condition_.notify_all();
}

void GraphPageStreamer::ClearUnpinned() {
    std::lock_guard lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end();) {
        if (!it->second.pinned && !requiredPages_.contains(it->first)) {
            residentBytes_ -= it->second.page->memoryBytes;
            it = cache_.erase(it);
            ++totalEvictions_;
        } else {
            ++it;
        }
    }
    for (auto it = refinementCache_.begin(); it != refinementCache_.end();) {
        if (!desiredRefinementBlocks_.contains(it->first)) {
            residentBytes_ -= it->second.block->memoryBytes;
            it = refinementCache_.erase(it);
            ++totalRefinementBlockEvictions_;
        } else {
            ++it;
        }
    }
}

void GraphPageStreamer::QueuePageLocked(std::uint32_t pageId, std::uint32_t priority,
                                        std::uint64_t generation) {
    if (pageId >= index_.graphPages.size() || cache_.contains(pageId) ||
        loadingPages_.contains(pageId) || !desiredPages_.contains(pageId)) {
        return;
    }

    const auto found = queuedPages_.find(pageId);
    if (found != queuedPages_.end()) {
        // Reprioritize in BOTH directions. A page that was strict demand one frame ago may become
        // merely speculative after a zoom reversal; leaving its old priority=0 would still make
        // the obsolete path outrun the new viewport. The sequence token invalidates the old heap
        // record without duplicating I/O or losing already-completed cache work.
        if (priority == found->second.priority) {
            return;
        }
    }

    const auto sequence = ++sequence_;
    queuedPages_[pageId] = {priority, generation, sequence};
    queue_.push({LoadKind::GraphPage, pageId, priority, generation, sequence});
}

void GraphPageStreamer::PumpPageQueueLocked(
    std::span<const PageSelection> selectedPages, std::uint64_t generation) {
    // Keep only a bounded amount of root I/O admitted at once. The full desired set remains in
    // memory as intent, but workers receive it in small priority-ordered waves. This avoids a
    // single wheel event inserting thousands of heap/hash entries on the render thread and makes
    // request cancellation cheap when the camera changes direction.
    constexpr std::size_t kMaxRequiredInFlight = 512u;
    constexpr std::size_t kMaxPrefetchInFlight = 128u;

    std::size_t requiredInFlight = 0;
    std::size_t prefetchInFlight = 0;
    for (const auto& [pageId, state] : queuedPages_) {
        static_cast<void>(state);
        if (requiredPages_.contains(pageId)) {
            ++requiredInFlight;
        } else {
            ++prefetchInFlight;
        }
    }
    for (const auto pageId : loadingPages_) {
        if (requiredPages_.contains(pageId)) {
            ++requiredInFlight;
        } else {
            ++prefetchInFlight;
        }
    }

    for (const auto& selection : selectedPages) {
        const auto pageId = selection.pageId;
        if (cache_.contains(pageId) || loadingPages_.contains(pageId)) {
            continue;
        }

        const auto queued = queuedPages_.find(pageId);
        if (queued != queuedPages_.end()) {
            // Required pages may have entered the queue earlier as speculation. Re-submit so
            // QueuePageLocked can raise their priority without duplicating the I/O request.
            if (selection.required && queued->second.priority != selection.priority) {
                QueuePageLocked(pageId, selection.priority, generation);
            }
            continue;
        }

        if (selection.required) {
            if (requiredInFlight >= kMaxRequiredInFlight) {
                continue;
            }
            QueuePageLocked(pageId, selection.priority, generation);
            ++requiredInFlight;
        } else {
            if (prefetchInFlight >= kMaxPrefetchInFlight) {
                continue;
            }
            QueuePageLocked(pageId, selection.priority, generation);
            ++prefetchInFlight;
        }
    }
}

void GraphPageStreamer::QueueRefinementBlockLocked(std::uint32_t blockId,
                                                    std::uint64_t generation) {
    if (blockId >= RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount) ||
        refinementCache_.contains(blockId) ||
        loadingRefinementBlocks_.contains(blockId) ||
        !queuedRefinementBlocks_.insert(blockId).second) {
        return;
    }
    // Current-view geometry misses outrank spatial prefetch but never block required root pages.
    queue_.push({LoadKind::RefinementBlock, blockId, 1u, generation, ++sequence_});
}

void GraphPageStreamer::Worker(std::stop_token stopToken) {
    try {
        // Keep the text sources open for the lifetime of the worker. Both drawable root pages
        // and refinement backing blocks share the same preprocessed graph scanner and decoded-node cache.
        TextSourceScanner graphScanner(graphPath_);
        TextSourceScanner rangeScanner(rangesPath_);
        std::ifstream indexStream(index_.indexPath, std::ios::binary);
        if (!indexStream) {
            throw std::runtime_error("could not open CH index for persistent root streaming");
        }

        while (!stopToken.stop_requested()) {
            LoadRequest request;
            std::uint64_t estimate = 0;
            bool requiredRoot = false;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, stopToken, [&]() { return !queue_.empty(); });
                if (stopToken.stop_requested()) {
                    break;
                }
                request = queue_.top();
                queue_.pop();

                if (request.kind == LoadKind::GraphPage) {
                    const auto queued = queuedPages_.find(request.id);
                    if (queued == queuedPages_.end() || queued->second.sequence != request.sequence) {
                        continue; // stale heap record from cancellation/reprioritization
                    }
                    queuedPages_.erase(queued);
                    if (cache_.contains(request.id) || loadingPages_.contains(request.id) ||
                        !desiredPages_.contains(request.id)) {
                        continue;
                    }
                    loadingPages_.insert(request.id);
                    requiredRoot = requiredPages_.contains(request.id);
                    estimate = EstimatePageBytes(request.id);
                } else {
                    queuedRefinementBlocks_.erase(request.id);
                    if (refinementCache_.contains(request.id) ||
                        loadingRefinementBlocks_.contains(request.id)) {
                        continue;
                    }
                    if (!desiredRefinementBlocks_.contains(request.id)) {
                        continue;
                    }
                    loadingRefinementBlocks_.insert(request.id);
                    estimate = EstimateRefinementBlockBytes(request.id);
                }

                if (!ReserveForLoadLocked(estimate, requiredRoot)) {
                    if (request.kind == LoadKind::GraphPage) {
                        loadingPages_.erase(request.id);
                    } else {
                        loadingRefinementBlocks_.erase(request.id);
                    }
                    ++budgetRejectedLoads_;
                    UpdateDemandReadyLocked();
                    condition_.notify_all();
                    continue;
                }
            }

            try {
                if (request.kind == LoadKind::GraphPage) {
                    auto page = std::make_shared<ResidentGraphPage>(
                        LoadPage(request.id, indexStream, graphScanner, rangeScanner));
                    std::unique_lock lock(mutex_);
                    ReleaseReservationLocked(estimate);
                    loadingPages_.erase(request.id);

                    if (!desiredPages_.contains(request.id)) {
                        UpdateDemandReadyLocked();
                        condition_.notify_all();
                        continue;
                    }

                    if (page->memoryBytes > dataCacheBudgetBytes_) {
                        ++budgetRejectedLoads_;
                        lastError_ = "single graph page exceeds the RAM cache budget";
                        UpdateDemandReadyLocked();
                        condition_.notify_all();
                        continue;
                    }
                    if (residentBytes_ + reservedLoadBytes_ + page->memoryBytes >
                        dataCacheBudgetBytes_) {
                        const auto target =
                            dataCacheBudgetBytes_ - page->memoryBytes - reservedLoadBytes_;
                        EvictToLocked(target);
                    }
                    if (residentBytes_ + reservedLoadBytes_ + page->memoryBytes >
                        dataCacheBudgetBytes_) {
                        ++budgetRejectedLoads_;
                        UpdateDemandReadyLocked();
                        condition_.notify_all();
                        continue;
                    }

                    // In the fixed-world grid spatialLevel 0 is the finest/base cell, not a
                    // whole-dataset ancestor. Do not permanently pin those pages; current-view
                    // demand and LRU are sufficient and keep the bounded cache fully reusable.
                    constexpr bool pin = false;
                    cache_[request.id] = {page, ++useCounter_, pin};
                    residentBytes_ += page->memoryBytes;
                    UpdatePeakCpuCacheBytes();
                    ++totalPageLoads_;
                    totalPageLoadMs_ += page->loadMilliseconds;
                    lastPageLoadMs_ = page->loadMilliseconds;
                    maxPageLoadMs_ = std::max(maxPageLoadMs_, page->loadMilliseconds);
                    estimatedSourceBytesRead_ += page->estimatedSourceBytesRead;
                    edgeSourceBlocksRead_ += page->edgeSourceBlocksRead;
                    nodeSourceBlocksRead_ += page->nodeSourceBlocksRead;
                    lastError_.clear();
                    UpdateDemandReadyLocked();
                    condition_.notify_all();
                } else {
                    auto block = std::make_shared<ResidentRefinementBlock>(
                        LoadRefinementBlock(request.id, graphScanner));
                    std::unique_lock lock(mutex_);
                    ReleaseReservationLocked(estimate);
                    loadingRefinementBlocks_.erase(request.id);

                    if (!desiredRefinementBlocks_.contains(request.id)) {
                        condition_.notify_all();
                        continue;
                    }

                    if (block->memoryBytes > dataCacheBudgetBytes_) {
                        ++budgetRejectedLoads_;
                        lastError_ = "single refinement block exceeds the RAM cache budget";
                        condition_.notify_all();
                        continue;
                    }
                    if (residentBytes_ + reservedLoadBytes_ + block->memoryBytes >
                        dataCacheBudgetBytes_) {
                        const auto target =
                            dataCacheBudgetBytes_ - block->memoryBytes - reservedLoadBytes_;
                        EvictToLocked(target);
                    }
                    if (residentBytes_ + reservedLoadBytes_ + block->memoryBytes >
                        dataCacheBudgetBytes_) {
                        ++budgetRejectedLoads_;
                        condition_.notify_all();
                        continue;
                    }

                    refinementCache_[request.id] = {block, ++useCounter_};
                    residentBytes_ += block->memoryBytes;
                    UpdatePeakCpuCacheBytes();
                    ++totalRefinementBlockLoads_;
                    totalRefinementBlockLoadMs_ += block->loadMilliseconds;
                    lastRefinementBlockLoadMs_ = block->loadMilliseconds;
                    estimatedSourceBytesRead_ += block->estimatedSourceBytesRead;
                    // One preprocessed graph edge source block was parsed for every refinement block.
                    ++edgeSourceBlocksRead_;
                    nodeSourceBlocksRead_ += block->nodeSourceBlocksRead;
                    lastError_.clear();
                    condition_.notify_all();
                }
            } catch (const std::exception& error) {
                std::unique_lock lock(mutex_);
                ReleaseReservationLocked(estimate);
                if (request.kind == LoadKind::GraphPage) {
                    loadingPages_.erase(request.id);
                    if (request.generation == generation_ || desiredPages_.contains(request.id)) {
                        lastError_ = error.what();
                    }
                } else {
                    loadingRefinementBlocks_.erase(request.id);
                    if (desiredRefinementBlocks_.contains(request.id)) {
                        lastError_ = error.what();
                    }
                }
                UpdateDemandReadyLocked();
                condition_.notify_all();
            }
        }
    } catch (const std::exception& error) {
        std::unique_lock lock(mutex_);
        lastError_ = error.what();
        currentDemandReady_ = false;
        condition_.notify_all();
    }
}

ResidentGraphPage GraphPageStreamer::LoadPage(
    std::uint32_t pageId, std::ifstream& indexStream, TextSourceScanner& graphScanner,
    TextSourceScanner& rangeScanner) {
    const auto beginTime = std::chrono::steady_clock::now();

    ResidentGraphPage result;
    result.pageId = pageId;
    if (index_.HasRootPayload()) {
        const auto& pageDesc = index_.graphPages.at(pageId);
        const auto expected = pageDesc.edgeCount;
        result.rootRecords.resize(expected);
        const auto byteOffset = index_.rootPayloadSectionOffset +
            pageDesc.rootPayloadRecordOffset * sizeof(index::CHIndexRootRecord);
        if (byteOffset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max())) {
            throw std::runtime_error("root payload offset exceeds stream range");
        }
        indexStream.clear();
        indexStream.seekg(static_cast<std::streamoff>(byteOffset), std::ios::beg);
        if (!indexStream) {
            throw std::runtime_error("could not seek CH index RootPayload");
        }
        indexStream.read(reinterpret_cast<char*>(result.rootRecords.data()),
                         static_cast<std::streamsize>(
                             result.rootRecords.size() * sizeof(index::CHIndexRootRecord)));
        if (!indexStream) {
            throw std::runtime_error("truncated CH index RootPayload tile");
        }
        if (result.rootRecords.size() != expected || result.rootRecords.empty()) {
            throw std::runtime_error("runtime graph page root payload count is invalid");
        }
        for (const auto& record : result.rootRecords) {
            if (record.globalEdgeId >= index_.edgeCount || record.birthLevel < 0 ||
                record.deathLevel < 0 || record.deathLevel > record.birthLevel) {
                throw std::runtime_error("runtime graph page root payload record is invalid");
            }
        }
        std::sort(result.rootRecords.begin(), result.rootRecords.end(),
                  [](const auto& a, const auto& b) {
                      if (a.birthLevel != b.birthLevel) {
                          return a.birthLevel > b.birthLevel;
                      }
                      return a.globalEdgeId < b.globalEdgeId;
                  });
        result.memoryBytes = sizeof(result) +
                             result.rootRecords.capacity() * sizeof(index::CHIndexRootRecord);
        result.estimatedSourceBytesRead =
            result.rootRecords.size() * sizeof(index::CHIndexRootRecord);
        result.loadMilliseconds = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - beginTime)
                                      .count();
        return result;
    }

    const auto edgeIds = index::CHIndex::ReadGraphPageEdgeIds(index_, pageId);
    if (edgeIds.empty()) {
        throw std::runtime_error("runtime graph page has no edge membership");
    }
    std::vector<TemporaryEdge> edges(edgeIds.size());
    for (std::size_t i = 0; i < edgeIds.size(); ++i) {
        edges[i].globalEdgeId = edgeIds[i];
    }

    const auto graphFileSize = std::filesystem::file_size(graphPath_);
    const auto rangeFileSize = std::filesystem::file_size(rangesPath_);
    const auto& pageDesc = index_.graphPages[pageId];

    for (const auto blockId : index_.SourceBlockRefs(pageDesc)) {
        if (blockId >= index_.graphEdgeBlocks.size() || blockId >= index_.rangeEdgeBlocks.size()) {
            throw std::runtime_error("graph page references an invalid source block");
        }
        const auto& graphBlock = index_.graphEdgeBlocks[blockId];
        const auto& rangeBlock = index_.rangeEdgeBlocks[blockId];
        const auto first = static_cast<std::uint32_t>(graphBlock.firstRecord);
        const auto end = first + graphBlock.recordCount;
        auto selected = static_cast<std::size_t>(
            std::lower_bound(edgeIds.begin(), edgeIds.end(), first) - edgeIds.begin());
        const auto selectedEnd = static_cast<std::size_t>(
            std::lower_bound(edgeIds.begin(), edgeIds.end(), end) - edgeIds.begin());
        if (selected == selectedEnd) {
            continue;
        }

        graphScanner.Seek(graphBlock.byteOffset);
        for (std::uint32_t local = 0; local < graphBlock.recordCount; ++local) {
            const auto globalEdge = first + local;
            const auto source = graphScanner.Read<std::uint32_t>();
            const auto target = graphScanner.Read<std::uint32_t>();
            graphScanner.Read<float>();
            const auto roadType = graphScanner.Read<std::int32_t>();
            graphScanner.Read<std::int32_t>();
            const auto childA = DecodeChild(graphScanner.Read<std::int64_t>());
            const auto childB = DecodeChild(graphScanner.Read<std::int64_t>());

            if (selected < selectedEnd && edgeIds[selected] == globalEdge) {
                auto& edge = edges[selected];
                edge.source = source;
                edge.target = target;
                edge.childA = childA;
                edge.childB = childB;
                edge.roadStyleType = data::RoadStyleIndexFromType(roadType);
                edge.graphLoaded = true;
                ++selected;
            }
        }

        rangeScanner.Seek(rangeBlock.byteOffset);
        selected = static_cast<std::size_t>(
            std::lower_bound(edgeIds.begin(), edgeIds.end(), first) - edgeIds.begin());
        for (std::uint32_t local = 0; local < rangeBlock.recordCount; ++local) {
            const auto globalEdge = rangeScanner.Read<std::uint32_t>();
            const auto birth = rangeScanner.Read<std::int32_t>();
            const auto death = rangeScanner.Read<std::int32_t>();
            if (selected < selectedEnd && edgeIds[selected] == globalEdge) {
                auto& edge = edges[selected];
                edge.birthLevel = birth;
                edge.deathLevel = death;
                edge.rangeLoaded = true;
                ++selected;
            }
        }

        result.estimatedSourceBytesRead +=
            BlockByteSize(index_.graphEdgeBlocks, blockId, graphFileSize);
        result.estimatedSourceBytesRead +=
            BlockByteSize(index_.rangeEdgeBlocks, blockId, rangeFileSize);
        ++result.edgeSourceBlocksRead;
    }

    std::vector<std::uint32_t> nodeIds;
    nodeIds.reserve(edges.size() * 2u);
    for (const auto& edge : edges) {
        if (!edge.graphLoaded || !edge.rangeLoaded) {
            throw std::runtime_error("graph page could not resolve all source edge records");
        }
        nodeIds.push_back(edge.source);
        nodeIds.push_back(edge.target);
    }
    std::sort(nodeIds.begin(), nodeIds.end());
    nodeIds.erase(std::unique(nodeIds.begin(), nodeIds.end()), nodeIds.end());

    result.nodes.resize(nodeIds.size());
    std::size_t selectedNode = 0;
    while (selectedNode < nodeIds.size()) {
        const auto blockId = nodeIds[selectedNode] / index_.nodeBlockRecordCount;
        if (blockId >= index_.nodeBlocks.size()) {
            throw std::runtime_error("graph page references an invalid node source block");
        }
        const auto& sourceBlock = index_.nodeBlocks[blockId];
        const auto first = static_cast<std::uint32_t>(sourceBlock.firstRecord);
        const auto end = first + sourceBlock.recordCount;
        const auto selectedEnd = static_cast<std::size_t>(
            std::lower_bound(nodeIds.begin(), nodeIds.end(), end) - nodeIds.begin());

        const auto access = AcquireNodeBlock(blockId, graphScanner);
        if (!access.block || access.block->firstRecord != first ||
            access.block->positions.size() != sourceBlock.recordCount) {
            throw std::runtime_error("decoded node block cache is inconsistent");
        }

        for (; selectedNode < selectedEnd; ++selectedNode) {
            const auto nodeId = nodeIds[selectedNode];
            if (nodeId < first || nodeId >= end) {
                throw std::runtime_error("graph page node id is outside its source block");
            }
            const auto local = static_cast<std::size_t>(nodeId - first);
            const auto& position = access.block->positions[local];
            result.nodes[selectedNode] = {nodeId, position[0], position[1]};
        }

        if (access.loadedFromSource) {
            result.estimatedSourceBytesRead += access.block->sourceBytesRead;
            ++result.nodeSourceBlocksRead;
        }
    }

    result.edges.reserve(edges.size());
    for (const auto& edge : edges) {
        const auto sourceIt = std::lower_bound(nodeIds.begin(), nodeIds.end(), edge.source);
        const auto targetIt = std::lower_bound(nodeIds.begin(), nodeIds.end(), edge.target);
        if (sourceIt == nodeIds.end() || *sourceIt != edge.source || targetIt == nodeIds.end() ||
            *targetIt != edge.target) {
            throw std::runtime_error("graph page node remap failed");
        }
        result.edges.push_back({
            edge.globalEdgeId,
            static_cast<std::uint32_t>(sourceIt - nodeIds.begin()),
            static_cast<std::uint32_t>(targetIt - nodeIds.begin()),
            edge.childA,
            edge.childB,
            edge.roadStyleType,
            0.0f,
            edge.birthLevel,
            edge.deathLevel,
        });
    }
    std::sort(result.edges.begin(), result.edges.end(), [](const auto& a, const auto& b) {
        if (a.birthLevel != b.birthLevel) {
            return a.birthLevel > b.birthLevel;
        }
        return a.globalEdgeId < b.globalEdgeId;
    });
    result.memoryBytes = sizeof(result) +
                         result.nodes.capacity() * sizeof(StreamedNode) +
                         result.edges.capacity() * sizeof(StreamedEdge);
    result.loadMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - beginTime)
            .count();
    return result;
}

std::uint64_t GraphPageStreamer::RefinementTileByteOffset(
    std::uint32_t tileId, TextSourceScanner& graphScanner,
    bool& builtPhysicalOffsetTable) {
    builtPhysicalOffsetTable = false;
    const auto tileRecords = RuntimeRefinementTileRecordCount(index_.edgeBlockRecordCount);
    const auto tileCount = RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount);
    if (tileRecords == 0u || tileId >= tileCount) {
        throw std::runtime_error("refinement runtime tile id out of range");
    }

    const auto firstEdge = static_cast<std::uint64_t>(tileId) * tileRecords;
    const auto sourceBlockId = static_cast<std::uint32_t>(firstEdge / index_.edgeBlockRecordCount);
    if (sourceBlockId >= index_.graphEdgeBlocks.size()) {
        throw std::runtime_error("refinement runtime tile maps outside source edge blocks");
    }

    {
        std::unique_lock lock(refinementOffsetMutex_);
        for (;;) {
            const auto cached = refinementTileByteOffsets_[tileId];
            if (cached != std::numeric_limits<std::uint64_t>::max()) {
                return cached;
            }
            if (!loadingRefinementOffsetBlocks_.contains(sourceBlockId)) {
                loadingRefinementOffsetBlocks_.insert(sourceBlockId);
                break;
            }
            refinementOffsetCondition_.wait(lock, [&]() {
                return !loadingRefinementOffsetBlocks_.contains(sourceBlockId);
            });
        }
    }

    std::vector<std::pair<std::uint32_t, std::uint64_t>> discoveredOffsets;
    try {
        const auto& sourceBlock = index_.graphEdgeBlocks[sourceBlockId];
        graphScanner.Seek(sourceBlock.byteOffset);
        discoveredOffsets.reserve(
            (sourceBlock.recordCount + tileRecords - 1u) / tileRecords);
        for (std::uint32_t local = 0; local < sourceBlock.recordCount; ++local) {
            std::uint64_t recordOffset = 0;
            if (local % tileRecords == 0u) {
                auto [source, offset] = graphScanner.ReadWithOffset<std::uint32_t>();
                (void)source;
                recordOffset = offset;
                const auto globalEdge = sourceBlock.firstRecord + local;
                const auto logicalTile = static_cast<std::uint32_t>(globalEdge / tileRecords);
                discoveredOffsets.emplace_back(logicalTile, recordOffset);
            } else {
                graphScanner.Read<std::uint32_t>();
            }
            graphScanner.Read<std::uint32_t>();
            graphScanner.Read<float>();
            graphScanner.Read<std::int32_t>();
            graphScanner.Read<std::int32_t>();
            graphScanner.Read<std::int64_t>();
            graphScanner.Read<std::int64_t>();
        }
    } catch (...) {
        std::lock_guard lock(refinementOffsetMutex_);
        loadingRefinementOffsetBlocks_.erase(sourceBlockId);
        refinementOffsetCondition_.notify_all();
        throw;
    }

    {
        std::lock_guard lock(refinementOffsetMutex_);
        for (const auto& [logicalTile, offset] : discoveredOffsets) {
            if (logicalTile < refinementTileByteOffsets_.size()) {
                refinementTileByteOffsets_[logicalTile] = offset;
            }
        }
        loadingRefinementOffsetBlocks_.erase(sourceBlockId);
        builtPhysicalOffsetTable = true;
    }
    refinementOffsetCondition_.notify_all();

    std::lock_guard lock(refinementOffsetMutex_);
    const auto result = refinementTileByteOffsets_[tileId];
    if (result == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("failed to derive refinement runtime tile byte offset");
    }
    return result;
}

ResidentRefinementBlock GraphPageStreamer::LoadRefinementBlock(
    std::uint32_t blockId, TextSourceScanner& graphScanner) {
    const auto beginTime = std::chrono::steady_clock::now();
    const auto tileRecords = RuntimeRefinementTileRecordCount(index_.edgeBlockRecordCount);
    const auto tileCount = RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount);
    if (tileRecords == 0u || blockId >= tileCount) {
        throw std::runtime_error("refinement runtime tile id out of range");
    }

    const auto firstEdge64 = static_cast<std::uint64_t>(blockId) * tileRecords;
    if (firstEdge64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("refinement runtime tile first edge does not fit in uint32");
    }
    const auto firstEdge = static_cast<std::uint32_t>(firstEdge64);
    const auto recordCount = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(tileRecords, index_.edgeCount - firstEdge64));
    const auto sourceBlockId = firstEdge / index_.edgeBlockRecordCount;
    if (sourceBlockId >= index_.graphEdgeBlocks.size()) {
        throw std::runtime_error("refinement runtime tile maps outside source edge blocks");
    }
    const auto& sourceBlock = index_.graphEdgeBlocks[sourceBlockId];
    if (firstEdge < sourceBlock.firstRecord ||
        firstEdge64 + recordCount > sourceBlock.firstRecord + sourceBlock.recordCount) {
        throw std::runtime_error("refinement runtime tile crosses a physical source block");
    }

    ResidentRefinementBlock result;
    result.blockId = blockId;
    std::vector<TemporaryEdge> edges(recordCount);

    bool builtOffsetTable = false;
    const auto tileByteOffset = RefinementTileByteOffset(blockId, graphScanner, builtOffsetTable);
    graphScanner.Seek(tileByteOffset);
    const auto edgeBounds = index::CHIndex::ReadEdgeSpatialBounds(index_, firstEdge, recordCount);
    for (std::uint32_t local = 0; local < recordCount; ++local) {
        auto& edge = edges[local];
        edge.globalEdgeId = firstEdge + local;
        edge.source = graphScanner.Read<std::uint32_t>();
        edge.target = graphScanner.Read<std::uint32_t>();
        graphScanner.Read<float>();
        edge.roadStyleType = data::RoadStyleIndexFromType(graphScanner.Read<std::int32_t>());
        graphScanner.Read<std::int32_t>();
        edge.childA = DecodeChild(graphScanner.Read<std::int64_t>());
        edge.childB = DecodeChild(graphScanner.Read<std::int64_t>());
        edge.graphLoaded = true;
    }

    const auto graphFileSize = std::filesystem::file_size(graphPath_);
    const auto physicalBytes = BlockByteSize(index_.graphEdgeBlocks, sourceBlockId, graphFileSize);
    // The first request touching a physical source block scans it once to derive all 1K tile
    // offsets. Later requests seek directly to their tile. Keep diagnostics conservative.
    result.estimatedSourceBytesRead = builtOffsetTable
                                          ? physicalBytes
                                          : std::max<std::uint64_t>(
                                                1u, physicalBytes * recordCount /
                                                        std::max<std::uint32_t>(sourceBlock.recordCount, 1u));

    std::vector<std::uint32_t> nodeIds;
    nodeIds.reserve(edges.size() * 2u);
    for (const auto& edge : edges) {
        nodeIds.push_back(edge.source);
        nodeIds.push_back(edge.target);
    }
    std::sort(nodeIds.begin(), nodeIds.end());
    nodeIds.erase(std::unique(nodeIds.begin(), nodeIds.end()), nodeIds.end());

    result.nodes.resize(nodeIds.size());
    std::size_t selectedNode = 0;
    while (selectedNode < nodeIds.size()) {
        const auto nodeBlockId = nodeIds[selectedNode] / index_.nodeBlockRecordCount;
        if (nodeBlockId >= index_.nodeBlocks.size()) {
            throw std::runtime_error("refinement edge references an invalid node source block");
        }
        const auto& nodeSourceBlock = index_.nodeBlocks[nodeBlockId];
        const auto first = static_cast<std::uint32_t>(nodeSourceBlock.firstRecord);
        const auto end = first + nodeSourceBlock.recordCount;
        const auto selectedEnd = static_cast<std::size_t>(
            std::lower_bound(nodeIds.begin(), nodeIds.end(), end) - nodeIds.begin());

        const auto access = AcquireNodeBlock(nodeBlockId, graphScanner);
        if (!access.block || access.block->firstRecord != first ||
            access.block->positions.size() != nodeSourceBlock.recordCount) {
            throw std::runtime_error("decoded node block cache is inconsistent");
        }

        for (; selectedNode < selectedEnd; ++selectedNode) {
            const auto nodeId = nodeIds[selectedNode];
            if (nodeId < first || nodeId >= end) {
                throw std::runtime_error("refinement node id is outside its source block");
            }
            const auto local = static_cast<std::size_t>(nodeId - first);
            const auto& position = access.block->positions[local];
            result.nodes[selectedNode] = {nodeId, position[0], position[1]};
        }

        if (access.loadedFromSource) {
            result.estimatedSourceBytesRead += access.block->sourceBytesRead;
            ++result.nodeSourceBlocksRead;
        }
    }

    result.edges.reserve(edges.size());
    for (const auto& edge : edges) {
        const auto sourceIt = std::lower_bound(nodeIds.begin(), nodeIds.end(), edge.source);
        const auto targetIt = std::lower_bound(nodeIds.begin(), nodeIds.end(), edge.target);
        if (sourceIt == nodeIds.end() || *sourceIt != edge.source || targetIt == nodeIds.end() ||
            *targetIt != edge.target) {
            throw std::runtime_error("refinement node remap failed");
        }
        const auto sourceLocal = static_cast<std::uint32_t>(sourceIt - nodeIds.begin());
        const auto targetLocal = static_cast<std::uint32_t>(targetIt - nodeIds.begin());
        float geometryError = 0.0f;
        if (edge.childA != data::InvalidEdgeId && edge.childB != data::InvalidEdgeId) {
            const auto localEdge = edge.globalEdgeId - firstEdge;
            if (localEdge >= edgeBounds.size()) {
                throw std::runtime_error("refinement edge is missing preprocessed spatial bounds");
            }
            const auto& source = result.nodes[sourceLocal];
            const auto& target = result.nodes[targetLocal];
            geometryError = ConservativeGeometryErrorCached(
                edgeBounds[localEdge], {source.x, source.y}, {target.x, target.y},
                geometryBoundaryX_, geometryBoundaryY_);
        }
        result.edges.push_back({
            edge.globalEdgeId,
            sourceLocal,
            targetLocal,
            edge.childA,
            edge.childB,
            edge.roadStyleType,
            geometryError,
            -1,
            -1,
        });
    }

    result.memoryBytes = sizeof(result) +
                         result.nodes.capacity() * sizeof(StreamedNode) +
                         result.edges.capacity() * sizeof(StreamedEdge);
    result.loadMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - beginTime)
            .count();
    return result;
}

GraphPageStreamer::NodeBlockAccess GraphPageStreamer::AcquireNodeBlock(
    std::uint32_t blockId, TextSourceScanner& graphScanner) {
    if (blockId >= index_.nodeBlocks.size()) {
        throw std::runtime_error("node source block id out of range");
    }

    {
        std::unique_lock lock(nodeCacheMutex_);
        for (;;) {
            const auto found = nodeBlockCache_.find(blockId);
            if (found != nodeBlockCache_.end()) {
                found->second.lastUse = ++nodeBlockUseCounter_;
                ++nodeBlockCacheHits_;
                return {found->second.block, false};
            }
            if (!loadingNodeBlocks_.contains(blockId)) {
                loadingNodeBlocks_.insert(blockId);
                ++nodeBlockCacheMisses_;
                break;
            }
            nodeCacheCondition_.wait(lock, [&]() {
                return !loadingNodeBlocks_.contains(blockId);
            });
        }
    }

    std::shared_ptr<DecodedNodeBlock> decoded;
    try {
        const auto& block = index_.nodeBlocks[blockId];
        decoded = std::make_shared<DecodedNodeBlock>();
        decoded->firstRecord = static_cast<std::uint32_t>(block.firstRecord);
        decoded->positions.resize(block.recordCount);

        const auto centerLatitude = (index_.minLatitude + index_.maxLatitude) * 0.5;
        const auto centerLongitude = (index_.minLongitude + index_.maxLongitude) * 0.5;
        const auto longitudeScale = std::cos(centerLatitude * kPi / 180.0);
        const auto halfWidth = std::max((index_.maxLongitude - index_.minLongitude) * 0.5 *
                                            longitudeScale,
                                        1e-12);
        const auto halfHeight =
            std::max((index_.maxLatitude - index_.minLatitude) * 0.5, 1e-12);
        const auto normalization = 0.95 / std::max(halfWidth, halfHeight);

        graphScanner.Seek(block.byteOffset);
        for (std::uint32_t local = 0; local < block.recordCount; ++local) {
            const auto id = graphScanner.Read<std::uint32_t>();
            graphScanner.Read<std::uint64_t>();
            const auto latitude = graphScanner.Read<double>();
            const auto longitude = graphScanner.Read<double>();
            graphScanner.Read<float>();
            graphScanner.Read<std::uint32_t>();
            const auto expectedId = decoded->firstRecord + local;
            if (id != expectedId) {
                throw std::runtime_error("preprocessed node source block is not ID-contiguous");
            }
            decoded->positions[local] = {
                static_cast<float>((longitude - centerLongitude) * longitudeScale * normalization),
                static_cast<float>((latitude - centerLatitude) * normalization),
            };
        }

        const auto nodeSectionEnd = blockId + 1u < index_.nodeBlocks.size()
                                        ? index_.nodeBlocks[blockId + 1u].byteOffset
                                        : index_.graphEdgeBlocks.front().byteOffset;
        decoded->sourceBytesRead = nodeSectionEnd - block.byteOffset;
        decoded->memoryBytes = sizeof(DecodedNodeBlock) +
                               decoded->positions.capacity() * sizeof(std::array<float, 2>);
    } catch (...) {
        std::lock_guard lock(nodeCacheMutex_);
        loadingNodeBlocks_.erase(blockId);
        nodeCacheCondition_.notify_all();
        throw;
    }

    {
        std::lock_guard lock(nodeCacheMutex_);
        loadingNodeBlocks_.erase(blockId);
        if (decoded->memoryBytes <= nodeBlockCacheBudgetBytes_) {
            if (nodeBlockCacheBytes_ + decoded->memoryBytes > nodeBlockCacheBudgetBytes_) {
                EvictNodeBlocksLocked(nodeBlockCacheBudgetBytes_ - decoded->memoryBytes);
            }
            nodeBlockCache_[blockId] = {decoded, ++nodeBlockUseCounter_};
            nodeBlockCacheBytes_ += decoded->memoryBytes;
        }
        nodeCacheCondition_.notify_all();
    }
    return {std::move(decoded), true};
}

void GraphPageStreamer::EvictNodeBlocksLocked(std::uint64_t targetBytes) {
    while (nodeBlockCacheBytes_ > targetBytes && !nodeBlockCache_.empty()) {
        auto candidate = nodeBlockCache_.end();
        for (auto it = nodeBlockCache_.begin(); it != nodeBlockCache_.end(); ++it) {
            if (candidate == nodeBlockCache_.end() ||
                it->second.lastUse < candidate->second.lastUse) {
                candidate = it;
            }
        }
        if (candidate == nodeBlockCache_.end()) {
            break;
        }
        nodeBlockCacheBytes_ -= candidate->second.block->memoryBytes;
        nodeBlockCache_.erase(candidate);
        ++nodeBlockCacheEvictions_;
    }
}

void GraphPageStreamer::RecomputeBudgetPartitionLocked() {
    // Keep the total CPU-side streaming cache hard-bounded. The decoded-node cache is a
    // sub-budget, not additional memory on top of the page/refinement cache.
    config_.ramBudgetBytes = std::max<std::uint64_t>(config_.ramBudgetBytes, 64ull * kMiB);

    // Text-backed refinement repeatedly resolves edge endpoints through decoded node blocks. With
    // multi-GiB RAM budgets the old 512 MiB node-cache ceiling caused almost every refinement wave
    // to reparse nodes that had just been evicted. Give nodes about 20% of the total cache and allow
    // that partition to grow to 4 GiB; the remaining majority still belongs to root/refinement data.
    const auto maxNodeBudget = std::max<std::uint64_t>(64ull * kMiB,
        std::min<std::uint64_t>(4ull * kGiB, config_.ramBudgetBytes / 3u));
    if (config_.nodeBlockCacheBudgetBytes == 0) {
        nodeBlockCacheBudgetBytes_ = std::clamp<std::uint64_t>(
            config_.ramBudgetBytes / 5u,
            std::min<std::uint64_t>(64ull * kMiB, maxNodeBudget), maxNodeBudget);
    } else {
        nodeBlockCacheBudgetBytes_ = std::clamp<std::uint64_t>(
            config_.nodeBlockCacheBudgetBytes, 4ull * kMiB,
            std::max<std::uint64_t>(4ull * kMiB, config_.ramBudgetBytes / 2u));
    }
    nodeBlockCacheBudgetBytes_ = std::min<std::uint64_t>(
        nodeBlockCacheBudgetBytes_, config_.ramBudgetBytes - 32ull * kMiB);
    config_.nodeBlockCacheBudgetBytes = nodeBlockCacheBudgetBytes_;

    dataCacheBudgetBytes_ = config_.ramBudgetBytes - nodeBlockCacheBudgetBytes_;
    softBudgetBytes_ = static_cast<std::uint64_t>(
        static_cast<double>(dataCacheBudgetBytes_) * config_.softBudgetRatio);
}

void GraphPageStreamer::UpdatePeakCpuCacheBytes() {
    std::uint64_t nodeBytes = 0;
    {
        std::lock_guard nodeLock(nodeCacheMutex_);
        nodeBytes = nodeBlockCacheBytes_;
    }
    const auto total = residentBytes_ + nodeBytes;
    auto peak = peakCpuCacheBytes_.load(std::memory_order_relaxed);
    while (peak < total &&
           !peakCpuCacheBytes_.compare_exchange_weak(
               peak, total, std::memory_order_relaxed)) {
    }
}

void GraphPageStreamer::ReconfigureBudgets(std::uint64_t ramBudgetBytes,
                                           std::uint64_t nodeBlockCacheBudgetBytes) {
    {
        std::scoped_lock lock(mutex_, nodeCacheMutex_);
        config_.ramBudgetBytes = ramBudgetBytes;
        config_.nodeBlockCacheBudgetBytes = nodeBlockCacheBudgetBytes;
        RecomputeBudgetPartitionLocked();
        queryCacheKey_.reset();

        // Fixed-grid pages are all normal LRU entries. The old quadtree-specific permanent
        // spatialLevel-0 pinning would pin arbitrary 4 km base cells and reduce useful cache
        // capacity, so budget changes simply clear any legacy pin state.
        pinnedBytes_ = 0;
        for (auto& [pageId, entry] : cache_) {
            static_cast<void>(pageId);
            entry.pinned = false;
        }

        if (residentBytes_ > dataCacheBudgetBytes_) {
            EvictToLocked(dataCacheBudgetBytes_, true);
        }
        if (nodeBlockCacheBytes_ > nodeBlockCacheBudgetBytes_) {
            EvictNodeBlocksLocked(nodeBlockCacheBudgetBytes_);
        }
        UpdateDemandReadyLocked();
    }
    condition_.notify_all();
    nodeCacheCondition_.notify_all();
    UpdatePeakCpuCacheBytes();
}

bool GraphPageStreamer::ReserveForLoadLocked(std::uint64_t bytes, bool required) {
    const auto limit = required ? dataCacheBudgetBytes_ : softBudgetBytes_;
    if (bytes > limit) {
        return false;
    }
    if (residentBytes_ + reservedLoadBytes_ + bytes > limit) {
        const auto committed = bytes + reservedLoadBytes_;
        const auto target = committed < limit ? limit - committed : 0;
        EvictToLocked(target);
    }
    if (residentBytes_ + reservedLoadBytes_ + bytes > limit) {
        return false;
    }
    reservedLoadBytes_ += bytes;
    return true;
}

void GraphPageStreamer::ReleaseReservationLocked(std::uint64_t bytes) {
    reservedLoadBytes_ = bytes > reservedLoadBytes_ ? 0 : reservedLoadBytes_ - bytes;
}

void GraphPageStreamer::EvictToLocked(std::uint64_t targetBytes, bool allowCurrentDemand) {
    while (residentBytes_ > targetBytes) {
        auto pageCandidate = cache_.end();
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (it->second.pinned ||
                (!allowCurrentDemand && requiredPages_.contains(it->first))) {
                continue;
            }
            if (pageCandidate == cache_.end() ||
                it->second.lastUse < pageCandidate->second.lastUse) {
                pageCandidate = it;
            }
        }

        auto refinementCandidate = refinementCache_.end();
        for (auto it = refinementCache_.begin(); it != refinementCache_.end(); ++it) {
            if (!allowCurrentDemand && desiredRefinementBlocks_.contains(it->first)) {
                continue;
            }
            if (refinementCandidate == refinementCache_.end() ||
                it->second.lastUse < refinementCandidate->second.lastUse) {
                refinementCandidate = it;
            }
        }

        const bool havePage = pageCandidate != cache_.end();
        const bool haveRefinement = refinementCandidate != refinementCache_.end();
        if (!havePage && !haveRefinement) {
            return;
        }

        if (haveRefinement &&
            (!havePage || refinementCandidate->second.lastUse < pageCandidate->second.lastUse)) {
            residentBytes_ -= refinementCandidate->second.block->memoryBytes;
            refinementCache_.erase(refinementCandidate);
            ++totalRefinementBlockEvictions_;
        } else {
            residentBytes_ -= pageCandidate->second.page->memoryBytes;
            cache_.erase(pageCandidate);
            ++totalEvictions_;
        }
    }
}

void GraphPageStreamer::UpdateDemandReadyLocked() {
    bool ready = true;
    for (const auto pageId : requiredPages_) {
        if (!cache_.contains(pageId)) {
            ready = false;
            break;
        }
    }

    if (ready && !currentDemandReady_) {
        lastDemandReadyMs_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - demandStart_)
                .count();
    }
    currentDemandReady_ = ready;
}

std::uint64_t GraphPageStreamer::EstimatePageBytes(std::uint32_t pageId) const {
    const auto& page = index_.graphPages.at(pageId);
    const std::uint64_t bytesPerEdge = index_.HasRootPayload()
                                           ? sizeof(index::CHIndexRootRecord) + 8u
                                           : 128u;
    return 64ull * 1024ull +
           static_cast<std::uint64_t>(page.edgeCount) * bytesPerEdge;
}

std::uint64_t GraphPageStreamer::EstimateRefinementBlockBytes(std::uint32_t blockId) const {
    const auto tileRecords = RuntimeRefinementTileRecordCount(index_.edgeBlockRecordCount);
    const auto tileCount = RuntimeRefinementTileCount(index_.edgeCount, index_.edgeBlockRecordCount);
    if (tileRecords == 0u || blockId >= tileCount) {
        return 0u;
    }
    const auto firstEdge = static_cast<std::uint64_t>(blockId) * tileRecords;
    const auto count = std::min<std::uint64_t>(tileRecords, index_.edgeCount - firstEdge);
    constexpr std::uint64_t kConservativeBytesPerEdge = 96;
    return 32ull * 1024ull + count * kConservativeBytesPerEdge;
}

} // namespace chmv::streaming::runtime
