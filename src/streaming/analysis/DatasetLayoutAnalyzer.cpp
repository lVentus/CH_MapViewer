#include "streaming/analysis/DatasetLayoutAnalyzer.h"

#include "data/ch/CHTypes.h"
#include "streaming/analysis/TextSourceScanner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chmv::streaming::analysis {
namespace {

constexpr std::uint32_t kInvalidPageId = std::numeric_limits<std::uint32_t>::max();
constexpr std::int32_t kMissingRange = std::numeric_limits<std::int32_t>::min();
constexpr std::uint64_t kProgressInterval = 1u << 18;

struct NodeCoordinate {
    float latitude = std::numeric_limits<float>::quiet_NaN();
    float longitude = std::numeric_limits<float>::quiet_NaN();
};

struct VirtualSortEntry {
    std::uint32_t bandIndex = 0;
    std::uint32_t mortonCell = 0;
    std::uint32_t edgeId = 0;
};


struct CellBounds {
    std::uint32_t minX = 0;
    std::uint32_t minY = 0;
    std::uint32_t maxX = 0;
    std::uint32_t maxY = 0;
};

struct RuntimeLodGroup {
    std::uint32_t groupIndex = 0;
    std::uint32_t levelMin = 0;
    std::uint32_t levelMax = 0;
    std::uint64_t edgeCount = 0;
};

std::uint64_t PackCellBounds(const CellBounds& bounds) {
    return static_cast<std::uint64_t>(bounds.minX) |
           (static_cast<std::uint64_t>(bounds.minY) << 16u) |
           (static_cast<std::uint64_t>(bounds.maxX) << 32u) |
           (static_cast<std::uint64_t>(bounds.maxY) << 48u);
}

CellBounds UnpackCellBounds(std::uint64_t packed) {
    return {
        static_cast<std::uint32_t>(packed & 0xffffu),
        static_cast<std::uint32_t>((packed >> 16u) & 0xffffu),
        static_cast<std::uint32_t>((packed >> 32u) & 0xffffu),
        static_cast<std::uint32_t>((packed >> 48u) & 0xffffu),
    };
}

CellBounds UnionBounds(CellBounds a, const CellBounds& b) {
    a.minX = std::min(a.minX, b.minX);
    a.minY = std::min(a.minY, b.minY);
    a.maxX = std::max(a.maxX, b.maxX);
    a.maxY = std::max(a.maxY, b.maxY);
    return a;
}

void ResolveHierarchyBounds(std::vector<std::uint64_t>& packedBounds,
                            const std::vector<std::uint64_t>& childrenByEdge) {
    struct Frame {
        std::uint32_t edgeId = 0;
        bool expanded = false;
    };

    std::vector<std::uint8_t> state(packedBounds.size(), 0);
    std::vector<Frame> stack;
    stack.reserve(128);

    for (std::uint32_t root = 0; root < packedBounds.size(); ++root) {
        if (state[root] == 2) {
            continue;
        }
        stack.push_back({root, false});
        while (!stack.empty()) {
            auto& frame = stack.back();
            const auto edgeId = frame.edgeId;
            if (state[edgeId] == 2) {
                stack.pop_back();
                continue;
            }

            if (!frame.expanded) {
                if (state[edgeId] == 1) {
                    throw std::runtime_error("cycle detected in shortcut hierarchy while resolving spatial bounds");
                }
                state[edgeId] = 1;
                frame.expanded = true;
                const auto packedChildren = childrenByEdge[edgeId];
                const auto childA = static_cast<std::uint32_t>(packedChildren & 0xffffffffu);
                const auto childB = static_cast<std::uint32_t>(packedChildren >> 32u);
                for (const auto child : {childB, childA}) {
                    if (child == data::InvalidEdgeId) {
                        continue;
                    }
                    if (child >= packedBounds.size()) {
                        throw std::runtime_error("shortcut child id out of range while resolving spatial bounds");
                    }
                    if (state[child] == 1) {
                        throw std::runtime_error("cycle detected in shortcut hierarchy while resolving spatial bounds");
                    }
                    if (state[child] == 0) {
                        stack.push_back({child, false});
                    }
                }
                continue;
            }

            auto bounds = UnpackCellBounds(packedBounds[edgeId]);
            const auto packedChildren = childrenByEdge[edgeId];
            const auto childA = static_cast<std::uint32_t>(packedChildren & 0xffffffffu);
            const auto childB = static_cast<std::uint32_t>(packedChildren >> 32u);
            if (childA != data::InvalidEdgeId) {
                bounds = UnionBounds(bounds, UnpackCellBounds(packedBounds[childA]));
            }
            if (childB != data::InvalidEdgeId) {
                bounds = UnionBounds(bounds, UnpackCellBounds(packedBounds[childB]));
            }
            packedBounds[edgeId] = PackCellBounds(bounds);
            state[edgeId] = 2;
            stack.pop_back();
        }
    }
}

std::vector<RuntimeLodGroup> BuildRuntimeLodGroups(
    const std::vector<LodDensityInfo>& density, std::uint32_t maxLevel,
    std::uint32_t targetEdgesPerPage) {
    std::vector<RuntimeLodGroup> groups;
    std::int64_t level = static_cast<std::int64_t>(maxLevel);
    while (level >= 0) {
        while (level >= 0 && density[static_cast<std::size_t>(level)].birthCount == 0) {
            --level;
        }
        if (level < 0) {
            break;
        }

        RuntimeLodGroup group;
        group.groupIndex = static_cast<std::uint32_t>(groups.size());
        group.levelMax = static_cast<std::uint32_t>(level);
        group.levelMin = group.levelMax;

        while (level >= 0) {
            const auto count = density[static_cast<std::size_t>(level)].birthCount;
            if (group.edgeCount != 0 && count != 0 &&
                group.edgeCount + count > targetEdgesPerPage) {
                break;
            }
            group.levelMin = static_cast<std::uint32_t>(level);
            group.edgeCount += count;
            --level;
            if (group.edgeCount >= targetEdgesPerPage) {
                break;
            }
        }
        groups.push_back(group);
    }
    return groups;
}

class Log2Histogram {
public:
    void Add(std::uint64_t value) {
        const auto bucket = value == 0 ? 0u : std::min<std::uint32_t>(63u, std::bit_width(value));
        ++buckets_[bucket];
        ++count_;
        max_ = std::max(max_, value);
    }

    [[nodiscard]] std::uint64_t QuantileUpperBound(double quantile) const {
        if (count_ == 0) {
            return 0;
        }
        const auto target = static_cast<std::uint64_t>(
            std::ceil(quantile * static_cast<double>(count_)));
        std::uint64_t cumulative = 0;
        for (std::uint32_t bucket = 0; bucket < buckets_.size(); ++bucket) {
            cumulative += buckets_[bucket];
            if (cumulative >= target) {
                if (bucket == 0) {
                    return 0;
                }
                if (bucket >= 64) {
                    return std::numeric_limits<std::uint64_t>::max();
                }
                return (std::uint64_t{1} << bucket) - 1;
            }
        }
        return max_;
    }

    [[nodiscard]] std::uint64_t Max() const { return max_; }

private:
    std::array<std::uint64_t, 65> buckets_{};
    std::uint64_t count_ = 0;
    std::uint64_t max_ = 0;
};

void ValidateConfig(const DatasetLayoutAnalysisConfig& config) {
    if (config.spatialGridSize == 0 ||
        (config.spatialGridSize & (config.spatialGridSize - 1)) != 0 ||
        config.spatialGridSize > 1024) {
        throw std::runtime_error("spatialGridSize must be a power of two in [1, 1024]");
    }
    if (config.lodBandWidth == 0) {
        throw std::runtime_error("lodBandWidth must be greater than zero");
    }
    if (config.sourceBlockNodeCount == 0) {
        throw std::runtime_error("sourceBlockNodeCount must be greater than zero");
    }
    if (config.sourceBlockEdgeCount == 0) {
        throw std::runtime_error("sourceBlockEdgeCount must be greater than zero");
    }
    if (config.virtualPageEdgeCount == 0) {
        throw std::runtime_error("virtualPageEdgeCount must be greater than zero");
    }
}

void ReportProgress(const DatasetLayoutAnalyzer::ProgressCallback& callback,
                    DatasetLayoutAnalysisStage stage, std::uint64_t current,
                    std::uint64_t total) {
    if (callback) {
        callback({stage, current, total});
    }
}

std::uint32_t DecodeChild(std::int64_t value) {
    if (value < 0) {
        return data::InvalidEdgeId;
    }
    if (value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("edge child id does not fit in uint32");
    }
    return static_cast<std::uint32_t>(value);
}

std::uint64_t PackChildren(std::uint32_t childA, std::uint32_t childB) {
    return static_cast<std::uint64_t>(childA) |
           (static_cast<std::uint64_t>(childB) << 32u);
}

std::uint32_t ChildA(std::uint64_t packed) {
    return static_cast<std::uint32_t>(packed & 0xffffffffu);
}

std::uint32_t ChildB(std::uint64_t packed) {
    return static_cast<std::uint32_t>(packed >> 32u);
}

std::uint32_t Morton2D(std::uint32_t x, std::uint32_t y, std::uint32_t bits) {
    std::uint32_t result = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        result |= ((x >> bit) & 1u) << (2u * bit);
        result |= ((y >> bit) & 1u) << (2u * bit + 1u);
    }
    return result;
}


bool TilesOverlap(const RuntimeGraphPageInfo& a, const RuntimeGraphPageInfo& b,
                  std::uint32_t gridBits) {
    const auto aScale = std::uint32_t{1} << (gridBits - a.spatialLevel);
    const auto bScale = std::uint32_t{1} << (gridBits - b.spatialLevel);

    const auto aMinX = a.tileX * aScale;
    const auto aMinY = a.tileY * aScale;
    const auto aMaxX = aMinX + aScale;
    const auto aMaxY = aMinY + aScale;
    const auto bMinX = b.tileX * bScale;
    const auto bMinY = b.tileY * bScale;
    const auto bMaxX = bMinX + bScale;
    const auto bMaxY = bMinY + bScale;

    return aMinX < bMaxX && aMaxX > bMinX && aMinY < bMaxY && aMaxY > bMinY;
}

std::pair<std::uint32_t, std::uint32_t> SpatialCell(
    double longitude, double latitude, double minLongitude, double maxLongitude,
    double minLatitude, double maxLatitude, std::uint32_t gridSize) {
    const auto normalize = [](double value, double minimum, double maximum) {
        if (!(maximum > minimum)) {
            return 0.0;
        }
        return std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
    };

    const auto xNorm = normalize(longitude, minLongitude, maxLongitude);
    const auto yNorm = normalize(latitude, minLatitude, maxLatitude);
    const auto toCell = [gridSize](double normalized) {
        const auto scaled = static_cast<std::uint64_t>(normalized * gridSize);
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(scaled, gridSize - 1));
    };
    return {toCell(xNorm), toCell(yNorm)};
}

std::uint32_t ExactHistogramQuantile(const std::vector<std::uint64_t>& histogram,
                                     std::uint64_t total, double quantile) {
    if (total == 0) {
        return 0;
    }
    const auto target = static_cast<std::uint64_t>(
        std::ceil(quantile * static_cast<double>(total)));
    std::uint64_t cumulative = 0;
    for (std::uint32_t value = 0; value < histogram.size(); ++value) {
        cumulative += histogram[value];
        if (cumulative >= target) {
            return value;
        }
    }
    return static_cast<std::uint32_t>(histogram.size() - 1);
}

std::uint32_t VectorQuantile(std::vector<std::uint32_t> values, double quantile) {
    if (values.empty()) {
        return 0;
    }
    const auto index = static_cast<std::size_t>(std::ceil(
        quantile * static_cast<double>(values.size()))) - 1;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

double Ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
               ? 0.0
               : static_cast<double>(numerator) / static_cast<double>(denominator);
}


} // namespace

DatasetLayoutAnalysisReport DatasetLayoutAnalyzer::Analyze(
    const std::filesystem::path& graphPath, const std::filesystem::path& rangesPath,
    const DatasetLayoutAnalysisConfig& config, ProgressCallback progressCallback) {
    ValidateConfig(config);

    DatasetLayoutAnalysisReport report;
    report.config = config;
    report.graphPath = graphPath;
    report.rangesPath = rangesPath;

    detail::TextSourceScanner graphScanner(graphPath);
    const auto nodeCount = graphScanner.Read<std::uint64_t>();
    const auto edgeCount = graphScanner.Read<std::uint64_t>();
    if (nodeCount > std::numeric_limits<std::uint32_t>::max() ||
        edgeCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("dataset exceeds the current 32-bit graph id representation");
    }

    report.nodeCount = nodeCount;
    report.edgeCount = edgeCount;

    std::vector<NodeCoordinate> nodes(static_cast<std::size_t>(nodeCount));
    double minLatitude = std::numeric_limits<double>::infinity();
    double maxLatitude = -std::numeric_limits<double>::infinity();
    double minLongitude = std::numeric_limits<double>::infinity();
    double maxLongitude = -std::numeric_limits<double>::infinity();
    std::uint64_t sequentialNodeRecords = 0;

    ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Nodes, 0, nodeCount);
    for (std::uint64_t record = 0; record < nodeCount; ++record) {
        const auto [id, nodeOffset] = graphScanner.ReadWithOffset<std::uint32_t>();
        if (id >= nodeCount) {
            throw std::runtime_error("node id out of range");
        }
        if (id == record) {
            ++sequentialNodeRecords;
        }
        if (record % config.sourceBlockNodeCount == 0) {
            const auto remaining = nodeCount - record;
            report.nodeSourceBlocks.push_back({
                static_cast<std::uint32_t>(report.nodeSourceBlocks.size()),
                record,
                static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    remaining, config.sourceBlockNodeCount)),
                nodeOffset,
            });
        }

        graphScanner.Read<std::uint64_t>();
        const auto latitude = graphScanner.Read<double>();
        const auto longitude = graphScanner.Read<double>();
        graphScanner.Read<float>();
        graphScanner.Read<std::uint32_t>();

        if (std::isfinite(nodes[id].latitude)) {
            throw std::runtime_error("duplicate node id: " + std::to_string(id));
        }
        nodes[id] = {static_cast<float>(latitude), static_cast<float>(longitude)};
        minLatitude = std::min(minLatitude, latitude);
        maxLatitude = std::max(maxLatitude, latitude);
        minLongitude = std::min(minLongitude, longitude);
        maxLongitude = std::max(maxLongitude, longitude);

        if ((record + 1) % kProgressInterval == 0 || record + 1 == nodeCount) {
            ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Nodes, record + 1,
                           nodeCount);
        }
    }
    report.nodeRecordSequentialRatio = Ratio(sequentialNodeRecords, nodeCount);
    report.minLatitude = minLatitude;
    report.minLongitude = minLongitude;
    report.maxLatitude = maxLatitude;
    report.maxLongitude = maxLongitude;

    const auto gridBits = static_cast<std::uint32_t>(std::countr_zero(config.spatialGridSize));
    std::vector<std::uint32_t> mortonByEdge(static_cast<std::size_t>(edgeCount));
    report.edgeSpatialBounds.resize(static_cast<std::size_t>(edgeCount));
    std::vector<std::uint64_t> childrenByEdge(static_cast<std::size_t>(edgeCount),
                                              PackChildren(data::InvalidEdgeId,
                                                           data::InvalidEdgeId));

    std::uint64_t spatialPairCount = 0;
    std::uint64_t sameCellCount = 0;
    std::uint64_t withinOneCellCount = 0;
    std::uint64_t withinFiveCellsCount = 0;
    std::uint32_t previousCellX = 0;
    std::uint32_t previousCellY = 0;
    bool hasPreviousCell = false;

    std::uint64_t parentChildReferenceCount = 0;
    std::uint64_t parentChildSameBlockCount = 0;
    std::uint64_t parentChildWithinOneBlockCount = 0;
    std::uint64_t parentChildWithinFiveBlocksCount = 0;
    Log2Histogram parentChildDistance;

    std::vector<std::uint32_t> currentNodeBlockRefs;
    std::vector<std::uint32_t> currentChildBlockRefs;
    auto finalizeStreamingBlock = [&]() {
        if (report.streamingEdgeBlocks.empty()) {
            return;
        }
        auto& block = report.streamingEdgeBlocks.back();
        std::sort(currentNodeBlockRefs.begin(), currentNodeBlockRefs.end());
        currentNodeBlockRefs.erase(
            std::unique(currentNodeBlockRefs.begin(), currentNodeBlockRefs.end()),
            currentNodeBlockRefs.end());
        block.nodeBlockRefOffset = static_cast<std::uint32_t>(report.edgeBlockNodeRefs.size());
        block.nodeBlockRefCount = static_cast<std::uint32_t>(currentNodeBlockRefs.size());
        report.edgeBlockNodeRefs.insert(report.edgeBlockNodeRefs.end(),
                                       currentNodeBlockRefs.begin(), currentNodeBlockRefs.end());

        std::sort(currentChildBlockRefs.begin(), currentChildBlockRefs.end());
        currentChildBlockRefs.erase(
            std::unique(currentChildBlockRefs.begin(), currentChildBlockRefs.end()),
            currentChildBlockRefs.end());
        block.childBlockRefOffset = static_cast<std::uint32_t>(report.edgeBlockChildRefs.size());
        block.childBlockRefCount = static_cast<std::uint32_t>(currentChildBlockRefs.size());
        report.edgeBlockChildRefs.insert(report.edgeBlockChildRefs.end(),
                                        currentChildBlockRefs.begin(), currentChildBlockRefs.end());
        currentNodeBlockRefs.clear();
        currentChildBlockRefs.clear();
    };

    ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Edges, 0, edgeCount);
    for (std::uint64_t edgeId64 = 0; edgeId64 < edgeCount; ++edgeId64) {
        const auto [source, sourceOffset] = graphScanner.ReadWithOffset<std::uint32_t>();
        const auto target = graphScanner.Read<std::uint32_t>();
        graphScanner.Read<float>();
        graphScanner.Read<std::int32_t>();
        graphScanner.Read<std::int32_t>();
        const auto childA = DecodeChild(graphScanner.Read<std::int64_t>());
        const auto childB = DecodeChild(graphScanner.Read<std::int64_t>());

        if (source >= nodeCount || target >= nodeCount) {
            throw std::runtime_error("edge endpoint out of range");
        }
        if (!std::isfinite(nodes[source].latitude) || !std::isfinite(nodes[target].latitude)) {
            throw std::runtime_error("edge references a missing node record");
        }

        const auto edgeId = static_cast<std::uint32_t>(edgeId64);
        if (edgeId % config.sourceBlockEdgeCount == 0) {
            if (edgeId != 0) {
                finalizeStreamingBlock();
            }
            const auto remaining = edgeCount - edgeId64;
            const auto recordCount = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                remaining, config.sourceBlockEdgeCount));
            report.graphSourceBlocks.push_back({
                static_cast<std::uint32_t>(report.graphSourceBlocks.size()),
                edgeId64,
                recordCount,
                sourceOffset,
            });

            StreamingEdgeBlockInfo block;
            block.blockId = static_cast<std::uint32_t>(report.streamingEdgeBlocks.size());
            block.firstEdgeId = edgeId64;
            block.edgeCount = recordCount;
            block.graphByteOffset = sourceOffset;
            block.minLatitude = std::numeric_limits<float>::infinity();
            block.minLongitude = std::numeric_limits<float>::infinity();
            block.maxLatitude = -std::numeric_limits<float>::infinity();
            block.maxLongitude = -std::numeric_limits<float>::infinity();
            block.minCellX = config.spatialGridSize - 1;
            block.minCellY = config.spatialGridSize - 1;
            block.maxCellX = 0;
            block.maxCellY = 0;
            report.streamingEdgeBlocks.push_back(block);
        }

        const auto midpointLatitude =
            (static_cast<double>(nodes[source].latitude) + nodes[target].latitude) * 0.5;
        const auto midpointLongitude =
            (static_cast<double>(nodes[source].longitude) + nodes[target].longitude) * 0.5;
        const auto [cellX, cellY] = SpatialCell(midpointLongitude, midpointLatitude,
                                               minLongitude, maxLongitude, minLatitude,
                                               maxLatitude, config.spatialGridSize);
        mortonByEdge[edgeId] = Morton2D(cellX, cellY, gridBits);

        auto& streamingBlock = report.streamingEdgeBlocks.back();
        const auto sourceNodeBlock = source / config.sourceBlockNodeCount;
        const auto targetNodeBlock = target / config.sourceBlockNodeCount;
        currentNodeBlockRefs.push_back(sourceNodeBlock);
        currentNodeBlockRefs.push_back(targetNodeBlock);
        streamingBlock.minLatitude = std::min(
            streamingBlock.minLatitude, std::min(nodes[source].latitude, nodes[target].latitude));
        streamingBlock.minLongitude = std::min(
            streamingBlock.minLongitude, std::min(nodes[source].longitude, nodes[target].longitude));
        streamingBlock.maxLatitude = std::max(
            streamingBlock.maxLatitude, std::max(nodes[source].latitude, nodes[target].latitude));
        streamingBlock.maxLongitude = std::max(
            streamingBlock.maxLongitude, std::max(nodes[source].longitude, nodes[target].longitude));
        const auto [sourceCellX, sourceCellY] = SpatialCell(
            nodes[source].longitude, nodes[source].latitude, minLongitude, maxLongitude,
            minLatitude, maxLatitude, config.spatialGridSize);
        const auto [targetCellX, targetCellY] = SpatialCell(
            nodes[target].longitude, nodes[target].latitude, minLongitude, maxLongitude,
            minLatitude, maxLatitude, config.spatialGridSize);
        streamingBlock.minCellX = std::min(streamingBlock.minCellX,
                                           std::min(sourceCellX, targetCellX));
        streamingBlock.minCellY = std::min(streamingBlock.minCellY,
                                           std::min(sourceCellY, targetCellY));
        streamingBlock.maxCellX = std::max(streamingBlock.maxCellX,
                                           std::max(sourceCellX, targetCellX));
        streamingBlock.maxCellY = std::max(streamingBlock.maxCellY,
                                           std::max(sourceCellY, targetCellY));
        report.edgeSpatialBounds[edgeId] = PackCellBounds({
            std::min(sourceCellX, targetCellX), std::min(sourceCellY, targetCellY),
            std::max(sourceCellX, targetCellX), std::max(sourceCellY, targetCellY)});

        if (hasPreviousCell) {
            ++spatialPairCount;
            const auto dx = static_cast<std::uint32_t>(
                std::abs(static_cast<std::int64_t>(cellX) - previousCellX));
            const auto dy = static_cast<std::uint32_t>(
                std::abs(static_cast<std::int64_t>(cellY) - previousCellY));
            const auto ringDistance = std::max(dx, dy);
            if (ringDistance == 0) {
                ++sameCellCount;
            }
            if (ringDistance <= 1) {
                ++withinOneCellCount;
            }
            if (ringDistance <= 5) {
                ++withinFiveCellsCount;
            }
        }
        previousCellX = cellX;
        previousCellY = cellY;
        hasPreviousCell = true;

        childrenByEdge[edgeId] = PackChildren(childA, childB);
        const auto isShortcut = childA != data::InvalidEdgeId && childB != data::InvalidEdgeId;
        if (isShortcut) {
            ++report.shortcutCount;
            ++streamingBlock.shortcutCount;
            for (const auto child : {childA, childB}) {
                if (child >= edgeCount) {
                    throw std::runtime_error("shortcut child edge id out of range");
                }
                ++parentChildReferenceCount;
                currentChildBlockRefs.push_back(child / config.sourceBlockEdgeCount);
                const auto distance = edgeId > child ? edgeId - child : child - edgeId;
                parentChildDistance.Add(distance);

                const auto parentBlock = edgeId / config.sourceBlockEdgeCount;
                const auto childBlock = child / config.sourceBlockEdgeCount;
                const auto blockDistance = parentBlock > childBlock
                                               ? parentBlock - childBlock
                                               : childBlock - parentBlock;
                if (blockDistance == 0) {
                    ++parentChildSameBlockCount;
                }
                if (blockDistance <= 1) {
                    ++parentChildWithinOneBlockCount;
                }
                if (blockDistance <= 5) {
                    ++parentChildWithinFiveBlocksCount;
                }
            }
        }

        if ((edgeId64 + 1) % kProgressInterval == 0 || edgeId64 + 1 == edgeCount) {
            ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Edges, edgeId64 + 1,
                           edgeCount);
        }
    }

    if (!report.streamingEdgeBlocks.empty()) {
        finalizeStreamingBlock();
    }

    ResolveHierarchyBounds(report.edgeSpatialBounds, childrenByEdge);

    report.sourceOrderSameSpatialCellRatio = Ratio(sameCellCount, spatialPairCount);
    report.sourceOrderWithinOneCellRatio = Ratio(withinOneCellCount, spatialPairCount);
    report.sourceOrderWithinFiveCellsRatio = Ratio(withinFiveCellsCount, spatialPairCount);
    report.parentChildReferenceCount = parentChildReferenceCount;
    report.parentChildSameSourceBlockRatio =
        Ratio(parentChildSameBlockCount, parentChildReferenceCount);
    report.parentChildWithinOneSourceBlockRatio =
        Ratio(parentChildWithinOneBlockCount, parentChildReferenceCount);
    report.parentChildWithinFiveSourceBlocksRatio =
        Ratio(parentChildWithinFiveBlocksCount, parentChildReferenceCount);
    report.parentChildIdDistanceP50UpperBound = parentChildDistance.QuantileUpperBound(0.50);
    report.parentChildIdDistanceP95UpperBound = parentChildDistance.QuantileUpperBound(0.95);
    report.parentChildIdDistanceP99UpperBound = parentChildDistance.QuantileUpperBound(0.99);
    report.parentChildIdDistanceMax = parentChildDistance.Max();

    detail::TextSourceScanner rangeScanner(rangesPath);
    std::vector<std::int32_t> birthByEdge(static_cast<std::size_t>(edgeCount), kMissingRange);
    std::vector<std::int32_t> deathByEdge(static_cast<std::size_t>(edgeCount), kMissingRange);
    std::vector<std::uint64_t> birthHistogram;
    std::vector<std::uint64_t> deathHistogram;
    std::vector<std::int64_t> aliveDifference;
    std::vector<std::uint64_t> lifetimeSpanHistogram;
    std::uint64_t sequentialRangeRecords = 0;
    std::uint32_t maxLevel = 0;

    ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Ranges, 0, edgeCount);
    for (std::uint64_t record = 0; record < edgeCount; ++record) {
        const auto [edgeId, recordOffset] = rangeScanner.ReadWithOffset<std::uint32_t>();
        const auto birthLevel = rangeScanner.Read<std::int32_t>();
        const auto deathLevel = rangeScanner.Read<std::int32_t>();
        if (edgeId >= edgeCount) {
            throw std::runtime_error("range edge id out of range");
        }
        if (edgeId == record) {
            ++sequentialRangeRecords;
        }
        if (record % config.sourceBlockEdgeCount == 0) {
            const auto remaining = edgeCount - record;
            report.rangeSourceBlocks.push_back({
                static_cast<std::uint32_t>(report.rangeSourceBlocks.size()),
                record,
                static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    remaining, config.sourceBlockEdgeCount)),
                recordOffset,
            });
        }
        if (birthByEdge[edgeId] != kMissingRange) {
            ++report.duplicateRangeRecordCount;
        }

        if (birthLevel == -1 && deathLevel == -1) {
            birthByEdge[edgeId] = -1;
            deathByEdge[edgeId] = -1;
            ++report.nonDrawableEdgeCount;
        } else if (birthLevel < 0 || deathLevel < 0 || deathLevel > birthLevel) {
            birthByEdge[edgeId] = -2;
            deathByEdge[edgeId] = -2;
            ++report.invalidRangeCount;
        } else {
            birthByEdge[edgeId] = birthLevel;
            deathByEdge[edgeId] = deathLevel;
            ++report.drawableEdgeCount;
            ++report.streamingEdgeBlocks[edgeId / config.sourceBlockEdgeCount].drawableEdgeCount;
            const auto birth = static_cast<std::uint32_t>(birthLevel);
            const auto death = static_cast<std::uint32_t>(deathLevel);
            maxLevel = std::max(maxLevel, birth);

            if (birthHistogram.size() <= birth) {
                birthHistogram.resize(static_cast<std::size_t>(birth) + 1, 0);
                deathHistogram.resize(static_cast<std::size_t>(birth) + 1, 0);
                aliveDifference.resize(static_cast<std::size_t>(birth) + 2, 0);
                lifetimeSpanHistogram.resize(static_cast<std::size_t>(birth) + 1, 0);
            }
            ++birthHistogram[birth];
            ++deathHistogram[death];
            ++aliveDifference[death];
            --aliveDifference[static_cast<std::size_t>(birth) + 1];
            const auto span = birth - death;
            ++lifetimeSpanHistogram[span];
        }

        if ((record + 1) % kProgressInterval == 0 || record + 1 == edgeCount) {
            ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Ranges, record + 1,
                           edgeCount);
        }
    }

    report.rangeRecordSequentialRatio = Ratio(sequentialRangeRecords, edgeCount);
    report.maxLevel = maxLevel;
    if (report.rangeSourceBlocks.size() != report.streamingEdgeBlocks.size()) {
        throw std::runtime_error("graph/range source block count mismatch");
    }
    for (std::size_t block = 0; block < report.streamingEdgeBlocks.size(); ++block) {
        report.streamingEdgeBlocks[block].rangeByteOffset = report.rangeSourceBlocks[block].byteOffset;
    }

    report.lodMaskWordsPerBlock = maxLevel / 64u + 1u;
    report.edgeBlockAliveMasks.assign(
        report.streamingEdgeBlocks.size() * report.lodMaskWordsPerBlock, 0);
    const auto setAliveInterval = [&](std::uint32_t blockId, std::uint32_t death,
                                      std::uint32_t birth) {
        const auto firstWord = death / 64u;
        const auto lastWord = birth / 64u;
        const auto base = static_cast<std::size_t>(blockId) * report.lodMaskWordsPerBlock;
        if (firstWord == lastWord) {
            const auto lowMask = ~std::uint64_t{0} << (death % 64u);
            const auto highBit = birth % 64u;
            const auto highMask = highBit == 63u
                                      ? ~std::uint64_t{0}
                                      : (std::uint64_t{1} << (highBit + 1u)) - 1u;
            report.edgeBlockAliveMasks[base + firstWord] |= lowMask & highMask;
            return;
        }
        report.edgeBlockAliveMasks[base + firstWord] |=
            ~std::uint64_t{0} << (death % 64u);
        for (auto word = firstWord + 1u; word < lastWord; ++word) {
            report.edgeBlockAliveMasks[base + word] = ~std::uint64_t{0};
        }
        const auto highBit = birth % 64u;
        report.edgeBlockAliveMasks[base + lastWord] |=
            highBit == 63u ? ~std::uint64_t{0}
                           : (std::uint64_t{1} << (highBit + 1u)) - 1u;
    };
    for (std::uint64_t edgeId = 0; edgeId < edgeCount; ++edgeId) {
        const auto birth = birthByEdge[static_cast<std::size_t>(edgeId)];
        const auto death = deathByEdge[static_cast<std::size_t>(edgeId)];
        if (birth >= 0 && death >= 0) {
            setAliveInterval(static_cast<std::uint32_t>(edgeId / config.sourceBlockEdgeCount),
                             static_cast<std::uint32_t>(death),
                             static_cast<std::uint32_t>(birth));
        }
    }

    for (const auto birth : birthByEdge) {
        if (birth == kMissingRange) {
            ++report.missingRangeRecordCount;
        }
    }

    if (report.drawableEdgeCount > 0) {
        std::int64_t alive = 0;
        report.lodDensity.reserve(static_cast<std::size_t>(maxLevel) + 1);
        for (std::uint32_t level = 0; level <= maxLevel; ++level) {
            alive += aliveDifference[level];
            report.lodDensity.push_back({
                level,
                level < birthHistogram.size() ? birthHistogram[level] : 0,
                level < deathHistogram.size() ? deathHistogram[level] : 0,
                static_cast<std::uint64_t>(std::max<std::int64_t>(alive, 0)),
            });
        }
    }

    report.lifetimeSpanP50 =
        ExactHistogramQuantile(lifetimeSpanHistogram, report.drawableEdgeCount, 0.50);
    report.lifetimeSpanP95 =
        ExactHistogramQuantile(lifetimeSpanHistogram, report.drawableEdgeCount, 0.95);
    report.lifetimeSpanP99 =
        ExactHistogramQuantile(lifetimeSpanHistogram, report.drawableEdgeCount, 0.99);
    report.lifetimeSpanMax = lifetimeSpanHistogram.empty()
                                 ? 0
                                 : static_cast<std::uint32_t>(lifetimeSpanHistogram.size() - 1);
    while (report.lifetimeSpanMax > 0 &&
           lifetimeSpanHistogram[report.lifetimeSpanMax] == 0) {
        --report.lifetimeSpanMax;
    }

    std::vector<std::uint64_t> birthDeltaHistogram(static_cast<std::size_t>(maxLevel) + 1, 0);
    std::uint64_t birthPairCount = 0;
    std::uint64_t sameBirthLevelCount = 0;
    std::uint64_t sameBirthBandCount = 0;
    std::int32_t previousBirth = -1;
    bool hasPreviousBirth = false;
    for (std::uint64_t edgeId = 0; edgeId < edgeCount; ++edgeId) {
        const auto birth = birthByEdge[static_cast<std::size_t>(edgeId)];
        if (birth < 0) {
            continue;
        }
        if (hasPreviousBirth) {
            ++birthPairCount;
            const auto delta = static_cast<std::uint32_t>(std::abs(
                static_cast<std::int64_t>(birth) - previousBirth));
            ++birthDeltaHistogram[delta];
            if (delta == 0) {
                ++sameBirthLevelCount;
            }
            if (static_cast<std::uint32_t>(birth) / config.lodBandWidth ==
                static_cast<std::uint32_t>(previousBirth) / config.lodBandWidth) {
                ++sameBirthBandCount;
            }
        }
        previousBirth = birth;
        hasPreviousBirth = true;
    }
    report.sourceOrderSameBirthLevelRatio = Ratio(sameBirthLevelCount, birthPairCount);
    report.sourceOrderSameLodBandRatio = Ratio(sameBirthBandCount, birthPairCount);
    report.sourceOrderBirthDeltaP50 = ExactHistogramQuantile(birthDeltaHistogram, birthPairCount, 0.50);
    report.sourceOrderBirthDeltaP95 = ExactHistogramQuantile(birthDeltaHistogram, birthPairCount, 0.95);
    report.sourceOrderBirthDeltaP99 = ExactHistogramQuantile(birthDeltaHistogram, birthPairCount, 0.99);

    ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Layout, 0,
                   report.drawableEdgeCount);
    std::vector<VirtualSortEntry> sortedEdges;
    sortedEdges.reserve(static_cast<std::size_t>(report.drawableEdgeCount));
    for (std::uint64_t edgeId64 = 0; edgeId64 < edgeCount; ++edgeId64) {
        const auto birth = birthByEdge[static_cast<std::size_t>(edgeId64)];
        if (birth < 0) {
            continue;
        }
        sortedEdges.push_back({
            static_cast<std::uint32_t>(birth) / config.lodBandWidth,
            mortonByEdge[static_cast<std::size_t>(edgeId64)],
            static_cast<std::uint32_t>(edgeId64),
        });
    }

    std::sort(sortedEdges.begin(), sortedEdges.end(), [](const auto& a, const auto& b) {
        if (a.bandIndex != b.bandIndex) {
            return a.bandIndex < b.bandIndex;
        }
        if (a.mortonCell != b.mortonCell) {
            return a.mortonCell < b.mortonCell;
        }
        return a.edgeId < b.edgeId;
    });

    std::vector<std::uint32_t> pageByEdge(static_cast<std::size_t>(edgeCount), kInvalidPageId);
    std::vector<std::uint32_t> sourceBlocksPerPage;
    std::vector<std::uint32_t> pageEdgeCounts;
    std::uint64_t processedLayoutEdges = 0;

    std::size_t bandBegin = 0;
    while (bandBegin < sortedEdges.size()) {
        const auto bandIndex = sortedEdges[bandBegin].bandIndex;
        std::size_t bandEnd = bandBegin + 1;
        while (bandEnd < sortedEdges.size() && sortedEdges[bandEnd].bandIndex == bandIndex) {
            ++bandEnd;
        }

        std::vector<std::uint32_t> cellCounts;
        std::size_t cellBegin = bandBegin;
        while (cellBegin < bandEnd) {
            const auto morton = sortedEdges[cellBegin].mortonCell;
            auto cellEnd = cellBegin + 1;
            while (cellEnd < bandEnd && sortedEdges[cellEnd].mortonCell == morton) {
                ++cellEnd;
            }
            cellCounts.push_back(static_cast<std::uint32_t>(cellEnd - cellBegin));
            cellBegin = cellEnd;
        }

        const auto bandEdgeCount = static_cast<std::uint64_t>(bandEnd - bandBegin);
        LodBandSpatialInfo spatialInfo;
        spatialInfo.bandIndex = bandIndex;
        spatialInfo.levelMin = bandIndex * config.lodBandWidth;
        spatialInfo.levelMax = spatialInfo.levelMin + config.lodBandWidth - 1;
        spatialInfo.edgeCount = bandEdgeCount;
        spatialInfo.occupiedCellCount = static_cast<std::uint32_t>(cellCounts.size());
        spatialInfo.meanEdgesPerOccupiedCell =
            cellCounts.empty() ? 0.0
                               : static_cast<double>(bandEdgeCount) /
                                     static_cast<double>(cellCounts.size());
        spatialInfo.p95EdgesPerCell = VectorQuantile(cellCounts, 0.95);
        spatialInfo.maxEdgesPerCell =
            cellCounts.empty() ? 0 : *std::max_element(cellCounts.begin(), cellCounts.end());
        report.lodBandSpatialDensity.push_back(spatialInfo);

        for (std::size_t pageBegin = bandBegin; pageBegin < bandEnd;
             pageBegin += config.virtualPageEdgeCount) {
            const auto pageEnd = std::min<std::size_t>(
                pageBegin + config.virtualPageEdgeCount, bandEnd);
            const auto pageId = static_cast<std::uint32_t>(report.virtualPages.size());
            std::vector<std::uint32_t> sourceBlocks;
            sourceBlocks.reserve(pageEnd - pageBegin);

            for (auto i = pageBegin; i < pageEnd; ++i) {
                const auto edgeId = sortedEdges[i].edgeId;
                pageByEdge[edgeId] = pageId;
                sourceBlocks.push_back(edgeId / config.sourceBlockEdgeCount);
            }
            std::sort(sourceBlocks.begin(), sourceBlocks.end());
            sourceBlocks.erase(std::unique(sourceBlocks.begin(), sourceBlocks.end()),
                               sourceBlocks.end());

            VirtualPageInfo page;
            page.pageId = pageId;
            page.bandIndex = bandIndex;
            page.levelMin = bandIndex * config.lodBandWidth;
            page.levelMax = page.levelMin + config.lodBandWidth - 1;
            page.edgeCount = static_cast<std::uint32_t>(pageEnd - pageBegin);
            page.sourceBlockCount = static_cast<std::uint32_t>(sourceBlocks.size());
            page.firstMortonCell = sortedEdges[pageBegin].mortonCell;
            page.lastMortonCell = sortedEdges[pageEnd - 1].mortonCell;
            report.virtualPages.push_back(page);
            sourceBlocksPerPage.push_back(page.sourceBlockCount);
            pageEdgeCounts.push_back(page.edgeCount);
        }

        processedLayoutEdges += bandEdgeCount;
        ReportProgress(progressCallback, DatasetLayoutAnalysisStage::Layout,
                       processedLayoutEdges, report.drawableEdgeCount);
        bandBegin = bandEnd;
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> pageChildSourceBlockPairs;
    pageChildSourceBlockPairs.reserve(static_cast<std::size_t>(report.drawableEdgeCount));
    for (std::uint64_t edgeId64 = 0; edgeId64 < edgeCount; ++edgeId64) {
        const auto pageId = pageByEdge[static_cast<std::size_t>(edgeId64)];
        if (pageId == kInvalidPageId) {
            continue;
        }
        const auto packedChildren = childrenByEdge[static_cast<std::size_t>(edgeId64)];
        const auto childA = ChildA(packedChildren);
        const auto childB = ChildB(packedChildren);
        if (childA == data::InvalidEdgeId || childB == data::InvalidEdgeId) {
            continue;
        }

        ++report.drawableShortcutCount;
        for (const auto child : {childA, childB}) {
            ++report.drawableShortcutChildReferenceCount;
            if (birthByEdge[child] >= 0) {
                ++report.drawableShortcutChildWithDrawableLifetimeCount;
            }
            pageChildSourceBlockPairs.emplace_back(
                pageId, child / config.sourceBlockEdgeCount);
        }
    }

    std::sort(pageChildSourceBlockPairs.begin(), pageChildSourceBlockPairs.end());
    pageChildSourceBlockPairs.erase(
        std::unique(pageChildSourceBlockPairs.begin(), pageChildSourceBlockPairs.end()),
        pageChildSourceBlockPairs.end());
    for (const auto& [pageId, sourceBlock] : pageChildSourceBlockPairs) {
        static_cast<void>(sourceBlock);
        ++report.virtualPages[pageId].directChildSourceBlockCount;
    }

    std::vector<std::uint32_t> childSourceBlocksPerPage;
    childSourceBlocksPerPage.reserve(report.virtualPages.size());
    std::uint64_t totalPageEdges = 0;
    for (const auto& page : report.virtualPages) {
        totalPageEdges += page.edgeCount;
        childSourceBlocksPerPage.push_back(page.directChildSourceBlockCount);
    }

    report.virtualPageCount = static_cast<std::uint32_t>(report.virtualPages.size());
    report.virtualPageMeanFill = report.virtualPages.empty()
                                     ? 0.0
                                     : static_cast<double>(totalPageEdges) /
                                           (static_cast<double>(report.virtualPages.size()) *
                                            config.virtualPageEdgeCount);
    report.virtualPageSourceBlockP50 = VectorQuantile(sourceBlocksPerPage, 0.50);
    report.virtualPageSourceBlockP95 = VectorQuantile(sourceBlocksPerPage, 0.95);
    report.virtualPageSourceBlockP99 = VectorQuantile(sourceBlocksPerPage, 0.99);
    report.virtualPageSourceBlockMax = sourceBlocksPerPage.empty()
                                           ? 0
                                           : *std::max_element(sourceBlocksPerPage.begin(),
                                                               sourceBlocksPerPage.end());
    report.virtualPageDirectChildSourceBlockP50 =
        VectorQuantile(childSourceBlocksPerPage, 0.50);
    report.virtualPageDirectChildSourceBlockP95 =
        VectorQuantile(childSourceBlocksPerPage, 0.95);
    report.virtualPageDirectChildSourceBlockP99 =
        VectorQuantile(childSourceBlocksPerPage, 0.99);
    report.virtualPageDirectChildSourceBlockMax =
        childSourceBlocksPerPage.empty()
            ? 0
            : *std::max_element(childSourceBlocksPerPage.begin(),
                                childSourceBlocksPerPage.end());

    // Runtime paging uses adaptive adjacent-LOD groups followed by hierarchical
    // spatial ownership. A spatial node is subdivided only while its payload is too
    // large. Edges whose full shortcut geometry crosses a child boundary remain in
    // the current ancestor node, so every drawable edge belongs to exactly one page.
    const auto runtimeLodGroups =
        BuildRuntimeLodGroups(report.lodDensity, maxLevel, config.virtualPageEdgeCount);
    std::vector<std::uint32_t> levelToRuntimeGroup(
        static_cast<std::size_t>(maxLevel) + 1u, kInvalidPageId);
    for (const auto& group : runtimeLodGroups) {
        for (std::uint32_t level = group.levelMin; level <= group.levelMax; ++level) {
            levelToRuntimeGroup[level] = group.groupIndex;
        }
    }

    std::vector<std::vector<std::uint32_t>> runtimeEdgesByGroup(runtimeLodGroups.size());
    for (std::uint32_t edgeId = 0; edgeId < edgeCount; ++edgeId) {
        const auto birth = birthByEdge[edgeId];
        if (birth < 0) {
            continue;
        }
        const auto groupId = levelToRuntimeGroup[static_cast<std::uint32_t>(birth)];
        if (groupId == kInvalidPageId || groupId >= runtimeEdgesByGroup.size()) {
            throw std::runtime_error("drawable edge has no runtime LOD group");
        }
        runtimeEdgesByGroup[groupId].push_back(edgeId);
    }

    std::vector<std::vector<std::uint32_t>> runtimePagesByGroup(runtimeLodGroups.size());

    auto appendAliveMask = [&](RuntimeGraphPageInfo& page,
                               std::span<const std::uint32_t> edgeIds) {
        page.lodMaskOffset =
            static_cast<std::uint32_t>(report.runtimeGraphPageAliveMasks.size());
        report.runtimeGraphPageAliveMasks.resize(
            report.runtimeGraphPageAliveMasks.size() + report.lodMaskWordsPerBlock, 0);

        for (const auto edgeId : edgeIds) {
            const auto birth = birthByEdge[edgeId];
            const auto death = deathByEdge[edgeId];
            if (birth < 0 || death < 0) {
                continue;
            }

            const auto firstWord = static_cast<std::uint32_t>(death) / 64u;
            const auto lastWord = static_cast<std::uint32_t>(birth) / 64u;
            const auto base = static_cast<std::size_t>(page.lodMaskOffset);
            if (firstWord == lastWord) {
                const auto lowMask = ~std::uint64_t{0} << (static_cast<std::uint32_t>(death) % 64u);
                const auto highBit = static_cast<std::uint32_t>(birth) % 64u;
                const auto highMask = highBit == 63u
                                          ? ~std::uint64_t{0}
                                          : (std::uint64_t{1} << (highBit + 1u)) - 1u;
                report.runtimeGraphPageAliveMasks[base + firstWord] |= lowMask & highMask;
            } else {
                report.runtimeGraphPageAliveMasks[base + firstWord] |=
                    ~std::uint64_t{0} << (static_cast<std::uint32_t>(death) % 64u);
                for (auto word = firstWord + 1u; word < lastWord; ++word) {
                    report.runtimeGraphPageAliveMasks[base + word] = ~std::uint64_t{0};
                }
                const auto highBit = static_cast<std::uint32_t>(birth) % 64u;
                report.runtimeGraphPageAliveMasks[base + lastWord] |=
                    highBit == 63u ? ~std::uint64_t{0}
                                   : (std::uint64_t{1} << (highBit + 1u)) - 1u;
            }
        }
    };

    auto emitRuntimePages = [&](const RuntimeLodGroup& group,
                                std::uint32_t spatialLevel,
                                std::uint32_t tileX,
                                std::uint32_t tileY,
                                std::vector<std::uint32_t>& edgeIds) {
        if (edgeIds.empty()) {
            return;
        }

        std::sort(edgeIds.begin(), edgeIds.end(), [&](std::uint32_t left, std::uint32_t right) {
            const auto leftMorton = mortonByEdge[left];
            const auto rightMorton = mortonByEdge[right];
            if (leftMorton != rightMorton) {
                return leftMorton < rightMorton;
            }
            const auto leftBirth = birthByEdge[left];
            const auto rightBirth = birthByEdge[right];
            if (leftBirth != rightBirth) {
                return leftBirth > rightBirth;
            }
            return left < right;
        });

        std::uint32_t subPage = 0;
        for (std::size_t pageBegin = 0; pageBegin < edgeIds.size();
             pageBegin += config.virtualPageEdgeCount, ++subPage) {
            const auto pageEnd = std::min<std::size_t>(
                pageBegin + config.virtualPageEdgeCount, edgeIds.size());

            RuntimeGraphPageInfo page;
            page.pageId = static_cast<std::uint32_t>(report.runtimeGraphPages.size());
            page.bandIndex = group.groupIndex;
            page.levelMin = group.levelMin;
            page.levelMax = group.levelMax;
            page.spatialLevel = spatialLevel;
            page.tileX = tileX;
            page.tileY = tileY;
            page.subPage = subPage;
            page.edgeCount = static_cast<std::uint32_t>(pageEnd - pageBegin);
            page.edgeIdOffset =
                static_cast<std::uint32_t>(report.runtimeGraphPageEdgeIds.size());

            std::vector<std::uint32_t> pageEdgeIds(
                edgeIds.begin() + static_cast<std::ptrdiff_t>(pageBegin),
                edgeIds.begin() + static_cast<std::ptrdiff_t>(pageEnd));
            std::vector<std::uint32_t> sourceRefs;
            sourceRefs.reserve(pageEdgeIds.size());
            std::vector<std::uint32_t> childSourceRefs;
            childSourceRefs.reserve(pageEdgeIds.size());

            for (const auto edgeId : pageEdgeIds) {
                sourceRefs.push_back(edgeId / config.sourceBlockEdgeCount);
                const auto packedChildren = childrenByEdge[edgeId];
                const auto childA = ChildA(packedChildren);
                const auto childB = ChildB(packedChildren);
                if (childA != data::InvalidEdgeId) {
                    childSourceRefs.push_back(childA / config.sourceBlockEdgeCount);
                }
                if (childB != data::InvalidEdgeId) {
                    childSourceRefs.push_back(childB / config.sourceBlockEdgeCount);
                }
            }

            std::sort(pageEdgeIds.begin(), pageEdgeIds.end());
            report.runtimeGraphPageEdgeIds.insert(report.runtimeGraphPageEdgeIds.end(),
                                                   pageEdgeIds.begin(), pageEdgeIds.end());

            std::sort(sourceRefs.begin(), sourceRefs.end());
            sourceRefs.erase(std::unique(sourceRefs.begin(), sourceRefs.end()), sourceRefs.end());
            page.sourceBlockRefOffset = static_cast<std::uint32_t>(
                report.runtimeGraphPageSourceBlockRefs.size());
            page.sourceBlockRefCount = static_cast<std::uint32_t>(sourceRefs.size());
            report.runtimeGraphPageSourceBlockRefs.insert(
                report.runtimeGraphPageSourceBlockRefs.end(), sourceRefs.begin(), sourceRefs.end());

            std::sort(childSourceRefs.begin(), childSourceRefs.end());
            childSourceRefs.erase(
                std::unique(childSourceRefs.begin(), childSourceRefs.end()), childSourceRefs.end());
            page.childSourceBlockRefOffset = static_cast<std::uint32_t>(
                report.runtimeGraphPageChildSourceBlockRefs.size());
            page.childSourceBlockRefCount = static_cast<std::uint32_t>(childSourceRefs.size());
            report.runtimeGraphPageChildSourceBlockRefs.insert(
                report.runtimeGraphPageChildSourceBlockRefs.end(),
                childSourceRefs.begin(), childSourceRefs.end());

            appendAliveMask(page, pageEdgeIds);
            report.runtimeGraphPages.push_back(page);
            runtimePagesByGroup[group.groupIndex].push_back(page.pageId);
        }
    };

    std::function<void(const RuntimeLodGroup&, std::vector<std::uint32_t>&&, std::uint32_t,
                       std::uint32_t, std::uint32_t)> partitionSpatially;
    partitionSpatially = [&](const RuntimeLodGroup& group,
                             std::vector<std::uint32_t>&& edgeIds,
                             std::uint32_t spatialLevel,
                             std::uint32_t tileX,
                             std::uint32_t tileY) {
        if (edgeIds.empty()) {
            return;
        }
        if (edgeIds.size() <= config.virtualPageEdgeCount || spatialLevel >= gridBits) {
            emitRuntimePages(group, spatialLevel, tileX, tileY, edgeIds);
            return;
        }

        std::vector<std::uint32_t> stay;
        std::array<std::vector<std::uint32_t>, 4> children;
        stay.reserve(edgeIds.size() / 8u + 1u);
        const auto childLevel = spatialLevel + 1u;
        const auto childShift = gridBits - childLevel;

        for (const auto edgeId : edgeIds) {
            const auto bounds = UnpackCellBounds(report.edgeSpatialBounds[edgeId]);
            const auto minTileX = bounds.minX >> childShift;
            const auto maxTileX = bounds.maxX >> childShift;
            const auto minTileY = bounds.minY >> childShift;
            const auto maxTileY = bounds.maxY >> childShift;
            if (minTileX == maxTileX && minTileY == maxTileY) {
                const auto childX = minTileX & 1u;
                const auto childY = minTileY & 1u;
                children[childX | (childY << 1u)].push_back(edgeId);
            } else {
                stay.push_back(edgeId);
            }
        }

        emitRuntimePages(group, spatialLevel, tileX, tileY, stay);
        for (std::uint32_t child = 0; child < children.size(); ++child) {
            if (children[child].empty()) {
                continue;
            }
            const auto childX = child & 1u;
            const auto childY = (child >> 1u) & 1u;
            partitionSpatially(group, std::move(children[child]), childLevel,
                               tileX * 2u + childX, tileY * 2u + childY);
        }
    };

    for (const auto& group : runtimeLodGroups) {
        auto& edges = runtimeEdgesByGroup[group.groupIndex];
        partitionSpatially(group, std::move(edges), 0, 0, 0);
    }

    // Page-level LOD hierarchy. Each page points to overlapping pages in the next
    // finer non-empty LOD group. This remains metadata only; edge records are never
    // duplicated between spatial nodes or LOD groups.
    for (std::size_t group = 0; group + 1 < runtimePagesByGroup.size(); ++group) {
        for (const auto parentId : runtimePagesByGroup[group]) {
            auto& parent = report.runtimeGraphPages[parentId];
            parent.lodChildPageRefOffset =
                static_cast<std::uint32_t>(report.runtimeGraphPageLodChildRefs.size());
            for (const auto childId : runtimePagesByGroup[group + 1]) {
                if (TilesOverlap(parent, report.runtimeGraphPages[childId], gridBits)) {
                    report.runtimeGraphPageLodChildRefs.push_back(childId);
                }
            }
            parent.lodChildPageRefCount = static_cast<std::uint32_t>(
                report.runtimeGraphPageLodChildRefs.size() - parent.lodChildPageRefOffset);
        }
    }

    std::vector<std::uint32_t> runtimeSourceBlocks;
    std::vector<std::uint32_t> runtimeChildSourceBlocks;
    runtimeSourceBlocks.reserve(report.runtimeGraphPages.size());
    runtimeChildSourceBlocks.reserve(report.runtimeGraphPages.size());
    for (const auto& page : report.runtimeGraphPages) {
        runtimeSourceBlocks.push_back(page.sourceBlockRefCount);
        runtimeChildSourceBlocks.push_back(page.childSourceBlockRefCount);
    }
    report.runtimeGraphPageSourceBlockP50 = VectorQuantile(runtimeSourceBlocks, 0.50);
    report.runtimeGraphPageSourceBlockP95 = VectorQuantile(runtimeSourceBlocks, 0.95);
    report.runtimeGraphPageSourceBlockP99 = VectorQuantile(runtimeSourceBlocks, 0.99);
    report.runtimeGraphPageSourceBlockMax = runtimeSourceBlocks.empty()
                                                ? 0
                                                : *std::max_element(runtimeSourceBlocks.begin(),
                                                                    runtimeSourceBlocks.end());
    report.runtimeGraphPageChildSourceBlockP50 = VectorQuantile(runtimeChildSourceBlocks, 0.50);
    report.runtimeGraphPageChildSourceBlockP95 = VectorQuantile(runtimeChildSourceBlocks, 0.95);
    report.runtimeGraphPageChildSourceBlockP99 = VectorQuantile(runtimeChildSourceBlocks, 0.99);
    report.runtimeGraphPageChildSourceBlockMax = runtimeChildSourceBlocks.empty()
                                                     ? 0
                                                     : *std::max_element(
                                                           runtimeChildSourceBlocks.begin(),
                                                           runtimeChildSourceBlocks.end());

    return report;
}


} // namespace chmv::streaming::analysis
