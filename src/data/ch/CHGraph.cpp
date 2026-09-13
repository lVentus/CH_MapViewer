#include "data/ch/CHGraph.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace chmv::data {

const CHEdge& CHGraph::Edge(std::uint32_t edgeId) const {
    if (edgeId >= edges_.size()) {
        throw std::out_of_range("edge id out of range");
    }
    return edges_[edgeId];
}

const EdgeRange& CHGraph::Range(std::uint32_t edgeId) const {
    if (edgeId >= ranges_.size()) {
        throw std::out_of_range("edge range id out of range");
    }
    return ranges_[edgeId];
}


const CHProjection& CHGraph::Projection() const {
    if (!projection_) {
        projection_ = ComputeCHProjection(nodes_);
    }
    return *projection_;
}

const OrderedRangeIndex& CHGraph::OrderedRanges() const {
    if (!orderedRangeIndex_) {
        BuildOrderedRangeIndex();
    }
    return *orderedRangeIndex_;
}

std::size_t CHGraph::DrawableEdgeCount() const {
    if (orderedRangeIndex_) {
        return orderedRangeIndex_->entries.size();
    }
    if (!drawableEdgeCount_) {
        drawableEdgeCount_ = static_cast<std::size_t>(std::count_if(
            ranges_.begin(), ranges_.end(), [](const EdgeRange& range) { return range.IsDrawable(); }));
    }
    return *drawableEdgeCount_;
}

std::size_t CHGraph::ShortcutCount() const {
    if (!shortcutCount_) {
        shortcutCount_ = static_cast<std::size_t>(std::count_if(
            edges_.begin(), edges_.end(), [](const CHEdge& edge) { return edge.IsShortcut(); }));
    }
    return *shortcutCount_;
}

void CHGraph::BuildOrderedRangeIndex() const {
    OrderedRangeIndex index;

    std::int32_t maxBirthLevel = -1;
    std::size_t drawableCount = 0;
    for (const auto& range : ranges_) {
        if (!range.IsDrawable() || range.birthLevel < range.deathLevel) {
            continue;
        }
        maxBirthLevel = std::max(maxBirthLevel, range.birthLevel);
        ++drawableCount;
    }

    if (maxBirthLevel < 0 || drawableCount == 0) {
        drawableEdgeCount_ = 0;
        orderedRangeIndex_ = std::move(index);
        return;
    }

    if (drawableCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("too many drawable edges for 32-bit ordered range indexing");
    }

    const auto levelCount = static_cast<std::size_t>(maxBirthLevel) + 1;
    std::vector<std::uint32_t> birthCounts(levelCount, 0);
    std::vector<std::int64_t> aliveDiff(levelCount + 1, 0);

    for (const auto& range : ranges_) {
        if (!range.IsDrawable() || range.birthLevel < range.deathLevel) {
            continue;
        }

        ++birthCounts[static_cast<std::size_t>(range.birthLevel)];
        ++aliveDiff[static_cast<std::size_t>(range.deathLevel)];
        --aliveDiff[static_cast<std::size_t>(range.birthLevel) + 1];
    }

    std::vector<std::uint32_t> bucketStart(levelCount, 0);
    std::uint32_t offset = 0;
    for (std::size_t level = levelCount; level-- > 0;) {
        bucketStart[level] = offset;
        offset += birthCounts[level];
    }

    index.entries.resize(drawableCount);
    auto writePosition = bucketStart;
    for (std::size_t edgeId = 0; edgeId < ranges_.size(); ++edgeId) {
        const auto& range = ranges_[edgeId];
        if (!range.IsDrawable() || range.birthLevel < range.deathLevel) {
            continue;
        }

        const auto birth = static_cast<std::size_t>(range.birthLevel);
        index.entries[writePosition[birth]++] = {
            static_cast<std::uint32_t>(edgeId),
            range.deathLevel,
        };
    }

    index.scanEndByLevel.resize(levelCount);
    std::uint32_t scanCount = 0;
    for (std::size_t level = levelCount; level-- > 0;) {
        scanCount += birthCounts[level];
        index.scanEndByLevel[level] = scanCount;
    }

    index.aliveCountByLevel.resize(levelCount);
    std::int64_t aliveCount = 0;
    for (std::size_t level = 0; level < levelCount; ++level) {
        aliveCount += aliveDiff[level];
        index.aliveCountByLevel[level] = static_cast<std::size_t>(aliveCount);
    }

    drawableEdgeCount_ = drawableCount;
    orderedRangeIndex_ = std::move(index);
}

} // namespace chmv::data
