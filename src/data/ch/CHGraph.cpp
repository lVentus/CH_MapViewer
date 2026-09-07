#include "data/ch/CHGraph.h"

#include <algorithm>
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

std::size_t CHGraph::DrawableEdgeCount() const {
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

} // namespace chmv::data
