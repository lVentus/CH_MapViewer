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
    return static_cast<std::size_t>(std::count_if(
        ranges_.begin(), ranges_.end(), [](const EdgeRange& range) { return range.IsDrawable(); }));
}

std::size_t CHGraph::ShortcutCount() const {
    return static_cast<std::size_t>(std::count_if(
        edges_.begin(), edges_.end(), [](const CHEdge& edge) { return edge.IsShortcut(); }));
}

} // namespace chmv::data
