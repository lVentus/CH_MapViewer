#pragma once

#include "data/ch/CHTypes.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chmv::data {

class CHGraph {
public:
    [[nodiscard]] const std::vector<CHNode>& Nodes() const { return nodes_; }
    [[nodiscard]] const std::vector<CHEdge>& Edges() const { return edges_; }
    [[nodiscard]] const std::vector<EdgeRange>& Ranges() const { return ranges_; }

    [[nodiscard]] std::vector<CHNode>& Nodes() { return nodes_; }
    [[nodiscard]] std::vector<CHEdge>& Edges() { return edges_; }
    [[nodiscard]] std::vector<EdgeRange>& Ranges() { return ranges_; }

    [[nodiscard]] const CHEdge& Edge(std::uint32_t edgeId) const;
    [[nodiscard]] const EdgeRange& Range(std::uint32_t edgeId) const;

    [[nodiscard]] std::size_t NodeCount() const { return nodes_.size(); }
    [[nodiscard]] std::size_t EdgeCount() const { return edges_.size(); }
    [[nodiscard]] std::size_t DrawableEdgeCount() const;
    [[nodiscard]] std::size_t ShortcutCount() const;

private:
    std::vector<CHNode> nodes_;
    std::vector<CHEdge> edges_;
    std::vector<EdgeRange> ranges_;
};

} // namespace chmv::data
