#pragma once

#include "reference/CPURangeFilter.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace chmv::data {
class CHGraph;
}

namespace chmv::pipeline {

struct CPUProcessingStats {
    std::size_t aliveEdgeCount = 0;
    std::size_t outputEdgeCount = 0;
    double rangeFilterMs = 0.0;
    double unfoldingMs = 0.0;
};

struct CPUProcessingResult {
    std::span<const std::uint32_t> aliveEdgeIds;
    std::span<const std::uint32_t> edgeIds;
    CPUProcessingStats stats;
};

class CPUReferencePipeline {
public:
    [[nodiscard]] CPUProcessingResult Process(const data::CHGraph& graph, std::int32_t lodLevel,
                                              bool fullyUnfold);

private:
    reference::CPURangeFilter rangeFilter_;
    std::vector<std::uint32_t> aliveEdges_;
    std::vector<std::uint32_t> unfoldedEdges_;
};

} // namespace chmv::pipeline
