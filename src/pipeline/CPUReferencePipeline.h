#pragma once

#include "geometry/GeometryRefinement.h"
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
    std::size_t rangeScannedEdgeCount = 0;
    std::size_t aliveEdgeCount = 0;
    std::size_t outputEdgeCount = 0;
    double rangeFilterMs = 0.0;
    double geometryRefinementMs = 0.0;
    bool cacheHit = false;
};

struct CPUProcessingResult {
    std::span<const std::uint32_t> aliveEdgeIds;
    std::span<const std::uint32_t> edgeIds;
    CPUProcessingStats stats;
};

class CPUReferencePipeline {
public:
    void SetGraph(const data::CHGraph& graph);
    [[nodiscard]] CPUProcessingResult Process(
        const data::CHGraph& graph, std::int32_t lodLevel,
        const geometry::RefinementParameters& refinementParameters);

private:
    reference::CPURangeFilter rangeFilter_;
    const data::CHGraph* graph_ = nullptr;
    std::vector<std::uint32_t> aliveEdges_;
    std::vector<std::uint32_t> refinedEdges_;
    std::vector<std::uint32_t> refinementStack_;
    CPUProcessingStats cachedStats_;
    std::int32_t cachedLodLevel_ = 0;
    geometry::RefinementParameters cachedRefinement_{};
    bool cacheValid_ = false;
};

} // namespace chmv::pipeline
