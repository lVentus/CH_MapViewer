#pragma once

#include "benchmark/GPUTimer.h"
#include "gpu/GPUBuffer.h"
#include "gpu/filtering/IRangeFilterStrategy.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"
#include "geometry/GeometryRefinement.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace chmv::pipeline {

struct GPUProcessingStats {
    std::optional<std::size_t> rangeCandidateEdgeCount;
    std::optional<double> rangeFilterMs;
    std::optional<double> geometryRefinementMs;
};

struct GPUProcessingResult {
    std::uint32_t aliveEdgeIdBuffer = 0;
    std::uint32_t aliveDrawCommandBuffer = 0;
    std::uint32_t edgeIdBuffer = 0;
    std::uint32_t drawCommandBuffer = 0;
};

class GPUDrivenPipeline {
public:
    void SetGraph(std::uint32_t edgeCount, std::uint32_t drawableEdgeCount);
    [[nodiscard]] GPUProcessingResult Process(
        std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
        gpu::filtering::IRangeFilterStrategy& rangeFilterStrategy,
        gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy,
        const geometry::RefinementParameters& refinementParameters);
    void ReadBackEdgeIds(std::uint32_t edgeIdBuffer, std::uint32_t drawCommandBuffer,
                         std::vector<std::uint32_t>& output) const;

    [[nodiscard]] GPUProcessingStats Stats() const;

private:
    gpu::GPUBuffer visibleEdgeBuffer_;
    gpu::GPUBuffer indirectBuffer_;
    benchmark::GPUTimer filterTimer_;
    benchmark::GPUTimer geometryRefinementTimer_;
    std::uint32_t edgeCount_ = 0;
    std::uint32_t visibleCapacity_ = 0;
    std::optional<std::size_t> rangeCandidateEdgeCount_;
    bool geometryRefinementActive_ = false;
};

} // namespace chmv::pipeline
