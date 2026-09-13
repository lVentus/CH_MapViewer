#pragma once

#include "gpu/GPUBuffer.h"
#include "gpu/filtering/IRangeFilterStrategy.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"
#include "geometry/GeometryRefinement.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace chmv::pipeline {

struct GPUProcessingStats {
    std::optional<std::size_t> rangeCandidateEdgeCount;
};

struct GPUProcessingResult {
    std::uint32_t aliveEdgeIdBuffer = 0;
    std::uint32_t aliveDrawCommandBuffer = 0;
    std::uint32_t edgeIdBuffer = 0;
    std::uint32_t drawCommandBuffer = 0;
};

struct GPURangeBenchmarkResult {
    std::string filterStrategy;
    std::int32_t lodLevel = 0;
    std::size_t warmupIterations = 0;
    std::size_t measuredIterations = 0;
    std::size_t batchSize = 0;
    std::optional<std::size_t> candidateEdgeCount;
    std::size_t aliveEdgeCount = 0;
    double measuredTotalMs = 0.0;
    double meanMs = 0.0;
    double medianMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    double p95Ms = 0.0;
    double p99Ms = 0.0;
    double stdDevMs = 0.0;
};

class GPUDrivenPipeline {
public:
    void SetGraph(std::uint32_t edgeCount, std::uint32_t drawableEdgeCount);
    [[nodiscard]] GPUProcessingResult Process(
        std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
        gpu::filtering::IRangeFilterStrategy& rangeFilterStrategy,
        gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy,
        const geometry::RefinementParameters& refinementParameters);
    [[nodiscard]] GPURangeBenchmarkResult RunRangeFilterBenchmark(
        std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
        gpu::filtering::IRangeFilterStrategy& rangeFilterStrategy);
    void ReadBackEdgeIds(std::uint32_t edgeIdBuffer, std::uint32_t drawCommandBuffer,
                         std::vector<std::uint32_t>& output) const;

    [[nodiscard]] GPUProcessingStats Stats() const;

private:
    gpu::GPUBuffer visibleEdgeBuffer_;
    gpu::GPUBuffer indirectBuffer_;
    std::uint32_t edgeCount_ = 0;
    std::uint32_t visibleCapacity_ = 0;
    std::optional<std::size_t> rangeCandidateEdgeCount_;
};

} // namespace chmv::pipeline
