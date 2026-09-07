#pragma once

#include "benchmark/GPUTimer.h"
#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace chmv::pipeline {

struct GPUProcessingStats {
    std::optional<double> rangeFilterMs;
    std::optional<double> unfoldingMs;
};

struct GPUProcessingResult {
    std::uint32_t aliveEdgeIdBuffer = 0;
    std::uint32_t aliveDrawCommandBuffer = 0;
    std::uint32_t edgeIdBuffer = 0;
    std::uint32_t drawCommandBuffer = 0;
};

class GPUDrivenPipeline {
public:
    explicit GPUDrivenPipeline(const std::filesystem::path& shaderDirectory);

    void SetGraph(std::uint32_t edgeCount);
    [[nodiscard]] GPUProcessingResult Process(
        std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
        gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy);
    void ReadBackEdgeIds(std::uint32_t edgeIdBuffer, std::uint32_t drawCommandBuffer,
                         std::vector<std::uint32_t>& output) const;

    [[nodiscard]] GPUProcessingStats Stats() const;

private:
    gpu::ComputeProgram filterProgram_;
    gpu::GPUBuffer visibleEdgeBuffer_;
    gpu::GPUBuffer indirectBuffer_;
    benchmark::GPUTimer filterTimer_;
    benchmark::GPUTimer unfoldingTimer_;
    std::uint32_t capacity_ = 0;
};

} // namespace chmv::pipeline
