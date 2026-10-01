#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "streaming/runtime/GraphPageStreamer.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace chmv::gpu::streaming {

struct RefinementRequestStats {
    std::uint32_t rawRequestCount = 0;
    std::uint32_t uniqueRequestCount = 0;
    std::uint32_t droppedRequestCount = 0;
};

class RefinementRequestCollector {
public:
    explicit RefinementRequestCollector(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::vector<chmv::streaming::runtime::RefinementEdgeRequest> Collect(
        std::uint32_t streamingLinkBuffer,
        std::uint32_t leafEdgeBuffer,
        std::uint32_t leafDrawCommandBuffer,
        std::uint32_t edgeCapacity);

    [[nodiscard]] RefinementRequestStats Stats() const { return stats_; }

private:
    void EnsureCapacity(std::uint32_t edgeCapacity);

    ComputeProgram collectProgram_;
    GPUBuffer requestBuffer_;
    GPUBuffer counterBuffer_;
    std::uint32_t edgeCapacity_ = 0;
    std::uint32_t requestCapacity_ = 0;
    RefinementRequestStats stats_{};
};

} // namespace chmv::gpu::streaming
