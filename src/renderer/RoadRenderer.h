#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/GraphicsProgram.h"

#include <cstdint>
#include <filesystem>

namespace chmv::data {
class CHGraph;
}

namespace chmv::gpu::unfolding {
class IUnfoldingStrategy;
}

namespace chmv::renderer {

class MapCamera2D;

class RoadRenderer {
public:
    explicit RoadRenderer(const std::filesystem::path& shaderDirectory);
    ~RoadRenderer();

    RoadRenderer(const RoadRenderer&) = delete;
    RoadRenderer& operator=(const RoadRenderer&) = delete;

    void SetGraph(const data::CHGraph& graph);
    void Draw(const MapCamera2D& camera, std::int32_t lodLevel, int framebufferWidth,
              int framebufferHeight, gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy);

    [[nodiscard]] std::int32_t MinLevel() const { return minLevel_; }
    [[nodiscard]] std::int32_t MaxLevel() const { return maxLevel_; }
    [[nodiscard]] bool HasGraph() const { return edgeCount_ != 0; }

private:
    gpu::ComputeProgram filterProgram_;
    gpu::GraphicsProgram roadProgram_;
    gpu::GPUBuffer nodeBuffer_;
    gpu::GPUBuffer edgeBuffer_;
    gpu::GPUBuffer visibleEdgeBuffer_;
    gpu::GPUBuffer indirectBuffer_;
    std::uint32_t vertexArray_ = 0;
    std::uint32_t edgeCount_ = 0;
    std::int32_t minLevel_ = 0;
    std::int32_t maxLevel_ = 0;
};

} // namespace chmv::renderer
