#pragma once

#include "gpu/GPUBuffer.h"
#include "gpu/GraphicsProgram.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>

namespace chmv::data {
class CHGraph;
}

namespace chmv::renderer {

class MapCamera2D;

struct RoadDrawData {
    std::uint32_t edgeIdBuffer = 0;
    std::uint32_t drawCommandBuffer = 0;
};

class RoadRenderer {
public:
    explicit RoadRenderer(const std::filesystem::path& shaderDirectory);
    ~RoadRenderer();

    RoadRenderer(const RoadRenderer&) = delete;
    RoadRenderer& operator=(const RoadRenderer&) = delete;

    void SetGraph(const data::CHGraph& graph);
    [[nodiscard]] RoadDrawData UploadEdgeIds(std::span<const std::uint32_t> edgeIds);
    void Draw(const MapCamera2D& camera, int framebufferWidth, int framebufferHeight,
              const RoadDrawData& drawData, float viewScale, float viewOffsetX,
              const std::array<float, 4>& color) const;
    void DrawViewport(const MapCamera2D& camera, int viewportX, int viewportY,
                      int viewportWidth, int viewportHeight, const RoadDrawData& drawData,
                      const std::array<float, 4>& color) const;

    [[nodiscard]] std::uint32_t EdgeBufferId() const { return edgeBuffer_.Id(); }
    [[nodiscard]] std::uint32_t EdgeCount() const { return edgeCount_; }
    [[nodiscard]] std::int32_t MinLevel() const { return minLevel_; }
    [[nodiscard]] std::int32_t MaxLevel() const { return maxLevel_; }
    [[nodiscard]] bool HasGraph() const { return edgeCount_ != 0; }

private:
    void EnsureUploadCapacity(std::size_t edgeCount);

    gpu::GraphicsProgram roadProgram_;
    gpu::GPUBuffer nodeBuffer_;
    gpu::GPUBuffer edgeBuffer_;
    gpu::GPUBuffer uploadedEdgeBuffer_;
    gpu::GPUBuffer uploadedIndirectBuffer_;
    std::uint32_t vertexArray_ = 0;
    std::uint32_t edgeCount_ = 0;
    std::size_t uploadCapacity_ = 0;
    std::int32_t minLevel_ = 0;
    std::int32_t maxLevel_ = 0;
};

} // namespace chmv::renderer
