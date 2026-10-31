#pragma once

#include "benchmark/GPUTimer.h"
#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/GraphicsProgram.h"
#include "renderer/RoadStyle.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace chmv::renderer {
class MapCamera2D;
}

namespace chmv::gpu::lod {

class LodPagePool;

// GPU half of the LOD page runtime: a compute pass that turns resident pages into a compact
// segment list, and a vertex-pulled draw of that list.
class LodPageRenderer {
public:
    explicit LodPageRenderer(const std::filesystem::path& shaderDirectory);
    ~LodPageRenderer();

    LodPageRenderer(const LodPageRenderer&) = delete;
    LodPageRenderer& operator=(const LodPageRenderer&) = delete;

    // Only needed when the level, the threshold or the slot list changed.
    void Filter(const LodPagePool& pool, std::span<const std::uint32_t> slots, std::uint32_t level,
                std::uint32_t threshold);
    void Draw(const LodPagePool& pool, const renderer::MapCamera2D& camera, int width, int height,
              const std::array<float, 4>& color, const renderer::RoadStyleConfig& styles);
    // Collects finished GPU queries without waiting; results are a few frames old.
    void Poll();

    [[nodiscard]] std::uint32_t DrawnSegments() const { return drawnSegments_; }
    [[nodiscard]] std::optional<double> FilterMilliseconds() const {
        return filterTimer_.LastMilliseconds();
    }
    [[nodiscard]] std::optional<double> DrawMilliseconds() const {
        return drawTimer_.LastMilliseconds();
    }
    // Synchronous copy of the current segment list (validation only).
    [[nodiscard]] std::vector<std::uint32_t> ReadSegments() const;

private:
    struct CountReadback {
        GPUBuffer buffer;
        void* fence = nullptr;
    };

    void EnsureCapacity(std::size_t slotCount);

    ComputeProgram filterProgram_;
    GraphicsProgram roadProgram_;
    GPUBuffer slotBuffer_;
    GPUBuffer segmentBuffer_;
    GPUBuffer commandBuffer_;
    std::size_t slotCapacity_ = 0;
    std::uint32_t vertexArray_ = 0;
    benchmark::GPUTimer filterTimer_;
    benchmark::GPUTimer drawTimer_;
    std::array<CountReadback, 3> countReadbacks_;
    std::size_t nextCountReadback_ = 0;
    std::uint32_t drawnSegments_ = 0;
};

} // namespace chmv::gpu::lod
