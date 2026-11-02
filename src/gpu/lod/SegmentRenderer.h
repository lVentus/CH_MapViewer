#pragma once

#include "benchmark/GPUTimer.h"
#include "cpu/CpuPasses.h"
#include "gpu/GPUBuffer.h"
#include "gpu/GraphicsProgram.h"
#include "renderer/RoadStyle.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

namespace chmv::renderer {
class MapCamera2D;
}

namespace chmv::gpu::lod {

// Draws segments the CPU computed, with the same road styles and stroke as the GPU pipeline:
// in CPU mode the GPU only rasterizes.
class SegmentRenderer {
public:
    explicit SegmentRenderer(const std::filesystem::path& shaderDirectory);
    ~SegmentRenderer();

    SegmentRenderer(const SegmentRenderer&) = delete;
    SegmentRenderer& operator=(const SegmentRenderer&) = delete;

    // Replaces the segments drawn. Returns why they cannot be drawn (too large for the GPU
    // memory left), or nothing; then nothing is drawn until the next upload.
    [[nodiscard]] std::string Upload(std::span<const cpu::CpuSegment> segments);
    void Draw(const renderer::MapCamera2D& camera, int width, int height,
              const renderer::RoadStyleConfig& styles);
    [[nodiscard]] std::optional<double> DrawMilliseconds() const {
        return drawTimer_.LastMilliseconds();
    }

private:
    GraphicsProgram program_;
    GPUBuffer segments_;
    std::uint32_t count_ = 0;
    // Segments one draw binds at most: the largest storage block, in multiples of 64 segments
    // (1280 bytes, a multiple of every binding offset alignment).
    std::uint32_t segmentsPerDraw_ = 0;
    std::uint32_t vertexArray_ = 0;
    benchmark::GPUTimer drawTimer_;
};

} // namespace chmv::gpu::lod
