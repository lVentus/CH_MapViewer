#include "gpu/lod/SegmentRenderer.h"

#include "renderer/MapCamera2D.h"
#include "streaming/lod/LodPageFormat.h"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace chmv::gpu::lod {
namespace {

[[nodiscard]] GLint FixedCameraCoordinate(float center) {
    constexpr double kLimit = 1073741823.0;
    const auto fixed =
        std::round(static_cast<double>(center) * chmv::streaming::lod::kFixedCoordinateScale);
    return static_cast<GLint>(std::clamp(fixed, -kLimit, kLimit));
}

} // namespace

SegmentRenderer::SegmentRenderer(const std::filesystem::path& shaderDirectory)
    : program_(shaderDirectory / "lod_segments_road.vert", shaderDirectory / "lod_page_road.frag") {
    glGenVertexArrays(1, &vertexArray_);
    GLint64 maxBlockBytes = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockBytes);
    // 16 MiB is the least GL 4.3 guarantees.
    const auto segments =
        std::max<GLint64>(maxBlockBytes, 16 << 20) / static_cast<GLint64>(sizeof(cpu::CpuSegment));
    segmentsPerDraw_ = static_cast<std::uint32_t>(
        std::min<GLint64>(segments / 64 * 64, std::numeric_limits<GLsizei>::max() / 6));
}

SegmentRenderer::~SegmentRenderer() {
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
    }
}

std::string SegmentRenderer::Upload(std::span<const cpu::CpuSegment> segments) {
    count_ = 0;
    if (segments.empty()) {
        return {};
    }
    if (segments.size() > std::numeric_limits<std::uint32_t>::max()) {
        return "CPU result too large to draw: " + std::to_string(segments.size()) + " segments";
    }
    if (segments_.SizeBytes() < segments.size_bytes()) {
        // Room to grow, so a view that gains a few segments does not reallocate; the exact size
        // if that room does not fit.
        try {
            segments_.Allocate(segments.size_bytes() + segments.size_bytes() / 2u, nullptr,
                               GL_DYNAMIC_DRAW);
        } catch (const std::runtime_error&) {
            try {
                segments_.Allocate(segments.size_bytes(), nullptr, GL_DYNAMIC_DRAW);
            } catch (const std::runtime_error& error) {
                return std::string("CPU result too large to draw: ") + error.what();
            }
        }
    }
    segments_.Update(0, segments.size_bytes(), segments.data());
    count_ = static_cast<std::uint32_t>(segments.size());
    return {};
}

void SegmentRenderer::Draw(const renderer::MapCamera2D& camera, int width, int height,
                           const renderer::RoadStyleConfig& styles) {
    if (width <= 0 || height <= 0 || count_ == 0u) {
        return;
    }
    const auto program = program_.Id();
    program_.Bind();
    glUniform2i(glGetUniformLocation(program, "uCameraFixed"),
                FixedCameraCoordinate(camera.CenterX()), FixedCameraCoordinate(camera.CenterY()));
    glUniform1f(glGetUniformLocation(program, "uFixedToClip"),
                static_cast<float>(static_cast<double>(camera.ViewScale()) /
                                   chmv::streaming::lod::kFixedCoordinateScale));
    glUniform1f(glGetUniformLocation(program, "uAspectScale"),
                static_cast<float>(height) / static_cast<float>(width));
    glUniform2f(glGetUniformLocation(program, "uViewportSize"), static_cast<float>(width),
                static_cast<float>(height));
    const auto colors = styles.TypeColors();
    const auto widths = styles.TypeWidths();
    glUniform4fv(glGetUniformLocation(program, "uRoadColors[0]"),
                 static_cast<GLsizei>(colors.size()), colors.front().data());
    glUniform1fv(glGetUniformLocation(program, "uRoadWidths[0]"),
                 static_cast<GLsizei>(widths.size()), widths.data());
    const auto depths = styles.TypeDepths();
    glUniform1fv(glGetUniformLocation(program, "uRoadDepths[0]"),
                 static_cast<GLsizei>(depths.size()), depths.data());

    // As LodPageRenderer::Draw: classes listed first on top.
    drawTimer_.Begin();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vertexArray_);
    for (std::uint32_t first = 0; first < count_; first += segmentsPerDraw_) {
        const auto count = std::min(segmentsPerDraw_, count_ - first);
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, segments_.Id(),
                          static_cast<GLintptr>(std::uint64_t{first} * sizeof(cpu::CpuSegment)),
                          static_cast<GLsizeiptr>(std::uint64_t{count} * sizeof(cpu::CpuSegment)));
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(count * 6u));
    }
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    drawTimer_.End();
}

} // namespace chmv::gpu::lod
