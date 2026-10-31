#include "gpu/lod/LodPageRenderer.h"

#include "gpu/lod/LodPagePool.h"
#include "renderer/MapCamera2D.h"
#include "streaming/lod/LodPageFormat.h"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>

namespace chmv::gpu::lod {
namespace {

// The command buffer holds the indirect draw arguments followed by the segment count.
constexpr std::size_t kCommandWords = 5;
constexpr std::uint32_t kMaxGroupsPerDispatch = 65535;

[[nodiscard]] GLsync AsSync(void* fence) {
    return static_cast<GLsync>(fence);
}

[[nodiscard]] GLint FixedCameraCoordinate(float center) {
    constexpr double kLimit = 1073741823.0;
    const auto fixed =
        std::round(static_cast<double>(center) * chmv::streaming::lod::kFixedCoordinateScale);
    return static_cast<GLint>(std::clamp(fixed, -kLimit, kLimit));
}

} // namespace

LodPageRenderer::LodPageRenderer(const std::filesystem::path& shaderDirectory)
    : filterProgram_(shaderDirectory / "lod_page_filter.comp"),
      roadProgram_(shaderDirectory / "lod_page_road.vert",
                   shaderDirectory / "streaming_road.frag") {
    glGenVertexArrays(1, &vertexArray_);
    const std::array<std::uint32_t, kCommandWords> empty{};
    commandBuffer_.Allocate(sizeof(empty), empty.data());
    for (auto& readback : countReadbacks_) {
        readback.buffer.Allocate(sizeof(empty), empty.data(), GL_STREAM_READ);
    }
    EnsureCapacity(256);
}

LodPageRenderer::~LodPageRenderer() {
    for (auto& readback : countReadbacks_) {
        if (readback.fence != nullptr) {
            glDeleteSync(AsSync(readback.fence));
        }
    }
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
    }
}

void LodPageRenderer::EnsureCapacity(std::size_t slotCount) {
    if (slotCount <= slotCapacity_) {
        return;
    }
    slotCapacity_ = std::max(slotCount, slotCapacity_ * 2u);
    slotBuffer_.Allocate(slotCapacity_ * 4u);
    segmentBuffer_.Allocate(slotCapacity_ * chmv::streaming::lod::kPageRecordCapacity * 4u);
}

void LodPageRenderer::Filter(const LodPagePool& pool, std::span<const std::uint32_t> slots,
                             std::uint32_t level, std::uint32_t threshold) {
    const std::array<std::uint32_t, kCommandWords> reset{0u, 1u, 0u, 0u, 0u};
    commandBuffer_.Update(0, sizeof(reset), reset.data());
    if (slots.empty()) {
        drawnSegments_ = 0;
        return;
    }

    EnsureCapacity(slots.size());
    slotBuffer_.Update(0, slots.size_bytes(), slots.data());

    filterTimer_.Begin();
    filterProgram_.Bind();
    pool.Buffer().BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    pool.Buffer().BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    slotBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    segmentBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    commandBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 4);
    const auto program = filterProgram_.Id();
    glUniform1ui(glGetUniformLocation(program, "uSlotWords"), pool.SlotWords());
    glUniform1ui(glGetUniformLocation(program, "uLevel"), level);
    glUniform1ui(glGetUniformLocation(program, "uThreshold"), threshold);
    const auto offsetLocation = glGetUniformLocation(program, "uSlotOffset");
    for (std::size_t offset = 0; offset < slots.size(); offset += kMaxGroupsPerDispatch) {
        const auto groups = std::min<std::size_t>(kMaxGroupsPerDispatch, slots.size() - offset);
        glUniform1ui(offsetLocation, static_cast<GLuint>(offset));
        filterProgram_.Dispatch(static_cast<unsigned int>(groups));
    }
    filterTimer_.End();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);

    auto& readback = countReadbacks_[nextCountReadback_];
    nextCountReadback_ = (nextCountReadback_ + 1u) % countReadbacks_.size();
    if (readback.fence != nullptr) {
        glDeleteSync(AsSync(readback.fence));
    }
    glBindBuffer(GL_COPY_READ_BUFFER, commandBuffer_.Id());
    glBindBuffer(GL_COPY_WRITE_BUFFER, readback.buffer.Id());
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                        kCommandWords * sizeof(std::uint32_t));
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    readback.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void LodPageRenderer::Poll() {
    // Oldest first, so the newest finished count wins.
    for (std::size_t i = 0; i < countReadbacks_.size(); ++i) {
        auto& readback = countReadbacks_[(nextCountReadback_ + i) % countReadbacks_.size()];
        if (readback.fence == nullptr ||
            glClientWaitSync(AsSync(readback.fence), 0, 0) == GL_TIMEOUT_EXPIRED) {
            continue;
        }
        glDeleteSync(AsSync(readback.fence));
        readback.fence = nullptr;
        std::array<std::uint32_t, kCommandWords> command{};
        glBindBuffer(GL_COPY_READ_BUFFER, readback.buffer.Id());
        glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(command), command.data());
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        drawnSegments_ = command[4];
    }
}

std::vector<std::uint32_t> LodPageRenderer::ReadSegments() const {
    std::array<std::uint32_t, kCommandWords> command{};
    glBindBuffer(GL_COPY_READ_BUFFER, commandBuffer_.Id());
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(command), command.data());
    std::vector<std::uint32_t> segments(command[4]);
    glBindBuffer(GL_COPY_READ_BUFFER, segmentBuffer_.Id());
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0,
                       static_cast<GLsizeiptr>(segments.size() * sizeof(std::uint32_t)),
                       segments.data());
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    return segments;
}

void LodPageRenderer::Draw(const LodPagePool& pool, const renderer::MapCamera2D& camera, int width,
                           int height, const std::array<float, 4>& color,
                           const renderer::RoadStyleConfig& styles) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const auto program = roadProgram_.Id();
    roadProgram_.Bind();
    pool.Buffer().BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    segmentBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    glUniform1ui(glGetUniformLocation(program, "uSlotWords"), pool.SlotWords());
    glUniform2i(glGetUniformLocation(program, "uCameraFixed"),
                FixedCameraCoordinate(camera.CenterX()), FixedCameraCoordinate(camera.CenterY()));
    glUniform1f(glGetUniformLocation(program, "uFixedToClip"),
                static_cast<float>(static_cast<double>(camera.ViewScale()) /
                                   chmv::streaming::lod::kFixedCoordinateScale));
    glUniform1f(glGetUniformLocation(program, "uAspectScale"),
                static_cast<float>(height) / static_cast<float>(width));
    glUniform2f(glGetUniformLocation(program, "uViewportSize"), static_cast<float>(width),
                static_cast<float>(height));
    glUniform4f(glGetUniformLocation(program, "uColor"), color[0], color[1], color[2], color[3]);

    const auto colorsLocation = glGetUniformLocation(program, "uRoadColors[0]");
    const auto widthsLocation = glGetUniformLocation(program, "uRoadWidths[0]");
    const bool styleValid = colorsLocation >= 0 && widthsLocation >= 0;
    glUniform1ui(glGetUniformLocation(program, "uRoadStyleDisabled"), styles.enabled ? 0u : 1u);
    glUniform1ui(glGetUniformLocation(program, "uRoadStyleConfigValid"), styleValid ? 1u : 0u);
    glUniform1f(glGetUniformLocation(program, "uRoadWidthScale"), styles.globalWidthScale);
    if (styleValid) {
        glUniform4fv(colorsLocation, static_cast<GLsizei>(styles.colors.size()),
                     styles.colors.front().data());
        glUniform1fv(widthsLocation, static_cast<GLsizei>(styles.widthsPixels.size()),
                     styles.widthsPixels.data());
    }

    drawTimer_.Begin();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, commandBuffer_.Id());
    glDrawArraysIndirect(GL_TRIANGLES, nullptr);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    drawTimer_.End();
}

} // namespace chmv::gpu::lod
