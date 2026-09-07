#include "renderer/RoadRenderer.h"

#include "data/ch/CHGraph.h"
#include "renderer/MapCamera2D.h"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace chmv::renderer {
namespace {

struct GPUNode {
    float x;
    float y;
};

struct alignas(16) GPUEdge {
    std::uint32_t source;
    std::uint32_t target;
    std::uint32_t childA;
    std::uint32_t childB;
    std::int32_t birthLevel;
    std::int32_t deathLevel;
    std::uint32_t padding0;
    std::uint32_t padding1;
};

struct DrawArraysIndirectCommand {
    std::uint32_t count;
    std::uint32_t instanceCount;
    std::uint32_t first;
    std::uint32_t baseInstance;
};

static_assert(sizeof(GPUNode) == 8);
static_assert(sizeof(GPUEdge) == 32);
static_assert(sizeof(DrawArraysIndirectCommand) == 16);

constexpr double kPi = 3.14159265358979323846;

void CheckBufferSize(std::size_t sizeBytes) {
    GLint64 maxBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
    if (sizeBytes > static_cast<std::size_t>(maxBlockSize)) {
        throw std::runtime_error(
            "dataset exceeds the maximum SSBO size of this GPU; streaming is not implemented yet");
    }
}

} // namespace

RoadRenderer::RoadRenderer(const std::filesystem::path& shaderDirectory)
    : roadProgram_(shaderDirectory / "road.vert", shaderDirectory / "road.frag") {
    glGenVertexArrays(1, &vertexArray_);

    const DrawArraysIndirectCommand command{0, 1, 0, 0};
    uploadedIndirectBuffer_.Allocate(sizeof(command), &command, GL_DYNAMIC_DRAW);
}

RoadRenderer::~RoadRenderer() {
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
    }
}

void RoadRenderer::SetGraph(const data::CHGraph& graph) {
    if (graph.NodeCount() == 0 || graph.EdgeCount() == 0) {
        edgeCount_ = 0;
        return;
    }

    double minLatitude = std::numeric_limits<double>::max();
    double maxLatitude = std::numeric_limits<double>::lowest();
    double minLongitude = std::numeric_limits<double>::max();
    double maxLongitude = std::numeric_limits<double>::lowest();

    for (const auto& node : graph.Nodes()) {
        minLatitude = std::min(minLatitude, node.latitude);
        maxLatitude = std::max(maxLatitude, node.latitude);
        minLongitude = std::min(minLongitude, node.longitude);
        maxLongitude = std::max(maxLongitude, node.longitude);
    }

    const auto centerLatitude = (minLatitude + maxLatitude) * 0.5;
    const auto centerLongitude = (minLongitude + maxLongitude) * 0.5;
    const auto longitudeScale = std::cos(centerLatitude * kPi / 180.0);
    const auto halfWidth = std::max((maxLongitude - minLongitude) * 0.5 * longitudeScale, 1e-12);
    const auto halfHeight = std::max((maxLatitude - minLatitude) * 0.5, 1e-12);
    const auto normalization = 0.95 / std::max(halfWidth, halfHeight);

    std::vector<GPUNode> nodes;
    nodes.reserve(graph.NodeCount());
    for (const auto& node : graph.Nodes()) {
        nodes.push_back({
            static_cast<float>((node.longitude - centerLongitude) * longitudeScale * normalization),
            static_cast<float>((node.latitude - centerLatitude) * normalization),
        });
    }

    minLevel_ = std::numeric_limits<std::int32_t>::max();
    maxLevel_ = std::numeric_limits<std::int32_t>::lowest();

    std::vector<GPUEdge> edges;
    edges.reserve(graph.EdgeCount());
    for (std::size_t edgeId = 0; edgeId < graph.EdgeCount(); ++edgeId) {
        const auto& edge = graph.Edges()[edgeId];
        const auto& range = graph.Ranges()[edgeId];
        edges.push_back({
            edge.source,
            edge.target,
            edge.childA,
            edge.childB,
            range.birthLevel,
            range.deathLevel,
            0,
            0,
        });

        if (range.IsDrawable()) {
            minLevel_ = std::min(minLevel_, range.deathLevel);
            maxLevel_ = std::max(maxLevel_, range.birthLevel);
        }
    }

    if (minLevel_ == std::numeric_limits<std::int32_t>::max()) {
        minLevel_ = 0;
        maxLevel_ = 0;
    }

    const auto nodeBytes = nodes.size() * sizeof(GPUNode);
    const auto edgeBytes = edges.size() * sizeof(GPUEdge);
    CheckBufferSize(nodeBytes);
    CheckBufferSize(edgeBytes);

    nodeBuffer_.Allocate(nodeBytes, nodes.data(), GL_STATIC_DRAW);
    edgeBuffer_.Allocate(edgeBytes, edges.data(), GL_STATIC_DRAW);
    edgeCount_ = static_cast<std::uint32_t>(edges.size());
    uploadCapacity_ = 0;
}

RoadDrawData RoadRenderer::UploadEdgeIds(std::span<const std::uint32_t> edgeIds) {
    EnsureUploadCapacity(edgeIds.size());
    if (!edgeIds.empty()) {
        uploadedEdgeBuffer_.Update(0, edgeIds.size_bytes(), edgeIds.data());
    }

    if (edgeIds.size() > std::numeric_limits<std::uint32_t>::max() / 2u) {
        throw std::runtime_error("too many CPU edges for indirect line rendering");
    }

    const DrawArraysIndirectCommand command{
        static_cast<std::uint32_t>(edgeIds.size() * 2),
        1,
        0,
        0,
    };
    uploadedIndirectBuffer_.Update(0, sizeof(command), &command);

    return {
        uploadedEdgeBuffer_.Id(),
        uploadedIndirectBuffer_.Id(),
    };
}

void RoadRenderer::Draw(const MapCamera2D& camera, int framebufferWidth, int framebufferHeight,
                        const RoadDrawData& drawData, float viewScale, float viewOffsetX,
                        const std::array<float, 4>& color) const {
    if (edgeCount_ == 0 || framebufferWidth <= 0 || framebufferHeight <= 0 ||
        drawData.edgeIdBuffer == 0 || drawData.drawCommandBuffer == 0) {
        return;
    }

    roadProgram_.Bind();
    nodeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    edgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, drawData.edgeIdBuffer);
    glUniform2f(glGetUniformLocation(roadProgram_.Id(), "uCameraCenter"), camera.CenterX(),
                camera.CenterY());
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uZoom"), camera.ViewScale());
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uAspectScale"),
                static_cast<float>(framebufferHeight) / static_cast<float>(framebufferWidth));
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uViewScale"), viewScale);
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uViewOffsetX"), viewOffsetX);
    glUniform4f(glGetUniformLocation(roadProgram_.Id(), "uColor"), color[0], color[1], color[2],
                color[3]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, drawData.drawCommandBuffer);
    glDrawArraysIndirect(GL_LINES, nullptr);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
}

void RoadRenderer::DrawViewport(const MapCamera2D& camera, int viewportX, int viewportY,
                                int viewportWidth, int viewportHeight,
                                const RoadDrawData& drawData,
                                const std::array<float, 4>& color) const {
    if (viewportWidth <= 0 || viewportHeight <= 0) {
        return;
    }

    GLint previousViewport[4]{};
    glGetIntegerv(GL_VIEWPORT, previousViewport);
    glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
    Draw(camera, viewportWidth, viewportHeight, drawData, 1.0f, 0.0f, color);
    glViewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
}

void RoadRenderer::EnsureUploadCapacity(std::size_t edgeCount) {
    if (uploadCapacity_ >= edgeCount && uploadCapacity_ != 0) {
        return;
    }

    const auto capacity = std::max<std::size_t>(edgeCount, 1);
    const auto sizeBytes = capacity * sizeof(std::uint32_t);
    CheckBufferSize(sizeBytes);
    uploadedEdgeBuffer_.Allocate(sizeBytes, nullptr, GL_DYNAMIC_DRAW);
    uploadCapacity_ = capacity;
}

} // namespace chmv::renderer
