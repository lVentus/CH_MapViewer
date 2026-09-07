#include "renderer/RoadRenderer.h"

#include "data/ch/CHGraph.h"
#include "renderer/MapCamera2D.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

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

constexpr std::uint32_t kFilterWorkGroupSize = 256;
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
    : filterProgram_(shaderDirectory / "range_filter.comp"),
      roadProgram_(shaderDirectory / "road.vert", shaderDirectory / "road.frag") {
    glGenVertexArrays(1, &vertexArray_);
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
    const auto visibleBytes = edges.size() * sizeof(std::uint32_t);
    CheckBufferSize(nodeBytes);
    CheckBufferSize(edgeBytes);
    CheckBufferSize(visibleBytes);

    nodeBuffer_.Allocate(nodeBytes, nodes.data(), GL_STATIC_DRAW);
    edgeBuffer_.Allocate(edgeBytes, edges.data(), GL_STATIC_DRAW);
    visibleEdgeBuffer_.Allocate(visibleBytes, nullptr, GL_DYNAMIC_DRAW);

    const DrawArraysIndirectCommand command{0, 1, 0, 0};
    indirectBuffer_.Allocate(sizeof(command), &command, GL_DYNAMIC_DRAW);
    edgeCount_ = static_cast<std::uint32_t>(edges.size());
}

void RoadRenderer::Draw(const MapCamera2D& camera, std::int32_t lodLevel, int framebufferWidth,
                        int framebufferHeight,
                        gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy) {
    if (edgeCount_ == 0 || framebufferWidth <= 0 || framebufferHeight <= 0) {
        return;
    }

    const std::uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indirectBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, &zero);

    filterProgram_.Bind();
    edgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    visibleEdgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    indirectBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    glUniform1ui(glGetUniformLocation(filterProgram_.Id(), "uEdgeCount"), edgeCount_);
    glUniform1i(glGetUniformLocation(filterProgram_.Id(), "uLevel"), lodLevel);
    filterProgram_.Dispatch((edgeCount_ + kFilterWorkGroupSize - 1) / kFilterWorkGroupSize);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    const auto unfolded = unfoldingStrategy.Execute({
        .edgeBuffer = edgeBuffer_.Id(),
        .edgeCount = edgeCount_,
        .inputEdgeBuffer = visibleEdgeBuffer_.Id(),
        .inputDrawCommandBuffer = indirectBuffer_.Id(),
        .outputCapacity = edgeCount_,
    });

    roadProgram_.Bind();
    nodeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    edgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, unfolded.edgeBuffer);
    glUniform2f(glGetUniformLocation(roadProgram_.Id(), "uCameraCenter"), camera.CenterX(),
                camera.CenterY());
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uZoom"), camera.ZoomFactor());
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uAspectScale"),
                static_cast<float>(framebufferHeight) / static_cast<float>(framebufferWidth));

    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, unfolded.drawCommandBuffer);
    glDrawArraysIndirect(GL_LINES, nullptr);
    glBindVertexArray(0);
}

} // namespace chmv::renderer
