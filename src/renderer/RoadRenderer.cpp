#include "renderer/RoadRenderer.h"

#include "data/ch/CHGraph.h"
#include "data/ch/CHTypes.h"
#include "renderer/MapCamera2D.h"
#include "streaming/runtime/GraphPageStreamer.h"

#include <glad/gl.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
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
    float geometryError;
    std::uint32_t padding;
};

struct alignas(16) GPUStreamingEdgeLink {
    std::uint32_t globalEdgeId;
    std::uint32_t globalChildA;
    std::uint32_t globalChildB;
    std::uint32_t localChildA;
    std::uint32_t localChildB;
    std::uint32_t padding0;
    std::uint32_t padding1;
    std::uint32_t padding2;
};

struct DrawArraysIndirectCommand {
    std::uint32_t count;
    std::uint32_t instanceCount;
    std::uint32_t first;
    std::uint32_t baseInstance;
};

static_assert(sizeof(GPUNode) == 8);
static_assert(sizeof(GPUEdge) == 32);
static_assert(sizeof(GPUStreamingEdgeLink) == 32);
static_assert(sizeof(DrawArraysIndirectCommand) == 16);


void CheckBufferSize(std::size_t sizeBytes) {
    GLint64 maxBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
    if (sizeBytes > static_cast<std::size_t>(maxBlockSize)) {
        throw std::runtime_error(
            "dataset exceeds the maximum SSBO size of this GPU; streaming is not implemented yet");
    }
}

bool PackedBoundsIntersects(std::uint64_t packed,
                            const streaming_runtime::StreamingSpatialWindow& window) {
    const auto minX = static_cast<std::uint32_t>(packed & 0xffffu);
    const auto minY = static_cast<std::uint32_t>((packed >> 16u) & 0xffffu);
    const auto maxX = static_cast<std::uint32_t>((packed >> 32u) & 0xffffu);
    const auto maxY = static_cast<std::uint32_t>((packed >> 48u) & 0xffffu);
    return maxX >= window.minX && minX <= window.maxX &&
           maxY >= window.minY && minY <= window.maxY;
}

} // namespace

RoadRenderer::RoadRenderer(const std::filesystem::path& shaderDirectory)
    : roadProgram_(shaderDirectory / "road.vert", shaderDirectory / "road.geom",
                   shaderDirectory / "road.frag") {
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
    streamingRanges_.clear();
    if (graph.NodeCount() == 0 || graph.EdgeCount() == 0) {
        edgeCount_ = 0;
        return;
    }

    const auto& projection = graph.Projection();

    std::vector<GPUNode> nodes;
    nodes.reserve(graph.NodeCount());
    for (const auto& node : graph.Nodes()) {
        const auto position = projection.Project(node.latitude, node.longitude);
        nodes.push_back({
            static_cast<float>(position.x),
            static_cast<float>(position.y),
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
            edge.geometryError,
            data::RoadStyleIndexFromType(edge.type),
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

void RoadRenderer::SetStreamingWorkingSet(
    std::span<const streaming_runtime::ResidentGraphPage* const> pages,
    std::span<const streaming_runtime::ResidentRefinementBlock* const> refinementBlocks,
    std::span<const std::uint64_t> edgeSpatialBounds,
    const streaming_runtime::StreamingSpatialWindow& refinementWindow) {
    std::size_t approximateNodeCount = 0;
    std::size_t approximateEdgeCount = 0;
    for (const auto* page : pages) {
        if (page) {
            approximateNodeCount += page->nodes.size();
            approximateEdgeCount += page->edges.size();
        }
    }
    for (const auto* block : refinementBlocks) {
        if (block) {
            approximateNodeCount += block->nodes.size();
            approximateEdgeCount += block->edges.size();
        }
    }

    if (approximateNodeCount == 0 || approximateEdgeCount == 0) {
        ClearGraph();
        return;
    }

    struct PendingEdge {
        std::uint32_t globalEdgeId = 0;
        std::uint32_t source = 0;
        std::uint32_t target = 0;
        std::uint32_t childA = data::InvalidEdgeId;
        std::uint32_t childB = data::InvalidEdgeId;
        std::uint32_t roadStyleType = 0;
        std::int32_t birthLevel = -1;
        std::int32_t deathLevel = -1;
    };

    std::vector<GPUNode> nodes;
    std::vector<PendingEdge> pendingEdges;
    nodes.reserve(approximateNodeCount);
    pendingEdges.reserve(approximateEdgeCount);

    std::unordered_map<std::uint32_t, std::uint32_t> nodeRemap;
    std::unordered_map<std::uint32_t, std::uint32_t> edgeRemap;
    nodeRemap.reserve(approximateNodeCount);
    edgeRemap.reserve(approximateEdgeCount);

    minLevel_ = std::numeric_limits<std::int32_t>::max();
    maxLevel_ = std::numeric_limits<std::int32_t>::lowest();

    const auto ingest = [&](const auto& sourceNodes, const auto& sourceEdges,
                            bool drawableRoots) {
        std::vector<std::uint32_t> localNodeRemap(sourceNodes.size());
        for (std::size_t i = 0; i < sourceNodes.size(); ++i) {
            const auto& node = sourceNodes[i];
            const auto [it, inserted] = nodeRemap.emplace(
                node.globalNodeId, static_cast<std::uint32_t>(nodes.size()));
            if (inserted) {
                if (nodes.size() >= std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("streaming working set exceeds 32-bit node indexing");
                }
                nodes.push_back({node.x, node.y});
            }
            localNodeRemap[i] = it->second;
        }

        for (const auto& edge : sourceEdges) {
            if (drawableRoots) {
                if (edge.globalEdgeId >= edgeSpatialBounds.size()) {
                    throw std::runtime_error("streamed root edge is missing spatial bounds");
                }
                if (!PackedBoundsIntersects(edgeSpatialBounds[edge.globalEdgeId],
                                            refinementWindow)) {
                    continue;
                }
            }
            if (edge.sourceLocal >= localNodeRemap.size() ||
                edge.targetLocal >= localNodeRemap.size()) {
                throw std::runtime_error(
                    "streamed working-set edge contains an invalid local node index");
            }
            if (edgeRemap.contains(edge.globalEdgeId)) {
                continue;
            }
            if (pendingEdges.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("streaming working set exceeds 32-bit edge indexing");
            }
            const auto localEdge = static_cast<std::uint32_t>(pendingEdges.size());
            edgeRemap.emplace(edge.globalEdgeId, localEdge);
            pendingEdges.push_back({
                edge.globalEdgeId,
                localNodeRemap[edge.sourceLocal],
                localNodeRemap[edge.targetLocal],
                edge.childA,
                edge.childB,
                edge.roadStyleType,
                drawableRoots ? edge.birthLevel : -1,
                drawableRoots ? edge.deathLevel : -1,
            });

            if (drawableRoots && edge.birthLevel >= edge.deathLevel && edge.birthLevel >= 0 &&
                edge.deathLevel >= 0) {
                minLevel_ = std::min(minLevel_, edge.deathLevel);
                maxLevel_ = std::max(maxLevel_, edge.birthLevel);
            }
        }
    };

    // Drawable root pages are ingested first so their lifetime range wins if a refinement source
    // block contains the same physical edge.
    for (const auto* page : pages) {
        if (page) {
            ingest(page->nodes, page->edges, true);
        }
    }
    for (const auto* block : refinementBlocks) {
        if (block) {
            ingest(block->nodes, block->edges, false);
        }
    }

    if (pendingEdges.empty()) {
        ClearGraph();
        return;
    }
    if (minLevel_ == std::numeric_limits<std::int32_t>::max()) {
        minLevel_ = 0;
        maxLevel_ = 0;
    }

    std::vector<GPUEdge> edges;
    std::vector<GPUStreamingEdgeLink> links;
    edges.reserve(pendingEdges.size());
    links.reserve(pendingEdges.size());
    streamingRanges_.clear();
    streamingRanges_.reserve(pendingEdges.size());
    for (const auto& edge : pendingEdges) {
        auto localChildA = data::InvalidEdgeId;
        auto localChildB = data::InvalidEdgeId;
        if (edge.childA != data::InvalidEdgeId) {
            const auto found = edgeRemap.find(edge.childA);
            if (found != edgeRemap.end()) {
                localChildA = found->second;
            }
        }
        if (edge.childB != data::InvalidEdgeId) {
            const auto found = edgeRemap.find(edge.childB);
            if (found != edgeRemap.end()) {
                localChildB = found->second;
            }
        }

        // The existing DFS shader expects an all-or-nothing local child pair. Missing children
        // therefore make this shortcut a temporary leaf; the request pass below identifies the
        // missing global children and streams their physical source blocks for a later frame.
        const bool completeShortcut = localChildA != data::InvalidEdgeId &&
                                      localChildB != data::InvalidEdgeId;
        streamingRanges_.push_back({edge.birthLevel, edge.deathLevel});
        edges.push_back({
            edge.source,
            edge.target,
            completeShortcut ? localChildA : data::InvalidEdgeId,
            completeShortcut ? localChildB : data::InvalidEdgeId,
            edge.birthLevel,
            edge.deathLevel,
            0.0f,
            edge.roadStyleType,
        });
        links.push_back({
            edge.globalEdgeId,
            edge.childA,
            edge.childB,
            localChildA,
            localChildB,
            0, 0, 0,
        });
    }

    const auto nodeBytes = nodes.size() * sizeof(GPUNode);
    const auto edgeBytes = edges.size() * sizeof(GPUEdge);
    const auto linkBytes = links.size() * sizeof(GPUStreamingEdgeLink);
    CheckBufferSize(nodeBytes);
    CheckBufferSize(edgeBytes);
    CheckBufferSize(linkBytes);
    nodeBuffer_.Allocate(nodeBytes, nodes.data(), GL_DYNAMIC_DRAW);
    edgeBuffer_.Allocate(edgeBytes, edges.data(), GL_DYNAMIC_DRAW);
    streamingLinkBuffer_.Allocate(linkBytes, links.data(), GL_DYNAMIC_DRAW);
    edgeCount_ = static_cast<std::uint32_t>(edges.size());
    uploadCapacity_ = 0;
}

void RoadRenderer::ClearGraph() {
    streamingRanges_.clear();
    edgeCount_ = 0;
    minLevel_ = 0;
    maxLevel_ = 0;
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
                        const std::array<float, 4>& color, const RoadStyleConfig& styles) const {
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
    glUniform2f(glGetUniformLocation(roadProgram_.Id(), "uViewportSize"),
                static_cast<float>(framebufferWidth), static_cast<float>(framebufferHeight));
    const auto roadStyleDisabledLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadStyleDisabled");
    const auto roadStyleConfigValidLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadStyleConfigValid");
    const auto roadColorsLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadColors[0]");
    const auto roadWidthsLocation =
        glGetUniformLocation(roadProgram_.Id(), "uRoadWidths[0]");
    const bool roadStyleConfigValid =
        roadColorsLocation >= 0 && roadWidthsLocation >= 0;
    glUniform1ui(roadStyleDisabledLocation, styles.enabled ? 0u : 1u);
    glUniform1ui(roadStyleConfigValidLocation, roadStyleConfigValid ? 1u : 0u);
    glUniform1f(glGetUniformLocation(roadProgram_.Id(), "uRoadWidthScale"),
                styles.globalWidthScale);
    if (roadStyleConfigValid) {
        glUniform4fv(roadColorsLocation,
                     static_cast<GLsizei>(styles.colors.size()), styles.colors.front().data());
        glUniform1fv(roadWidthsLocation,
                     static_cast<GLsizei>(styles.widthsPixels.size()), styles.widthsPixels.data());
    }

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
                                const std::array<float, 4>& color,
                                const RoadStyleConfig& styles) const {
    if (viewportWidth <= 0 || viewportHeight <= 0) {
        return;
    }

    GLint previousViewport[4]{};
    glGetIntegerv(GL_VIEWPORT, previousViewport);
    glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
    Draw(camera, viewportWidth, viewportHeight, drawData, 1.0f, 0.0f, color, styles);
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
