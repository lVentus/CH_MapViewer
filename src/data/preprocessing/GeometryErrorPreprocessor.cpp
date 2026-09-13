#include "data/preprocessing/GeometryErrorPreprocessor.h"

#include "data/ch/CHGraph.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace chmv::data::preprocessing {
namespace {

constexpr std::size_t kProgressInterval = 1u << 16;

void ReportProgress(const GeometryErrorPreprocessor::ProgressCallback& callback,
                    std::size_t current, std::size_t total) {
    if (!callback) {
        return;
    }
    callback(total == 0 ? 1.0f
                        : static_cast<float>(static_cast<double>(current) /
                                             static_cast<double>(total)));
}

bool Connects(const CHEdge& edge, std::uint32_t a, std::uint32_t b) {
    return (edge.source == a && edge.target == b) ||
           (edge.source == b && edge.target == a);
}

std::uint32_t FindSplitNode(const CHEdge& parent, const CHEdge& childA, const CHEdge& childB) {
    const std::uint32_t candidates[] = {childA.source, childA.target};
    for (const auto candidate : candidates) {
        if (candidate != childB.source && candidate != childB.target) {
            continue;
        }
        if (candidate != parent.source && candidate != parent.target) {
            const bool validChildren =
                (Connects(childA, parent.source, candidate) &&
                 Connects(childB, candidate, parent.target)) ||
                (Connects(childB, parent.source, candidate) &&
                 Connects(childA, candidate, parent.target));
            if (validChildren) {
                return candidate;
            }
        }
    }

    throw std::runtime_error("shortcut child edges do not share a valid split node");
}

double PointSegmentDistance(ProjectedPoint point, ProjectedPoint segmentStart,
                            ProjectedPoint segmentEnd) {
    const auto dx = segmentEnd.x - segmentStart.x;
    const auto dy = segmentEnd.y - segmentStart.y;
    const auto lengthSquared = dx * dx + dy * dy;
    if (lengthSquared <= 1e-30) {
        return std::hypot(point.x - segmentStart.x, point.y - segmentStart.y);
    }

    const auto px = point.x - segmentStart.x;
    const auto py = point.y - segmentStart.y;
    const auto t = std::clamp((px * dx + py * dy) / lengthSquared, 0.0, 1.0);
    const auto closestX = segmentStart.x + t * dx;
    const auto closestY = segmentStart.y + t * dy;
    return std::hypot(point.x - closestX, point.y - closestY);
}

float ComputeShortcutError(const CHEdge& edge, const std::vector<CHEdge>& edges,
                           const std::vector<CHNode>& nodes, const CHProjection& projection) {
    if (!edge.IsShortcut()) {
        return 0.0f;
    }
    if (edge.childA >= edges.size() || edge.childB >= edges.size()) {
        throw std::runtime_error("shortcut references an invalid child edge");
    }

    const auto& childA = edges[edge.childA];
    const auto& childB = edges[edge.childB];
    const auto splitNode = FindSplitNode(edge, childA, childB);
    if (edge.source >= nodes.size() || edge.target >= nodes.size() || splitNode >= nodes.size()) {
        throw std::runtime_error("shortcut geometry references an invalid node");
    }

    const auto& source = nodes[edge.source];
    const auto& target = nodes[edge.target];
    const auto& split = nodes[splitNode];
    const auto localError = PointSegmentDistance(
        projection.Project(split.latitude, split.longitude),
        projection.Project(source.latitude, source.longitude),
        projection.Project(target.latitude, target.longitude));

    return static_cast<float>(localError +
                              std::max(childA.geometryError, childB.geometryError));
}

bool ChildrenPrecedeParents(const std::vector<CHEdge>& edges) {
    for (std::size_t edgeId = 0; edgeId < edges.size(); ++edgeId) {
        const auto& edge = edges[edgeId];
        if (!edge.IsShortcut()) {
            continue;
        }
        if (edge.childA >= edges.size() || edge.childB >= edges.size()) {
            throw std::runtime_error("shortcut references an invalid child edge");
        }
        if (edge.childA >= edgeId || edge.childB >= edgeId) {
            return false;
        }
    }
    return true;
}

void ProcessOrdered(std::vector<CHEdge>& edges, const std::vector<CHNode>& nodes,
                    const CHProjection& projection,
                    const GeometryErrorPreprocessor::ProgressCallback& progressCallback) {
    for (std::size_t edgeId = 0; edgeId < edges.size(); ++edgeId) {
        auto& edge = edges[edgeId];
        edge.geometryError = ComputeShortcutError(edge, edges, nodes, projection);
        if ((edgeId + 1) % kProgressInterval == 0 || edgeId + 1 == edges.size()) {
            ReportProgress(progressCallback, edgeId + 1, edges.size());
        }
    }
}

void ProcessGeneral(std::vector<CHEdge>& edges, const std::vector<CHNode>& nodes,
                    const CHProjection& projection,
                    const GeometryErrorPreprocessor::ProgressCallback& progressCallback) {
    std::vector<std::uint8_t> state(edges.size(), 0);
    std::vector<std::uint32_t> stack;
    stack.reserve(128);
    std::size_t completed = 0;

    for (std::size_t root = 0; root < edges.size(); ++root) {
        if (state[root] == 2) {
            continue;
        }

        stack.push_back(static_cast<std::uint32_t>(root));
        while (!stack.empty()) {
            const auto edgeId = stack.back();
            auto& edge = edges[edgeId];

            if (state[edgeId] == 2) {
                stack.pop_back();
                continue;
            }

            if (!edge.IsShortcut()) {
                edge.geometryError = 0.0f;
                state[edgeId] = 2;
                stack.pop_back();
                ++completed;
            } else {
                if (edge.childA >= edges.size() || edge.childB >= edges.size()) {
                    throw std::runtime_error("shortcut references an invalid child edge");
                }

                if (state[edgeId] == 0) {
                    state[edgeId] = 1;
                }

                const std::uint32_t children[] = {edge.childA, edge.childB};
                bool waitingForChild = false;
                for (const auto child : children) {
                    if (state[child] == 1) {
                        throw std::runtime_error("cycle detected in shortcut decomposition");
                    }
                    if (state[child] == 0) {
                        stack.push_back(child);
                        waitingForChild = true;
                        break;
                    }
                }

                if (waitingForChild) {
                    continue;
                }

                edge.geometryError = ComputeShortcutError(edge, edges, nodes, projection);
                state[edgeId] = 2;
                stack.pop_back();
                ++completed;
            }

            if (completed % kProgressInterval == 0 || completed == edges.size()) {
                ReportProgress(progressCallback, completed, edges.size());
            }
        }
    }
}

} // namespace

void GeometryErrorPreprocessor::Run(CHGraph& graph, ProgressCallback progressCallback) {
    auto& edges = graph.Edges();
    const auto& nodes = graph.Nodes();
    if (edges.empty() || nodes.empty()) {
        ReportProgress(progressCallback, 1, 1);
        return;
    }

    const auto& projection = graph.Projection();
    ReportProgress(progressCallback, 0, edges.size());

    if (ChildrenPrecedeParents(edges)) {
        ProcessOrdered(edges, nodes, projection, progressCallback);
    } else {
        ProcessGeneral(edges, nodes, projection, progressCallback);
    }
}

} // namespace chmv::data::preprocessing
