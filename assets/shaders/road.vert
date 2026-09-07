#version 430 core

struct Edge {
    uvec4 topology;
    ivec4 levels;
};

layout(std430, binding = 0) readonly buffer Nodes {
    vec2 nodePositions[];
};

layout(std430, binding = 1) readonly buffer Edges {
    Edge edges[];
};

layout(std430, binding = 2) readonly buffer VisibleEdges {
    uint visibleEdgeIds[];
};

uniform vec2 uCameraCenter;
uniform float uZoom;
uniform float uAspectScale;

void main() {
    uint edgeId = visibleEdgeIds[uint(gl_VertexID) >> 1u];
    Edge edge = edges[edgeId];
    uint nodeId = (gl_VertexID & 1) == 0 ? edge.topology.x : edge.topology.y;
    vec2 position = (nodePositions[nodeId] - uCameraCenter) * uZoom;
    position.x *= uAspectScale;
    gl_Position = vec4(position, 0.0, 1.0);
}
