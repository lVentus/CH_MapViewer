#version 430 core

// Vertex pulling: six vertices (two triangles) per drawn segment, no vertex buffers and no
// geometry shader. Positions are integers relative to the camera, so precision does not depend
// on how large the map is.

layout(std430, binding = 0) readonly buffer PageWords { uint pageWords[]; };
layout(std430, binding = 3) readonly buffer Segments { uint segments[]; };

layout(location = 1) flat out uint gRoadType;

uniform uint uSlotWords;
uniform ivec2 uCameraFixed;
uniform float uFixedToClip;
uniform float uAspectScale;
uniform vec2 uViewportSize;
// Width in pixels and depth per road type, from the road style config: the classes listed first
// lie on top, whatever order the segments come in.
uniform float uRoadWidths[64];
uniform float uRoadDepths[64];

vec2 clipPosition(uint vertexWord) {
    ivec2 fixedPosition = ivec2(int(pageWords[vertexWord]), int(pageWords[vertexWord + 1u]));
    vec2 position = vec2(fixedPosition - uCameraFixed) * uFixedToClip;
    position.x *= uAspectScale;
    return position;
}

void main() {
    uint segmentIndex = uint(gl_VertexID) / 6u;
    uint corner = uint(gl_VertexID) - segmentIndex * 6u;
    uint item = segments[segmentIndex];
    uint slot = item >> 8u;
    uint local = item & 0xffu;

    uint base = slot * uSlotWords;
    uint header = pageWords[base];
    uint recordCount = header & 0xffffu;
    uint vertexCount = header >> 16u;
    uint endpoints = pageWords[base + 4u + local * 4u];
    uint vertexBase = base + 4u + recordCount * 4u;
    vec2 a = clipPosition(vertexBase + (endpoints & 0xffffu) * 2u);
    vec2 b = clipPosition(vertexBase + (endpoints >> 16u) * 2u);

    uint typeWord = pageWords[vertexBase + vertexCount * 2u + (local >> 2u)];
    uint roadType = min((typeWord >> ((local & 3u) * 8u)) & 0xffu, 63u);
    float widthPx = uRoadWidths[roadType];

    vec2 viewport = max(uViewportSize, vec2(1.0));
    vec2 pixelDelta = (b - a) * viewport * 0.5;
    float lengthPx = length(pixelDelta);
    vec2 normalPx = lengthPx > 1e-5 ? vec2(-pixelDelta.y, pixelDelta.x) / lengthPx
                                    : vec2(0.0, 1.0);
    vec2 offset = normalPx * (widthPx / viewport);

    // Corners: (a,-) (a,+) (b,-) | (b,-) (a,+) (b,+)
    bool atB = corner == 2u || corner == 3u || corner == 5u;
    float side = (corner == 1u || corner == 4u || corner == 5u) ? 1.0 : -1.0;
    gl_Position = vec4((atB ? b : a) + side * offset, uRoadDepths[roadType] * 2.0 - 1.0, 1.0);
    gRoadType = roadType;
}
