#version 430 core

// CPU results: five words per segment (both ends in fixed coordinates, road type), drawn the
// same way as lod_page_road.vert draws page records: six vertices per segment, positions
// relative to the camera.

layout(std430, binding = 0) readonly buffer Segments { int segments[]; };

layout(location = 1) flat out uint gRoadType;

uniform ivec2 uCameraFixed;
uniform float uFixedToClip;
uniform float uAspectScale;
uniform vec2 uViewportSize;
// Width in pixels and depth per road type, from the road style config: the classes listed first
// lie on top, whatever order the segments come in.
uniform float uRoadWidths[64];
uniform float uRoadDepths[64];

vec2 clipPosition(uint word) {
    vec2 position = vec2(ivec2(segments[word], segments[word + 1u]) - uCameraFixed) * uFixedToClip;
    position.x *= uAspectScale;
    return position;
}

void main() {
    uint segmentIndex = uint(gl_VertexID) / 6u;
    uint corner = uint(gl_VertexID) - segmentIndex * 6u;
    uint base = segmentIndex * 5u;
    vec2 a = clipPosition(base);
    vec2 b = clipPosition(base + 2u);
    uint roadType = min(uint(segments[base + 4u]), 63u);
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
