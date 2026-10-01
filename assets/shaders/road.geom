#version 450 core

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in float vLodAlpha[];
flat in uint vRoadType[];
layout(location = 0) out float gLodAlpha;
flat out uint gRoadType;

uniform vec2 uViewportSize;
uniform uint uRoadStyleDisabled;
uniform uint uRoadStyleConfigValid;
uniform float uRoadWidthScale;
uniform float uRoadWidths[64];

float fallbackRoadWidth(uint roadType) {
    if (roadType == 63u) return 1.15;
    float t = float(min(roadType, 62u)) / 62.0;
    return max(0.90, 5.60 - 4.25 * t);
}

void emitRoadVertex(vec4 clip, vec2 offset) {
    gl_Position = clip + vec4(offset, 0.0, 0.0);
    EmitVertex();
}

void main() {
    vec2 viewport = max(uViewportSize, vec2(1.0));
    vec2 p0 = gl_in[0].gl_Position.xy;
    vec2 p1 = gl_in[1].gl_Position.xy;
    vec2 pixelDelta = (p1 - p0) * viewport * 0.5;
    float lengthPx = length(pixelDelta);
    vec2 normalPx = lengthPx > 1e-5
                        ? vec2(-pixelDelta.y, pixelDelta.x) / lengthPx
                        : vec2(0.0, 1.0);
    uint roadType = min(vRoadType[0], 63u);
    float baseWidth = uRoadStyleConfigValid != 0u
                          ? uRoadWidths[roadType]
                          : fallbackRoadWidth(roadType);
    float widthPx = uRoadStyleDisabled == 0u
                        ? max(baseWidth * max(uRoadWidthScale, 0.01), 0.5)
                        : 1.0;
    vec2 offset = normalPx * (widthPx / viewport);

    gRoadType = roadType;
    gLodAlpha = vLodAlpha[0];
    emitRoadVertex(gl_in[0].gl_Position, -offset);
    emitRoadVertex(gl_in[0].gl_Position,  offset);
    gLodAlpha = vLodAlpha[1];
    emitRoadVertex(gl_in[1].gl_Position, -offset);
    emitRoadVertex(gl_in[1].gl_Position,  offset);
    EndPrimitive();
}
