#version 450 core

const uint CACHED_SEGMENT_BIT = 0x80000000u;
const uint CACHED_SEGMENT_MASK = 0x7fffffffu;

struct RootRecord {
    uint globalEdgeId;
    int birthLevel;
    int deathLevel;
    uint boundsLo;
    uint boundsHi;
    uint padding0;
    uint padding1;
    uint padding2;
    vec4 segment;
};

layout(std430, binding = 0) readonly buffer DrawItems { uint drawItems[]; };
layout(std430, binding = 1) readonly buffer BlockPageTable { uint blockSlots[]; };
layout(std430, binding = 2) readonly buffer BlockEndpoints { vec4 endpoints[]; };
layout(std430, binding = 3) readonly buffer RefinedSegments { vec4 cachedSegments[]; };
layout(std430, binding = 4) readonly buffer RootRecords { RootRecord roots[]; };
layout(std430, binding = 5) readonly buffer DrawAlphas { float drawAlphas[]; };
layout(std430, binding = 6) readonly buffer RefinedRoadTypes { uint cachedRoadTypeWords[]; };

layout(location = 0) out float vLodAlpha;
flat out uint vRoadType;

uniform uint uBlockRecordCount;
uniform float uCameraCenterX;
uniform float uCameraCenterY;
uniform float uZoom;
uniform float uAspectScale;

uint rootRoadType(RootRecord root) {
    uint metadata = ((root.boundsLo >> 14u) & 0x3u) |
                    (((root.boundsLo >> 30u) & 0x3u) << 2u) |
                    (((root.boundsHi >> 14u) & 0x3u) << 4u) |
                    (((root.boundsHi >> 30u) & 0x3u) << 6u);
    return (metadata & 0xc0u) == 0x80u ? (metadata & 0x3fu) : 0u;
}

void main() {
    uint itemIndex = uint(gl_VertexID) >> 1u;
    uint item = drawItems[itemIndex];
    vec4 segment;
    uint roadType;

    if ((item & CACHED_SEGMENT_BIT) != 0u) {
        uint segmentIndex = item & CACHED_SEGMENT_MASK;
        segment = cachedSegments[segmentIndex];
        uint packed = cachedRoadTypeWords[segmentIndex >> 2u];
        roadType = (packed >> ((segmentIndex & 3u) * 8u)) & 63u;
    } else {
        RootRecord root = roots[item];
        segment = root.segment;
        roadType = rootRoadType(root);
    }

    vec2 position = (gl_VertexID & 1) == 0 ? segment.xy : segment.zw;
    vec2 centered = position - vec2(uCameraCenterX, uCameraCenterY);
    centered *= uZoom;
    centered.x *= uAspectScale;
    gl_Position = vec4(centered, 0.0, 1.0);
    vLodAlpha = drawAlphas[itemIndex];
    vRoadType = roadType;
}
