// Output of the expansion kernels: the segment list the road shader draws, and its count in the
// indirect draw command (lod_draw_finish.comp turns the count into vertices).

#include "lod_page_chord.glsl"

layout(std430, binding = 3) writeonly buffer Segments { uint segments[]; };
layout(std430, binding = 5) buffer DrawCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint baseInstance;
    uint segmentCount;
};

uniform uint uSegmentCapacity;

void drawSegment(uint item) {
    uint index = atomicAdd(segmentCount, 1u);
    if (index < uSegmentCapacity) {
        segments[index] = item;
    }
}

shared uint sDrawnCount;
shared uint sDrawnBase;

// Called by every invocation of the workgroup; one global atomic per workgroup.
void drawSegments(bool draw, uint item) {
    if (gl_LocalInvocationIndex == 0u) {
        sDrawnCount = 0u;
    }
    barrier();
    uint offset = 0u;
    if (draw) {
        offset = atomicAdd(sDrawnCount, 1u);
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u && sDrawnCount != 0u) {
        sDrawnBase = atomicAdd(segmentCount, sDrawnCount);
    }
    barrier();
    if (draw && sDrawnBase + offset < uSegmentCapacity) {
        segments[sDrawnBase + offset] = item;
    }
}
