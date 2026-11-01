// Shared by the LOD page kernels: the page pool as the shaders see it (LodPageFormat.h) and the
// status counters of a pass (LodPageLists.h).
//
// A pool slot holds a page blob: header word (record count | vertex count << 16), three spare
// words, one uvec4 per record, two words per vertex, one road type byte per record. In hierarchy
// mode the page's children block follows at word uPageWords of the slot.
//
// Lists ([count, groupsX, groupsY, groupsZ, items...]) carry their own indirect dispatch
// arguments, and record items are slot << 8 | record.
//
// Storage bindings: 0/1 pool, 2 input list, 3 output list, 4 status, 5-7 kernel specific.

layout(std430, binding = 0) readonly buffer PageWords { uint pageWords[]; };
layout(std430, binding = 1) readonly buffer PageRecords { uvec4 pageRecords[]; };
layout(std430, binding = 4) buffer Status { uint status[]; };

const uint kStatusCandidateOverflow = 0u;
const uint kStatusSegmentOverflow = 1u;
const uint kStatusMissingPage = 2u;
const uint kStatusStackOverflow = 3u;
const uint kStatusFrontierOverflow = 4u;

const uint kNoItem = 0xffffffffu;

uniform uint uSlotWords;

uint slotBase(uint slot) {
    return slot * uSlotWords;
}

uint recordCount(uint slot) {
    return pageWords[slotBase(slot)] & 0xffffu;
}

uvec4 pageRecord(uint item) {
    return pageRecords[slotBase(item >> 8u) / 4u + 1u + (item & 0xffu)];
}

bool inLevelRange(uint packedRange, uint level) {
    return level >= (packedRange & 0xffffu) && level <= (packedRange >> 16u);
}

// The record is a root at this level, or lies below a parent there.
bool levelMatch(uvec4 record, uint level) {
    return inLevelRange(record.y, level) || inLevelRange(record.z, level);
}

// Dispatches with more than 65535 workgroups use a second dimension.
uint linearGroup() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

uint linearInvocation() {
    return linearGroup() * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
}
