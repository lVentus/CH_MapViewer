// Walking the hierarchy (needs the PageChildren extension in the pool): a node is a
// representative record, its children are found through the children block of its page, and
// pages are found through the slot of every page. Comes after lod_page_segments.glsl, whose
// cull box (lod_page_chord.glsl) lies inside the rectangle the planner used, so every page a
// visible node lives in is resident.

layout(std430, binding = 6) readonly buffer SlotOfPage { uint slotOfPage[]; };
// ErrorBoundMantissa() of LodPageFormat.h, four per entry.
layout(std140, binding = 0) uniform ErrorBounds { uvec4 boundMantissas[256]; };

uniform uint uPageWords;
uniform uint uThreshold;

const uint kMaxErrorBound = (1u << 30u) - 1u;

// ScaleErrorBound() of LodPageFormat.h.
uint errorBound(uint code) {
    if (code == 0u) {
        return 0u;
    }
    uint octave = code >> 10u;
    if (octave >= 41u) {
        return kMaxErrorBound;
    }
    uint shift = 42u - octave;
    if (shift >= 32u) {
        return 1u;
    }
    uint step = code & 1023u;
    uint mantissa = boundMantissas[step >> 2u][step & 3u];
    uint rest = (mantissa & ((1u << shift) - 1u)) != 0u ? 1u : 0u;
    return min((mantissa >> shift) + rest, kMaxErrorBound);
}

// The node's whole geometry lies within its error of its chord.
bool nodeVisible(uint item, uvec4 record) {
    ivec2 a;
    ivec2 b;
    chordEnds(item, record, a, b);
    return chordTouches(a, b, int(errorBound(record.w & 0xffffu)));
}

// Expanded unless the node is already accurate enough or is an original edge.
bool expands(uint item, uvec4 record) {
    if ((record.w & 0xffffu) <= uThreshold) {
        return false;
    }
    uint block = slotBase(item >> 8u) + uPageWords;
    uint local = item & 0xffu;
    return ((pageWords[block + (local >> 5u)] >> (local & 31u)) & 1u) != 0u;
}

// Child 0 or 1 of an expanding node as a pool item, or kNoItem if its page is not resident.
// The planner loads every page holding a node inside the cull box, so such a child is outside.
uint childItem(uint item, uint which) {
    uint block = slotBase(item >> 8u) + uPageWords;
    uint local = item & 0xffu;
    uint rank = uint(bitCount(pageWords[block + (local >> 5u)] & ((1u << (local & 31u)) - 1u)));
    for (uint word = 0u; word < (local >> 5u); ++word) {
        rank += uint(bitCount(pageWords[block + word]));
    }
    uint child = pageWords[block + 8u + rank * 2u + which];
    uint slot = child == kNoItem ? kNoItem : slotOfPage[child >> 8u];
    if (slot == kNoItem) {
        atomicAdd(status[kStatusMissingPage], 1u);
        return kNoItem;
    }
    return (slot << 8u) | (child & 0xffu);
}
