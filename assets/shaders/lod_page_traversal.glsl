// Walking the hierarchy (needs the PageChildren extension in the pool): a node is a
// representative record, its children are found through the children block of its page, and
// pages are found through the slot of every page.

layout(std430, binding = 6) readonly buffer SlotOfPage { uint slotOfPage[]; };
// ErrorBoundMantissa() of LodPageFormat.h, four per entry.
layout(std140, binding = 0) uniform ErrorBounds { uvec4 boundMantissas[256]; };

uniform uint uPageWords;
uniform uint uThreshold;
// Fixed coordinates, inside the rectangle the planner used, so every page a visible node lives
// in is resident.
uniform ivec4 uCullBox;

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

ivec2 vertexPosition(uint base, uint vertexBase, uint index) {
    uint word = base + vertexBase + index * 2u;
    return ivec2(int(pageWords[word]), int(pageWords[word + 1u]));
}

// The node's whole geometry lies within its error of its chord.
bool nodeVisible(uint item, uvec4 record) {
    uint base = slotBase(item >> 8u);
    uint vertexBase = 4u + (pageWords[base] & 0xffffu) * 4u;
    ivec2 a = vertexPosition(base, vertexBase, record.x & 0xffffu);
    ivec2 b = vertexPosition(base, vertexBase, record.x >> 16u);
    int bound = int(errorBound(record.w & 0xffffu));
    ivec2 low = min(a, b);
    ivec2 high = max(a, b);
    return high.x >= uCullBox.x - bound && low.x <= uCullBox.z + bound &&
           high.y >= uCullBox.y - bound && low.y <= uCullBox.w + bound;
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
