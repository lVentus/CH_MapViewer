// The chord of a record (its two end points, from the vertices of its page) against the cull
// box of a pass: the planner's query box in fixed coordinates (TraversalCullBox() of
// LodPagePlanner.h). With uCullSegments, a segment whose chord misses the box is not drawn.

uniform ivec4 uCullBox;
uniform uint uCullSegments;

ivec2 vertexPosition(uint base, uint vertexBase, uint index) {
    uint word = base + vertexBase + index * 2u;
    return ivec2(int(pageWords[word]), int(pageWords[word + 1u]));
}

void chordEnds(uint item, uvec4 record, out ivec2 a, out ivec2 b) {
    uint base = slotBase(item >> 8u);
    uint vertexBase = 4u + (pageWords[base] & 0xffffu) * 4u;
    a = vertexPosition(base, vertexBase, record.x & 0xffffu);
    b = vertexPosition(base, vertexBase, record.x >> 16u);
}

// The chord's box, grown by `grow`, touches the cull box.
bool chordTouches(ivec2 a, ivec2 b, int grow) {
    ivec2 low = min(a, b);
    ivec2 high = max(a, b);
    return high.x >= uCullBox.x - grow && low.x <= uCullBox.z + grow &&
           high.y >= uCullBox.y - grow && low.y <= uCullBox.w + grow;
}

// A segment the expansion would draw is kept unless culling is on and it lies outside.
bool keepDrawn(uint item, uvec4 record) {
    if (uCullSegments == 0u) {
        return true;
    }
    ivec2 a;
    ivec2 b;
    chordEnds(item, record, a, b);
    return chordTouches(a, b, 0);
}
