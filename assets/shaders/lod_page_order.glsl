// Record order of the birth-ordered filter (lod_filter_sort.comp).

layout(std430, binding = 5) readonly buffer Order { uint order[]; };

uint orderedRecord(uint slot, uint position) {
    return (order[slot * 64u + (position >> 2u)] >> ((position & 3u) * 8u)) & 0xffu;
}

uint highestLevel(uvec4 record) {
    return max(record.y >> 16u, record.z >> 16u);
}
