#pragma once

#include "streaming/lod/LodPageFormat.h"

#include <cstdint>
#include <span>
#include <vector>

namespace chmv::streaming::lod {

// What the camera currently asks for. Everything a view needs follows from these five values,
// so page selection never has to wait for the GPU.
struct LodPageQuery {
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
};

// Conservative per-page (or per-group) test: false only if no record inside can be drawn.
template <typename Bounds>
[[nodiscard]] inline bool MayDraw(const Bounds& bounds, const LodPageQuery& query) {
    if (bounds.maxX < query.minX || bounds.minX > query.maxX || bounds.maxY < query.minY ||
        bounds.minY > query.maxY || query.level < bounds.minLevel ||
        query.level > bounds.maxLevel || query.threshold < bounds.minErrorCode ||
        query.threshold >= bounds.maxParentCode) {
        return false;
    }
    const auto band = LevelBand(query.level, bounds.minLevel, bounds.maxLevel);
    const auto code = query.threshold >> 8u;
    return code >= bounds.bands.minError[band] && code <= bounds.bands.maxParent[band];
}

// Appends the ids of all pages that may contain drawn records, in file order (coarse layers
// first, then Morton order).
void CollectLodPages(std::span<const PageGroup> groups, std::span<const PageEntry> pages,
                     const LodPageQuery& query, std::vector<std::uint32_t>& out);

} // namespace chmv::streaming::lod
