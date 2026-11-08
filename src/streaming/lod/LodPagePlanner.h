#pragma once

#include "streaming/lod/LodPageFormat.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace chmv::renderer {
class LODController;
class MapCamera2D;
} // namespace chmv::renderer

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

    [[nodiscard]] bool operator==(const LodPageQuery&) const = default;
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

// What a frame shows, from the camera alone: the planner query (the view grown by half the
// widest road, and the level of the LOD curve; its threshold is left to the caller, which
// derives it from toleranceCode and the expansion mode) and the view rectangle (minX, minY,
// maxX, maxY) in normalized coordinates.
struct LodPageView {
    LodPageQuery query;
    std::uint32_t toleranceCode = 0;
    // The error code of one pixel.
    std::uint32_t pixelCode = 0;
    std::array<double, 4> rectangle{};
};

// zoomOutOctaves > 0 plans the same rectangle with the level and tolerance of a camera that
// many octaves farther away (a coarser view of the same place).
[[nodiscard]] LodPageView PlanView(const renderer::MapCamera2D& camera,
                                   const renderer::LODController& lod, int width, int height,
                                   float pixelTolerance, std::int32_t maxLevel,
                                   double zoomOutOctaves = 0.0);

// The query box in fixed coordinates, rounded inward (and two units more), so a node that a
// traversal keeps always lies in a page the planner chose with the float box.
[[nodiscard]] std::array<std::int32_t, 4> TraversalCullBox(const LodPageQuery& query);

} // namespace chmv::streaming::lod
