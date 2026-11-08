#include "streaming/lod/LodPagePlanner.h"

#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "renderer/RoadStyle.h"

#include <algorithm>
#include <cmath>

namespace chmv::streaming::lod {

void CollectLodPages(std::span<const PageGroup> groups, std::span<const PageEntry> pages,
                     const LodPageQuery& query, std::vector<std::uint32_t>& out) {
    for (const auto& group : groups) {
        if (!MayDraw(group, query)) {
            continue;
        }
        const auto end = group.firstPage + group.pageCount;
        for (auto page = group.firstPage; page < end; ++page) {
            if (MayDraw(pages[page], query)) {
                out.push_back(page);
            }
        }
    }
}

LodPageView PlanView(const renderer::MapCamera2D& camera, const renderer::LODController& lod,
                     int width, int height, float pixelTolerance, std::int32_t maxLevel,
                     double zoomOutOctaves) {
    // Roads just outside the screen still reach into it with half their stroke width.
    constexpr double kStrokeMarginPixels = renderer::kMaxRoadWidthPixels / 2.0;
    const auto viewScale = std::max(static_cast<double>(camera.ViewScale()), 1e-12);
    const auto halfY = 1.0 / viewScale;
    const auto halfX = halfY * static_cast<double>(width) / static_cast<double>(height);
    const auto centerX = static_cast<double>(camera.CenterX());
    const auto centerY = static_cast<double>(camera.CenterY());
    const auto pixel = 2.0 / (viewScale * static_cast<double>(height));
    const auto margin = kStrokeMarginPixels * pixel;
    LodPageView view;
    view.query.minX = static_cast<float>(centerX - halfX - margin);
    view.query.minY = static_cast<float>(centerY - halfY - margin);
    view.query.maxX = static_cast<float>(centerX + halfX + margin);
    view.query.maxY = static_cast<float>(centerY + halfY + margin);
    const auto farther = std::exp2(zoomOutOctaves);
    view.query.level = static_cast<std::uint32_t>(std::clamp<long>(
        std::lround(lod.ContinuousLevel(static_cast<float>(camera.ZoomFactor() / farther))), 0L,
        maxLevel));
    view.toleranceCode = EncodeErrorTolerance(static_cast<double>(std::max(pixelTolerance, 0.01f)) *
                                              pixel * farther);
    view.pixelCode = EncodeErrorTolerance(pixel);
    view.rectangle = {centerX - halfX, centerY - halfY, centerX + halfX, centerY + halfY};
    return view;
}

std::array<std::int32_t, 4> TraversalCullBox(const LodPageQuery& query) {
    const auto inward = [](float value, double direction) {
        constexpr double kLimit = 1073741824.0;
        const auto fixed = static_cast<double>(value) * kFixedCoordinateScale;
        const auto rounded = direction > 0.0 ? std::ceil(fixed) + 2.0 : std::floor(fixed) - 2.0;
        return static_cast<std::int32_t>(std::clamp(rounded, -kLimit, kLimit));
    };
    return {inward(query.minX, 1.0), inward(query.minY, 1.0), inward(query.maxX, -1.0),
            inward(query.maxY, -1.0)};
}

} // namespace chmv::streaming::lod
