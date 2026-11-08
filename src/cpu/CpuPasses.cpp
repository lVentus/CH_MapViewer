#include "cpu/CpuPasses.h"

#include "streaming/lod/LodPageFormat.h"

#include <algorithm>
#include <iterator>

namespace chmv::cpu {

bool ChordTouches(std::int32_t ax, std::int32_t ay, std::int32_t bx, std::int32_t by,
                  const std::array<double, 4>& rectangle) {
    const auto normalized = [](std::int32_t value) {
        return static_cast<double>(value) / streaming::lod::kFixedCoordinateScale;
    };
    return std::max(normalized(ax), normalized(bx)) >= rectangle[0] &&
           std::min(normalized(ax), normalized(bx)) <= rectangle[2] &&
           std::max(normalized(ay), normalized(by)) >= rectangle[1] &&
           std::min(normalized(ay), normalized(by)) <= rectangle[3];
}

bool ChordInCullBox(std::int32_t ax, std::int32_t ay, std::int32_t bx, std::int32_t by,
                    const std::array<std::int32_t, 4>& cull) {
    return std::max(ax, bx) >= cull[0] && std::min(ax, bx) <= cull[2] &&
           std::max(ay, by) >= cull[1] && std::min(ay, by) <= cull[3];
}

std::vector<std::uint32_t> EdgesInView(std::span<const CpuSegment> segments,
                                       std::span<const std::uint32_t> edges,
                                       const std::array<double, 4>& rectangle) {
    std::vector<std::uint32_t> inView;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        if (ChordTouches(segment.ax, segment.ay, segment.bx, segment.by, rectangle)) {
            inView.push_back(edges[i]);
        }
    }
    std::sort(inView.begin(), inView.end());
    return inView;
}

EdgeDifference CompareEdges(std::span<const std::uint32_t> expected,
                            std::span<const std::uint32_t> actual) {
    std::vector<std::uint32_t> difference;
    std::set_difference(expected.begin(), expected.end(), actual.begin(), actual.end(),
                        std::back_inserter(difference));
    EdgeDifference result;
    result.missing = difference.size();
    difference.clear();
    std::set_difference(actual.begin(), actual.end(), expected.begin(), expected.end(),
                        std::back_inserter(difference));
    result.extra = difference.size();
    return result;
}

} // namespace chmv::cpu
