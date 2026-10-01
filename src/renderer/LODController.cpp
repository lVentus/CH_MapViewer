#include "renderer/LODController.h"

#include <algorithm>
#include <cmath>

namespace chmv::renderer {
namespace {

// Map zoom in screen-scale octaves. The previous log1p curve was concave, so a small overview
// zoom change consumed a disproportionately large fraction of the CH hierarchy. A convex curve in
// normalized log2(screen scale) keeps coarse/global LODs stable for longer, then catches up as the
// view becomes local. Endpoints remain unchanged: zoom=10 -> maxLevel, zoom=1000 -> minLevel.
constexpr float kMaximumMappedZoom = 1000.0f;
constexpr float kOverviewDetailExponent = 1.35f;
// The coarsest ~1/9 of the CH hierarchy contributes almost no useful visual change at the
// fitted overview. Automatic LOD therefore starts at 8/9 of the hierarchy instead of spending
// zoom range on those nearly-empty top levels. Manual LOD can still select the full range.
constexpr float kOverviewLevelFraction = 8.0f / 9.0f;
constexpr float kMinOverviewZoom = 0.01f;

} // namespace

void LODController::SetLevelRange(std::int32_t minLevel, std::int32_t maxLevel) {
    minLevel_ = minLevel;
    maxLevel_ = std::max(minLevel, maxLevel);
    manualLevel_ = std::clamp(manualLevel_, minLevel_, maxLevel_);
}

void LODController::SetManualLevel(std::int32_t level) {
    manualLevel_ = std::clamp(level, minLevel_, maxLevel_);
}

void LODController::SetOverviewZoom(float zoom) {
    if (std::isfinite(zoom) && zoom > 0.0f) {
        overviewZoom_ = std::max(zoom, kMinOverviewZoom);
    }
}

float LODController::ContinuousLevel(float zoom) const {
    if (!automatic_ || maxLevel_ <= minLevel_) {
        return static_cast<float>(manualLevel_);
    }

    // Anchor the curve at this dataset's actual fit-to-view zoom rather than at a global numeric
    // zoom such as 10x. EUR and BW have very different physical extents, so their Reset() zooms
    // differ even though equal zoom values now represent equal physical scale.
    const auto overviewZoom = std::max(overviewZoom_, kMinOverviewZoom);
    const auto mappingEndZoom = std::max(kMaximumMappedZoom, overviewZoom * 1.0001f);
    const auto clampedZoom = std::clamp(zoom, overviewZoom, mappingEndZoom);
    const auto zoomOctaves = std::log2(clampedZoom / overviewZoom);
    const auto maxOctaves = std::log2(mappingEndZoom / overviewZoom);
    const auto normalizedScale = maxOctaves > 0.0f
                                     ? std::clamp(zoomOctaves / maxOctaves, 0.0f, 1.0f)
                                     : 1.0f;

    // levelFraction is 1 at maxLevel (coarsest) and 0 at minLevel (finest). At the fitted
    // overview we intentionally begin at 8/9 instead of 1, then use the full remaining physical
    // zoom range to reach minLevel at 1000x (3 km vertical span with the current camera scale).
    const auto curve = std::pow(normalizedScale, kOverviewDetailExponent);
    const auto levelFraction = kOverviewLevelFraction * (1.0f - curve);
    const auto level = static_cast<float>(minLevel_) +
                       levelFraction * static_cast<float>(maxLevel_ - minLevel_);
    return std::clamp(level, static_cast<float>(minLevel_), static_cast<float>(maxLevel_));
}

std::int32_t LODController::Level(float zoom) const {
    return static_cast<std::int32_t>(std::lround(ContinuousLevel(zoom)));
}

} // namespace chmv::renderer
