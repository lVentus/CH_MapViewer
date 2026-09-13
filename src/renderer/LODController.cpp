#include "renderer/LODController.h"

#include <algorithm>
#include <cmath>

namespace chmv::renderer {
namespace {

constexpr float kOverviewZoom = 10.0f / 3.0f;
constexpr float kFullDetailZoom = 320.0f / 3.0f;

} // namespace

void LODController::SetLevelRange(std::int32_t minLevel, std::int32_t maxLevel) {
    minLevel_ = minLevel;
    maxLevel_ = std::max(minLevel, maxLevel);
    manualLevel_ = std::clamp(manualLevel_, minLevel_, maxLevel_);
}

void LODController::SetManualLevel(std::int32_t level) {
    manualLevel_ = std::clamp(level, minLevel_, maxLevel_);
}

std::int32_t LODController::Level(float zoom) const {
    if (!automatic_ || maxLevel_ <= minLevel_) {
        return manualLevel_;
    }

    const auto clampedZoom = std::clamp(zoom, kOverviewZoom, kFullDetailZoom);
    const auto detail = std::log2(clampedZoom / kOverviewZoom) /
                        std::log2(kFullDetailZoom / kOverviewZoom);
    const auto level = static_cast<float>(maxLevel_) -
                       detail * static_cast<float>(maxLevel_ - minLevel_);
    return static_cast<std::int32_t>(std::lround(level));
}

} // namespace chmv::renderer
