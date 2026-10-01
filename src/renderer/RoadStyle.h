#pragma once

#include "data/ch/CHTypes.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace chmv::renderer {

namespace detail {
[[nodiscard]] inline constexpr std::array<float, 3> HueToRgb(float h) {
    while (h < 0.0f) {
        h += 1.0f;
    }
    while (h >= 1.0f) {
        h -= 1.0f;
    }
    const float scaled = h * 6.0f;
    const int sector = static_cast<int>(scaled);
    const float f = scaled - static_cast<float>(sector);
    switch (sector) {
    case 0: return {1.0f, f, 0.0f};
    case 1: return {1.0f - f, 1.0f, 0.0f};
    case 2: return {0.0f, 1.0f, f};
    case 3: return {0.0f, 1.0f - f, 1.0f};
    case 4: return {f, 0.0f, 1.0f};
    default: return {1.0f, 0.0f, 1.0f - f};
    }
}

[[nodiscard]] inline constexpr std::array<float, 4> DefaultRoadColor(std::size_t type) {
    if (type == data::RoadStyleFallbackType) {
        return {0.70f, 0.72f, 0.76f, 0.95f};
    }
    const float t = static_cast<float>(type) /
                    static_cast<float>(data::RoadStyleTypeCount - 2u);
    const float hue = 0.02f + 0.78f * t; // red -> orange -> yellow -> green -> cyan -> blue -> purple
    const auto base = HueToRgb(hue);
    const float whiten = 0.08f + 0.18f * t;
    return {
        base[0] * (1.0f - whiten) + whiten,
        base[1] * (1.0f - whiten) + whiten,
        base[2] * (1.0f - whiten) + whiten,
        0.96f
    };
}

[[nodiscard]] inline constexpr float DefaultRoadWidth(std::size_t type) {
    if (type == data::RoadStyleFallbackType) {
        return 1.15f;
    }
    const float t = static_cast<float>(type) /
                    static_cast<float>(data::RoadStyleTypeCount - 2u);
    return std::max(0.90f, 5.60f - 4.25f * t);
}
} // namespace detail

struct RoadStyleConfig {
    bool enabled = true;
    float globalWidthScale = 1.0f;
    std::array<std::array<float, 4>, data::RoadStyleTypeCount> colors{};
    std::array<float, data::RoadStyleTypeCount> widthsPixels{};

    RoadStyleConfig() { ResetDefaults(); }

    void ResetType(std::size_t type) {
        if (type >= data::RoadStyleTypeCount) {
            return;
        }
        colors[type] = detail::DefaultRoadColor(type);
        widthsPixels[type] = detail::DefaultRoadWidth(type);
    }

    void ResetDefaults() {
        enabled = true;
        globalWidthScale = 1.0f;
        for (std::size_t i = 0; i < data::RoadStyleTypeCount; ++i) {
            ResetType(i);
        }
    }
};

} // namespace chmv::renderer
