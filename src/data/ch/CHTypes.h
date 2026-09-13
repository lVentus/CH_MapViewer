#pragma once

#include <cstdint>
#include <limits>

namespace chmv::data {

constexpr std::uint32_t InvalidEdgeId = std::numeric_limits<std::uint32_t>::max();

struct CHNode {
    std::uint64_t osmId = 0;
    double latitude = 0.0;
    double longitude = 0.0;
    float elevation = 0.0f;
    std::uint32_t level = 0;
};

struct CHEdge {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    float weight = 0.0f;
    std::int32_t type = 0;
    std::int32_t maxSpeed = 0;
    std::uint32_t childA = InvalidEdgeId;
    std::uint32_t childB = InvalidEdgeId;
    float geometryError = 0.0f;

    [[nodiscard]] bool IsShortcut() const {
        return childA != InvalidEdgeId && childB != InvalidEdgeId;
    }
};

struct EdgeRange {
    std::int32_t birthLevel = -1;
    std::int32_t deathLevel = -1;

    [[nodiscard]] bool IsDrawable() const {
        return birthLevel >= 0 && deathLevel >= 0;
    }

    [[nodiscard]] bool IsAlive(std::int32_t level) const {
        return IsDrawable() && level <= birthLevel && level >= deathLevel;
    }
};

} // namespace chmv::data
