#pragma once

#include "data/ch/CHTypes.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace chmv::renderer {

// One way of drawing roads.
struct RoadClass {
    std::string name;
    std::array<float, 4> color{1.0f, 1.0f, 1.0f, 1.0f};
    float widthPixels = 1.0f;
};

// Road styling from a config file (assets/road_styles.cfg): a handful of road classes, each
// covering some edge types (the type ids of OsmGraphCreator's all.cfg). Every type the pages can
// store (data::RoadStyleIndexFromType) belongs to exactly one class.
struct RoadStyleConfig {
    std::vector<RoadClass> classes;
    std::array<std::uint8_t, data::RoadStyleTypeCount> classOfType{};

    // What the road shaders take: color and width per type, and the depth that puts the
    // classes listed first on top of the later ones (between 0 and 1, smaller is nearer). With
    // it the picture does not depend on the order in which the segments are drawn.
    [[nodiscard]] std::array<std::array<float, 4>, data::RoadStyleTypeCount> TypeColors() const;
    [[nodiscard]] std::array<float, data::RoadStyleTypeCount> TypeWidths() const;
    [[nodiscard]] std::array<float, data::RoadStyleTypeCount> TypeDepths() const;
};

// Roads wider than this would need a larger query margin around the view.
inline constexpr float kMaxRoadWidthPixels = 32.0f;

// One style for every road, for when the config file cannot be used.
[[nodiscard]] RoadStyleConfig UniformRoadStyle();
// Throws std::runtime_error naming the file and line of the first problem.
[[nodiscard]] RoadStyleConfig LoadRoadStyles(const std::filesystem::path& path);

} // namespace chmv::renderer
