#pragma once

#include <span>

namespace chmv::data {

struct CHNode;

struct ProjectedPoint {
    double x = 0.0;
    double y = 0.0;
};

struct CHProjection {
    double centerLatitude = 0.0;
    double centerLongitude = 0.0;
    double longitudeScale = 1.0;
    double normalization = 1.0;

    [[nodiscard]] ProjectedPoint Project(double latitude, double longitude) const;
};

[[nodiscard]] CHProjection ComputeCHProjection(std::span<const CHNode> nodes);

} // namespace chmv::data
