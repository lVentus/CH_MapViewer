#include "data/ch/CHProjection.h"

#include "data/ch/CHTypes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace chmv::data {
namespace {

constexpr double kPi = 3.14159265358979323846;

} // namespace

ProjectedPoint CHProjection::Project(double latitude, double longitude) const {
    return {
        (longitude - centerLongitude) * longitudeScale * normalization,
        (latitude - centerLatitude) * normalization,
    };
}

CHProjection ComputeCHProjection(std::span<const CHNode> nodes) {
    if (nodes.empty()) {
        throw std::runtime_error("cannot compute a projection for an empty graph");
    }

    double minLatitude = std::numeric_limits<double>::max();
    double maxLatitude = std::numeric_limits<double>::lowest();
    double minLongitude = std::numeric_limits<double>::max();
    double maxLongitude = std::numeric_limits<double>::lowest();

    for (const auto& node : nodes) {
        minLatitude = std::min(minLatitude, node.latitude);
        maxLatitude = std::max(maxLatitude, node.latitude);
        minLongitude = std::min(minLongitude, node.longitude);
        maxLongitude = std::max(maxLongitude, node.longitude);
    }

    CHProjection projection;
    projection.centerLatitude = (minLatitude + maxLatitude) * 0.5;
    projection.centerLongitude = (minLongitude + maxLongitude) * 0.5;
    projection.longitudeScale = std::cos(projection.centerLatitude * kPi / 180.0);

    const auto halfWidth =
        std::max((maxLongitude - minLongitude) * 0.5 * projection.longitudeScale, 1e-12);
    const auto halfHeight = std::max((maxLatitude - minLatitude) * 0.5, 1e-12);
    projection.normalization = 0.95 / std::max(halfWidth, halfHeight);
    return projection;
}

} // namespace chmv::data
