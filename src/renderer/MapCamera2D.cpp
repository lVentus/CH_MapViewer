#include "renderer/MapCamera2D.h"

#include <algorithm>
#include <cmath>

namespace chmv::renderer {
namespace {

// Canonical physical zoom is calibrated so that 1000x corresponds to an approximate
// 3 km vertical world span. ViewHalfHeightMeters() is reference / zoom, therefore a
// 1,500,000 m reference half-span gives 2 * 1,500,000 / 1000 = 3,000 m vertically.
// Equal zoom values still have the same approximate physical scale across datasets.
constexpr double kReferenceHalfSpanMeters = 1500000.0;
constexpr float kZoomPerScrollStep = 1.03f;
constexpr float kZoomResponse = 14.0f;
constexpr float kMinZoom = 0.01f;
constexpr float kMaxZoom = 1048576.0f;

} // namespace

void MapCamera2D::SetMetersPerNormalizedUnit(double metersPerNormalizedUnit) {
    if (!(metersPerNormalizedUnit > 0.0) || !std::isfinite(metersPerNormalizedUnit)) {
        return;
    }
    metersPerNormalizedUnit_ = metersPerNormalizedUnit;
}

float MapCamera2D::FitZoom() const {
    const auto zoom = kReferenceHalfSpanMeters /
                      std::max(metersPerNormalizedUnit_, 1e-12);
    return static_cast<float>(std::clamp(zoom, static_cast<double>(kMinZoom),
                                         static_cast<double>(kMaxZoom)));
}

void MapCamera2D::Reset() {
    centerX_ = 0.0f;
    centerY_ = 0.0f;
    zoom_ = FitZoom();
    targetZoom_ = zoom_;
}

float MapCamera2D::ViewScale() const {
    return static_cast<float>(static_cast<double>(zoom_) * metersPerNormalizedUnit_ /
                              kReferenceHalfSpanMeters);
}

double MapCamera2D::ViewHalfHeightMeters() const {
    return kReferenceHalfSpanMeters / std::max(static_cast<double>(zoom_), 1e-12);
}

void MapCamera2D::Zoom(float scrollSteps) {
    targetZoom_ *= std::pow(kZoomPerScrollStep, scrollSteps);
    targetZoom_ = std::clamp(targetZoom_, kMinZoom, kMaxZoom);
}

void MapCamera2D::Update(float deltaSeconds) {
    const auto dt = std::clamp(deltaSeconds, 0.0f, 0.1f);
    const auto response = 1.0f - std::exp(-kZoomResponse * dt);
    zoom_ += (targetZoom_ - zoom_) * response;

    if (std::abs(targetZoom_ - zoom_) < 0.0001f) {
        zoom_ = targetZoom_;
    }
}

void MapCamera2D::PanPixels(double deltaX, double deltaY, int framebufferHeight) {
    if (framebufferHeight <= 0) {
        return;
    }

    const auto worldPerPixel = 2.0f / (ViewScale() * static_cast<float>(framebufferHeight));
    centerX_ -= static_cast<float>(deltaX) * worldPerPixel;
    centerY_ += static_cast<float>(deltaY) * worldPerPixel;
}

} // namespace chmv::renderer
