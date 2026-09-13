#include "renderer/MapCamera2D.h"

#include <algorithm>
#include <cmath>

namespace chmv::renderer {
namespace {

constexpr float kOverviewZoom = 10.0f;
constexpr float kZoomPerScrollStep = 1.03f;
constexpr float kZoomResponse = 14.0f;
constexpr float kMaxZoom = 1048576.0f;

} // namespace

void MapCamera2D::Reset() {
    centerX_ = 0.0f;
    centerY_ = 0.0f;
    zoom_ = kOverviewZoom;
    targetZoom_ = kOverviewZoom;
}

float MapCamera2D::ViewScale() const {
    return zoom_ / kOverviewZoom;
}

void MapCamera2D::Zoom(float scrollSteps) {
    targetZoom_ *= std::pow(kZoomPerScrollStep, scrollSteps);
    targetZoom_ = std::clamp(targetZoom_, 1.0f, kMaxZoom);
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
