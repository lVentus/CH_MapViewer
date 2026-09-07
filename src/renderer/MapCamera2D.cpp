#include "renderer/MapCamera2D.h"

#include <algorithm>
#include <cmath>

namespace chmv::renderer {
namespace {

constexpr float kZoomPerScrollStep = 1.25f;
constexpr float kMaxZoom = 1048576.0f;

} // namespace

void MapCamera2D::Reset() {
    centerX_ = 0.0f;
    centerY_ = 0.0f;
    zoom_ = 1.0f;
}

void MapCamera2D::Zoom(float scrollSteps) {
    zoom_ *= std::pow(kZoomPerScrollStep, scrollSteps);
    zoom_ = std::clamp(zoom_, 1.0f, kMaxZoom);
}

void MapCamera2D::PanPixels(double deltaX, double deltaY, int framebufferHeight) {
    if (framebufferHeight <= 0) {
        return;
    }

    const auto worldPerPixel = 2.0f / (zoom_ * static_cast<float>(framebufferHeight));
    centerX_ -= static_cast<float>(deltaX) * worldPerPixel;
    centerY_ += static_cast<float>(deltaY) * worldPerPixel;
}

} // namespace chmv::renderer
