#pragma once

#include <cstdint>

namespace chmv::renderer {

class MapCamera2D {
public:
    void Reset();
    void Zoom(float scrollSteps);
    void PanPixels(double deltaX, double deltaY, int framebufferHeight);

    [[nodiscard]] float CenterX() const { return centerX_; }
    [[nodiscard]] float CenterY() const { return centerY_; }
    [[nodiscard]] float ZoomFactor() const { return zoom_; }

private:
    float centerX_ = 0.0f;
    float centerY_ = 0.0f;
    float zoom_ = 1.0f;
};

} // namespace chmv::renderer
