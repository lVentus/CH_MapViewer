#pragma once

#include <cstdint>

namespace chmv::renderer {

class MapCamera2D {
public:
    // Root/road coordinates are normalized per dataset. Configure how many physical metres one
    // normalized coordinate unit represents so the public ZoomFactor has the same physical
    // meaning for BW, EUR, or any later dataset.
    void SetMetersPerNormalizedUnit(double metersPerNormalizedUnit);
    void Reset();
    void Zoom(float scrollSteps);
    void Update(float deltaSeconds);
    void PanPixels(double deltaX, double deltaY, int framebufferHeight);

    [[nodiscard]] float CenterX() const { return centerX_; }
    [[nodiscard]] float CenterY() const { return centerY_; }
    // Canonical physical zoom. Equal values mean equal approximate metres-per-pixel regardless of
    // the dataset bounds used by preprocessing.
    [[nodiscard]] float ZoomFactor() const { return zoom_; }
    [[nodiscard]] float TargetZoomFactor() const { return targetZoom_; }
    // Scale consumed by shaders in the dataset-normalized coordinate system.
    [[nodiscard]] float ViewScale() const;
    [[nodiscard]] double MetersPerNormalizedUnit() const { return metersPerNormalizedUnit_; }
    [[nodiscard]] double ViewHalfHeightMeters() const;

private:
    [[nodiscard]] float FitZoom() const;

    float centerX_ = 0.0f;
    float centerY_ = 0.0f;
    float zoom_ = 10.0f;
    float targetZoom_ = 10.0f;
    double metersPerNormalizedUnit_ = 2003750.8342789244;
};

} // namespace chmv::renderer
