#pragma once

namespace chmv::geometry {

enum class RefinementMode {
    None,
    Full,
    Adaptive,
};

struct RefinementParameters {
    RefinementMode mode = RefinementMode::None;
    float screenPixelScale = 0.0f;
    float maxScreenErrorPixels = 1.0f;

    [[nodiscard]] bool Active() const { return mode != RefinementMode::None; }
    friend bool operator==(const RefinementParameters&, const RefinementParameters&) = default;
};

} // namespace chmv::geometry
