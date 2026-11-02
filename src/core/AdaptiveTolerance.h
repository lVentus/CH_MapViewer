#pragma once

#include "benchmark/GPUTimer.h"
#include "renderer/ToleranceController.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace chmv::renderer {
class MapCamera2D;
}

namespace chmv::core {

// What the last frames cost, by how it grows with the tolerance step (see ToleranceInput), and
// the step they drew.
struct FrameCosts {
    int drawnStep = 0;
    double passFixedMs = 0.0;
    double passScaledMs = 0.0;
    double drawScaledMs = 0.0;
    std::optional<std::array<std::uint64_t, renderer::kToleranceSteps>> counts;
};

// The screen tolerance of each frame: the manual one, or the one the tolerance controller picks
// from the measured frames; twice that while the camera moves, if asked for. Times every frame
// on the GPU from its first command to the end of its drawing.
class AdaptiveTolerance {
public:
    // Starts the frame's timing and returns its tolerance in pixels. `adaptive`: the expansion
    // uses a tolerance at all.
    [[nodiscard]] float BeginFrame(const renderer::MapCamera2D& camera, int width, int height,
                                   float manualPixels, bool adaptive, const FrameCosts& costs,
                                   double seconds);
    // Ends the frame's timing; `passRan`: it filtered and expanded, `drawnStep`: what it drew.
    void EndFrame(bool passRan, int drawnStep);

    [[nodiscard]] renderer::ToleranceSettings& Settings() { return settings_; }
    [[nodiscard]] const renderer::ToleranceStatus& Status() const { return controller_.Status(); }
    // Tolerance of the current frame, and whether it is doubled because the camera moves.
    [[nodiscard]] float Pixels() const { return pixels_; }
    [[nodiscard]] bool Coarsened() const { return coarsened_; }
    // The last measured frame, if any.
    [[nodiscard]] std::optional<double> LastFrameMs() const {
        return frameTimer_.LastMilliseconds();
    }

private:
    struct FrameRecord {
        std::uint64_t sequence = 0;
        bool passRan = false;
        int step = 0;
    };

    benchmark::GPUTimer frameTimer_;
    renderer::ToleranceController controller_;
    renderer::MotionCoarsening motion_;
    renderer::ToleranceSettings settings_;
    renderer::ToleranceMode mode_ = renderer::ToleranceMode::Manual;
    // Frames in flight, by measurement number, until their times arrive.
    std::array<FrameRecord, 16> frames_{};
    std::uint64_t usedSequence_ = 0;
    std::array<float, 5> lastCamera_{};
    // The step asked for, doubling included, to tell when it is drawn.
    int requestedStep_ = 0;
    float pixels_ = 1.0f;
    bool coarsened_ = false;
};

} // namespace chmv::core
