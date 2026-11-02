#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace chmv::renderer {

// Screen tolerances the automatic modes choose from: step k is 2^(k/4) pixels, so 1 px to 8 px
// in quarter octaves. One step is 256 error codes (1024 codes per octave, LodPageFormat.h), so
// the error code of step k is the code of one pixel plus 256 k.
inline constexpr int kToleranceSteps = 13;
inline constexpr std::uint32_t kCodesPerToleranceStep = 256;
// Coarser while moving: twice the tolerance, four steps.
inline constexpr int kMotionSteps = 4;

[[nodiscard]] float StepPixels(int step);
// The step nearest to a tolerance, within 0 .. kToleranceSteps - 1.
[[nodiscard]] int NearestStep(float pixels);
// The step an error threshold stands for, given the code of one pixel.
[[nodiscard]] int StepOfThreshold(std::uint32_t threshold, std::uint32_t pixelCode);

enum class ToleranceMode { Manual, Predictive, Feedback };

// What the panel sets: manual or automatic tolerance, the frame rate the automatic modes aim at,
// and twice the tolerance while the camera moves.
struct ToleranceSettings {
    ToleranceMode mode = ToleranceMode::Manual;
    float targetFps = 60.0f;
    bool coarserWhileMoving = false;
};

// One measured frame: its time from the first to the last GPU timestamp of the frame (so CPU work
// the GPU waited for inside the frame counts too), whether it ran a pass (the view changed) and
// the step it drew.
struct ToleranceSample {
    double frameMs = 0.0;
    bool passRan = false;
    int step = 0;
};

// What the controller gets every frame.
struct ToleranceInput {
    double seconds = 0.0;
    // 1000 / target frame rate.
    double budgetMs = 0.0;
    // The camera moves, so the coming frames run a pass each.
    bool moving = false;
    // Step of what is drawn now; the costs below are measured on it once it has been drawn for a
    // few frames (GPU times arrive late).
    int drawnStep = 0;
    // The step asked for last time is the one drawn (its pages have arrived); only then is a
    // finer step worth trying.
    bool requestDrawn = true;
    // Last measured costs (ms) by how they grow with the segments: a pass has a part that does
    // not (filter, counting) and one that does (expansion); drawing does too.
    double passFixedMs = 0.0;
    double passScaledMs = 0.0;
    double drawScaledMs = 0.0;
    // Segments each step would draw in the current view. Missing when they cannot be counted;
    // then each step is taken to save a few percent.
    std::optional<std::array<std::uint64_t, kToleranceSteps>> counts;
    // A frame measured since the last update.
    std::optional<ToleranceSample> sample;
};

// What the controller decided and why, for the panel.
struct ToleranceStatus {
    int step = 0;
    bool counted = false;
    double budgetMs = 0.0;
    double lastFrameMs = 0.0;
    // Time of a frame outside pass and drawing (UI, clearing, waiting for the CPU).
    double otherMs = 0.0;
    // Measured over predicted extra time of frames that run a pass (planning, uploads).
    double passCorrection = 1.0;
    // Predicted frame time per step, with a pass and without.
    std::array<double, kToleranceSteps> movingMs{};
    std::array<double, kToleranceSteps> stillMs{};
};

// Chooses the screen tolerance from the measured frame time, so that frames stay within the
// budget of the target frame rate: 1 px when the machine is fast enough, coarser otherwise.
//
// Predictive: from the segments each step would draw and the measured cost of the current one,
// it predicts every step's frame time. When the frames get too slow (or are about to: the camera
// starts moving) it goes at once to the finest step predicted within 90% of the budget; when
// there is room again (0.5 s below 70% of the budget) it goes finer one step every 100 ms, as
// long as the next step is predicted within 75%. A finer step that turned out too slow is not
// tried again for a while, twice as long each time.
//
// Feedback: a PI controller on the measured frame time alone, for comparison.
class ToleranceController {
public:
    void SetMode(ToleranceMode mode);
    [[nodiscard]] ToleranceMode Mode() const { return mode_; }
    // Starts over from this step (the manual tolerance when switching to automatic).
    void Reset(int step);
    // The step for the coming frame.
    int Update(const ToleranceInput& input);
    [[nodiscard]] const ToleranceStatus& Status() const { return status_; }

private:
    void Learn(const ToleranceInput& input);
    void Predict(const ToleranceInput& input);
    void UpdatePredictive(const ToleranceInput& input);
    void UpdateFeedback(const ToleranceInput& input);
    [[nodiscard]] double Ratio(const ToleranceInput& input, int step) const;

    ToleranceMode mode_ = ToleranceMode::Manual;
    ToleranceStatus status_;
    int step_ = 0;
    double now_ = 0.0;
    std::optional<double> other_;
    double passCorrection_ = 1.0;
    // Time the measured frames have stayed below 70% of the budget.
    double calmSeconds_ = 0.0;
    bool lastSampleCalm_ = false;
    double lastChange_ = -1e9;
    // The drawn step and for how many updates it has not changed.
    int lastDrawnStep_ = -1000;
    int settledUpdates_ = 0;
    // When the last finer step was first drawn; a quick way back marks it as too slow. Pending
    // while its pages are still on their way.
    double lastRefine_ = -1e9;
    bool refinePending_ = false;
    bool refinedWhileMoving_ = false;
    // A step too slow while moving may still do standing still, and the other way round.
    int blockedStep_ = -1;
    bool blockedWhileMoving_ = false;
    double blockedUntil_ = 0.0;
    double backoffSeconds_ = 1.0;
    // Feedback state.
    double integral_ = 0.0;
};

// Twice the tolerance while the camera moves and for a moment after it stops.
class MotionCoarsening {
public:
    [[nodiscard]] bool Update(bool moving, double seconds);

private:
    double stillSeconds_ = 1e9;
};

} // namespace chmv::renderer
