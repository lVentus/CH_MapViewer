#include "renderer/ToleranceController.h"

#include <algorithm>
#include <cmath>

namespace chmv::renderer {
namespace {

constexpr int kCoarsest = kToleranceSteps - 1;
// A step fits when its predicted frame time leaves 10% of the budget for noise; a finer step is
// only taken when it leaves 25%, so that noise in the predictions does not flip between two.
constexpr double kFitShare = 0.9;
constexpr double kRefineShare = 0.75;
// Finer again only after the frames stayed below 70% of the budget this long, one step at a time.
constexpr double kCalmShare = 0.7;
constexpr double kCalmSeconds = 0.5;
constexpr double kRefineSeconds = 0.1;
// Going back within this time after making a step finer means that step was too slow.
constexpr double kFailSeconds = 1.0;
constexpr double kMaxBackoffSeconds = 16.0;
constexpr double kBackoffResetSeconds = 5.0;
// Without counts: what one step is taken to save (about a quarter per octave).
constexpr double kStepSaving = 0.93;
// GPU times arrive a few frames late: costs count as those of the drawn step only after it has
// been drawn this many frames.
constexpr int kSettleUpdates = 6;
// Weight of a new measurement in the running averages.
constexpr double kLearnRate = 0.2;
// Feedback: the frame time it aims at, and the gains in steps per budget of error.
constexpr double kFeedbackShare = 0.85;
constexpr double kFeedbackProportional = 2.0;
constexpr double kFeedbackIntegral = 3.0;
// Coarser while moving lasts this long after the camera stops.
constexpr double kMotionHoldSeconds = 0.15;

[[nodiscard]] int ClampStep(int step) {
    return std::clamp(step, 0, kCoarsest);
}

} // namespace

float StepPixels(int step) {
    return static_cast<float>(std::exp2(static_cast<double>(step) / 4.0));
}

int NearestStep(float pixels) {
    const auto step = std::lround(4.0 * std::log2(std::max(static_cast<double>(pixels), 1e-3)));
    return ClampStep(static_cast<int>(step));
}

int StepOfThreshold(std::uint32_t threshold, std::uint32_t pixelCode) {
    const auto codes = static_cast<double>(threshold) - static_cast<double>(pixelCode);
    return static_cast<int>(std::lround(codes / kCodesPerToleranceStep));
}

void ToleranceController::SetMode(ToleranceMode mode) {
    mode_ = mode;
}

void ToleranceController::Reset(int step) {
    step_ = ClampStep(step);
    integral_ = static_cast<double>(step_) / kFeedbackIntegral;
    calmSeconds_ = 0.0;
    lastSampleCalm_ = false;
    lastChange_ = now_;
    lastRefine_ = -1e9;
    refinePending_ = false;
    blockedStep_ = -1;
    backoffSeconds_ = 1.0;
}

double ToleranceController::Ratio(const ToleranceInput& input, int step) const {
    // Counts only exist for the steps themselves; doubled while moving, the drawn step can lie
    // beyond them.
    const auto drawn = input.drawnStep;
    if (input.counts && drawn == ClampStep(drawn)) {
        const auto& counts = *input.counts;
        const auto base = counts[static_cast<std::size_t>(drawn)];
        if (base != 0u) {
            return static_cast<double>(counts[static_cast<std::size_t>(step)]) /
                   static_cast<double>(base);
        }
    }
    return std::pow(kStepSaving, step - drawn);
}

void ToleranceController::Learn(const ToleranceInput& input) {
    if (!input.sample) {
        return;
    }
    const auto& sample = *input.sample;
    status_.lastFrameMs = sample.frameMs;
    // Only frames of the current kind tell whether there is room: standing still, the one pass
    // that a finer step itself causes is no reason to wait.
    if (sample.passRan == input.moving) {
        lastSampleCalm_ = sample.frameMs <= kCalmShare * input.budgetMs;
    }
    // The costs belong to the drawn step once it has settled; a frame of another step says
    // nothing about them.
    if (sample.step != input.drawnStep || settledUpdates_ < kSettleUpdates) {
        return;
    }
    if (!sample.passRan) {
        const auto other = std::max(sample.frameMs - input.drawScaledMs, 0.0);
        other_ = other_ ? *other_ + kLearnRate * (other - *other_) : other;
        return;
    }
    const auto pass = input.passFixedMs + input.passScaledMs;
    if (other_ && pass > 0.01) {
        const auto correction = (sample.frameMs - *other_ - input.drawScaledMs) / pass;
        passCorrection_ =
            std::clamp(passCorrection_ + kLearnRate * (correction - passCorrection_), 0.5, 4.0);
    }
}

void ToleranceController::Predict(const ToleranceInput& input) {
    const auto other = other_.value_or(0.0);
    for (int step = 0; step < kToleranceSteps; ++step) {
        const auto ratio = Ratio(input, step);
        const auto draw = input.drawScaledMs * ratio;
        const auto index = static_cast<std::size_t>(step);
        status_.stillMs[index] = other + draw;
        status_.movingMs[index] =
            other + passCorrection_ * (input.passFixedMs + input.passScaledMs * ratio) + draw;
    }
    status_.otherMs = other;
    status_.passCorrection = passCorrection_;
    status_.counted = input.counts.has_value();
}

void ToleranceController::UpdatePredictive(const ToleranceInput& input) {
    calmSeconds_ = lastSampleCalm_ ? calmSeconds_ + input.seconds : 0.0;
    if (refinePending_ && input.requestDrawn) {
        refinePending_ = false;
        lastRefine_ = now_;
    }
    // Predictions only from settled costs; until then only measured frames count.
    const bool settled = settledUpdates_ >= kSettleUpdates;
    const auto& predicted = input.moving ? status_.movingMs : status_.stillMs;
    int wanted = step_;
    if (settled) {
        wanted = kCoarsest;
        for (int step = 0; step < kToleranceSteps; ++step) {
            if (predicted[static_cast<std::size_t>(step)] <= kFitShare * input.budgetMs) {
                wanted = step;
                break;
            }
        }
    }
    // A frame of this step and kind that was measured too slow: coarser at least by one.
    if (input.sample && input.sample->step == step_ && input.sample->passRan == input.moving &&
        input.sample->frameMs > input.budgetMs) {
        wanted = std::max(wanted, std::min(step_ + 1, kCoarsest));
    }

    if (wanted > step_) {
        if (now_ - lastRefine_ < kFailSeconds && input.moving == refinedWhileMoving_) {
            blockedStep_ = step_;
            blockedWhileMoving_ = input.moving;
            blockedUntil_ = now_ + backoffSeconds_;
            backoffSeconds_ = std::min(backoffSeconds_ * 2.0, kMaxBackoffSeconds);
            lastRefine_ = -1e9;
        }
        step_ = wanted;
        lastChange_ = now_;
        calmSeconds_ = 0.0;
        refinePending_ = false;
    } else if (wanted < step_ && input.requestDrawn && calmSeconds_ >= kCalmSeconds &&
               now_ - lastChange_ >= kRefineSeconds &&
               predicted[static_cast<std::size_t>(step_ - 1)] <= kRefineShare * input.budgetMs &&
               !(step_ - 1 == blockedStep_ && input.moving == blockedWhileMoving_ &&
                 now_ < blockedUntil_)) {
        --step_;
        lastChange_ = now_;
        lastRefine_ = -1e9;
        refinePending_ = true;
        refinedWhileMoving_ = input.moving;
    }
    // The step that was too slow held this time: it gets a fresh start.
    if (step_ == blockedStep_ && lastRefine_ > 0.0 && now_ - lastRefine_ > kBackoffResetSeconds) {
        blockedStep_ = -1;
        backoffSeconds_ = 1.0;
    }
}

void ToleranceController::UpdateFeedback(const ToleranceInput& input) {
    if (!input.sample || input.budgetMs <= 0.0) {
        return;
    }
    const auto error = input.sample->frameMs / input.budgetMs - kFeedbackShare;
    integral_ = std::clamp(integral_ + error * input.seconds, 0.0,
                           static_cast<double>(kCoarsest) / kFeedbackIntegral);
    const auto output = std::clamp(kFeedbackProportional * error + kFeedbackIntegral * integral_,
                                   0.0, static_cast<double>(kCoarsest));
    step_ = static_cast<int>(std::lround(output));
}

int ToleranceController::Update(const ToleranceInput& input) {
    now_ += input.seconds;
    if (input.drawnStep != lastDrawnStep_) {
        lastDrawnStep_ = input.drawnStep;
        settledUpdates_ = 0;
    } else {
        ++settledUpdates_;
    }
    status_.budgetMs = input.budgetMs;
    Learn(input);
    Predict(input);
    switch (mode_) {
    case ToleranceMode::Manual:
        step_ = ClampStep(input.drawnStep);
        break;
    case ToleranceMode::Predictive:
        UpdatePredictive(input);
        break;
    case ToleranceMode::Feedback:
        UpdateFeedback(input);
        break;
    }
    status_.step = step_;
    return step_;
}

bool MotionCoarsening::Update(bool moving, double seconds) {
    stillSeconds_ = moving ? 0.0 : stillSeconds_ + seconds;
    return stillSeconds_ < kMotionHoldSeconds;
}

} // namespace chmv::renderer
