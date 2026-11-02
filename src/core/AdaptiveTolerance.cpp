#include "core/AdaptiveTolerance.h"

#include "renderer/MapCamera2D.h"

#include <glad/gl.h>

#include <algorithm>

namespace chmv::core {

float AdaptiveTolerance::BeginFrame(const renderer::MapCamera2D& camera, int width, int height,
                                    float manualPixels, bool adaptive, const FrameCosts& costs,
                                    double seconds) {
    // Begin collects the times that have arrived. The start stamp goes to the GPU right away, so
    // that CPU work before the frame's first drawing (a CPU pass) counts too.
    frameTimer_.Begin();
    glFlush();
    const std::array<float, 5> view{camera.CenterX(), camera.CenterY(), camera.ZoomFactor(),
                                    static_cast<float>(width), static_cast<float>(height)};
    const bool moving = view != lastCamera_;
    lastCamera_ = view;

    if (settings_.mode != mode_) {
        // Automatic modes start from the manual tolerance, or from where the other one was.
        const bool wasAutomatic = mode_ != renderer::ToleranceMode::Manual;
        controller_.SetMode(settings_.mode);
        controller_.Reset(wasAutomatic ? controller_.Status().step
                                       : renderer::NearestStep(manualPixels));
        mode_ = settings_.mode;
    }
    renderer::ToleranceInput input;
    input.seconds = seconds;
    input.budgetMs = 1000.0 / std::clamp(static_cast<double>(settings_.targetFps), 1.0, 1000.0);
    input.moving = moving;
    input.drawnStep = costs.drawnStep;
    input.requestDrawn = costs.drawnStep == requestedStep_;
    input.passFixedMs = costs.passFixedMs;
    input.passScaledMs = costs.passScaledMs;
    input.drawScaledMs = costs.drawScaledMs;
    input.counts = costs.counts;
    const auto sequence = frameTimer_.LastSequence();
    const auto& record = frames_[sequence % frames_.size()];
    if (sequence > usedSequence_ && record.sequence == sequence) {
        if (const auto frameMs = frameTimer_.LastMilliseconds()) {
            input.sample = renderer::ToleranceSample{*frameMs, record.passRan, record.step};
        }
        usedSequence_ = sequence;
    }
    // Only adaptive expansions draw with a tolerance; the others say nothing about it.
    const auto step = adaptive ? controller_.Update(input) : controller_.Status().step;

    const bool automatic = adaptive && mode_ != renderer::ToleranceMode::Manual;
    const auto pixels = automatic ? renderer::StepPixels(step) : manualPixels;
    coarsened_ = motion_.Update(moving, seconds) && settings_.coarserWhileMoving && adaptive;
    pixels_ = coarsened_ ? pixels * 2.0f : pixels;
    requestedStep_ = step + (coarsened_ ? renderer::kMotionSteps : 0);
    return pixels_;
}

void AdaptiveTolerance::EndFrame(bool passRan, int drawnStep) {
    const auto sequence = frameTimer_.End();
    if (sequence != 0u) {
        frames_[sequence % frames_.size()] = {sequence, passRan, drawnStep};
    }
}

} // namespace chmv::core
