#include "benchmark/LodPageZoomSweep.h"

#include "cpu/CpuPasses.h"
#include "gpu/lod/LodPageRuntime.h"
#include "renderer/MapCamera2D.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace chmv::benchmark {
namespace {

constexpr double kSettleTimeoutSeconds = 20.0;
constexpr double kZoomSeconds = 12.0;
constexpr double kHoldSeconds = 3.0;
// 1000x is a 3 km vertical span (MapCamera2D calibration).
constexpr float kNearZoom = 1000.0f;
constexpr float kZoomPerScrollStep = 1.03f;

[[nodiscard]] double Percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1u));
    return values[index];
}

} // namespace

const char* LodPageZoomSweep::Name(Phase phase) {
    switch (phase) {
    case Phase::Idle:
        return "idle";
    case Phase::Settle:
        return "overview";
    case Phase::ZoomIn:
        return "zoom in";
    case Phase::HoldNear:
        return "hold near";
    case Phase::ZoomOut:
        return "zoom out";
    case Phase::HoldFar:
        return "hold far";
    }
    return "?";
}

void LodPageZoomSweep::Start(const std::filesystem::path& csvPath, renderer::MapCamera2D& camera) {
    camera.Reset();
    overviewZoom_ = camera.ZoomFactor();
    csvPath_ = csvPath;
    csv_ = std::ofstream(csvPath);
    csv_ << "time_s,phase,zoom,level,draw_level,threshold,required_pages,missing_pages,"
            "draw_pages,segments,uploaded_bytes,plan_ms,upload_ms,update_ms,gpu_filter_ms,"
            "gpu_unfold_ms,gpu_draw_ms,frame_ms,complete,cpu_level,cpu_threshold,cpu_segments,"
            "cpu_prepare_ms,cpu_filter_ms,cpu_expand_ms,cpu_output_ms,cpu_threads,tolerance_px,"
            "frame_gpu_ms\n";
    stats_ = {};
    summary_.clear();
    totalSeconds_ = 0.0;
    Enter(Phase::Settle);
}

void LodPageZoomSweep::Cancel() {
    if (Running()) {
        csv_.close();
        phase_ = Phase::Idle;
        summary_ = "Cancelled.";
    }
}

void LodPageZoomSweep::Enter(Phase phase) {
    phase_ = phase;
    phaseSeconds_ = 0.0;
}

float LodPageZoomSweep::TargetZoom() const {
    const auto ratio = static_cast<double>(kNearZoom) / static_cast<double>(overviewZoom_);
    const auto t = std::clamp(phaseSeconds_ / kZoomSeconds, 0.0, 1.0);
    switch (phase_) {
    case Phase::ZoomIn:
        return static_cast<float>(overviewZoom_ * std::pow(ratio, t));
    case Phase::HoldNear:
        return kNearZoom;
    case Phase::ZoomOut:
        return static_cast<float>(overviewZoom_ * std::pow(ratio, 1.0 - t));
    default:
        return overviewZoom_;
    }
}

void LodPageZoomSweep::Advance(renderer::MapCamera2D& camera, float deltaSeconds) {
    if (!Running()) {
        return;
    }
    phaseSeconds_ += deltaSeconds;
    totalSeconds_ += deltaSeconds;
    const auto steps =
        std::log(TargetZoom() / camera.TargetZoomFactor()) / std::log(kZoomPerScrollStep);
    camera.Zoom(steps);
    camera.Update(deltaSeconds);
}

void LodPageZoomSweep::Record(const SweepFrame& frame, const renderer::MapCamera2D& camera,
                              double frameMs) {
    if (!Running()) {
        return;
    }
    // GPU columns are -1 when only the CPU runs; CPU columns are -1 when it has no result, and
    // its timings when it did not run this frame (the view did not change).
    const gpu::lod::LodPageRuntimeStats none;
    const auto& stats = frame.gpu != nullptr ? *frame.gpu : none;
    const auto gpu = [&](double value) { return frame.gpu != nullptr ? value : -1.0; };
    const auto gpuCount = [&](std::uint64_t value) {
        return frame.gpu != nullptr ? static_cast<long long>(value) : -1LL;
    };
    const auto* result =
        frame.cpu != nullptr && frame.cpu->unavailable.empty() ? frame.cpu : nullptr;
    const auto cpuCount = [&](std::uint64_t value) {
        return result != nullptr ? static_cast<long long>(value) : -1LL;
    };
    const auto* timed = frame.cpuComputed ? result : nullptr;
    const auto cpuTime = [&](double value) { return timed != nullptr ? value : -1.0; };
    // Complete once the GPU has every page it needs and the CPU (when it runs) has a result.
    const bool complete =
        (frame.gpu == nullptr || stats.complete) && (frame.cpu == nullptr || result != nullptr);
    const auto segments = frame.gpu != nullptr ? std::uint64_t{stats.drawnSegments}
                          : result != nullptr  ? std::uint64_t{result->segments.size()}
                                               : 0u;
    csv_ << std::fixed << std::setprecision(4) << totalSeconds_ << ',' << Name(phase_) << ','
         << camera.ZoomFactor() << ',' << gpuCount(stats.level) << ','
         << gpuCount(stats.drawLevel) << ',' << gpuCount(stats.threshold) << ','
         << gpuCount(stats.requiredPages) << ',' << gpuCount(stats.missingRequiredPages) << ','
         << gpuCount(stats.drawPages) << ',' << gpuCount(stats.drawnSegments) << ','
         << gpuCount(stats.uploadedBytes) << ',' << gpu(stats.planMs) << ','
         << gpu(stats.uploadMs) << ',' << gpu(stats.updateMs) << ','
         << gpu(stats.gpuFilterMs.value_or(-1.0)) << ','
         << gpu(stats.gpuUnfoldMs.value_or(-1.0)) << ',' << gpu(stats.gpuDrawMs.value_or(-1.0))
         << ',' << frameMs << ',' << (complete ? 1 : 0) << ','
         << cpuCount(result != nullptr ? result->level : 0u) << ','
         << cpuCount(result != nullptr ? result->threshold : 0u) << ','
         << cpuCount(result != nullptr ? result->segments.size() : 0u) << ','
         << cpuTime(timed != nullptr ? timed->prepareMs : 0.0) << ','
         << cpuTime(timed != nullptr ? timed->filterMs : 0.0) << ','
         << cpuTime(timed != nullptr ? timed->expandMs : 0.0) << ','
         << cpuTime(timed != nullptr ? timed->outputMs : 0.0) << ','
         << (frame.cpu != nullptr ? static_cast<int>(frame.cpu->threads) : -1) << ','
         << frame.tolerancePixels << ',' << frame.frameGpuMs.value_or(-1.0) << '\n';

    auto& phaseStats = stats_[static_cast<std::size_t>(phase_)];
    phaseStats.frameMs.push_back(frameMs);
    phaseStats.incompleteFrames += complete ? 0u : 1u;
    phaseStats.uploadedBytes += stats.uploadedBytes;
    phaseStats.maxSegments =
        std::max<std::uint64_t>(phaseStats.maxSegments, segments);
    if (phaseStats.completeAfterMs < 0.0 && complete) {
        phaseStats.completeAfterMs = phaseSeconds_ * 1000.0;
    }

    switch (phase_) {
    case Phase::Settle:
        if (complete || phaseSeconds_ > kSettleTimeoutSeconds) {
            Enter(Phase::ZoomIn);
        }
        break;
    case Phase::ZoomIn:
        if (phaseSeconds_ >= kZoomSeconds) {
            Enter(Phase::HoldNear);
        }
        break;
    case Phase::HoldNear:
        if (phaseSeconds_ >= kHoldSeconds) {
            Enter(Phase::ZoomOut);
        }
        break;
    case Phase::ZoomOut:
        if (phaseSeconds_ >= kZoomSeconds) {
            Enter(Phase::HoldFar);
        }
        break;
    case Phase::HoldFar:
        if (phaseSeconds_ >= kHoldSeconds) {
            Finish();
        }
        break;
    case Phase::Idle:
        break;
    }
}

void LodPageZoomSweep::Finish() {
    csv_.close();
    phase_ = Phase::Idle;

    std::ostringstream summary;
    summary << std::fixed << std::setprecision(1);
    for (std::size_t i = 1; i < kPhaseCount; ++i) {
        const auto& phaseStats = stats_[i];
        const auto frames = phaseStats.frameMs.size();
        if (frames == 0u) {
            continue;
        }
        const auto mean =
            std::accumulate(phaseStats.frameMs.begin(), phaseStats.frameMs.end(), 0.0) /
            static_cast<double>(frames);
        summary << Name(static_cast<Phase>(i)) << ": " << frames << " frames, frame " << mean
                << " ms mean / " << Percentile(phaseStats.frameMs, 0.95) << " p95 / "
                << Percentile(phaseStats.frameMs, 1.0) << " max, incomplete "
                << 100.0 * static_cast<double>(phaseStats.incompleteFrames) /
                       static_cast<double>(frames)
                << "%, uploaded "
                << static_cast<double>(phaseStats.uploadedBytes) / (1024.0 * 1024.0)
                << " MiB, up to " << phaseStats.maxSegments << " segments";
        if (i == 1u || i == 3u || i == 5u) {
            summary << ", complete after ";
            if (phaseStats.completeAfterMs >= 0.0) {
                summary << phaseStats.completeAfterMs << " ms";
            } else {
                summary << "never";
            }
        }
        summary << '\n';
    }
    summary << "Per-frame log: " << csvPath_.string();
    summary_ = summary.str();
    std::cout << "[LOD page zoom sweep]\n" << summary_ << '\n';
}

} // namespace chmv::benchmark
