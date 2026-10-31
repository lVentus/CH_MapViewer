#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace chmv::gpu::lod {
struct LodPageRuntimeStats;
}

namespace chmv::renderer {
class MapCamera2D;
}

namespace chmv::benchmark {

// Scripted camera path for the LOD page runtime: wait until the overview is complete, zoom to
// street level at a steady rate, hold, zoom back out, hold. Every frame goes to a CSV next to
// the dataset; the summary answers "how long until the view is complete" and "what does a frame
// cost" for each part of the path.
class LodPageZoomSweep {
public:
    void Start(const std::filesystem::path& csvPath, renderer::MapCamera2D& camera);
    void Cancel();
    // Replaces user camera input while running.
    void Advance(renderer::MapCamera2D& camera, float deltaSeconds);
    // Records the frame that was just produced; frameMs is the time since the previous frame.
    void Record(const gpu::lod::LodPageRuntimeStats& stats, const renderer::MapCamera2D& camera,
                double frameMs);

    [[nodiscard]] bool Running() const { return phase_ != Phase::Idle; }
    [[nodiscard]] const char* PhaseName() const { return Name(phase_); }
    [[nodiscard]] const std::string& Summary() const { return summary_; }
    [[nodiscard]] const std::filesystem::path& CsvPath() const { return csvPath_; }

private:
    enum class Phase { Idle, Settle, ZoomIn, HoldNear, ZoomOut, HoldFar };
    static constexpr std::size_t kPhaseCount = 6;

    struct PhaseStats {
        std::vector<double> frameMs;
        std::uint64_t incompleteFrames = 0;
        std::uint64_t uploadedBytes = 0;
        std::uint32_t maxSegments = 0;
        double completeAfterMs = -1.0;
    };

    [[nodiscard]] static const char* Name(Phase phase);
    [[nodiscard]] float TargetZoom() const;
    void Enter(Phase phase);
    void Finish();

    Phase phase_ = Phase::Idle;
    double phaseSeconds_ = 0.0;
    double totalSeconds_ = 0.0;
    float overviewZoom_ = 1.0f;
    std::filesystem::path csvPath_;
    std::ofstream csv_;
    std::array<PhaseStats, kPhaseCount> stats_{};
    std::string summary_;
};

} // namespace chmv::benchmark
