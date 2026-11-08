#pragma once

#include "benchmark/GPUTimer.h"
#include "benchmark/TimingStats.h"
#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/GraphicsProgram.h"
#include "gpu/lod/LodPageLists.h"
#include "renderer/RoadStyle.h"
#include "renderer/ToleranceController.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPagePlanner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace chmv::gpu::filtering {
class IRangeFilterStrategy;
}

namespace chmv::gpu::unfolding {
class IUnfoldingStrategy;
}

namespace chmv::renderer {
class MapCamera2D;
}

namespace chmv::gpu::lod {

class LodPagePool;

// What one filter and expansion pass works on: the resident slots of the planned pages, and the
// directory and query (level, threshold, box) they were planned with. cullSegments drops drawn
// segments whose chord misses the query box.
struct LodPagePass {
    std::span<const std::uint32_t> slots;
    std::span<const streaming::lod::PageEntry> directory;
    streaming::lod::LodPageQuery query;
    std::uint32_t hierarchyHeight = 0;
    bool cullSegments = false;
};

// Repeated passes of the selected strategies after a few unmeasured ones, timed two ways: GPU
// timestamps around the filter and the expansion of every pass, and the wall time of batches of
// passes ended by glFinish (per pass; dependable where timestamps are not, as on some
// integrated GPUs). Drawing the result is timed the same two ways.
struct LodPageBenchmarkResult {
    std::uint32_t iterations = 0;
    std::uint32_t pages = 0;
    std::uint32_t candidates = 0;
    std::uint32_t segments = 0;
    benchmark::TimingStats filterGpu;
    benchmark::TimingStats unfoldGpu;
    benchmark::TimingStats passGpu;
    benchmark::TimingStats passWall;
    benchmark::TimingStats drawGpu;
    benchmark::TimingStats drawWall;
};

// GPU half of the LOD page runtime: the selected filter and expansion turn resident pages into a
// segment list, which a vertex-pulled draw renders.
class LodPageRenderer {
public:
    explicit LodPageRenderer(const std::filesystem::path& shaderDirectory);
    ~LodPageRenderer();

    LodPageRenderer(const LodPageRenderer&) = delete;
    LodPageRenderer& operator=(const LodPageRenderer&) = delete;

    // Only needed when the pass or a strategy changed. With stepBase (the error code of one
    // pixel) it also counts the segments each tolerance step would draw from the same pages.
    void Process(const LodPagePool& pool, const LodPagePass& pass,
                 filtering::IRangeFilterStrategy& filter, unfolding::IUnfoldingStrategy& unfold,
                 std::optional<std::uint32_t> stepBase = std::nullopt);
    void Draw(const LodPagePool& pool, const renderer::MapCamera2D& camera, int width, int height,
              const renderer::RoadStyleConfig& styles);
    // Runs the pass and the draw `iterations` times each (and the batches again) and waits for
    // the timings.
    [[nodiscard]] LodPageBenchmarkResult
    Benchmark(const LodPagePool& pool, const LodPagePass& pass,
              filtering::IRangeFilterStrategy& filter, unfolding::IUnfoldingStrategy& unfold,
              std::uint32_t iterations, const renderer::MapCamera2D& camera, int width, int height,
              const renderer::RoadStyleConfig& styles);
    // Collects finished GPU results without waiting; they are a few frames old.
    void Poll();
    // True once after a finished pass ran out of room; the buffers have grown, so it is worth
    // running again.
    [[nodiscard]] bool TakeRetry();

    [[nodiscard]] std::uint32_t DrawnSegments() const { return drawnSegments_; }
    [[nodiscard]] std::uint32_t Candidates() const { return candidates_; }
    [[nodiscard]] const std::array<std::uint32_t, kStatusWords>& Status() const { return status_; }
    [[nodiscard]] std::optional<double> FilterMilliseconds() const {
        return filterTimer_.LastMilliseconds();
    }
    [[nodiscard]] std::optional<double> UnfoldMilliseconds() const {
        return unfoldTimer_.LastMilliseconds();
    }
    [[nodiscard]] std::optional<double> DrawMilliseconds() const {
        return drawTimer_.LastMilliseconds();
    }
    [[nodiscard]] std::optional<double> CountMilliseconds() const {
        return countTimer_.LastMilliseconds();
    }
    // Segments per tolerance step of the last finished pass, if it counted them.
    [[nodiscard]] const std::optional<std::array<std::uint32_t, renderer::kToleranceSteps>>&
    StepCounts() const {
        return stepCounts_;
    }

    // Synchronous copies of the last pass, for validation.
    [[nodiscard]] std::vector<std::uint32_t> ReadSegments() const;
    [[nodiscard]] std::vector<std::uint32_t> ReadCandidates() const;
    [[nodiscard]] std::array<std::uint32_t, kStatusWords> ReadStatus() const;

private:
    struct Readback {
        GPUBuffer buffer;
        void* fence = nullptr;
        std::uint32_t segmentScale = 0;
        bool counted = false;
    };

    struct Capacities {
        std::uint32_t candidates = 0;
        std::uint32_t segments = 0;
    };

    void EnsureCapacity(std::size_t slotCount, std::size_t poolSlots);
    // Uploads the slot list and empties the candidates, segments and status.
    [[nodiscard]] Capacities StartPass(const LodPagePool& pool, const LodPagePass& pass);
    void Filter(const LodPagePool& pool, const LodPagePass& pass, Capacities capacities,
                filtering::IRangeFilterStrategy& filter);
    void Unfold(const LodPagePool& pool, const LodPagePass& pass, Capacities capacities,
                unfolding::IUnfoldingStrategy& unfold);
    void CountSteps(const LodPagePool& pool, const LodPagePass& pass, std::uint32_t stepBase);

    GraphicsProgram roadProgram_;
    ComputeProgram finishProgram_;
    ComputeProgram countProgram_;
    GPUBuffer slotList_;
    GPUBuffer candidateList_;
    GPUBuffer segmentBuffer_;
    GPUBuffer commandBuffer_;
    GPUBuffer statusBuffer_;
    GPUBuffer errorBounds_;
    GPUBuffer stepCountBuffer_;
    std::vector<std::uint32_t> slotListWords_;
    // Pages the lists have room for; segments get segmentScale_ times the records of those,
    // all within the largest storage block the GPU binds.
    std::size_t slotCapacity_ = 0;
    std::size_t candidateCapacity_ = 0;
    std::size_t segmentCapacity_ = 0;
    std::uint32_t segmentScale_ = 1;
    std::uint32_t allocatedScale_ = 0;
    std::int64_t maxBlockBytes_ = 0;
    std::uint32_t vertexArray_ = 0;
    benchmark::GPUTimer filterTimer_;
    benchmark::GPUTimer unfoldTimer_;
    benchmark::GPUTimer drawTimer_;
    benchmark::GPUTimer countTimer_;
    std::array<Readback, 3> readbacks_;
    std::size_t nextReadback_ = 0;
    std::uint32_t drawnSegments_ = 0;
    std::uint32_t candidates_ = 0;
    std::array<std::uint32_t, kStatusWords> status_{};
    std::optional<std::array<std::uint32_t, renderer::kToleranceSteps>> stepCounts_;
    bool retry_ = false;
};

} // namespace chmv::gpu::lod
