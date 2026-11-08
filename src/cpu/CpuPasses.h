#pragma once

// What the CPU passes compute, whichever method they use: the drawn segments with their CH
// edges, and how long each step took. They draw exactly what the GPU draws for the same view,
// level and threshold (compared inside the view, where both are complete).

#include "geometry/GeometryRefinement.h"
#include "renderer/ToleranceController.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace chmv::cpu {

// Counterparts of the GPU filters: test every record (FullScan), only the prefix of records
// ordered by their highest level (BirthOrdered), pick pages / grid cells by their box and
// levels before testing records (Culling), or count per page first and then write every
// candidate to its place (TwoPass).
enum class CpuFilter {
    FullScan,
    BirthOrdered,
    Culling,
    TwoPass,
};

// Counterparts of the GPU expansions: roots as they are, the dual-interval rule (Intervals,
// needs page records), or walking the hierarchy depth first, a depth at a time, or from one
// work queue shared by all threads (Queue).
enum class CpuExpansion {
    Roots,
    Intervals,
    Dfs,
    Bfs,
    Queue,
};

// The CPU counterpart of a GPU filter + expansion choice; empty where there is none yet.
// cullSegments drops drawn segments whose chord misses the view, as on the GPU.
struct CpuStrategy {
    std::optional<CpuFilter> filter;
    std::optional<CpuExpansion> expansion;
    geometry::RefinementMode mode = geometry::RefinementMode::None;
    bool cullSegments = false;
};

// Who computes what is drawn: the GPU pipeline, a CPU method (the GPU only rasterizes), or both
// side by side (GPU left, CPU right) and compared.
enum class ProcessingMode {
    Gpu,
    Cpu,
    Compare,
};

// Which CPU method runs: the same pages the GPU uses, or the classic one on the whole CH.
enum class CpuMethod {
    Pages,
    Classic,
};

// Fixed coordinates of both ends and the road style type.
struct CpuSegment {
    std::int32_t ax = 0;
    std::int32_t ay = 0;
    std::int32_t bx = 0;
    std::int32_t by = 0;
    std::uint32_t type = 0;
};

static_assert(sizeof(CpuSegment) == 20);

struct CpuResult {
    std::vector<CpuSegment> segments;
    // The CH edge of every segment.
    std::vector<std::uint32_t> edges;
    std::uint64_t candidates = 0;
    std::uint32_t pages = 0;
    // Children in pages the view did not need (they lie outside it, as on the GPU).
    std::uint64_t missingPages = 0;
    // Planning and getting pages (page method only; on the GPU side this is the planner and
    // the upload), the filter, the expansion, and building the segment list for drawing.
    double prepareMs = 0.0;
    double filterMs = 0.0;
    double expandMs = 0.0;
    double outputMs = 0.0;
    unsigned threads = 1;
    // Level and error code of the query this result answers.
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
    // Segments each tolerance step would draw from the same pages, when asked for (page method
    // only), and the time counting took.
    std::optional<std::array<std::uint64_t, renderer::kToleranceSteps>> stepCounts;
    double countMs = 0.0;
    // Why nothing was computed (no counterpart, data still loading, ...).
    std::string unavailable;

    [[nodiscard]] double TotalMs() const { return prepareMs + filterMs + expandMs + outputMs; }
};

// The CH edges of the segments whose chord touches the view rectangle (normalized
// coordinates), sorted: the part of a result that any two methods must agree on.
[[nodiscard]] std::vector<std::uint32_t> EdgesInView(std::span<const CpuSegment> segments,
                                                     std::span<const std::uint32_t> edges,
                                                     const std::array<double, 4>& rectangle);

// Chord of a segment against the view rectangle, as the GPU result is checked.
[[nodiscard]] bool ChordTouches(std::int32_t ax, std::int32_t ay, std::int32_t bx, std::int32_t by,
                                const std::array<double, 4>& rectangle);

// Chord of a segment against a cull box in fixed coordinates, as segment culling tests it
// (lod_page_chord.glsl).
[[nodiscard]] bool ChordInCullBox(std::int32_t ax, std::int32_t ay, std::int32_t bx,
                                  std::int32_t by, const std::array<std::int32_t, 4>& cull);

// Sorted multisets of edges: how many of `expected` are missing from `actual`, and how many of
// `actual` are not in `expected`.
struct EdgeDifference {
    std::uint64_t missing = 0;
    std::uint64_t extra = 0;
};

// GPU against CPU inside the view, both computed for the same query.
struct CpuGpuComparison {
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
    std::uint64_t gpuInView = 0;
    std::uint64_t cpuInView = 0;
    std::uint64_t onlyGpu = 0;
    std::uint64_t onlyCpu = 0;
    // Why the two were not compared (GPU result truncated, page file unreadable); empty when
    // they were.
    std::string notCompared;

    [[nodiscard]] bool Identical() const {
        return notCompared.empty() && onlyGpu == 0u && onlyCpu == 0u;
    }
};

[[nodiscard]] EdgeDifference CompareEdges(std::span<const std::uint32_t> expected,
                                          std::span<const std::uint32_t> actual);

} // namespace chmv::cpu
