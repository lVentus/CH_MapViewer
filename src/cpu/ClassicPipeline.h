#pragma once

#include "cpu/CpuPasses.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPagePlanner.h"

#include <cstdint>
#include <memory>
#include <stop_token>
#include <vector>

namespace chmv::cpu {

class CpuWorkers;

// The classic CPU method, independent of the pages: the whole CH in RAM (read from the CH layer
// of the page file), roots found by their [death, birth] interval, expanded with a stack (DFS)
// or a frontier (BFS). It cuts with the same error codes and culls with the same error bounds
// as the page traversals, so inside the view it draws exactly the same edges.
class ClassicGraph {
public:
    // RAM it needs for a file, while loading included.
    [[nodiscard]] static std::uint64_t EstimateBytes(const streaming::lod::FileHeader& header);
    // Throws when the file cannot be read; stops early (and throws) when stop is requested.
    [[nodiscard]] static std::unique_ptr<ClassicGraph> Load(const streaming::lod::LodPageFile& file,
                                                            const std::stop_token& stop);

    // Filters: every edge (FullScan), the edges ordered by birth level (BirthOrdered), or a loose
    // grid of the edges' boxes, grown by their error, per cell ordered by birth (Culling).
    // Expansions: Roots, Dfs, Bfs (the dual-interval rule needs the page records).
    [[nodiscard]] CpuResult Run(CpuWorkers& workers, const streaming::lod::LodPageQuery& query,
                                CpuFilter filter, CpuExpansion expansion) const;

    [[nodiscard]] std::uint64_t Bytes() const;

private:
    // Loose multi-level grid: an edge lives in the one cell of the finest level whose cell is at
    // least as large as its grown box, the cell holding the box's centre, so a cell's edges stay
    // within half a cell of it.
    struct Grid {
        std::int64_t originX = 0;
        std::int64_t originY = 0;
        // Cells of level k are 2^(shift - k) fixed units wide.
        std::uint32_t shift = 0;
        std::uint32_t levels = 0;
        std::vector<std::uint64_t> levelStart;
        // First entry of every cell, one more at the end.
        std::vector<std::uint32_t> cellStart;
        // Edges, per cell highest birth first, with their birth levels.
        std::vector<std::uint32_t> entries;
        std::vector<std::int16_t> entryBirth;
    };

    ClassicGraph() = default;
    void DeriveStyles();
    void BuildFilters();
    [[nodiscard]] bool Alive(std::uint32_t edge, std::uint32_t level) const;
    [[nodiscard]] bool Visible(std::uint32_t edge, const std::array<std::int32_t, 4>& cull) const;
    // The edge's chord box grown by its error bound.
    [[nodiscard]] std::array<std::int64_t, 4> GrownBox(std::uint32_t edge) const;

    std::vector<std::int32_t> x_;
    std::vector<std::int32_t> y_;
    std::vector<std::uint32_t> source_;
    std::vector<std::uint32_t> target_;
    std::vector<std::uint32_t> childA_;
    std::vector<std::uint32_t> childB_;
    std::vector<std::int16_t> birth_;
    std::vector<std::int16_t> death_;
    std::vector<std::uint16_t> errorCode_;
    std::vector<std::uint8_t> type_;
    // Edges that can be roots, highest birth first; byBirthCount_[R] of them have birth >= R.
    std::vector<std::uint32_t> byBirth_;
    std::vector<std::uint32_t> byBirthCount_;
    Grid grid_;
};

} // namespace chmv::cpu
