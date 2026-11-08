#pragma once

#include "gpu/ComputeProgram.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace chmv::gpu::lod {

// GPU lists of the filter and expansion passes share one layout, [count, groupsX, groupsY,
// groupsZ, items...], so a list carries the indirect dispatch arguments of the pass that reads
// it (see lod_page_common.glsl). Record items are slot << 8 | record.
inline constexpr std::uint32_t kListHeaderWords = 4;

// Counters a pass leaves in its status buffer; all zero means the result is complete.
enum class LodPageStatus : std::uint32_t {
    CandidateOverflow, // candidates beyond the list capacity were dropped
    SegmentOverflow,   // drawn segments beyond the segment capacity were dropped
    MissingPage,       // children skipped because their page is not resident (outside the
                       // view when it is complete, since the planner loads every page of it)
    StackOverflow,     // a depth-first walk ran out of stack
    FrontierOverflow,  // a breadth-first frontier or the work queue ran out of space
    Count,
};

inline constexpr std::size_t kStatusWords = 8;
static_assert(static_cast<std::size_t>(LodPageStatus::Count) <= kStatusWords);

// Workgroups beyond 65535 continue in the second dimension.
[[nodiscard]] std::array<std::uint32_t, 3> GroupGrid(std::uint64_t groups);

// Turns the count of a list into its dispatch arguments, on the GPU (lod_list_groups.comp).
class ListDispatcher {
public:
    explicit ListDispatcher(const std::filesystem::path& shaderDirectory);

    // Clamps the count of `list` to `capacity` items and stores the grid for workgroups of
    // `groupSize` items. A nonzero `next` list is emptied for the pass that appends to it.
    void Prepare(std::uint32_t list, std::uint32_t capacity, std::uint32_t groupSize,
                 std::uint32_t next = 0) const;
    // Dispatches the bound program with the grid stored in `list`.
    static void Dispatch(std::uint32_t list);

private:
    ComputeProgram program_;
};

} // namespace chmv::gpu::lod
