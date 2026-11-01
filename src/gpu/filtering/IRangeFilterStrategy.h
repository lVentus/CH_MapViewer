#pragma once

#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPagePlanner.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace chmv::gpu::lod {
class LodPagePool;
}

namespace chmv::gpu::filtering {

// One filter pass over resident LOD pages. A filter finds the candidates of a level: every
// record of the planned pages whose root or child level interval contains the level, as a list
// of slot << 8 | record (LodPageLists.h). Which of them are drawn is the expansion's business,
// so every filter must find the same set; they differ only in the work spent on it.
struct RangeFilterInput {
    const lod::LodPagePool* pool = nullptr;
    // Pool slots of the pages the CPU planner chose for the query, also as a GPU list whose
    // grid has one workgroup per slot.
    std::span<const std::uint32_t> slots;
    std::uint32_t slotList = 0;
    // Directory (indexed by page) and query the planner chose those pages with.
    std::span<const streaming::lod::PageEntry> directory;
    streaming::lod::LodPageQuery query;
    // Empty on entry; room for slots.size() * kPageRecordCapacity items.
    std::uint32_t candidateList = 0;
    std::uint32_t candidateCapacity = 0;
    std::uint32_t statusBuffer = 0;
};

class IRangeFilterStrategy {
public:
    virtual ~IRangeFilterStrategy() = default;

    [[nodiscard]] virtual std::string_view Name() const = 0;
    // Forgets everything derived from earlier passes.
    virtual void Reset() {}
    virtual void Execute(const RangeFilterInput& input) = 0;
};

} // namespace chmv::gpu::filtering
