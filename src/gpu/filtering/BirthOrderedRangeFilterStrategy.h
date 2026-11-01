#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/filtering/IRangeFilterStrategy.h"
#include "gpu/lod/LodPageLists.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace chmv::gpu::filtering {

// Keeps the records of every resident page sorted by the highest level at which they can still
// be drawn (sorted on the GPU once per upload). The records of a level are then a prefix of
// that order, found by binary search, and only the prefixes are tested: coarse views touch a
// small part of each page.
class BirthOrderedRangeFilterStrategy final : public IRangeFilterStrategy {
public:
    explicit BirthOrderedRangeFilterStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Birth-ordered"; }
    void Reset() override;
    void Execute(const RangeFilterInput& input) override;

private:
    void SortNewPages(const RangeFilterInput& input);

    ComputeProgram sortProgram_;
    ComputeProgram countProgram_;
    ComputeProgram offsetsProgram_;
    ComputeProgram gatherProgram_;
    lod::ListDispatcher lists_;
    GPUBuffer order_;
    GPUBuffer sortList_;
    GPUBuffer prefix_;
    std::size_t sortListCapacity_ = 0;
    std::size_t prefixCapacity_ = 0;
    // Page every slot's order was built for.
    std::vector<std::uint32_t> orderedPages_;
    std::vector<std::uint32_t> sortSlots_;
    std::uint64_t poolGeneration_ = 0;
};

} // namespace chmv::gpu::filtering
