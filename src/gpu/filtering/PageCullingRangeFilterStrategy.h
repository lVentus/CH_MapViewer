#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/filtering/IRangeFilterStrategy.h"
#include "gpu/lod/LodPageLists.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace chmv::gpu::filtering {

// Lets the GPU choose the pages itself: every resident slot keeps a copy of its page's directory
// entry, one invocation per slot applies the planner's page test to the query, and only the
// pages that pass are scanned. The CPU slot list is not used, so the planner could in principle
// stop at choosing what to load.
class PageCullingRangeFilterStrategy final : public IRangeFilterStrategy {
public:
    explicit PageCullingRangeFilterStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "GPU page culling"; }
    void Reset() override;
    void Execute(const RangeFilterInput& input) override;

private:
    void UpdateEntries(const RangeFilterInput& input);

    ComputeProgram cullProgram_;
    ComputeProgram scanProgram_;
    lod::ListDispatcher lists_;
    GPUBuffer entryBuffer_;
    GPUBuffer slotList_;
    // CPU copy of the entry buffer, and the page every slot's entry was written for.
    std::vector<std::uint32_t> entries_;
    std::vector<std::uint32_t> entryPages_;
    std::uint64_t poolGeneration_ = 0;
    const streaming::lod::PageEntry* directory_ = nullptr;
};

} // namespace chmv::gpu::filtering
