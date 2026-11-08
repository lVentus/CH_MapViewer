#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/filtering/IRangeFilterStrategy.h"

#include <cstddef>
#include <filesystem>

namespace chmv::gpu::filtering {

// Counts first, then writes: one pass tests every record of the planned pages and keeps a mask
// and a count per page, a prefix sum turns the counts into offsets, and a second pass writes
// each candidate to its page's offset plus its rank in the mask. Unlike the full scan it needs
// no atomic to reserve room and its output order is fixed, at the price of a second pass (over
// the masks only) and the prefix sum.
class TwoPassRangeFilterStrategy final : public IRangeFilterStrategy {
public:
    explicit TwoPassRangeFilterStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Two-pass scatter"; }
    void Execute(const RangeFilterInput& input) override;

private:
    ComputeProgram maskProgram_;
    ComputeProgram offsetsProgram_;
    ComputeProgram scatterProgram_;
    GPUBuffer masks_;
    GPUBuffer prefix_;
    // Pages the buffers have room for.
    std::size_t capacity_ = 0;
};

} // namespace chmv::gpu::filtering
