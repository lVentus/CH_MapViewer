#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/lod/LodPageLists.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <cstdint>
#include <filesystem>

namespace chmv::gpu::unfolding {

// Binds what every expansion kernel reads and writes (lod_page_common.glsl and
// lod_page_segments.glsl): the pool, `inputList` at 2, segments, status, draw command and their
// uniforms. The program must be bound.
void BindExpansionKernel(std::uint32_t program, const UnfoldingInput& input,
                         std::uint32_t inputList);
// The rest for kernels that walk the hierarchy (lod_page_traversal.glsl).
void BindTraversalKernel(std::uint32_t program, const UnfoldingInput& input);

// One invocation per candidate, deciding from the record's intervals alone
// (lod_unfold_intervals.comp).
class IntervalKernel {
public:
    explicit IntervalKernel(const std::filesystem::path& shaderDirectory);

    // Without child intervals only roots are drawn.
    void Run(const UnfoldingInput& input, bool childIntervals) const;

private:
    ComputeProgram program_;
    lod::ListDispatcher lists_;
};

} // namespace chmv::gpu::unfolding
