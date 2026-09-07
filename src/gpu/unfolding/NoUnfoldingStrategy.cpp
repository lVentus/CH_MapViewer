#include "gpu/unfolding/NoUnfoldingStrategy.h"

namespace chmv::gpu::unfolding {

UnfoldingOutput NoUnfoldingStrategy::Execute(const UnfoldingInput& input) {
    return {
        .edgeBuffer = input.inputEdgeBuffer,
        .drawCommandBuffer = input.inputDrawCommandBuffer,
    };
}

} // namespace chmv::gpu::unfolding
