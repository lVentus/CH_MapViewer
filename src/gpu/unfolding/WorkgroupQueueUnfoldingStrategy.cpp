#include "gpu/unfolding/WorkgroupQueueUnfoldingStrategy.h"

#include "gpu/unfolding/ExpansionKernels.h"

#include <algorithm>
#include <string>

namespace chmv::gpu::unfolding {
namespace {

// As for the depth-first walk, which the overflow path is.
constexpr std::uint32_t kMaxStackSize = 128;
constexpr std::uint32_t kGroupSize = 256;

} // namespace

WorkgroupQueueUnfoldingStrategy::WorkgroupQueueUnfoldingStrategy(
    const std::filesystem::path& shaderDirectory)
    : shaderDirectory_(shaderDirectory), lists_(shaderDirectory) {}

void WorkgroupQueueUnfoldingStrategy::Execute(const UnfoldingInput& input) {
    const auto stackSize = std::clamp(input.hierarchyHeight + 1u, 2u, kMaxStackSize);
    if (stackSize != stackSize_) {
        program_.Load(shaderDirectory_ / "lod_unfold_workgroup.comp",
                      "#define STACK_SIZE " + std::to_string(stackSize) + "\n");
        stackSize_ = stackSize;
    }
    lists_.Prepare(input.candidateList, input.candidateCapacity, kGroupSize);
    program_.Bind();
    BindExpansionKernel(program_.Id(), input, input.candidateList);
    BindTraversalKernel(program_.Id(), input);
    lod::ListDispatcher::Dispatch(input.candidateList);
}

} // namespace chmv::gpu::unfolding
