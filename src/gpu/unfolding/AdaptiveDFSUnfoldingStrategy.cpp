#include "gpu/unfolding/AdaptiveDFSUnfoldingStrategy.h"

#include "gpu/unfolding/ExpansionKernels.h"

#include <glad/gl.h>

#include <algorithm>
#include <string>

namespace chmv::gpu::unfolding {
namespace {

// A walk never holds more than one pending sibling per depth, plus the node itself. Deeper
// hierarchies than this report stack overflows instead of using ever more local memory.
constexpr std::uint32_t kMaxStackSize = 128;
constexpr std::uint32_t kGroupSize = 64;

} // namespace

AdaptiveDFSUnfoldingStrategy::AdaptiveDFSUnfoldingStrategy(
    const std::filesystem::path& shaderDirectory)
    : shaderDirectory_(shaderDirectory), lists_(shaderDirectory) {}

void AdaptiveDFSUnfoldingStrategy::Execute(const UnfoldingInput& input) {
    const auto stackSize = std::clamp(input.hierarchyHeight + 1u, 2u, kMaxStackSize);
    if (stackSize != stackSize_) {
        program_.Load(shaderDirectory_ / "lod_unfold_dfs.comp",
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
