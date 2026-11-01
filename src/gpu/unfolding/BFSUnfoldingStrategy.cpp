#include "gpu/unfolding/BFSUnfoldingStrategy.h"

#include "gpu/unfolding/ExpansionKernels.h"

#include <glad/gl.h>

#include <algorithm>

namespace chmv::gpu::unfolding {
namespace {

constexpr std::uint32_t kGroupSize = 256;

} // namespace

BFSUnfoldingStrategy::BFSUnfoldingStrategy(const std::filesystem::path& shaderDirectory,
                                           geometry::RefinementMode mode)
    : mode_(mode), program_(shaderDirectory / "lod_unfold_bfs.comp"), lists_(shaderDirectory) {}

void BFSUnfoldingStrategy::Execute(const UnfoldingInput& input) {
    // A frontier never holds more nodes than are drawn in the end, give or take the culled
    // children of the last expansions; overflows are counted in the status.
    const std::size_t capacity = std::max(input.segmentCapacity, 1024u);
    if (capacity > frontierCapacity_) {
        for (auto& frontier : frontiers_) {
            frontier.Allocate((lod::kListHeaderWords + capacity) * 4u);
        }
        frontierCapacity_ = capacity;
    }

    const auto id = program_.Id();
    const auto run = [&](std::uint32_t current, std::uint32_t currentCapacity,
                         const GPUBuffer& next, bool fromCandidates) {
        lists_.Prepare(current, currentCapacity, kGroupSize, next.Id());
        program_.Bind();
        BindExpansionKernel(id, input, current);
        BindTraversalKernel(id, input);
        next.BindBase(GL_SHADER_STORAGE_BUFFER, 7);
        glUniform1ui(glGetUniformLocation(id, "uFromCandidates"), fromCandidates ? 1u : 0u);
        glUniform1ui(glGetUniformLocation(id, "uFrontierCapacity"),
                     static_cast<std::uint32_t>(frontierCapacity_));
        lod::ListDispatcher::Dispatch(current);
    };

    // The roots are at depth 0 and nothing lies deeper than the hierarchy height; one more
    // dispatch than strictly needed costs nothing when its frontier is empty.
    run(input.candidateList, input.candidateCapacity, frontiers_[0], true);
    for (std::uint32_t depth = 1; depth <= input.hierarchyHeight + 1u; ++depth) {
        const auto& current = frontiers_[(depth - 1u) % 2u];
        run(current.Id(), static_cast<std::uint32_t>(frontierCapacity_),
            frontiers_[depth % 2u], false);
    }
}

} // namespace chmv::gpu::unfolding
