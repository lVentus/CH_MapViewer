#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/lod/LodPageLists.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <array>
#include <cstddef>
#include <filesystem>

namespace chmv::gpu::unfolding {

// Breadth first, one hierarchy depth per dispatch: the nodes of a depth are spread over all
// invocations and their children collected in the next frontier, so no invocation waits for a
// long walk of another. Full goes down to the original edges, Adaptive stops at the screen
// tolerance.
class BFSUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    BFSUnfoldingStrategy(const std::filesystem::path& shaderDirectory,
                         geometry::RefinementMode mode);

    [[nodiscard]] std::string_view Name() const override {
        return mode_ == geometry::RefinementMode::Full ? "Full BFS" : "Adaptive BFS";
    }
    [[nodiscard]] geometry::RefinementMode Mode() const override { return mode_; }
    [[nodiscard]] bool Traverses() const override { return true; }
    void Execute(const UnfoldingInput& input) override;

private:
    geometry::RefinementMode mode_;
    ComputeProgram program_;
    lod::ListDispatcher lists_;
    std::array<GPUBuffer, 2> frontiers_;
    std::size_t frontierCapacity_ = 0;
};

} // namespace chmv::gpu::unfolding
