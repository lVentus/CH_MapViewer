#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/lod/LodPageLists.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <cstdint>
#include <filesystem>

namespace chmv::gpu::unfolding {

// One invocation per root walks its subtree depth first with a private stack, skipping subtrees
// outside the view and drawing every node within the screen tolerance.
class AdaptiveDFSUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    explicit AdaptiveDFSUnfoldingStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Adaptive DFS"; }
    [[nodiscard]] geometry::RefinementMode Mode() const override {
        return geometry::RefinementMode::Adaptive;
    }
    [[nodiscard]] bool Traverses() const override { return true; }
    void Execute(const UnfoldingInput& input) override;

private:
    std::filesystem::path shaderDirectory_;
    ComputeProgram program_;
    lod::ListDispatcher lists_;
    // The stack is sized for the hierarchy at hand, so the program is built on first use.
    std::uint32_t stackSize_ = 0;
};

} // namespace chmv::gpu::unfolding
