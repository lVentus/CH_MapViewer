#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/lod/LodPageLists.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <cstdint>
#include <filesystem>

namespace chmv::gpu::unfolding {

// One workgroup per 256 candidates walks their subtrees together from a queue in shared memory
// (lod_unfold_workgroup.comp): a long walk is shared by the whole workgroup instead of one
// invocation, the GPU balances the workgroups, and it takes a single dispatch.
class WorkgroupQueueUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    explicit WorkgroupQueueUnfoldingStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Adaptive workgroup queue"; }
    [[nodiscard]] geometry::RefinementMode Mode() const override {
        return geometry::RefinementMode::Adaptive;
    }
    [[nodiscard]] bool Traverses() const override { return true; }
    void Execute(const UnfoldingInput& input) override;

private:
    std::filesystem::path shaderDirectory_;
    ComputeProgram program_;
    lod::ListDispatcher lists_;
    // The overflow stack is sized for the hierarchy at hand, so the program is built on first
    // use.
    std::uint32_t stackSize_ = 0;
};

} // namespace chmv::gpu::unfolding
