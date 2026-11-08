#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace chmv::gpu::unfolding {

// A fixed number of workgroups stays resident and takes nodes from one global queue, appending
// the children of what they expand, until nothing is pending (lod_unfold_queue.comp). One
// dispatch, and no invocation sits idle while work is left anywhere; every node costs a few
// global atomics instead.
class PersistentQueueUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    explicit PersistentQueueUnfoldingStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Adaptive persistent queue"; }
    [[nodiscard]] geometry::RefinementMode Mode() const override {
        return geometry::RefinementMode::Adaptive;
    }
    [[nodiscard]] bool Traverses() const override { return true; }
    void Execute(const UnfoldingInput& input) override;

private:
    ComputeProgram startProgram_;
    ComputeProgram program_;
    GPUBuffer queue_;
    std::size_t capacity_ = 0;
    std::size_t maxCapacity_ = 0;
};

} // namespace chmv::gpu::unfolding
