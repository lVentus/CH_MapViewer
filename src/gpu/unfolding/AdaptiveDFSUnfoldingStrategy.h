#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <cstdint>
#include <filesystem>

namespace chmv::gpu::unfolding {

class AdaptiveDFSUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    explicit AdaptiveDFSUnfoldingStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Adaptive DFS (screen-space)"; }
    [[nodiscard]] geometry::RefinementMode Mode() const override {
        return geometry::RefinementMode::Adaptive;
    }
    [[nodiscard]] UnfoldingOutput Execute(const UnfoldingInput& input) override;

private:
    void EnsureCapacity(std::uint32_t capacity);

    ComputeProgram prepareDispatchProgram_;
    ComputeProgram unfoldProgram_;
    ComputeProgram finalizeProgram_;
    GPUBuffer outputEdgeBuffer_;
    GPUBuffer outputCounterBuffer_;
    GPUBuffer statusBuffer_;
    GPUBuffer dispatchBuffer_;
    GPUBuffer indirectBuffer_;
    std::uint32_t capacity_ = 0;
};

} // namespace chmv::gpu::unfolding
