#pragma once

#include "gpu/unfolding/ExpansionKernels.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <filesystem>

namespace chmv::gpu::unfolding {

// Expansion without traversal: the builder stored for every record the levels at which it is a
// root, the levels at which it lies below a parent, its own error and its parent's error. One
// invocation per candidate then knows on its own whether it belongs to the cut, so there is no
// stack, no frontier and no divergence.
class DualIntervalUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    explicit DualIntervalUnfoldingStrategy(const std::filesystem::path& shaderDirectory)
        : kernel_(shaderDirectory) {}

    [[nodiscard]] std::string_view Name() const override { return "Dual interval"; }
    [[nodiscard]] geometry::RefinementMode Mode() const override {
        return geometry::RefinementMode::Adaptive;
    }
    void Execute(const UnfoldingInput& input) override { kernel_.Run(input, true); }

private:
    IntervalKernel kernel_;
};

} // namespace chmv::gpu::unfolding
