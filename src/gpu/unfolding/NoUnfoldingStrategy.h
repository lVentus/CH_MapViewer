#pragma once

#include "gpu/unfolding/ExpansionKernels.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <filesystem>

namespace chmv::gpu::unfolding {

// No expansion: the roots of the level are drawn as they are.
class NoUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    explicit NoUnfoldingStrategy(const std::filesystem::path& shaderDirectory)
        : kernel_(shaderDirectory) {}

    [[nodiscard]] std::string_view Name() const override { return "None"; }
    [[nodiscard]] geometry::RefinementMode Mode() const override {
        return geometry::RefinementMode::None;
    }
    void Execute(const UnfoldingInput& input) override { kernel_.Run(input, false); }

private:
    IntervalKernel kernel_;
};

} // namespace chmv::gpu::unfolding
