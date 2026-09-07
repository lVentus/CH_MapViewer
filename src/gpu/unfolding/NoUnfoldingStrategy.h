#pragma once

#include "gpu/unfolding/IUnfoldingStrategy.h"

namespace chmv::gpu::unfolding {

class NoUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    [[nodiscard]] std::string_view Name() const override { return "No unfolding"; }
    [[nodiscard]] bool FullyUnfolds() const override { return false; }
    [[nodiscard]] UnfoldingOutput Execute(const UnfoldingInput& input) override;
};

} // namespace chmv::gpu::unfolding
