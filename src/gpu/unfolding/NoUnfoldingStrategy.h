#pragma once

#include "gpu/unfolding/IUnfoldingStrategy.h"

namespace chmv::gpu::unfolding {

class NoUnfoldingStrategy final : public IUnfoldingStrategy {
public:
    [[nodiscard]] std::string_view Name() const override { return "No unfolding"; }
    [[nodiscard]] UnfoldingOutput Execute(const UnfoldingInput& input) override;
};

} // namespace chmv::gpu::unfolding
