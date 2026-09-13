#pragma once

#include "geometry/GeometryRefinement.h"

#include <cstdint>
#include <string_view>

namespace chmv::gpu::unfolding {

struct UnfoldingInput {
    std::uint32_t edgeBuffer = 0;
    std::uint32_t edgeCount = 0;
    std::uint32_t inputEdgeBuffer = 0;
    std::uint32_t inputDrawCommandBuffer = 0;
    std::uint32_t outputCapacity = 0;
    float screenPixelScale = 0.0f;
    float maxScreenErrorPixels = 1.0f;
};

struct UnfoldingOutput {
    std::uint32_t edgeBuffer = 0;
    std::uint32_t drawCommandBuffer = 0;
};

class IUnfoldingStrategy {
public:
    virtual ~IUnfoldingStrategy() = default;

    [[nodiscard]] virtual std::string_view Name() const = 0;
    [[nodiscard]] virtual geometry::RefinementMode Mode() const = 0;
    [[nodiscard]] bool RefinesGeometry() const { return Mode() != geometry::RefinementMode::None; }
    [[nodiscard]] bool FullyUnfolds() const { return Mode() == geometry::RefinementMode::Full; }
    [[nodiscard]] virtual UnfoldingOutput Execute(const UnfoldingInput& input) = 0;
};

} // namespace chmv::gpu::unfolding
