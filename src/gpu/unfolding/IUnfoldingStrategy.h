#pragma once

#include <cstdint>
#include <string_view>

namespace chmv::gpu::unfolding {

struct UnfoldingInput {
    std::uint32_t edgeBuffer = 0;
    std::uint32_t edgeCount = 0;
    std::uint32_t inputEdgeBuffer = 0;
    std::uint32_t inputDrawCommandBuffer = 0;
    std::uint32_t outputCapacity = 0;
};

struct UnfoldingOutput {
    std::uint32_t edgeBuffer = 0;
    std::uint32_t drawCommandBuffer = 0;
};

class IUnfoldingStrategy {
public:
    virtual ~IUnfoldingStrategy() = default;

    [[nodiscard]] virtual std::string_view Name() const = 0;
    [[nodiscard]] virtual UnfoldingOutput Execute(const UnfoldingInput& input) = 0;
};

} // namespace chmv::gpu::unfolding
