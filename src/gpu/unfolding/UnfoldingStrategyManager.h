#pragma once

#include "gpu/unfolding/IUnfoldingStrategy.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace chmv::gpu::unfolding {

class UnfoldingStrategyManager {
public:
    void Register(std::unique_ptr<IUnfoldingStrategy> strategy);
    void Select(std::size_t index);

    [[nodiscard]] IUnfoldingStrategy& Current();
    [[nodiscard]] const IUnfoldingStrategy& Current() const;
    [[nodiscard]] std::size_t CurrentIndex() const { return currentIndex_; }
    [[nodiscard]] std::span<const std::unique_ptr<IUnfoldingStrategy>> Strategies() const {
        return strategies_;
    }

private:
    std::vector<std::unique_ptr<IUnfoldingStrategy>> strategies_;
    std::size_t currentIndex_ = 0;
};

} // namespace chmv::gpu::unfolding
