#pragma once

#include "gpu/filtering/IRangeFilterStrategy.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace chmv::data {
class CHGraph;
}

namespace chmv::gpu::filtering {

class RangeFilterStrategyManager {
public:
    void Register(std::unique_ptr<IRangeFilterStrategy> strategy);
    void SetGraph(const data::CHGraph& graph);
    void Select(std::size_t index);

    [[nodiscard]] IRangeFilterStrategy& Current();
    [[nodiscard]] const IRangeFilterStrategy& Current() const;
    [[nodiscard]] std::size_t CurrentIndex() const { return currentIndex_; }
    [[nodiscard]] std::span<const std::unique_ptr<IRangeFilterStrategy>> Strategies() const {
        return strategies_;
    }

private:
    std::vector<std::unique_ptr<IRangeFilterStrategy>> strategies_;
    std::size_t currentIndex_ = 0;
};

} // namespace chmv::gpu::filtering
