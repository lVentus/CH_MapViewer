#include "gpu/filtering/RangeFilterStrategyManager.h"

#include "data/ch/CHGraph.h"

#include <stdexcept>
#include <utility>

namespace chmv::gpu::filtering {

void RangeFilterStrategyManager::Register(std::unique_ptr<IRangeFilterStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("cannot register a null range filter strategy");
    }
    strategies_.push_back(std::move(strategy));
}

void RangeFilterStrategyManager::SetGraph(const data::CHGraph& graph) {
    for (auto& strategy : strategies_) {
        strategy->SetGraph(graph);
        strategy->Reset();
    }
}

void RangeFilterStrategyManager::Select(std::size_t index) {
    if (index >= strategies_.size()) {
        throw std::out_of_range("range filter strategy index out of range");
    }
    if (currentIndex_ == index) {
        return;
    }
    currentIndex_ = index;
    strategies_[currentIndex_]->Reset();
}

IRangeFilterStrategy& RangeFilterStrategyManager::Current() {
    if (strategies_.empty()) {
        throw std::runtime_error("no range filter strategies are registered");
    }
    return *strategies_[currentIndex_];
}

const IRangeFilterStrategy& RangeFilterStrategyManager::Current() const {
    if (strategies_.empty()) {
        throw std::runtime_error("no range filter strategies are registered");
    }
    return *strategies_[currentIndex_];
}

} // namespace chmv::gpu::filtering
