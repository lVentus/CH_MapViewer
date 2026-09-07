#include "gpu/unfolding/UnfoldingStrategyManager.h"

#include <stdexcept>

namespace chmv::gpu::unfolding {

void UnfoldingStrategyManager::Register(std::unique_ptr<IUnfoldingStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("cannot register a null unfolding strategy");
    }
    strategies_.push_back(std::move(strategy));
}

void UnfoldingStrategyManager::Select(std::size_t index) {
    if (index >= strategies_.size()) {
        throw std::out_of_range("unfolding strategy index out of range");
    }
    currentIndex_ = index;
}

IUnfoldingStrategy& UnfoldingStrategyManager::Current() {
    if (strategies_.empty()) {
        throw std::runtime_error("no unfolding strategies are registered");
    }
    return *strategies_[currentIndex_];
}

const IUnfoldingStrategy& UnfoldingStrategyManager::Current() const {
    if (strategies_.empty()) {
        throw std::runtime_error("no unfolding strategies are registered");
    }
    return *strategies_[currentIndex_];
}

} // namespace chmv::gpu::unfolding
