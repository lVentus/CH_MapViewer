#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace chmv::benchmark {

class GPUTimer {
public:
    GPUTimer();
    ~GPUTimer();

    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    void Begin();
    void End();

    [[nodiscard]] std::optional<double> LastMilliseconds() const { return lastMilliseconds_; }

private:
    void CollectReadyResults();

    static constexpr std::size_t QueryCount = 4;
    std::array<std::uint32_t, QueryCount> queries_{};
    std::array<bool, QueryCount> pending_{};
    std::size_t nextQuery_ = 0;
    int activeQuery_ = -1;
    std::optional<double> lastMilliseconds_;
};

} // namespace chmv::benchmark
