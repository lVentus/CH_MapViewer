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
    [[nodiscard]] std::uint64_t ElapsedNanoseconds(std::uint64_t start,
                                                    std::uint64_t end) const;

    static constexpr std::size_t QueryCount = 8;
    std::array<std::uint32_t, QueryCount> startQueries_{};
    std::array<std::uint32_t, QueryCount> endQueries_{};
    std::array<bool, QueryCount> pending_{};
    std::array<std::uint64_t, QueryCount> sequence_{};
    std::size_t nextQuery_ = 0;
    int activeQuery_ = -1;
    int timestampBits_ = 0;
    std::uint64_t nextSequence_ = 1;
    std::uint64_t lastSequence_ = 0;
    std::optional<double> lastMilliseconds_;
};

} // namespace chmv::benchmark
