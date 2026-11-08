#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chmv::benchmark {

// Distribution of repeated measurements, in milliseconds. Percentiles are nearest rank.
struct TimingStats {
    std::uint32_t samples = 0;
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double stddev = 0.0;
};

[[nodiscard]] inline TimingStats Summarize(std::vector<double> values) {
    TimingStats stats;
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    const auto count = values.size();
    const auto rank = [&](double q) {
        const auto index = static_cast<std::size_t>(std::ceil(q * static_cast<double>(count)));
        return values[std::clamp<std::size_t>(index, 1u, count) - 1u];
    };
    double sum = 0.0;
    for (const auto value : values) {
        sum += value;
    }
    stats.samples = static_cast<std::uint32_t>(count);
    stats.mean = sum / static_cast<double>(count);
    stats.median = count % 2u != 0u ? values[count / 2u]
                                    : (values[count / 2u - 1u] + values[count / 2u]) / 2.0;
    stats.min = values.front();
    stats.max = values.back();
    stats.p95 = rank(0.95);
    stats.p99 = rank(0.99);
    double squares = 0.0;
    for (const auto value : values) {
        squares += (value - stats.mean) * (value - stats.mean);
    }
    stats.stddev = std::sqrt(squares / static_cast<double>(count));
    return stats;
}

} // namespace chmv::benchmark
