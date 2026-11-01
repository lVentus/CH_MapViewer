#pragma once

#include <cstdint>

namespace chmv::data {

struct SystemMemoryInfo {
    std::uint64_t totalPhysicalBytes = 0;
    std::uint64_t availablePhysicalBytes = 0;
};

// Physical RAM as the OS reports it right now; zeros if it cannot tell.
[[nodiscard]] SystemMemoryInfo QuerySystemMemoryInfo();

} // namespace chmv::data
