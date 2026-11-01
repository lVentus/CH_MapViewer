#pragma once

#include <cstdint>

namespace chmv::gpu {

enum class GPUMemoryArchitecture : std::uint8_t {
    Unknown,
    Dedicated,
    Unified,
};

struct GPUMemoryInfo {
    GPUMemoryArchitecture architecture = GPUMemoryArchitecture::Unknown;
    std::uint64_t reportedAvailableBytes = 0;
    std::uint64_t reportedTotalBytes = 0;
};

// OpenGL has no universal dedicated-vs-shared memory query: this combines the vendor and
// renderer strings with the memory extensions some drivers offer. Needs a current context.
[[nodiscard]] GPUMemoryInfo QueryGpuMemoryInfo();

} // namespace chmv::gpu
