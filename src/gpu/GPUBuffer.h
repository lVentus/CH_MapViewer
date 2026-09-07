#pragma once

#include <cstddef>
#include <cstdint>

namespace chmv::gpu {

class GPUBuffer {
public:
    GPUBuffer();
    ~GPUBuffer();

    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    GPUBuffer(GPUBuffer&& other) noexcept;
    GPUBuffer& operator=(GPUBuffer&& other) noexcept;

    void Allocate(std::size_t sizeBytes, const void* data = nullptr, std::uint32_t usage = 0);
    void Update(std::size_t offsetBytes, std::size_t sizeBytes, const void* data) const;
    void BindBase(std::uint32_t target, std::uint32_t binding) const;

    [[nodiscard]] std::uint32_t Id() const { return id_; }
    [[nodiscard]] std::size_t SizeBytes() const { return sizeBytes_; }

private:
    std::uint32_t id_ = 0;
    std::size_t sizeBytes_ = 0;
};

} // namespace chmv::gpu
