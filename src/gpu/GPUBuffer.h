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
    // On systems exposing ARB_buffer_storage, allocate a persistently mapped, coherent buffer and
    // hint that the backing store should remain client-accessible.  On UMA/iGPU systems this gives
    // the driver a portable opportunity to keep the SSBO in shared system memory so CPU producers
    // can write the same storage the GPU consumes.  Returns true when the shared path is active;
    // otherwise falls back to Allocate().
    [[nodiscard]] bool AllocateHostShared(std::size_t sizeBytes, const void* data = nullptr);
    void Update(std::size_t offsetBytes, std::size_t sizeBytes, const void* data) const;
    void BindBase(std::uint32_t target, std::uint32_t binding) const;

    [[nodiscard]] static bool HostSharedStorageSupported();
    [[nodiscard]] bool IsHostShared() const { return hostShared_; }
    [[nodiscard]] void* MappedData(std::size_t offsetBytes = 0) const;
    // Call after writing one or more ranges through MappedData().  Coherent persistent mappings do
    // not need glFlushMappedBufferRange, but a client-mapped-buffer barrier is still required before
    // subsequent GPU commands consume those writes.
    void CommitHostWrites() const;

    [[nodiscard]] std::uint32_t Id() const { return id_; }
    [[nodiscard]] std::size_t SizeBytes() const { return sizeBytes_; }

private:
    void DestroyStorage();
    void RecreateBufferObject();

    std::uint32_t id_ = 0;
    std::size_t sizeBytes_ = 0;
    void* mappedData_ = nullptr;
    bool hostShared_ = false;
    bool immutableStorage_ = false;
};

} // namespace chmv::gpu
