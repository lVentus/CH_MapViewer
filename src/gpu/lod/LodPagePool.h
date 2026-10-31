#pragma once

#include "gpu/GPUBuffer.h"

#include <cstdint>
#include <span>
#include <vector>

namespace chmv::gpu::lod {

// Fixed-size page slots in one storage buffer, recycled least-recently-used first. A page that
// was touched in the current frame is never evicted in that frame.
class LodPagePool {
public:
    static constexpr std::uint32_t kNoSlot = 0xffffffffu;

    // slotCount is reduced to what one storage buffer binding can address on this GPU.
    void Initialize(std::uint32_t pageCount, std::uint32_t slotCount, std::uint32_t slotWords);

    void BeginFrame() { ++frame_; }
    [[nodiscard]] std::uint32_t SlotOf(std::uint32_t page) const { return slotOfPage_[page]; }
    [[nodiscard]] bool Resident(std::uint32_t page) const { return slotOfPage_[page] != kNoSlot; }
    [[nodiscard]] std::uint32_t PageOfSlot(std::uint32_t slot) const { return pageOfSlot_[slot]; }
    void Touch(std::uint32_t page);
    // Returns false when every slot is pinned by the current frame.
    bool Upload(std::uint32_t page, std::span<const std::uint32_t> words);

    [[nodiscard]] const GPUBuffer& Buffer() const { return buffer_; }
    [[nodiscard]] std::uint32_t SlotCount() const {
        return static_cast<std::uint32_t>(pageOfSlot_.size());
    }
    [[nodiscard]] std::uint32_t SlotWords() const { return slotWords_; }
    [[nodiscard]] std::uint32_t ResidentCount() const { return residentCount_; }
    [[nodiscard]] std::uint64_t Evictions() const { return evictions_; }

private:
    void Unlink(std::uint32_t slot);
    void PushFront(std::uint32_t slot);

    GPUBuffer buffer_;
    std::uint32_t slotWords_ = 0;
    std::vector<std::uint32_t> slotOfPage_;
    std::vector<std::uint32_t> pageOfSlot_;
    std::vector<std::uint64_t> lastUsed_;
    std::vector<std::uint32_t> previous_;
    std::vector<std::uint32_t> next_;
    std::vector<std::uint32_t> free_;
    std::uint32_t head_ = kNoSlot;
    std::uint32_t tail_ = kNoSlot;
    std::uint32_t residentCount_ = 0;
    std::uint64_t frame_ = 0;
    std::uint64_t evictions_ = 0;
};

} // namespace chmv::gpu::lod
