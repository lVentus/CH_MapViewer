#pragma once

#include "gpu/GPUBuffer.h"

#include <cstdint>
#include <span>
#include <vector>

namespace chmv::gpu::lod {

// Fixed-size page slots in one storage buffer, recycled least-recently-used first. A page that
// was touched in the current frame is never evicted in that frame. With children words, every
// slot also holds the page's children block behind the page, and a second buffer tells the GPU
// which slot each page is in, so traversals can follow references between pages.
class LodPagePool {
public:
    static constexpr std::uint32_t kNoSlot = 0xffffffffu;

    // slotCount is reduced to what one storage buffer binding can address on this GPU.
    void Initialize(std::uint32_t pageCount, std::uint32_t slotCount, std::uint32_t pageWords,
                    std::uint32_t childWords = 0);

    void BeginFrame() { ++frame_; }
    [[nodiscard]] std::uint32_t SlotOf(std::uint32_t page) const { return slotOfPage_[page]; }
    [[nodiscard]] bool Resident(std::uint32_t page) const { return slotOfPage_[page] != kNoSlot; }
    [[nodiscard]] std::uint32_t PageOfSlot(std::uint32_t slot) const { return pageOfSlot_[slot]; }
    void Touch(std::uint32_t page);
    // Returns false when every slot is pinned by the current frame.
    bool Upload(std::uint32_t page, std::span<const std::uint32_t> words,
                std::span<const std::uint32_t> children = {});

    // Binds the pool for a kernel using lod_page_common.glsl (words at 0, records at 1,
    // uSlotWords) and, with children, lod_page_traversal.glsl (slot of every page at 6,
    // uPageWords). The program must be bound.
    void BindForKernel(std::uint32_t program) const;

    [[nodiscard]] const GPUBuffer& Buffer() const { return buffer_; }
    // Slot of every page, kNoSlot if not resident; only with children words.
    [[nodiscard]] const GPUBuffer& SlotOfPageBuffer() const { return slotOfPageBuffer_; }
    [[nodiscard]] std::uint32_t SlotCount() const {
        return static_cast<std::uint32_t>(pageOfSlot_.size());
    }
    [[nodiscard]] std::uint32_t PageCount() const {
        return static_cast<std::uint32_t>(slotOfPage_.size());
    }
    [[nodiscard]] std::uint32_t SlotWords() const { return slotWords_; }
    // Where the children block starts inside a slot.
    [[nodiscard]] std::uint32_t ChildOffsetWords() const { return pageWords_; }
    [[nodiscard]] bool HasChildren() const { return slotWords_ > pageWords_; }
    [[nodiscard]] std::uint32_t ResidentCount() const { return residentCount_; }
    [[nodiscard]] std::uint64_t Evictions() const { return evictions_; }
    // Changes with every Initialize (of any pool), so per-slot caches know when to start over.
    [[nodiscard]] std::uint64_t Generation() const { return generation_; }

private:
    void Unlink(std::uint32_t slot);
    void PushFront(std::uint32_t slot);
    void PublishSlot(std::uint32_t page, std::uint32_t slot);

    GPUBuffer buffer_;
    GPUBuffer slotOfPageBuffer_;
    std::uint32_t slotWords_ = 0;
    std::uint32_t pageWords_ = 0;
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
    std::uint64_t generation_ = 0;
};

} // namespace chmv::gpu::lod
