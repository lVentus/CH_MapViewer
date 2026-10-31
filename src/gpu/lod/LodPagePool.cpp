#include "gpu/lod/LodPagePool.h"

#include <glad/gl.h>

#include <algorithm>
#include <stdexcept>

namespace chmv::gpu::lod {
namespace {

// The whole pool is bound as one storage block, and the shaders address it with 32-bit word
// offsets and 24-bit slot numbers.
[[nodiscard]] std::uint32_t MaxSlots(std::uint32_t slotWords) {
    GLint64 maxBlockBytes = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockBytes);
    const auto slotBytes = std::uint64_t{slotWords} * 4u;
    auto limit = std::min<std::uint64_t>(0xffffffffull / slotWords, 1ull << 24u);
    if (maxBlockBytes > 0) {
        limit = std::min(limit, static_cast<std::uint64_t>(maxBlockBytes) / slotBytes);
    }
    return static_cast<std::uint32_t>(limit);
}

} // namespace

void LodPagePool::Initialize(std::uint32_t pageCount, std::uint32_t slotCount,
                             std::uint32_t slotWords) {
    slotCount = std::max(1u, std::min(slotCount, MaxSlots(slotWords)));
    slotWords_ = slotWords;
    buffer_.Allocate(static_cast<std::size_t>(slotCount) * slotWords * 4u);
    slotOfPage_.assign(pageCount, kNoSlot);
    pageOfSlot_.assign(slotCount, kNoSlot);
    lastUsed_.assign(slotCount, 0);
    previous_.assign(slotCount, kNoSlot);
    next_.assign(slotCount, kNoSlot);
    free_.resize(slotCount);
    for (std::uint32_t i = 0; i < slotCount; ++i) {
        free_[i] = slotCount - 1u - i;
    }
    head_ = kNoSlot;
    tail_ = kNoSlot;
    residentCount_ = 0;
    frame_ = 1;
    evictions_ = 0;
}

void LodPagePool::Touch(std::uint32_t page) {
    const auto slot = slotOfPage_[page];
    if (slot == kNoSlot || lastUsed_[slot] == frame_) {
        return;
    }
    lastUsed_[slot] = frame_;
    Unlink(slot);
    PushFront(slot);
}

bool LodPagePool::Upload(std::uint32_t page, std::span<const std::uint32_t> words) {
    if (words.size() > slotWords_) {
        throw std::runtime_error("LOD page larger than a pool slot");
    }
    if (slotOfPage_[page] != kNoSlot) {
        Touch(page);
        return true;
    }

    std::uint32_t slot = kNoSlot;
    if (!free_.empty()) {
        slot = free_.back();
        free_.pop_back();
        ++residentCount_;
    } else {
        if (tail_ == kNoSlot || lastUsed_[tail_] == frame_) {
            return false;
        }
        slot = tail_;
        Unlink(slot);
        slotOfPage_[pageOfSlot_[slot]] = kNoSlot;
        ++evictions_;
    }

    buffer_.Update(static_cast<std::size_t>(slot) * slotWords_ * 4u, words.size() * 4u,
                   words.data());
    slotOfPage_[page] = slot;
    pageOfSlot_[slot] = page;
    lastUsed_[slot] = frame_;
    PushFront(slot);
    return true;
}

void LodPagePool::Unlink(std::uint32_t slot) {
    const auto before = previous_[slot];
    const auto after = next_[slot];
    if (before != kNoSlot) {
        next_[before] = after;
    } else if (head_ == slot) {
        head_ = after;
    }
    if (after != kNoSlot) {
        previous_[after] = before;
    } else if (tail_ == slot) {
        tail_ = before;
    }
    previous_[slot] = kNoSlot;
    next_[slot] = kNoSlot;
}

void LodPagePool::PushFront(std::uint32_t slot) {
    previous_[slot] = kNoSlot;
    next_[slot] = head_;
    if (head_ != kNoSlot) {
        previous_[head_] = slot;
    }
    head_ = slot;
    if (tail_ == kNoSlot) {
        tail_ = slot;
    }
}

} // namespace chmv::gpu::lod
