#include "gpu/lod/LodPagePool.h"

#include <glad/gl.h>

#include <algorithm>
#include <atomic>
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

std::atomic<std::uint64_t> nextGeneration{1};

} // namespace

void LodPagePool::Initialize(std::uint32_t pageCount, std::uint32_t slotCount,
                             std::uint32_t pageWords, std::uint32_t childWords) {
    const auto slotWords = pageWords + childWords;
    slotCount = std::max(1u, std::min(slotCount, MaxSlots(slotWords)));
    slotWords_ = slotWords;
    pageWords_ = pageWords;
    buffer_.Allocate(static_cast<std::size_t>(slotCount) * slotWords * 4u);
    if (childWords != 0u) {
        const std::vector<std::uint32_t> none(pageCount, kNoSlot);
        slotOfPageBuffer_.Allocate(none.size() * 4u, none.data());
    }
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
    generation_ = nextGeneration++;
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

bool LodPagePool::Upload(std::uint32_t page, std::span<const std::uint32_t> words,
                         std::span<const std::uint32_t> children) {
    if (words.size() > pageWords_ || children.size() > slotWords_ - pageWords_) {
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
        PublishSlot(pageOfSlot_[slot], kNoSlot);
        ++evictions_;
    }

    const auto base = static_cast<std::size_t>(slot) * slotWords_ * 4u;
    buffer_.Update(base, words.size() * 4u, words.data());
    if (!children.empty()) {
        buffer_.Update(base + std::size_t{pageWords_} * 4u, children.size() * 4u,
                       children.data());
    }
    PublishSlot(page, slot);
    slotOfPage_[page] = slot;
    pageOfSlot_[slot] = page;
    lastUsed_[slot] = frame_;
    PushFront(slot);
    return true;
}

void LodPagePool::BindForKernel(std::uint32_t program) const {
    buffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    buffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glUniform1ui(glGetUniformLocation(program, "uSlotWords"), slotWords_);
    if (HasChildren()) {
        slotOfPageBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
        glUniform1ui(glGetUniformLocation(program, "uPageWords"), pageWords_);
    }
}

void LodPagePool::PublishSlot(std::uint32_t page, std::uint32_t slot) {
    if (HasChildren()) {
        slotOfPageBuffer_.Update(std::size_t{page} * 4u, 4u, &slot);
    }
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
