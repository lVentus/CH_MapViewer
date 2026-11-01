#include "gpu/filtering/PageCullingRangeFilterStrategy.h"

#include "gpu/filtering/FullScanRangeFilterStrategy.h"
#include "gpu/lod/LodPagePool.h"

#include <glad/gl.h>

#include <algorithm>
#include <bit>
#include <cstring>

namespace chmv::gpu::filtering {
namespace {

// Layout of one slot entry in lod_filter_cull.comp.
constexpr std::size_t kEntryWords = 16;

void PackEntry(const streaming::lod::PageEntry& page, std::uint32_t* entry) {
    entry[0] = std::bit_cast<std::uint32_t>(page.minX);
    entry[1] = std::bit_cast<std::uint32_t>(page.minY);
    entry[2] = std::bit_cast<std::uint32_t>(page.maxX);
    entry[3] = std::bit_cast<std::uint32_t>(page.maxY);
    entry[4] = page.minLevel | (static_cast<std::uint32_t>(page.maxLevel) << 16u);
    entry[5] = page.minErrorCode | (static_cast<std::uint32_t>(page.maxParentCode) << 16u);
    entry[6] = 0u;
    entry[7] = 0u;
    std::memcpy(entry + 8, page.bands.minError.data(), page.bands.minError.size());
    std::memcpy(entry + 12, page.bands.maxParent.data(), page.bands.maxParent.size());
}

// An empty slot never passes: its level range is empty.
void PackEmpty(std::uint32_t* entry) {
    std::fill(entry, entry + kEntryWords, 0u);
    entry[4] = 0xffffu;
}

} // namespace

PageCullingRangeFilterStrategy::PageCullingRangeFilterStrategy(
    const std::filesystem::path& shaderDirectory)
    : cullProgram_(shaderDirectory / "lod_filter_cull.comp"),
      scanProgram_(shaderDirectory / "lod_filter_scan.comp"), lists_(shaderDirectory) {}

void PageCullingRangeFilterStrategy::Reset() {
    entries_.clear();
    entryPages_.clear();
    poolGeneration_ = 0;
    directory_ = nullptr;
}

void PageCullingRangeFilterStrategy::UpdateEntries(const RangeFilterInput& input) {
    const auto& pool = *input.pool;
    const auto slots = pool.SlotCount();
    if (poolGeneration_ != pool.Generation() || directory_ != input.directory.data()) {
        entries_.assign(std::size_t{slots} * kEntryWords, 0u);
        for (std::uint32_t slot = 0; slot < slots; ++slot) {
            PackEmpty(entries_.data() + std::size_t{slot} * kEntryWords);
        }
        entryPages_.assign(slots, lod::LodPagePool::kNoSlot);
        entryBuffer_.Allocate(entries_.size() * 4u, entries_.data());
        slotList_.Allocate((std::size_t{lod::kListHeaderWords} + slots) * 4u);
        poolGeneration_ = pool.Generation();
        directory_ = input.directory.data();
    }

    std::uint32_t first = slots;
    std::uint32_t last = 0;
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        const auto page = pool.PageOfSlot(slot);
        if (page == entryPages_[slot]) {
            continue;
        }
        auto* entry = entries_.data() + std::size_t{slot} * kEntryWords;
        if (page == lod::LodPagePool::kNoSlot) {
            PackEmpty(entry);
        } else {
            PackEntry(input.directory[page], entry);
        }
        entryPages_[slot] = page;
        first = std::min(first, slot);
        last = slot;
    }
    if (first <= last) {
        const auto offset = std::size_t{first} * kEntryWords;
        entryBuffer_.Update(offset * 4u, (std::size_t{last + 1u} * kEntryWords - offset) * 4u,
                            entries_.data() + offset);
    }
}

void PageCullingRangeFilterStrategy::Execute(const RangeFilterInput& input) {
    UpdateEntries(input);
    const auto slots = input.pool->SlotCount();
    const std::uint32_t empty = 0;
    slotList_.Update(0, sizeof(empty), &empty);

    cullProgram_.Bind();
    const auto id = cullProgram_.Id();
    entryBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    slotList_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    glUniform1ui(glGetUniformLocation(id, "uPoolSlots"), slots);
    glUniform4f(glGetUniformLocation(id, "uQueryBox"), input.query.minX, input.query.minY,
                input.query.maxX, input.query.maxY);
    glUniform1ui(glGetUniformLocation(id, "uLevel"), input.query.level);
    glUniform1ui(glGetUniformLocation(id, "uThreshold"), input.query.threshold);
    const auto grid = lod::GroupGrid((std::uint64_t{slots} + 255u) / 256u);
    cullProgram_.Dispatch(grid[0], grid[1], grid[2]);

    lists_.Prepare(slotList_.Id(), slots, 1u);
    FullScanRangeFilterStrategy::Scan(scanProgram_, input, slotList_.Id());
}

} // namespace chmv::gpu::filtering
