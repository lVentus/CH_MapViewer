#include "gpu/filtering/BirthOrderedRangeFilterStrategy.h"

#include "gpu/lod/LodPagePool.h"

#include <glad/gl.h>

#include <algorithm>

namespace chmv::gpu::filtering {
namespace {

// One byte per record.
constexpr std::size_t kOrderWords = streaming::lod::kPageRecordCapacity / 4u;
constexpr std::uint32_t kGroupSize = 256;

} // namespace

BirthOrderedRangeFilterStrategy::BirthOrderedRangeFilterStrategy(
    const std::filesystem::path& shaderDirectory)
    : sortProgram_(shaderDirectory / "lod_filter_sort.comp"),
      countProgram_(shaderDirectory / "lod_filter_count.comp"),
      offsetsProgram_(shaderDirectory / "lod_filter_offsets.comp"),
      gatherProgram_(shaderDirectory / "lod_filter_gather.comp"), lists_(shaderDirectory) {}

void BirthOrderedRangeFilterStrategy::Reset() {
    orderedPages_.clear();
    poolGeneration_ = 0;
}

void BirthOrderedRangeFilterStrategy::SortNewPages(const RangeFilterInput& input) {
    const auto& pool = *input.pool;
    if (poolGeneration_ != pool.Generation()) {
        order_.Allocate(std::size_t{pool.SlotCount()} * kOrderWords * 4u);
        orderedPages_.assign(pool.SlotCount(), lod::LodPagePool::kNoSlot);
        poolGeneration_ = pool.Generation();
    }

    // [count, grid, slots...]
    sortSlots_.assign(lod::kListHeaderWords, 0u);
    for (const auto slot : input.slots) {
        const auto page = pool.PageOfSlot(slot);
        if (orderedPages_[slot] != page) {
            orderedPages_[slot] = page;
            sortSlots_.push_back(slot);
        }
    }
    const auto count = sortSlots_.size() - lod::kListHeaderWords;
    if (count == 0u) {
        return;
    }
    const auto grid = lod::GroupGrid(count);
    sortSlots_[0] = static_cast<std::uint32_t>(count);
    std::copy(grid.begin(), grid.end(), sortSlots_.begin() + 1);
    if (sortSlots_.size() > sortListCapacity_) {
        sortListCapacity_ = std::max(sortSlots_.size(), sortListCapacity_ * 2u);
        sortList_.Allocate(sortListCapacity_ * 4u);
    }
    sortList_.Update(0, sortSlots_.size() * 4u, sortSlots_.data());

    sortProgram_.Bind();
    pool.BindForKernel(sortProgram_.Id());
    sortList_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    order_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    sortProgram_.Dispatch(grid[0], grid[1], grid[2]);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void BirthOrderedRangeFilterStrategy::Execute(const RangeFilterInput& input) {
    SortNewPages(input);
    const auto slots = input.slots.size();
    if (slots + lod::kListHeaderWords > prefixCapacity_) {
        prefixCapacity_ = std::max(slots + lod::kListHeaderWords, prefixCapacity_ * 2u);
        prefix_.Allocate(prefixCapacity_ * 4u);
    }
    const auto& pool = *input.pool;

    // How many records of each page can still be drawn at the level.
    countProgram_.Bind();
    pool.BindForKernel(countProgram_.Id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.slotList);
    order_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    prefix_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    glUniform1ui(glGetUniformLocation(countProgram_.Id(), "uLevel"), input.query.level);
    const auto grid = lod::GroupGrid((slots + kGroupSize - 1u) / kGroupSize);
    countProgram_.Dispatch(grid[0], grid[1], grid[2]);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Where each page's records go, and how many there are in all.
    offsetsProgram_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.slotList);
    prefix_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    offsetsProgram_.Dispatch(1);
    lists_.Prepare(prefix_.Id(), input.candidateCapacity, kGroupSize);

    // Test only those records.
    gatherProgram_.Bind();
    const auto id = gatherProgram_.Id();
    pool.BindForKernel(id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.slotList);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, input.candidateList);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, input.statusBuffer);
    order_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    prefix_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    glUniform1ui(glGetUniformLocation(id, "uLevel"), input.query.level);
    glUniform1ui(glGetUniformLocation(id, "uCandidateCapacity"), input.candidateCapacity);
    lod::ListDispatcher::Dispatch(prefix_.Id());
}

} // namespace chmv::gpu::filtering
