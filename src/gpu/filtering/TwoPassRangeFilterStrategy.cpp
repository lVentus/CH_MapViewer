#include "gpu/filtering/TwoPassRangeFilterStrategy.h"

#include "gpu/lod/LodPageLists.h"
#include "gpu/lod/LodPagePool.h"
#include "streaming/lod/LodPageFormat.h"

#include <glad/gl.h>

#include <algorithm>

namespace chmv::gpu::filtering {
namespace {

// One bit per record.
constexpr std::size_t kMaskWords = streaming::lod::kPageRecordCapacity / 32u;

} // namespace

TwoPassRangeFilterStrategy::TwoPassRangeFilterStrategy(const std::filesystem::path& shaderDirectory)
    : maskProgram_(shaderDirectory / "lod_filter_mask.comp"),
      offsetsProgram_(shaderDirectory / "lod_filter_offsets.comp"),
      scatterProgram_(shaderDirectory / "lod_filter_scatter.comp") {}

void TwoPassRangeFilterStrategy::Execute(const RangeFilterInput& input) {
    const auto slots = input.slots.size();
    if (slots > capacity_) {
        capacity_ = std::max(slots, capacity_ * 2u);
        masks_.Allocate(capacity_ * kMaskWords * 4u);
        prefix_.Allocate((lod::kListHeaderWords + capacity_) * 4u);
    }
    const auto& pool = *input.pool;

    // Which records of each page are candidates, and how many.
    maskProgram_.Bind();
    pool.BindForKernel(maskProgram_.Id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.slotList);
    masks_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    prefix_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    glUniform1ui(glGetUniformLocation(maskProgram_.Id(), "uLevel"), input.query.level);
    lod::ListDispatcher::Dispatch(input.slotList);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Where each page's candidates go, and how many there are in all.
    offsetsProgram_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.slotList);
    prefix_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    offsetsProgram_.Dispatch(1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Every candidate to its place.
    scatterProgram_.Bind();
    const auto id = scatterProgram_.Id();
    pool.BindForKernel(id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.slotList);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, input.candidateList);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, input.statusBuffer);
    masks_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    prefix_.BindBase(GL_SHADER_STORAGE_BUFFER, 6);
    glUniform1ui(glGetUniformLocation(id, "uCandidateCapacity"), input.candidateCapacity);
    lod::ListDispatcher::Dispatch(input.slotList);
}

} // namespace chmv::gpu::filtering
