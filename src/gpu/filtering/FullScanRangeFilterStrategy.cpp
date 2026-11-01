#include "gpu/filtering/FullScanRangeFilterStrategy.h"

#include "gpu/lod/LodPageLists.h"
#include "gpu/lod/LodPagePool.h"

#include <glad/gl.h>

namespace chmv::gpu::filtering {

FullScanRangeFilterStrategy::FullScanRangeFilterStrategy(
    const std::filesystem::path& shaderDirectory)
    : scanProgram_(shaderDirectory / "lod_filter_scan.comp") {}

void FullScanRangeFilterStrategy::Execute(const RangeFilterInput& input) {
    Scan(scanProgram_, input, input.slotList);
}

void FullScanRangeFilterStrategy::Scan(const ComputeProgram& program,
                                       const RangeFilterInput& input, std::uint32_t slotList) {
    program.Bind();
    const auto id = program.Id();
    input.pool->BindForKernel(id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, slotList);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, input.candidateList);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, input.statusBuffer);
    glUniform1ui(glGetUniformLocation(id, "uLevel"), input.query.level);
    glUniform1ui(glGetUniformLocation(id, "uCandidateCapacity"), input.candidateCapacity);
    lod::ListDispatcher::Dispatch(slotList);
}

} // namespace chmv::gpu::filtering
