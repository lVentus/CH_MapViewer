#include "gpu/unfolding/ExpansionKernels.h"

#include "gpu/lod/LodPagePool.h"

#include <glad/gl.h>

namespace chmv::gpu::unfolding {

void BindExpansionKernel(std::uint32_t program, const UnfoldingInput& input,
                         std::uint32_t inputList) {
    input.pool->BindForKernel(program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, inputList);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, input.segmentBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, input.statusBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, input.drawCommand);
    glUniform1ui(glGetUniformLocation(program, "uSegmentCapacity"), input.segmentCapacity);
    glUniform1ui(glGetUniformLocation(program, "uLevel"), input.level);
    glUniform1ui(glGetUniformLocation(program, "uThreshold"), input.threshold);
    glUniform4i(glGetUniformLocation(program, "uCullBox"), input.cullBox[0], input.cullBox[1],
                input.cullBox[2], input.cullBox[3]);
    glUniform1ui(glGetUniformLocation(program, "uCullSegments"), input.cullSegments ? 1u : 0u);
}

void BindTraversalKernel(std::uint32_t, const UnfoldingInput& input) {
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, input.errorBounds);
}

IntervalKernel::IntervalKernel(const std::filesystem::path& shaderDirectory)
    : program_(shaderDirectory / "lod_unfold_intervals.comp"), lists_(shaderDirectory) {}

void IntervalKernel::Run(const UnfoldingInput& input, bool childIntervals) const {
    lists_.Prepare(input.candidateList, input.candidateCapacity, 256u);
    program_.Bind();
    BindExpansionKernel(program_.Id(), input, input.candidateList);
    glUniform1ui(glGetUniformLocation(program_.Id(), "uChildIntervals"), childIntervals ? 1u : 0u);
    lod::ListDispatcher::Dispatch(input.candidateList);
}

} // namespace chmv::gpu::unfolding
