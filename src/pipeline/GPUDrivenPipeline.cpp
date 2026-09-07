#include "pipeline/GPUDrivenPipeline.h"

#include <glad/gl.h>

#include <cstddef>
#include <stdexcept>

namespace chmv::pipeline {
namespace {

struct DrawArraysIndirectCommand {
    std::uint32_t count;
    std::uint32_t instanceCount;
    std::uint32_t first;
    std::uint32_t baseInstance;
};

static_assert(sizeof(DrawArraysIndirectCommand) == 16);

constexpr std::uint32_t kFilterWorkGroupSize = 256;

void CheckBufferSize(std::size_t sizeBytes) {
    GLint64 maxBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
    if (sizeBytes > static_cast<std::size_t>(maxBlockSize)) {
        throw std::runtime_error(
            "dataset exceeds the maximum SSBO size of this GPU; streaming is not implemented yet");
    }
}

} // namespace

GPUDrivenPipeline::GPUDrivenPipeline(const std::filesystem::path& shaderDirectory)
    : filterProgram_(shaderDirectory / "range_filter.comp") {
    const DrawArraysIndirectCommand command{0, 1, 0, 0};
    indirectBuffer_.Allocate(sizeof(command), &command, GL_DYNAMIC_DRAW);
}

void GPUDrivenPipeline::SetGraph(std::uint32_t edgeCount) {
    if (capacity_ == edgeCount) {
        return;
    }

    if (edgeCount == 0) {
        capacity_ = 0;
        return;
    }

    const auto sizeBytes = static_cast<std::size_t>(edgeCount) * sizeof(std::uint32_t);
    CheckBufferSize(sizeBytes);
    visibleEdgeBuffer_.Allocate(sizeBytes, nullptr, GL_DYNAMIC_DRAW);
    capacity_ = edgeCount;
}

GPUProcessingResult GPUDrivenPipeline::Process(
    std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
    gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy) {
    if (edgeCount == 0) {
        return {};
    }

    if (capacity_ != edgeCount) {
        SetGraph(edgeCount);
    }

    const std::uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indirectBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, &zero);

    filterTimer_.Begin();
    filterProgram_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, graphEdgeBuffer);
    visibleEdgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    indirectBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    glUniform1ui(glGetUniformLocation(filterProgram_.Id(), "uEdgeCount"), edgeCount);
    glUniform1i(glGetUniformLocation(filterProgram_.Id(), "uLevel"), lodLevel);
    filterProgram_.Dispatch((edgeCount + kFilterWorkGroupSize - 1) / kFilterWorkGroupSize);
    filterTimer_.End();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    const gpu::unfolding::UnfoldingInput input{
        graphEdgeBuffer,
        edgeCount,
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
        edgeCount,
    };

    unfoldingTimer_.Begin();
    const auto unfolded = unfoldingStrategy.Execute(input);
    unfoldingTimer_.End();

    return {
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
        unfolded.edgeBuffer,
        unfolded.drawCommandBuffer,
    };
}

void GPUDrivenPipeline::ReadBackEdgeIds(std::uint32_t edgeIdBuffer,
                                        std::uint32_t drawCommandBuffer,
                                        std::vector<std::uint32_t>& output) const {
    output.clear();
    if (edgeIdBuffer == 0 || drawCommandBuffer == 0) {
        return;
    }

    DrawArraysIndirectCommand command{};
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, drawCommandBuffer);
    glGetBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(command), &command);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    const auto edgeCount = command.count / 2;
    output.resize(edgeCount);
    if (edgeCount == 0) {
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeIdBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       static_cast<GLsizeiptr>(edgeCount * sizeof(std::uint32_t)), output.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

GPUProcessingStats GPUDrivenPipeline::Stats() const {
    return {
        filterTimer_.LastMilliseconds(),
        unfoldingTimer_.LastMilliseconds(),
    };
}

} // namespace chmv::pipeline
