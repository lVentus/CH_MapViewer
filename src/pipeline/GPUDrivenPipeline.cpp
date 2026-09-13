#include "pipeline/GPUDrivenPipeline.h"

#include <glad/gl.h>

#include <algorithm>
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

void CheckBufferSize(std::size_t sizeBytes) {
    GLint64 maxBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
    if (sizeBytes > static_cast<std::size_t>(maxBlockSize)) {
        throw std::runtime_error(
            "dataset exceeds the maximum SSBO size of this GPU; streaming is not implemented yet");
    }
}

} // namespace

void GPUDrivenPipeline::SetGraph(std::uint32_t edgeCount, std::uint32_t drawableEdgeCount) {
    if (edgeCount_ == edgeCount && visibleCapacity_ == drawableEdgeCount) {
        return;
    }

    edgeCount_ = edgeCount;
    visibleCapacity_ = drawableEdgeCount;
    rangeCandidateEdgeCount_.reset();

    const auto storageCount = std::max<std::uint32_t>(drawableEdgeCount, 1);
    const auto sizeBytes = static_cast<std::size_t>(storageCount) * sizeof(std::uint32_t);
    CheckBufferSize(sizeBytes);
    visibleEdgeBuffer_.Allocate(sizeBytes, nullptr, GL_DYNAMIC_DRAW);

    const DrawArraysIndirectCommand command{0, 1, 0, 0};
    indirectBuffer_.Allocate(sizeof(command), &command, GL_DYNAMIC_DRAW);
}

GPUProcessingResult GPUDrivenPipeline::Process(
    std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
    gpu::filtering::IRangeFilterStrategy& rangeFilterStrategy,
    gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy,
    const geometry::RefinementParameters& refinementParameters) {
    if (edgeCount == 0) {
        return {};
    }

    if (edgeCount_ != edgeCount) {
        throw std::runtime_error("GPU pipeline graph state is out of date");
    }

    const std::uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indirectBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, &zero);

    filterTimer_.Begin();
    const auto filterStats = rangeFilterStrategy.Execute({
        graphEdgeBuffer,
        edgeCount,
        lodLevel,
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
    });
    filterTimer_.End();
    rangeCandidateEdgeCount_ = filterStats.candidateEdgeCount;

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    geometryRefinementActive_ = unfoldingStrategy.RefinesGeometry();
    if (!geometryRefinementActive_) {
        return {
            visibleEdgeBuffer_.Id(),
            indirectBuffer_.Id(),
            visibleEdgeBuffer_.Id(),
            indirectBuffer_.Id(),
        };
    }

    const gpu::unfolding::UnfoldingInput input{
        graphEdgeBuffer,
        edgeCount,
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
        edgeCount,
        refinementParameters.screenPixelScale,
        refinementParameters.maxScreenErrorPixels,
    };

    geometryRefinementTimer_.Begin();
    const auto refined = unfoldingStrategy.Execute(input);
    geometryRefinementTimer_.End();

    return {
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
        refined.edgeBuffer,
        refined.drawCommandBuffer,
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

    const auto outputEdgeCount = command.count / 2;
    output.resize(outputEdgeCount);
    if (outputEdgeCount == 0) {
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeIdBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       static_cast<GLsizeiptr>(outputEdgeCount * sizeof(std::uint32_t)),
                       output.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

GPUProcessingStats GPUDrivenPipeline::Stats() const {
    return {
        rangeCandidateEdgeCount_,
        filterTimer_.LastMilliseconds(),
        geometryRefinementActive_ ? geometryRefinementTimer_.LastMilliseconds()
                                  : std::optional<double>{},
    };
}

} // namespace chmv::pipeline
