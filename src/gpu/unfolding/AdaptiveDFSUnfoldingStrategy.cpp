#include "gpu/unfolding/AdaptiveDFSUnfoldingStrategy.h"

#include <glad/gl.h>

#include <cstddef>

namespace chmv::gpu::unfolding {
namespace {

struct DispatchIndirectCommand {
    std::uint32_t groupsX;
    std::uint32_t groupsY;
    std::uint32_t groupsZ;
};

struct DrawArraysIndirectCommand {
    std::uint32_t count;
    std::uint32_t instanceCount;
    std::uint32_t first;
    std::uint32_t baseInstance;
};

static_assert(sizeof(DispatchIndirectCommand) == 12);
static_assert(sizeof(DrawArraysIndirectCommand) == 16);

} // namespace

AdaptiveDFSUnfoldingStrategy::AdaptiveDFSUnfoldingStrategy(
    const std::filesystem::path& shaderDirectory)
    : prepareDispatchProgram_(shaderDirectory / "unfold_prepare_dispatch.comp"),
      unfoldProgram_(shaderDirectory / "unfold_adaptive_dfs.comp"),
      finalizeProgram_(shaderDirectory / "unfold_finalize.comp") {
    const std::uint32_t zero = 0;
    outputCounterBuffer_.Allocate(sizeof(zero), &zero, GL_DYNAMIC_DRAW);

    const std::uint32_t status[2] = {0, 0};
    statusBuffer_.Allocate(sizeof(status), status, GL_DYNAMIC_DRAW);

    const DispatchIndirectCommand dispatch{0, 1, 1};
    dispatchBuffer_.Allocate(sizeof(dispatch), &dispatch, GL_DYNAMIC_DRAW);

    const DrawArraysIndirectCommand command{0, 1, 0, 0};
    indirectBuffer_.Allocate(sizeof(command), &command, GL_DYNAMIC_DRAW);
}

UnfoldingOutput AdaptiveDFSUnfoldingStrategy::Execute(const UnfoldingInput& input) {
    if (input.edgeCount == 0 || input.outputCapacity == 0) {
        return {};
    }

    EnsureCapacity(input.outputCapacity);

    const std::uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputCounterBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statusBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, statusBuffer_.SizeBytes(),
                         GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

    prepareDispatchProgram_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input.inputDrawCommandBuffer);
    dispatchBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    prepareDispatchProgram_.Dispatch(1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    unfoldProgram_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input.edgeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, input.inputEdgeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.inputDrawCommandBuffer);
    outputEdgeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    outputCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 4);
    statusBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    glUniform1ui(glGetUniformLocation(unfoldProgram_.Id(), "uEdgeCount"), input.edgeCount);
    glUniform1ui(glGetUniformLocation(unfoldProgram_.Id(), "uOutputCapacity"), capacity_);
    glUniform1f(glGetUniformLocation(unfoldProgram_.Id(), "uScreenPixelScale"),
                input.screenPixelScale);
    glUniform1f(glGetUniformLocation(unfoldProgram_.Id(), "uMaxScreenErrorPixels"),
                input.maxScreenErrorPixels);

    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, dispatchBuffer_.Id());
    glDispatchComputeIndirect(0);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    finalizeProgram_.Bind();
    outputCounterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    indirectBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glUniform1ui(glGetUniformLocation(finalizeProgram_.Id(), "uOutputCapacity"), capacity_);
    finalizeProgram_.Dispatch(1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    return {
        .edgeBuffer = outputEdgeBuffer_.Id(),
        .drawCommandBuffer = indirectBuffer_.Id(),
    };
}

void AdaptiveDFSUnfoldingStrategy::EnsureCapacity(std::uint32_t capacity) {
    if (capacity_ == capacity) {
        return;
    }

    outputEdgeBuffer_.Allocate(static_cast<std::size_t>(capacity) * sizeof(std::uint32_t), nullptr,
                               GL_DYNAMIC_DRAW);
    capacity_ = capacity;
}

} // namespace chmv::gpu::unfolding
