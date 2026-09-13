#include "gpu/filtering/BirthOrderedRangeFilterStrategy.h"

#include "data/ch/CHGraph.h"

#include <glad/gl.h>

#include <cstddef>
#include <stdexcept>

namespace chmv::gpu::filtering {
namespace {

struct DispatchIndirectCommand {
    std::uint32_t groupsX;
    std::uint32_t groupsY;
    std::uint32_t groupsZ;
};

static_assert(sizeof(DispatchIndirectCommand) == 12);
static_assert(sizeof(data::OrderedRangeEntry) == 8);

void CheckBufferSize(std::size_t sizeBytes) {
    GLint64 maxBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
    if (sizeBytes > static_cast<std::size_t>(maxBlockSize)) {
        throw std::runtime_error(
            "ordered range index exceeds the maximum SSBO size of this GPU; "
            "streaming is not implemented yet");
    }
}

} // namespace

BirthOrderedRangeFilterStrategy::BirthOrderedRangeFilterStrategy(
    const std::filesystem::path& shaderDirectory)
    : prepareDispatchProgram_(shaderDirectory / "range_filter_prepare_dispatch.comp"),
      filterProgram_(shaderDirectory / "range_filter_ordered.comp") {
    const DispatchIndirectCommand dispatch{0, 1, 1};
    dispatchBuffer_.Allocate(sizeof(dispatch), &dispatch, GL_DYNAMIC_DRAW);
}

void BirthOrderedRangeFilterStrategy::SetGraph(const data::CHGraph& graph) {
    const auto& index = graph.OrderedRanges();
    levelCount_ = static_cast<std::uint32_t>(index.scanEndByLevel.size());

    if (index.entries.empty() || index.scanEndByLevel.empty()) {
        return;
    }

    const auto rangeBytes = index.entries.size() * sizeof(data::OrderedRangeEntry);
    const auto scanEndBytes = index.scanEndByLevel.size() * sizeof(std::uint32_t);
    CheckBufferSize(rangeBytes);
    CheckBufferSize(scanEndBytes);
    orderedRangeBuffer_.Allocate(rangeBytes, index.entries.data(), GL_STATIC_DRAW);
    scanEndBuffer_.Allocate(scanEndBytes, index.scanEndByLevel.data(), GL_STATIC_DRAW);
}

RangeFilterExecutionStats BirthOrderedRangeFilterStrategy::Execute(const RangeFilterInput& input) {
    if (levelCount_ == 0) {
        return {};
    }

    prepareDispatchProgram_.Bind();
    scanEndBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    dispatchBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glUniform1i(glGetUniformLocation(prepareDispatchProgram_.Id(), "uLevel"), input.lodLevel);
    glUniform1ui(glGetUniformLocation(prepareDispatchProgram_.Id(), "uLevelCount"),
                 levelCount_);
    prepareDispatchProgram_.Dispatch(1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    filterProgram_.Bind();
    orderedRangeBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    scanEndBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.outputEdgeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, input.outputDrawCommandBuffer);
    glUniform1i(glGetUniformLocation(filterProgram_.Id(), "uLevel"), input.lodLevel);
    glUniform1ui(glGetUniformLocation(filterProgram_.Id(), "uLevelCount"),
                 levelCount_);

    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, dispatchBuffer_.Id());
    glDispatchComputeIndirect(0);

    return {};
}

} // namespace chmv::gpu::filtering
