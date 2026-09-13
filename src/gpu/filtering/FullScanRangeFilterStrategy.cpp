#include "gpu/filtering/FullScanRangeFilterStrategy.h"

#include "data/ch/CHGraph.h"

#include <glad/gl.h>

namespace chmv::gpu::filtering {
namespace {

constexpr std::uint32_t kWorkGroupSize = 256;

} // namespace

FullScanRangeFilterStrategy::FullScanRangeFilterStrategy(
    const std::filesystem::path& shaderDirectory)
    : filterProgram_(shaderDirectory / "range_filter.comp") {}

void FullScanRangeFilterStrategy::SetGraph(const data::CHGraph& graph) {
    edgeCount_ = static_cast<std::uint32_t>(graph.EdgeCount());
}

RangeFilterExecutionStats FullScanRangeFilterStrategy::Execute(const RangeFilterInput& input) {
    filterProgram_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input.graphEdgeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, input.outputEdgeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input.outputDrawCommandBuffer);
    glUniform1ui(glGetUniformLocation(filterProgram_.Id(), "uEdgeCount"), input.graphEdgeCount);
    glUniform1i(glGetUniformLocation(filterProgram_.Id(), "uLevel"), input.lodLevel);
    filterProgram_.Dispatch((input.graphEdgeCount + kWorkGroupSize - 1) / kWorkGroupSize);
    return {edgeCount_};
}

RangeFilterDiagnostics FullScanRangeFilterStrategy::ReadBackDiagnostics(std::int32_t) const {
    return {
        edgeCount_,
        (edgeCount_ + kWorkGroupSize - 1) / kWorkGroupSize,
        kWorkGroupSize,
    };
}

} // namespace chmv::gpu::filtering
