#include "gpu/streaming/RefinementRequestCollector.h"

#include <glad/gl.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace chmv::gpu::streaming {
namespace {

constexpr std::uint32_t kLocalSize = 256;

} // namespace

RefinementRequestCollector::RefinementRequestCollector(
    const std::filesystem::path& shaderDirectory)
    : collectProgram_(shaderDirectory / "streaming_refinement_requests.comp") {
    const std::uint32_t zero[2] = {0, 0};
    counterBuffer_.Allocate(sizeof(zero), zero, GL_DYNAMIC_DRAW);
}

std::vector<chmv::streaming::runtime::RefinementEdgeRequest> RefinementRequestCollector::Collect(
    std::uint32_t streamingLinkBuffer,
    std::uint32_t leafEdgeBuffer,
    std::uint32_t leafDrawCommandBuffer,
    std::uint32_t edgeCapacity) {
    stats_ = {};
    if (streamingLinkBuffer == 0 || leafEdgeBuffer == 0 || leafDrawCommandBuffer == 0 ||
        edgeCapacity == 0) {
        return {};
    }

    EnsureCapacity(edgeCapacity);

    const std::uint32_t zero[2] = {0, 0};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, zero);

    collectProgram_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, streamingLinkBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, leafEdgeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, leafDrawCommandBuffer);
    requestBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    counterBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 4);
    glUniform1ui(glGetUniformLocation(collectProgram_.Id(), "uEdgeCapacity"), edgeCapacity_);
    glUniform1ui(glGetUniformLocation(collectProgram_.Id(), "uRequestCapacity"),
                 requestCapacity_);

    const auto groups = (edgeCapacity_ + kLocalSize - 1u) / kLocalSize;
    collectProgram_.Dispatch(std::max(groups, 1u));
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    std::uint32_t counters[2] = {0, 0};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterBuffer_.Id());
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(counters), counters);
    stats_.rawRequestCount = counters[0];
    stats_.droppedRequestCount = counters[1];

    const auto readableCount = std::min(counters[0], requestCapacity_);
    std::vector<chmv::streaming::runtime::RefinementEdgeRequest> requests(readableCount);
    if (readableCount != 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, requestBuffer_.Id());
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(requests.size() * sizeof(chmv::streaming::runtime::RefinementEdgeRequest)),
                           requests.data());
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    std::sort(requests.begin(), requests.end(), [](const auto& left, const auto& right) {
        if (left.parentEdgeId != right.parentEdgeId) {
            return left.parentEdgeId < right.parentEdgeId;
        }
        return left.childEdgeId < right.childEdgeId;
    });
    requests.erase(std::unique(requests.begin(), requests.end()), requests.end());
    stats_.uniqueRequestCount = static_cast<std::uint32_t>(requests.size());
    return requests;
}

void RefinementRequestCollector::EnsureCapacity(std::uint32_t edgeCapacity) {
    if (edgeCapacity_ == edgeCapacity && requestCapacity_ != 0) {
        return;
    }
    if (edgeCapacity > std::numeric_limits<std::uint32_t>::max() / 2u) {
        throw std::runtime_error("streaming refinement request capacity exceeds uint32");
    }

    edgeCapacity_ = edgeCapacity;
    requestCapacity_ = std::max(edgeCapacity * 2u, 1u);
    requestBuffer_.Allocate(
        static_cast<std::size_t>(requestCapacity_) *
            sizeof(chmv::streaming::runtime::RefinementEdgeRequest), nullptr,
        GL_DYNAMIC_READ);
}

} // namespace chmv::gpu::streaming
