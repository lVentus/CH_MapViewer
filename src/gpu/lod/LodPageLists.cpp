#include "gpu/lod/LodPageLists.h"

#include <glad/gl.h>

#include <algorithm>

namespace chmv::gpu::lod {

std::array<std::uint32_t, 3> GroupGrid(std::uint64_t groups) {
    constexpr std::uint64_t kMaxGroups = 65535;
    return {static_cast<std::uint32_t>(std::min(groups, kMaxGroups)),
            static_cast<std::uint32_t>((groups + kMaxGroups - 1u) / kMaxGroups), 1u};
}

ListDispatcher::ListDispatcher(const std::filesystem::path& shaderDirectory)
    : program_(shaderDirectory / "lod_list_groups.comp") {}

void ListDispatcher::Prepare(std::uint32_t list, std::uint32_t capacity, std::uint32_t groupSize,
                             std::uint32_t next) const {
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program_.Bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, list);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, next != 0u ? next : list);
    const auto id = program_.Id();
    glUniform1ui(glGetUniformLocation(id, "uCapacity"), capacity);
    glUniform1ui(glGetUniformLocation(id, "uGroupSize"), groupSize);
    glUniform1ui(glGetUniformLocation(id, "uEmptyNext"), next != 0u ? 1u : 0u);
    program_.Dispatch(1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
}

void ListDispatcher::Dispatch(std::uint32_t list) {
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, list);
    glDispatchComputeIndirect(static_cast<GLintptr>(sizeof(std::uint32_t)));
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
}

} // namespace chmv::gpu::lod
