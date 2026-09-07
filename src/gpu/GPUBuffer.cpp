#include "gpu/GPUBuffer.h"

#include <glad/gl.h>

#include <utility>

namespace chmv::gpu {

GPUBuffer::GPUBuffer() {
    glGenBuffers(1, &id_);
}

GPUBuffer::~GPUBuffer() {
    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
    }
}

GPUBuffer::GPUBuffer(GPUBuffer&& other) noexcept
    : id_(std::exchange(other.id_, 0)), sizeBytes_(std::exchange(other.sizeBytes_, 0)) {}

GPUBuffer& GPUBuffer::operator=(GPUBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
    }

    id_ = std::exchange(other.id_, 0);
    sizeBytes_ = std::exchange(other.sizeBytes_, 0);
    return *this;
}

void GPUBuffer::Allocate(std::size_t sizeBytes, const void* data, std::uint32_t usage) {
    const auto glUsage = usage == 0 ? GL_DYNAMIC_DRAW : usage;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(sizeBytes), data, glUsage);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    sizeBytes_ = sizeBytes;
}

void GPUBuffer::Update(std::size_t offsetBytes, std::size_t sizeBytes, const void* data) const {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offsetBytes),
                    static_cast<GLsizeiptr>(sizeBytes), data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GPUBuffer::BindBase(std::uint32_t target, std::uint32_t binding) const {
    glBindBufferBase(target, binding, id_);
}

} // namespace chmv::gpu
