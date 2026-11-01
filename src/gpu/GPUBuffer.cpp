#include "gpu/GPUBuffer.h"

#include <glad/gl.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#ifndef GL_MAP_PERSISTENT_BIT
#define GL_MAP_PERSISTENT_BIT 0x0040
#endif
#ifndef GL_MAP_COHERENT_BIT
#define GL_MAP_COHERENT_BIT 0x0080
#endif
#ifndef GL_DYNAMIC_STORAGE_BIT
#define GL_DYNAMIC_STORAGE_BIT 0x0100
#endif
#ifndef GL_CLIENT_STORAGE_BIT
#define GL_CLIENT_STORAGE_BIT 0x0200
#endif
#ifndef GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT
#define GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT 0x00004000
#endif

namespace chmv::gpu {
namespace {

#if defined(_WIN32)
#define CHMV_GL_APIENTRY __stdcall
#else
#define CHMV_GL_APIENTRY
#endif
using BufferStorageProc = void (CHMV_GL_APIENTRY *)(GLenum, GLsizeiptr, const void*, GLbitfield);
#undef CHMV_GL_APIENTRY

[[nodiscard]] bool HasGlExtension(std::string_view name) {
    GLint count = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &count);
    for (GLint i = 0; i < count; ++i) {
        const auto* extension = reinterpret_cast<const char*>(
            glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
        if (extension != nullptr && name == extension) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] BufferStorageProc GetBufferStorageProc() {
    static const auto proc = reinterpret_cast<BufferStorageProc>(
        glfwGetProcAddress("glBufferStorage"));
    return proc;
}

} // namespace

GPUBuffer::GPUBuffer() {
    glGenBuffers(1, &id_);
}

GPUBuffer::~GPUBuffer() {
    DestroyStorage();
    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
    }
}

GPUBuffer::GPUBuffer(GPUBuffer&& other) noexcept
    : id_(std::exchange(other.id_, 0)),
      sizeBytes_(std::exchange(other.sizeBytes_, 0)),
      mappedData_(std::exchange(other.mappedData_, nullptr)),
      hostShared_(std::exchange(other.hostShared_, false)),
      immutableStorage_(std::exchange(other.immutableStorage_, false)) {}

GPUBuffer& GPUBuffer::operator=(GPUBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    DestroyStorage();
    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
    }

    id_ = std::exchange(other.id_, 0);
    sizeBytes_ = std::exchange(other.sizeBytes_, 0);
    mappedData_ = std::exchange(other.mappedData_, nullptr);
    hostShared_ = std::exchange(other.hostShared_, false);
    immutableStorage_ = std::exchange(other.immutableStorage_, false);
    return *this;
}

void GPUBuffer::DestroyStorage() {
    if (id_ == 0) return;
    if (mappedData_ != nullptr) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        mappedData_ = nullptr;
    }
    sizeBytes_ = 0;
    hostShared_ = false;
    immutableStorage_ = false;
}

void GPUBuffer::RecreateBufferObject() {
    DestroyStorage();
    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
    }
    glGenBuffers(1, &id_);
}

void GPUBuffer::Allocate(std::size_t sizeBytes, const void* data, std::uint32_t usage) {
    // Immutable storage cannot be re-specified.  Recreate the GL object on every allocation so
    // budget reconfiguration works identically for ordinary and host-shared buffers.
    RecreateBufferObject();
    const auto glUsage = usage == 0 ? GL_DYNAMIC_DRAW : usage;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(sizeBytes), data, glUsage);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    // A buffer the driver could not allocate must not go unnoticed: every later use of it would
    // be undefined.
    bool outOfMemory = false;
    for (int i = 0; i < 8; ++i) {
        const auto error = glGetError();
        if (error == GL_NO_ERROR) {
            break;
        }
        outOfMemory = outOfMemory || error == GL_OUT_OF_MEMORY;
    }
    if (outOfMemory) {
        throw std::runtime_error("out of GPU memory allocating " +
                                 std::to_string(sizeBytes / (1024u * 1024u)) + " MiB");
    }
    sizeBytes_ = sizeBytes;
}

bool GPUBuffer::HostSharedStorageSupported() {
    return HasGlExtension("GL_ARB_buffer_storage") && GetBufferStorageProc() != nullptr;
}

bool GPUBuffer::AllocateHostShared(std::size_t sizeBytes, const void* data) {
    if (!HostSharedStorageSupported()) {
        Allocate(sizeBytes, data, GL_DYNAMIC_DRAW);
        return false;
    }

    RecreateBufferObject();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    const GLbitfield storageFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT |
                                    GL_MAP_COHERENT_BIT | GL_DYNAMIC_STORAGE_BIT |
                                    GL_CLIENT_STORAGE_BIT;
    GetBufferStorageProc()(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(sizeBytes), nullptr,
                           storageFlags);
    const GLbitfield mapFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    mappedData_ = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                   static_cast<GLsizeiptr>(sizeBytes), mapFlags);
    if (mappedData_ == nullptr) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        Allocate(sizeBytes, data, GL_DYNAMIC_DRAW);
        return false;
    }

    sizeBytes_ = sizeBytes;
    hostShared_ = true;
    immutableStorage_ = true;
    if (data != nullptr && sizeBytes != 0u) {
        std::memcpy(mappedData_, data, sizeBytes);
        CommitHostWrites();
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return true;
}

void GPUBuffer::Update(std::size_t offsetBytes, std::size_t sizeBytes, const void* data) const {
    if (sizeBytes == 0u) return;
    if (data == nullptr || offsetBytes > sizeBytes_ || sizeBytes > sizeBytes_ - offsetBytes) {
        throw std::runtime_error("GPU buffer update is out of range");
    }
    if (mappedData_ != nullptr) {
        std::memcpy(static_cast<std::byte*>(mappedData_) + offsetBytes, data, sizeBytes);
        CommitHostWrites();
        return;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offsetBytes),
                    static_cast<GLsizeiptr>(sizeBytes), data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void* GPUBuffer::MappedData(std::size_t offsetBytes) const {
    if (mappedData_ == nullptr || offsetBytes > sizeBytes_) return nullptr;
    return static_cast<std::byte*>(mappedData_) + offsetBytes;
}

void GPUBuffer::CommitHostWrites() const {
    if (mappedData_ == nullptr) return;
    std::atomic_thread_fence(std::memory_order_release);
    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
}

void GPUBuffer::BindBase(std::uint32_t target, std::uint32_t binding) const {
    glBindBufferBase(target, binding, id_);
}

} // namespace chmv::gpu
