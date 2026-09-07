#include "gpu/ComputeProgram.h"

#include <glad/gl.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chmv::gpu {

ComputeProgram::ComputeProgram(const std::filesystem::path& shaderPath) {
    Load(shaderPath);
}

ComputeProgram::~ComputeProgram() {
    Reset();
}

ComputeProgram::ComputeProgram(ComputeProgram&& other) noexcept
    : program_(std::exchange(other.program_, 0)) {}

ComputeProgram& ComputeProgram::operator=(ComputeProgram&& other) noexcept {
    if (this != &other) {
        Reset();
        program_ = std::exchange(other.program_, 0);
    }
    return *this;
}

void ComputeProgram::Load(const std::filesystem::path& shaderPath) {
    Reset();

    const auto source = ReadFile(shaderPath);
    const char* sourcePtr = source.c_str();

    const auto shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &sourcePtr, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(length));
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("compute shader compilation failed: " + std::string(log.data()));
    }

    program_ = glCreateProgram();
    glAttachShader(program_, shader);
    glLinkProgram(program_);
    glDeleteShader(shader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program_, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(length));
        glGetProgramInfoLog(program_, length, nullptr, log.data());
        const auto message = std::string(log.data());
        Reset();
        throw std::runtime_error("compute program link failed: " + message);
    }
}

void ComputeProgram::Bind() const {
    glUseProgram(program_);
}

void ComputeProgram::Dispatch(unsigned int groupsX, unsigned int groupsY,
                              unsigned int groupsZ) const {
    glDispatchCompute(groupsX, groupsY, groupsZ);
}

std::string ComputeProgram::ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open shader: " + path.string());
    }

    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

void ComputeProgram::Reset() {
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
}

} // namespace chmv::gpu
