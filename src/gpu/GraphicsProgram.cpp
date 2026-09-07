#include "gpu/GraphicsProgram.h"

#include <glad/gl.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chmv::gpu {

GraphicsProgram::GraphicsProgram(const std::filesystem::path& vertexShaderPath,
                                 const std::filesystem::path& fragmentShaderPath) {
    Load(vertexShaderPath, fragmentShaderPath);
}

GraphicsProgram::~GraphicsProgram() {
    Reset();
}

GraphicsProgram::GraphicsProgram(GraphicsProgram&& other) noexcept
    : program_(std::exchange(other.program_, 0)) {}

GraphicsProgram& GraphicsProgram::operator=(GraphicsProgram&& other) noexcept {
    if (this != &other) {
        Reset();
        program_ = std::exchange(other.program_, 0);
    }
    return *this;
}

void GraphicsProgram::Load(const std::filesystem::path& vertexShaderPath,
                           const std::filesystem::path& fragmentShaderPath) {
    Reset();

    const auto vertexSource = ReadFile(vertexShaderPath);
    const auto fragmentSource = ReadFile(fragmentShaderPath);
    const auto vertexShader = Compile(GL_VERTEX_SHADER, vertexSource);
    const auto fragmentShader = Compile(GL_FRAGMENT_SHADER, fragmentSource);

    program_ = glCreateProgram();
    glAttachShader(program_, vertexShader);
    glAttachShader(program_, fragmentShader);
    glLinkProgram(program_);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program_, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(length));
        glGetProgramInfoLog(program_, length, nullptr, log.data());
        const auto message = std::string(log.data());
        Reset();
        throw std::runtime_error("graphics program link failed: " + message);
    }
}

void GraphicsProgram::Bind() const {
    glUseProgram(program_);
}

std::string GraphicsProgram::ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open shader: " + path.string());
    }

    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

unsigned int GraphicsProgram::Compile(unsigned int type, const std::string& source) {
    const char* sourcePtr = source.c_str();
    const auto shader = glCreateShader(type);
    glShaderSource(shader, 1, &sourcePtr, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) {
        return shader;
    }

    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> log(static_cast<std::size_t>(length));
    glGetShaderInfoLog(shader, length, nullptr, log.data());
    glDeleteShader(shader);
    throw std::runtime_error("graphics shader compilation failed: " + std::string(log.data()));
}

void GraphicsProgram::Reset() {
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
}

} // namespace chmv::gpu
