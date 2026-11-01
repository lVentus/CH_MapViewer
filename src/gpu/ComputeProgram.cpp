#include "gpu/ComputeProgram.h"

#include <glad/gl.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace chmv::gpu {

ComputeProgram::ComputeProgram(const std::filesystem::path& shaderPath,
                               std::string_view defines) {
    Load(shaderPath, defines);
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

void ComputeProgram::Load(const std::filesystem::path& shaderPath, std::string_view defines) {
    Reset();

    auto source = ReadSource(shaderPath);
    if (!defines.empty()) {
        const auto versionEnd = source.find('\n');
        source.insert(versionEnd == std::string::npos ? source.size() : versionEnd + 1u, defines);
    }
    const char* sourcePtr = source.c_str();

    const auto shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &sourcePtr, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("compute shader compilation failed (" + shaderPath.string() +
                                 "): " + log.c_str());
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
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program_, static_cast<GLsizei>(log.size()), nullptr, log.data());
        const std::string message = log.c_str();
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

std::string ComputeProgram::ReadSource(const std::filesystem::path& path, int depth) {
    if (depth > 8) {
        throw std::runtime_error("shader includes nested too deeply: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open shader: " + path.string());
    }

    constexpr std::string_view kInclude = "#include \"";
    std::string source;
    std::string line;
    while (std::getline(input, line)) {
        if (line.starts_with(kInclude)) {
            const auto end = line.find('"', kInclude.size());
            if (end == std::string::npos) {
                throw std::runtime_error("malformed #include in shader: " + path.string());
            }
            source += ReadSource(
                path.parent_path() / line.substr(kInclude.size(), end - kInclude.size()),
                depth + 1);
        } else {
            source += line;
            source += '\n';
        }
    }
    return source;
}

void ComputeProgram::Reset() {
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
}

} // namespace chmv::gpu
