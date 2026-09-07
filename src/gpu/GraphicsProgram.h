#pragma once

#include <filesystem>
#include <string>

namespace chmv::gpu {

class GraphicsProgram {
public:
    GraphicsProgram() = default;
    GraphicsProgram(const std::filesystem::path& vertexShaderPath,
                    const std::filesystem::path& fragmentShaderPath);
    ~GraphicsProgram();

    GraphicsProgram(const GraphicsProgram&) = delete;
    GraphicsProgram& operator=(const GraphicsProgram&) = delete;
    GraphicsProgram(GraphicsProgram&& other) noexcept;
    GraphicsProgram& operator=(GraphicsProgram&& other) noexcept;

    void Load(const std::filesystem::path& vertexShaderPath,
              const std::filesystem::path& fragmentShaderPath);
    void Bind() const;

    [[nodiscard]] unsigned int Id() const { return program_; }

private:
    static std::string ReadFile(const std::filesystem::path& path);
    static unsigned int Compile(unsigned int type, const std::string& source);
    void Reset();

    unsigned int program_ = 0;
};

} // namespace chmv::gpu
