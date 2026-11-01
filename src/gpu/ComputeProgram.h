#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace chmv::gpu {

// A compute shader program. Sources may pull in files next to them with #include "name", and
// `defines` (lines like "#define STACK_SIZE 29\n") are inserted right after #version.
class ComputeProgram {
public:
    ComputeProgram() = default;
    explicit ComputeProgram(const std::filesystem::path& shaderPath,
                            std::string_view defines = {});
    ~ComputeProgram();

    ComputeProgram(const ComputeProgram&) = delete;
    ComputeProgram& operator=(const ComputeProgram&) = delete;
    ComputeProgram(ComputeProgram&& other) noexcept;
    ComputeProgram& operator=(ComputeProgram&& other) noexcept;

    void Load(const std::filesystem::path& shaderPath, std::string_view defines = {});
    void Bind() const;
    void Dispatch(unsigned int groupsX, unsigned int groupsY = 1, unsigned int groupsZ = 1) const;

    [[nodiscard]] unsigned int Id() const { return program_; }

private:
    static std::string ReadSource(const std::filesystem::path& path, int depth = 0);
    void Reset();

    unsigned int program_ = 0;
};

} // namespace chmv::gpu
