#pragma once

#include <filesystem>
#include <string>

namespace chmv::gpu {

class ComputeProgram {
public:
    ComputeProgram() = default;
    explicit ComputeProgram(const std::filesystem::path& shaderPath);
    ~ComputeProgram();

    ComputeProgram(const ComputeProgram&) = delete;
    ComputeProgram& operator=(const ComputeProgram&) = delete;
    ComputeProgram(ComputeProgram&& other) noexcept;
    ComputeProgram& operator=(ComputeProgram&& other) noexcept;

    void Load(const std::filesystem::path& shaderPath);
    void Bind() const;
    void Dispatch(unsigned int groupsX, unsigned int groupsY = 1, unsigned int groupsZ = 1) const;

    [[nodiscard]] unsigned int Id() const { return program_; }

private:
    static std::string ReadFile(const std::filesystem::path& path);
    void Reset();

    unsigned int program_ = 0;
};

} // namespace chmv::gpu
