#pragma once

#include "streaming/lod/LodPageBuilder.h"
#include "streaming/lod/LodPageFormat.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace chmv::streaming::lod {

// Random-access binary writer for the file under construction.
class LodPageOutput {
public:
    explicit LodPageOutput(const std::filesystem::path& path);

    void WriteAt(std::uint64_t offset, const void* data, std::uint64_t bytes);
    void Close();

private:
    std::filesystem::path path_;
    std::ofstream stream_;
};

// The part of the hierarchy the render layer is derived from. Everything else in the source
// goes straight to the CH-layer columns of the output file while parsing.
struct LodPageHierarchy {
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    // Fixed-point projected coordinates (kFixedCoordinateScale).
    std::vector<std::int32_t> nodeX;
    std::vector<std::int32_t> nodeY;
    std::vector<std::uint32_t> source;
    std::vector<std::uint32_t> target;
    std::vector<std::uint32_t> childA;
    std::vector<std::uint32_t> childB;
    std::vector<std::int16_t> birth;
    std::vector<std::int16_t> death;
    std::vector<std::int16_t> type;
    std::int32_t maxLevel = 0;

    [[nodiscard]] bool IsShortcut(std::uint32_t edge) const {
        return childA[edge] != 0xffffffffu && childB[edge] != 0xffffffffu;
    }
    [[nodiscard]] bool HasRootInterval(std::uint32_t edge) const {
        return birth[edge] >= 0 && death[edge] >= 0 && death[edge] <= birth[edge];
    }
};

// Places the CH-layer sections in `header` (counts come from the source header), parses both
// source files, streams the CH-layer columns into `output` and fills the projection fields.
[[nodiscard]] LodPageHierarchy ReadLodPageSource(const std::filesystem::path& graphPath,
                                                 const std::filesystem::path& rangesPath,
                                                 FileHeader& header, LodPageOutput& output,
                                                 std::uint64_t& nextSectionOffset,
                                                 const LodPageBuilder::ProgressCallback& progress);

[[nodiscard]] inline constexpr std::uint64_t AlignSection(std::uint64_t offset) {
    return (offset + kSectionAlignment - 1u) / kSectionAlignment * kSectionAlignment;
}

} // namespace chmv::streaming::lod
