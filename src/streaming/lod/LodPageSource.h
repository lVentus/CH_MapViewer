#pragma once

#include "streaming/lod/LodPageBuilder.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPageIo.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

namespace chmv::streaming::lod {

// What stays in memory after parsing: every node's projected position and CH level. All edge
// data goes straight into the CH-layer columns of the output file.
struct LodPageNodes {
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    // Fixed-point projected coordinates (kFixedCoordinateScale).
    std::vector<std::int32_t> x;
    std::vector<std::int32_t> y;
    std::vector<std::uint16_t> level;
    std::int32_t maxRangeLevel = 0;
    std::uint32_t maxNodeLevel = 0;

    [[nodiscard]] std::uint64_t Bytes() const { return nodeCount * 10u; }
};

using SectionHashes = std::array<StreamHash, kSectionCount>;
// Called with the node and edge counts before anything large is allocated; may throw.
using SourceCountsCallback = std::function<void(std::uint64_t nodes, std::uint64_t edges)>;

// Places the CH-layer sections in `header`, parses both source files, streams the CH-layer
// columns into `output` (hashing them) and fills the projection fields.
[[nodiscard]] LodPageNodes
ReadLodPageSource(const std::filesystem::path& graphPath, const std::filesystem::path& rangesPath,
                  FileHeader& header, LodPageOutput& output, SectionHashes& hashes,
                  std::uint64_t& nextSectionOffset, const SourceCountsCallback& checkCounts,
                  const LodPageBuilder::ProgressCallback& progress);

[[nodiscard]] inline constexpr std::uint64_t AlignSection(std::uint64_t offset) {
    return (offset + kSectionAlignment - 1u) / kSectionAlignment * kSectionAlignment;
}

} // namespace chmv::streaming::lod
