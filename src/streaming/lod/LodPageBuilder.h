#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>

namespace chmv::streaming::lod {

enum class BuildStage {
    ReadNodes,
    ReadEdges,
    ReadRanges,
    ComputeHierarchy,
    LayoutPages,
    WritePages,
    Verify,
};

struct BuildProgress {
    BuildStage stage = BuildStage::ReadNodes;
    std::uint64_t current = 0;
    std::uint64_t total = 0;
};

struct BuildResult {
    std::filesystem::path path;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    std::uint64_t recordCount = 0;
    std::uint64_t pageCount = 0;
    std::uint64_t coverGapEdges = 0;
    std::uint64_t extraContextRecords = 0;
    std::uint32_t verifiedCuts = 0;
    double seconds = 0.0;
};

// Builds the self-contained LOD page file from a .sch/.ranges pair. The result is written
// under a temporary name, checked against a DFS reference cut, and only then moved to
// outputPath, so an interrupted build never leaves a half-written file behind.
class LodPageBuilder {
public:
    using ProgressCallback = std::function<void(const BuildProgress&)>;

    // Rough peak heap use of Build(), used to refuse datasets that would not fit.
    [[nodiscard]] static std::uint64_t EstimateMemoryBytes(std::uint64_t nodeCount,
                                                           std::uint64_t edgeCount);

    // The callback may throw to cancel the build.
    [[nodiscard]] static BuildResult Build(const std::filesystem::path& graphPath,
                                           const std::filesystem::path& rangesPath,
                                           const std::filesystem::path& outputPath,
                                           const ProgressCallback& progress = {});
};

} // namespace chmv::streaming::lod
