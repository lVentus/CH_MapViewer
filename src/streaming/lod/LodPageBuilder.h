#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>

namespace chmv::streaming::lod {

enum class BuildStage {
    ReadNodes,
    ReadEdges,
    ReadRanges,
    ResolveShortcuts,
    ComputeErrors,
    ComputeContexts,
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
    std::uint32_t pagePasses = 0;
    std::uint64_t memoryBudgetBytes = 0;
    double seconds = 0.0;
};

// Builds the self-contained LOD page file from a .sch/.ranges pair, out of core: edge data
// lives in the output file and in scratch files next to it, memory holds only node positions
// and a few bytes per edge. The result is written under a temporary name, checked against
// DFS reference cuts, and only then moved to outputPath, so an interrupted build never leaves
// a half-written file behind.
class LodPageBuilder {
public:
    using ProgressCallback = std::function<void(const BuildProgress&)>;

    // Least free RAM a build needs; more only makes it use fewer passes.
    [[nodiscard]] static std::uint64_t EstimateMemoryBytes(std::uint64_t nodeCount,
                                                           std::uint64_t edgeCount);
    // Free disk space next to the output: the file itself plus scratch files.
    [[nodiscard]] static std::uint64_t EstimateDiskBytes(std::uint64_t nodeCount,
                                                         std::uint64_t edgeCount);

    // memoryBudgetBytes = 0 uses three quarters of the RAM that is free right now. The callback
    // may throw to cancel the build.
    [[nodiscard]] static BuildResult Build(const std::filesystem::path& graphPath,
                                           const std::filesystem::path& rangesPath,
                                           const std::filesystem::path& outputPath,
                                           const ProgressCallback& progress = {},
                                           std::uint64_t memoryBudgetBytes = 0);
};

} // namespace chmv::streaming::lod
