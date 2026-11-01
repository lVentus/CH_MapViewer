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
    WriteHierarchy,
    Verify,
    Compress,
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
    std::uint64_t visitOnlyRecords = 0;
    std::uint32_t verifiedCuts = 0;
    std::uint32_t pagePasses = 0;
    std::uint64_t memoryBudgetBytes = 0;
    // Size of the file, and of the same file uncompressed.
    std::uint64_t fileBytes = 0;
    std::uint64_t uncompressedBytes = 0;
    double seconds = 0.0;
};

// Builds the self-contained LOD page file from a .sch/.ranges pair, out of core: edge data
// lives in the output file and in scratch files next to it, memory holds only node positions
// and a few bytes per edge. The result is written uncompressed under a temporary name, checked
// against DFS reference cuts, compressed into a second temporary file that must decode to
// exactly the checked bytes, and only then moved to outputPath, so an interrupted build never
// leaves a half-written file behind.
class LodPageBuilder {
public:
    using ProgressCallback = std::function<void(const BuildProgress&)>;

    // Least free RAM a build needs; more only makes it use fewer passes.
    [[nodiscard]] static std::uint64_t EstimateMemoryBytes(std::uint64_t nodeCount,
                                                           std::uint64_t edgeCount);
    // Free disk space next to the output: the uncompressed file, its scratch files and the
    // compressed copy.
    [[nodiscard]] static std::uint64_t EstimateDiskBytes(std::uint64_t nodeCount,
                                                         std::uint64_t edgeCount);

    // memoryBudgetBytes = 0 uses three quarters of the RAM that is free right now. The callback
    // may throw to cancel the build.
    [[nodiscard]] static BuildResult Build(const std::filesystem::path& graphPath,
                                           const std::filesystem::path& rangesPath,
                                           const std::filesystem::path& outputPath,
                                           const ProgressCallback& progress = {},
                                           std::uint64_t memoryBudgetBytes = 0);
    // Compresses an uncompressed page file (version 3) in place, without its sources; the
    // result must decode to exactly the same bytes. Reports BuildStage::Compress.
    static void Compress(const std::filesystem::path& path, const ProgressCallback& progress = {});
};

} // namespace chmv::streaming::lod
