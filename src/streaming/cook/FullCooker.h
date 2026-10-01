#pragma once

#include "streaming/analysis/DatasetLayoutAnalyzer.h"

#include <cstdint>
#include <filesystem>
#include <functional>

namespace chmv::streaming::cook {

enum class FullCookStage {
    AnalyzeSource,
    LoadSource,
    ReorderNodes,
    ReorderEdges,
    WriteGraph,
    WriteRanges,
    AnalyzeCooked,
    WriteIndex,
};

struct FullCookProgress {
    FullCookStage stage = FullCookStage::AnalyzeSource;
    std::uint64_t current = 0;
    std::uint64_t total = 0;
};

struct FullCookPaths {
    std::filesystem::path directory;
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;
    std::filesystem::path indexPath;
    std::filesystem::path manifestPath;
    std::filesystem::path originalGraphPath;
    std::filesystem::path originalRangesPath;
};

struct FullCookOptions {
    // Heap working-set budget for external-memory sorting/partitioning. Zero selects an automatic budget.
    std::uint64_t memoryBudgetBytes = 0;
    // Optional scratch location. Empty keeps temporary files beside the dataset.
    std::filesystem::path tempDirectory;
    // Detailed CSV analysis is useful for thesis measurements but is skipped automatically for very large datasets.
    bool writeDetailedAnalysis = true;
    // Test/debug override; normal builds select external mode automatically when the source exceeds the cook budget.
    bool forceExternalMemory = false;
};

struct FullCookResult {
    FullCookPaths paths;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    bool reused = false;
};

class FullCooker {
public:
    using ProgressCallback = std::function<void(const FullCookProgress&)>;

    [[nodiscard]] static FullCookPaths DefaultPaths(
        const std::filesystem::path& sourceGraphPath,
        const std::filesystem::path& sourceRangesPath);

    [[nodiscard]] static bool HasValidCook(
        const std::filesystem::path& sourceGraphPath,
        const std::filesystem::path& sourceRangesPath,
        const analysis::DatasetLayoutAnalysisConfig& config = {});

    [[nodiscard]] static FullCookResult Cook(
        const std::filesystem::path& sourceGraphPath,
        const std::filesystem::path& sourceRangesPath,
        const analysis::DatasetLayoutAnalysisConfig& config = {},
        bool force = false,
        ProgressCallback progressCallback = {},
        const FullCookOptions& options = {});
};

} // namespace chmv::streaming::cook
