#pragma once

#include "streaming/analysis/DatasetLayoutAnalyzer.h"

#include <cstdint>
#include <filesystem>
#include <functional>

namespace chmv::streaming::preprocess {

enum class PreprocessStage {
    AnalyzeSource,
    LoadSource,
    ReorderNodes,
    ResolveGeometryBounds,
    ReorderEdges,
    WriteGraph,
    WriteRanges,
    ValidateOutput,
    WriteIndex,
};

struct PreprocessProgress {
    PreprocessStage stage = PreprocessStage::AnalyzeSource;
    std::uint64_t current = 0;
    std::uint64_t total = 0;
};

struct PreprocessPaths {
    std::filesystem::path directory;
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;
    std::filesystem::path indexPath;
    std::filesystem::path manifestPath;
    std::filesystem::path originalGraphPath;
    std::filesystem::path originalRangesPath;
};

struct PreprocessOptions {
    // Heap working-set budget for external-memory preprocessing. Zero automatically
    // derives a budget from currently available physical memory; there is no fixed cap.
    std::uint64_t memoryBudgetBytes = 0;
    // Optional scratch location. Empty keeps temporary files beside the dataset.
    std::filesystem::path tempDirectory;
    // Preserve the thesis/debug reports for datasets where the detailed analyzer fits safely in
    // memory. Large datasets automatically keep only the lightweight preprocess/index reports,
    // so reporting never turns a fast out-of-core preprocess back into a full-memory workload.
    bool writeDetailedAnalysis = true;
    // A full post-write text reparse is intentionally optional. The normal path validates
    // IDs/references/ranges while generating the output and performs lightweight index checks.
    bool strictOutputValidation = false;
};

struct PreprocessResult {
    PreprocessPaths paths;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    bool reused = false;
};

class DatasetPreprocessor {
public:
    using ProgressCallback = std::function<void(const PreprocessProgress&)>;

    [[nodiscard]] static PreprocessPaths DefaultPaths(
        const std::filesystem::path& sourceGraphPath,
        const std::filesystem::path& sourceRangesPath);

    [[nodiscard]] static bool HasValidPreprocess(
        const std::filesystem::path& sourceGraphPath,
        const std::filesystem::path& sourceRangesPath,
        const analysis::DatasetLayoutAnalysisConfig& config = {});

    [[nodiscard]] static PreprocessResult Run(
        const std::filesystem::path& sourceGraphPath,
        const std::filesystem::path& sourceRangesPath,
        const analysis::DatasetLayoutAnalysisConfig& config = {},
        bool force = false,
        ProgressCallback progressCallback = {},
        const PreprocessOptions& options = {});
};

} // namespace chmv::streaming::preprocess
