#pragma once

#include "streaming/analysis/DatasetLayoutAnalyzer.h"
#include "streaming/cook/FullCooker.h"

#include <cstdint>
#include <filesystem>

namespace chmv::streaming::cook::detail {

struct ExternalFullCookResult {
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    std::uint64_t drawableEdgeCount = 0;
    std::uint32_t graphPageCount = 0;
};

[[nodiscard]] ExternalFullCookResult RunExternalFullCook(
    const std::filesystem::path& sourceGraphPath,
    const std::filesystem::path& sourceRangesPath,
    const std::filesystem::path& outputGraphPath,
    const std::filesystem::path& outputRangesPath,
    const std::filesystem::path& outputIndexPath,
    const analysis::DatasetLayoutAnalysisConfig& config,
    std::uint64_t memoryBudgetBytes,
    const std::filesystem::path& tempRoot,
    const FullCooker::ProgressCallback& progressCallback);

} // namespace chmv::streaming::cook::detail
