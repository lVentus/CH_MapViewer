#include "streaming/preprocess/DatasetPreprocessor.h"
#include "streaming/preprocess/ExternalPreprocess.h"

#include "streaming/index/CHIndex.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace chmv::streaming::preprocess {
namespace {

constexpr std::uint32_t kManifestVersion = 7;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

struct FileStamp {
    std::uint64_t size = 0;
    std::int64_t writeTime = 0;
};

FileStamp GetFileStamp(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        return {};
    }
    return {
        std::filesystem::file_size(path),
        static_cast<std::int64_t>(
            std::filesystem::last_write_time(path).time_since_epoch().count()),
    };
}

std::uint64_t QueryAvailablePhysicalMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        return status.ullAvailPhys;
    }
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t valueKiB = 0;
    std::string unit;
    while (meminfo >> key >> valueKiB >> unit) {
        if (key == "MemAvailable:") {
            return valueKiB * 1024ull;
        }
    }
#endif
    return 0;
}

std::uint64_t ResolvePreprocessMemoryBudget(std::uint64_t requested) {
    if (requested != 0) {
        return std::max<std::uint64_t>(requested, 64ull * kMiB);
    }

    const auto available = QueryAvailablePhysicalMemory();
    if (available == 0) {
        return 1024ull * kMiB;
    }

    // Use the memory that is actually free now, not a fixed machine-size cap. Keep a
    // substantial reserve for the OS file cache, graphics driver and the viewer itself.
    const auto reserve = std::min<std::uint64_t>(4ull * kGiB, available / 3u);
    const auto afterReserve = available > reserve ? available - reserve : available / 2u;
    const auto sixtyPercent = available * 3u / 5u;
    return std::max<std::uint64_t>(256ull * kMiB,
                                   std::min(afterReserve, sixtyPercent));
}

bool CanRunDetailedAnalyzer(std::uint64_t nodeCount, std::uint64_t edgeCount,
                            std::uint64_t drawableEdgeCount) {
    const auto estimate = nodeCount * 24ull + edgeCount * 64ull +
                          drawableEdgeCount * 20ull + (256ull << 20u);
    const auto available = QueryAvailablePhysicalMemory();
    return available != 0 && available > estimate + 2ull * kGiB;
}

void WriteFastPreprocessReport(
    const PreprocessPaths& paths,
    const analysis::DatasetLayoutAnalysisConfig& config,
    const index::CHIndexData& indexData,
    std::uint64_t drawableEdgeCount) {
    const auto directory = analysis::DatasetLayoutAnalyzer::DefaultReportDirectory(paths.graphPath);
    std::filesystem::create_directories(directory);

    std::uint64_t pageEdges = 0;
    std::uint32_t maxSpatialLevel = 0;
    std::unordered_map<std::uint32_t, std::uint64_t> pagesBySpatialLevel;
    std::unordered_map<std::uint64_t, std::uint64_t> pagesByLodRange;
    for (const auto& page : indexData.graphPages) {
        pageEdges += page.edgeCount;
        maxSpatialLevel = std::max(maxSpatialLevel, page.spatialLevel);
        ++pagesBySpatialLevel[page.spatialLevel];
        const auto lodKey = (static_cast<std::uint64_t>(page.levelMin) << 32u) | page.levelMax;
        ++pagesByLodRange[lodKey];
    }

    {
        const auto path = directory / "preprocess_summary.txt";
        std::ofstream output(path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "CH_MapViewer preprocess summary\n\n";
        output << "Nodes: " << indexData.nodeCount << '\n';
        output << "Edges: " << indexData.edgeCount << '\n';
        output << "Drawable lifetime edges: " << drawableEdgeCount << '\n';
        output << "Runtime root tiles: " << indexData.graphPages.size() << '\n';
        output << "Runtime root-tile edge sum: " << pageEdges << '\n';
        output << "Mean runtime root-tile fill: " << std::fixed << std::setprecision(2)
               << (indexData.graphPages.empty() || index::kRuntimeRootTileTargetEdges == 0
                       ? 0.0
                       : 100.0 * static_cast<double>(pageEdges) /
                             (static_cast<double>(indexData.graphPages.size()) *
                              static_cast<double>(index::kRuntimeRootTileTargetEdges)))
               << "%\n";
        output << "Spatial cell->page refs: " << indexData.spatialPageRefs.size() << '\n';
        output << "RootPayload storage: dense contiguous LOD-group Morton stream\n";
        output << "Runtime residency: fine-grained logical root tiles (no payload rewrite)\n";
        output << "Fixed base cell: " << std::fixed << std::setprecision(3)
               << (indexData.spatialCellSizeMeters / 1000.0) << " km\n";
        output << "World fixed-grid dimension: " << indexData.spatialGridSize << " x "
               << indexData.spatialGridSize << '\n';
        output << "Maximum direct spatial scale: L" << maxSpatialLevel << " ("
               << (indexData.spatialCellSizeMeters * static_cast<double>(std::uint64_t{1}
                                                                         << maxSpatialLevel) /
                   1000.0)
               << " km cells)\n";
        output << "LOD band width: " << config.lodBandWidth << '\n';
        output << "Source node blocks: " << indexData.nodeBlocks.size() << '\n';
        output << "Source edge blocks: " << indexData.graphEdgeBlocks.size() << '\n';
        output << "Runtime root tile target edges: " << index::kRuntimeRootTileTargetEdges << '\n';
        output << "Source/full-cook block target edges: " << config.virtualPageEdgeCount << '\n';
        output << "\nSpatial assignment is direct fixed-grid classification. RootPayload stays dense and contiguous; logical runtime tiles are small residency/I/O views into that payload, and exact spatial cells map directly to the tiles they touch.\n";
    }

    {
        const auto path = directory / "preprocess_runtime_pages.csv";
        std::ofstream output(path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "page_id,lod_group_index,level_min,level_max,spatial_level,tile_x,tile_y,sub_page,edge_count,source_block_count\n";
        for (const auto& page : indexData.graphPages) {
            output << page.pageId << ',' << page.bandIndex << ',' << page.levelMin << ','
                   << page.levelMax << ',' << page.spatialLevel << ',' << page.tileX << ','
                   << page.tileY << ',' << page.subPage << ',' << page.edgeCount << ','
                   << page.sourceBlockRefCount << '\n';
        }
    }

    {
        const auto path = directory / "preprocess_spatial_levels.csv";
        std::ofstream output(path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "spatial_level,cell_size_meters,page_count\n";
        std::vector<std::pair<std::uint32_t, std::uint64_t>> rows(pagesBySpatialLevel.begin(),
                                                                  pagesBySpatialLevel.end());
        std::sort(rows.begin(), rows.end());
        for (const auto& [level, count] : rows) {
            output << level << ','
                   << config.spatialCellSizeMeters * static_cast<double>(std::uint64_t{1} << level)
                   << ',' << count << '\n';
        }
    }

    {
        const auto path = directory / "preprocess_lod_page_counts.csv";
        std::ofstream output(path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "level_min,level_max,page_count\n";
        std::vector<std::pair<std::uint64_t, std::uint64_t>> rows(pagesByLodRange.begin(),
                                                                  pagesByLodRange.end());
        std::sort(rows.begin(), rows.end());
        for (const auto& [key, count] : rows) {
            output << static_cast<std::uint32_t>(key >> 32u) << ','
                   << static_cast<std::uint32_t>(key) << ',' << count << '\n';
        }
    }
}

std::filesystem::path TemporaryPath(const std::filesystem::path& finalPath) {
    return std::filesystem::path(finalPath.string() + ".tmp");
}

void RemoveIfExists(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

void CommitTemporaryFile(const std::filesystem::path& temporaryPath,
                         const std::filesystem::path& finalPath) {
    RemoveIfExists(finalPath);
    std::filesystem::rename(temporaryPath, finalPath);
}

void RequirePreprocessDiskSpace(const std::filesystem::path& sourceGraphPath,
                                const std::filesystem::path& sourceRangesPath,
                                const std::filesystem::path& destinationDirectory) {
    std::error_code error;
    const auto graphBytes = std::filesystem::file_size(sourceGraphPath, error);
    if (error) {
        return;
    }
    const auto rangeBytes = std::filesystem::file_size(sourceRangesPath, error);
    if (error) {
        return;
    }
    const auto sourceBytes = graphBytes + rangeBytes;

    auto probe = destinationDirectory;
    while (!probe.empty() && !std::filesystem::exists(probe, error)) {
        probe = probe.parent_path();
    }
    if (probe.empty() || error) {
        return;
    }
    const auto space = std::filesystem::space(probe, error);
    if (error) {
        return;
    }

    // The new direct-grid path no longer writes one temporary copy per spatial depth,
    // but it still keeps compact mapped metadata/remap arrays while writing the new text pair.
    const auto margin = std::max<std::uint64_t>(2ull * kGiB, sourceBytes / 3u);
    const auto requiredBytes = sourceBytes + margin;
    if (space.available < requiredBytes) {
        throw std::runtime_error(
            "not enough free disk space for preprocessing: need about " +
            std::to_string(requiredBytes / kMiB) + " MiB free, but only " +
            std::to_string(space.available / kMiB) + " MiB is available");
    }
}

std::pair<std::filesystem::path, std::filesystem::path> AuthoritativeSourcePaths(
    const PreprocessPaths& paths) {
    const bool graphBackup = std::filesystem::is_regular_file(paths.originalGraphPath);
    const bool rangesBackup = std::filesystem::is_regular_file(paths.originalRangesPath);
    if (graphBackup != rangesBackup) {
        throw std::runtime_error(
            "preprocess source backup is incomplete; expected both .sch.org and .sch.ranges.org");
    }
    if (graphBackup) {
        return {paths.originalGraphPath, paths.originalRangesPath};
    }
    return {paths.graphPath, paths.rangesPath};
}

void BackupOriginalSources(const PreprocessPaths& paths) {
    if (std::filesystem::is_regular_file(paths.originalGraphPath) &&
        std::filesystem::is_regular_file(paths.originalRangesPath)) {
        return;
    }
    if (std::filesystem::exists(paths.originalGraphPath) ||
        std::filesystem::exists(paths.originalRangesPath)) {
        throw std::runtime_error(
            "cannot create preprocess backup because an incomplete .org pair exists");
    }

    std::filesystem::rename(paths.graphPath, paths.originalGraphPath);
    try {
        std::filesystem::rename(paths.rangesPath, paths.originalRangesPath);
    } catch (...) {
        std::filesystem::rename(paths.originalGraphPath, paths.graphPath);
        throw;
    }
}

bool ParseStampedLine(const std::string& line, const std::string& key,
                      const std::filesystem::path& sourcePath) {
    const auto prefix = key + '=';
    if (!line.starts_with(prefix)) {
        return false;
    }
    const auto payload = line.substr(prefix.size());
    const auto first = payload.find('|');
    const auto second = first == std::string::npos ? std::string::npos
                                                    : payload.find('|', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
        return false;
    }
    try {
        const auto size = static_cast<std::uint64_t>(
            std::stoull(payload.substr(first + 1, second - first - 1)));
        const auto time = static_cast<std::int64_t>(std::stoll(payload.substr(second + 1)));
        const auto stamp = GetFileStamp(sourcePath);
        return stamp.size == size && stamp.writeTime == time;
    } catch (...) {
        return false;
    }
}

void WriteManifest(const PreprocessPaths& paths,
                   const analysis::DatasetLayoutAnalysisConfig& config,
                   std::uint64_t nodeCount, std::uint64_t edgeCount,
                   const std::filesystem::path& outputPath) {
    const auto sourceGraphStamp = GetFileStamp(paths.originalGraphPath);
    const auto sourceRangesStamp = GetFileStamp(paths.originalRangesPath);
    const auto graphStamp = GetFileStamp(paths.graphPath);
    const auto rangesStamp = GetFileStamp(paths.rangesPath);
    std::ofstream output(outputPath, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not write preprocess manifest: " + outputPath.string());
    }
    output << "format_version=" << kManifestVersion << '\n';
    output << "source_graph=" << paths.originalGraphPath.filename().generic_string() << '|'
           << sourceGraphStamp.size << '|' << sourceGraphStamp.writeTime << '\n';
    output << "source_ranges=" << paths.originalRangesPath.filename().generic_string() << '|'
           << sourceRangesStamp.size << '|' << sourceRangesStamp.writeTime << '\n';
    output << "preprocessed_graph=" << paths.graphPath.filename().generic_string() << '|'
           << graphStamp.size << '|' << graphStamp.writeTime << '\n';
    output << "preprocessed_ranges=" << paths.rangesPath.filename().generic_string() << '|'
           << rangesStamp.size << '|' << rangesStamp.writeTime << '\n';
    output << "node_count=" << nodeCount << '\n';
    output << "edge_count=" << edgeCount << '\n';
    output << std::setprecision(17);
    output << "spatial_cell_meters=" << config.spatialCellSizeMeters << '\n';
    output << "lod_band_width=" << config.lodBandWidth << '\n';
    output << "source_block_node_count=" << config.sourceBlockNodeCount << '\n';
    output << "source_block_edge_count=" << config.sourceBlockEdgeCount << '\n';
    output << "virtual_page_edge_count=" << config.virtualPageEdgeCount << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize preprocess manifest");
    }
}

bool ManifestMatches(const PreprocessPaths& paths,
                     const analysis::DatasetLayoutAnalysisConfig& config) {
    if (!std::filesystem::is_regular_file(paths.manifestPath) ||
        !std::filesystem::is_regular_file(paths.graphPath) ||
        !std::filesystem::is_regular_file(paths.rangesPath) ||
        !std::filesystem::is_regular_file(paths.originalGraphPath) ||
        !std::filesystem::is_regular_file(paths.originalRangesPath) ||
        !std::filesystem::is_regular_file(paths.indexPath)) {
        return false;
    }

    std::ifstream input(paths.manifestPath);
    if (!input) {
        return false;
    }

    bool version = false;
    bool sourceGraph = false;
    bool sourceRanges = false;
    bool graph = false;
    bool ranges = false;
    bool cellSize = false;
    bool band = false;
    bool nodeBlock = false;
    bool edgeBlock = false;
    bool pageEdges = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line == "format_version=" + std::to_string(kManifestVersion)) {
            version = true;
        } else if (line.starts_with("source_graph=")) {
            sourceGraph = ParseStampedLine(line, "source_graph", paths.originalGraphPath);
        } else if (line.starts_with("source_ranges=")) {
            sourceRanges = ParseStampedLine(line, "source_ranges", paths.originalRangesPath);
        } else if (line.starts_with("preprocessed_graph=")) {
            graph = ParseStampedLine(line, "preprocessed_graph", paths.graphPath);
        } else if (line.starts_with("preprocessed_ranges=")) {
            ranges = ParseStampedLine(line, "preprocessed_ranges", paths.rangesPath);
        } else if (line.starts_with("spatial_cell_meters=")) {
            try {
                const auto value = std::stod(line.substr(std::string("spatial_cell_meters=").size()));
                cellSize = std::abs(value - config.spatialCellSizeMeters) <=
                           std::max(1e-9, std::abs(config.spatialCellSizeMeters) * 1e-12);
            } catch (...) {
                cellSize = false;
            }
        } else if (line == "lod_band_width=" + std::to_string(config.lodBandWidth)) {
            band = true;
        } else if (line == "source_block_node_count=" +
                                   std::to_string(config.sourceBlockNodeCount)) {
            nodeBlock = true;
        } else if (line == "source_block_edge_count=" +
                                   std::to_string(config.sourceBlockEdgeCount)) {
            edgeBlock = true;
        } else if (line == "virtual_page_edge_count=" +
                                   std::to_string(config.virtualPageEdgeCount)) {
            pageEdges = true;
        }
    }

    const bool manifestMatches = version && sourceGraph && sourceRanges && graph && ranges &&
                                 cellSize && band && nodeBlock && edgeBlock && pageEdges;
    if (!manifestMatches) {
        return false;
    }

    return index::CHIndex::IsValid(paths.indexPath, paths.graphPath, paths.rangesPath, config);
}

} // namespace

PreprocessPaths DatasetPreprocessor::DefaultPaths(
    const std::filesystem::path& sourceGraphPath,
    const std::filesystem::path& sourceRangesPath) {
    PreprocessPaths paths;
    paths.directory = sourceGraphPath.parent_path();
    paths.graphPath = sourceGraphPath;
    paths.rangesPath = sourceRangesPath;
    paths.indexPath = index::CHIndex::DefaultPath(paths.graphPath);
    const auto baseName = sourceGraphPath.extension() == ".sch"
                              ? sourceGraphPath.stem().string()
                              : sourceGraphPath.filename().string();
    paths.manifestPath = paths.directory / (baseName + ".preprocess_manifest");
    paths.originalGraphPath = std::filesystem::path(sourceGraphPath.string() + ".org");
    paths.originalRangesPath = std::filesystem::path(sourceRangesPath.string() + ".org");
    return paths;
}

bool DatasetPreprocessor::HasValidPreprocess(
    const std::filesystem::path& sourceGraphPath,
    const std::filesystem::path& sourceRangesPath,
    const analysis::DatasetLayoutAnalysisConfig& config) {
    try {
        return ManifestMatches(DefaultPaths(sourceGraphPath, sourceRangesPath), config);
    } catch (...) {
        return false;
    }
}

PreprocessResult DatasetPreprocessor::Run(
    const std::filesystem::path& sourceGraphPath,
    const std::filesystem::path& sourceRangesPath,
    const analysis::DatasetLayoutAnalysisConfig& config,
    bool force, ProgressCallback progressCallback,
    const PreprocessOptions& options) {
    const auto paths = DefaultPaths(sourceGraphPath, sourceRangesPath);
    if (!force && ManifestMatches(paths, config)) {
        const auto indexData = index::CHIndex::Load(paths.indexPath);
        std::uint64_t drawableEdgeCount = 0;
        for (const auto& page : indexData.graphPages) {
            drawableEdgeCount += page.edgeCount;
        }

        // Reporting is metadata, not a reason to preprocess again. Always recreate the cheap
        // index-derived reports when loading an existing preprocess result. If an older fast
        // preprocess omitted the thesis reports, regenerate them once when the dataset fits the
        // analyzer's safe memory budget.
        WriteFastPreprocessReport(paths, config, indexData, drawableEdgeCount);
        if (options.writeDetailedAnalysis) {
            const auto reportDirectory =
                analysis::DatasetLayoutAnalyzer::DefaultReportDirectory(paths.graphPath);
            const auto detailedSummary = reportDirectory / "summary.txt";
            if (!std::filesystem::is_regular_file(detailedSummary) &&
                CanRunDetailedAnalyzer(indexData.nodeCount, indexData.edgeCount,
                                       drawableEdgeCount)) {
                auto report = analysis::DatasetLayoutAnalyzer::Analyze(
                    paths.graphPath, paths.rangesPath, config);
                analysis::DatasetLayoutAnalyzer::WriteReport(report, reportDirectory);
            }
        }
        return {paths, indexData.nodeCount, indexData.edgeCount, true};
    }

    const auto [authoritativeGraphPath, authoritativeRangesPath] =
        AuthoritativeSourcePaths(paths);
    if (!std::filesystem::is_regular_file(authoritativeGraphPath) ||
        !std::filesystem::is_regular_file(authoritativeRangesPath)) {
        throw std::runtime_error("preprocess requires an existing .sch/.ranges source pair");
    }

    RequirePreprocessDiskSpace(authoritativeGraphPath, authoritativeRangesPath, paths.directory);
    const auto graphTemporary = TemporaryPath(paths.graphPath);
    const auto rangesTemporary = TemporaryPath(paths.rangesPath);
    const auto indexTemporary = TemporaryPath(paths.indexPath);
    const auto manifestTemporary = TemporaryPath(paths.manifestPath);
    RemoveIfExists(graphTemporary);
    RemoveIfExists(rangesTemporary);
    RemoveIfExists(indexTemporary);
    RemoveIfExists(manifestTemporary);

    const auto memoryBudgetBytes = ResolvePreprocessMemoryBudget(options.memoryBudgetBytes);
    const auto tempBase = options.tempDirectory.empty() ? paths.directory : options.tempDirectory;
    const auto tempRoot = tempBase / (paths.graphPath.filename().string() + ".preprocess_tmp");
    std::error_code tempError;
    std::filesystem::remove_all(tempRoot, tempError);

    try {
        std::cout << "Preprocess memory budget: "
                  << (memoryBudgetBytes / kMiB) << " MiB"
                  << (options.memoryBudgetBytes == 0
                          ? " (auto from currently available RAM)\n"
                          : " (explicit override)\n");
        const auto result = detail::RunExternalPreprocess(
            authoritativeGraphPath, authoritativeRangesPath, graphTemporary, rangesTemporary,
            indexTemporary, config, memoryBudgetBytes, tempRoot,
            options.strictOutputValidation, progressCallback);

        BackupOriginalSources(paths);
        CommitTemporaryFile(graphTemporary, paths.graphPath);
        CommitTemporaryFile(rangesTemporary, paths.rangesPath);
        CommitTemporaryFile(indexTemporary, paths.indexPath);
        if (!index::CHIndex::IsValid(paths.indexPath, paths.graphPath, paths.rangesPath, config)) {
            throw std::runtime_error(
                "committed preprocess index does not match the preprocessed dataset");
        }

        WriteManifest(paths, config, result.nodeCount, result.edgeCount, manifestTemporary);
        CommitTemporaryFile(manifestTemporary, paths.manifestPath);

        const auto committedIndex = index::CHIndex::Load(paths.indexPath);
        WriteFastPreprocessReport(paths, config, committedIndex, result.drawableEdgeCount);

        if (options.writeDetailedAnalysis) {
            if (CanRunDetailedAnalyzer(result.nodeCount, result.edgeCount,
                                       result.drawableEdgeCount)) {
                auto report = analysis::DatasetLayoutAnalyzer::Analyze(
                    paths.graphPath, paths.rangesPath, config);
                analysis::DatasetLayoutAnalyzer::WriteReport(
                    report, analysis::DatasetLayoutAnalyzer::DefaultReportDirectory(paths.graphPath));
            } else {
                std::cout << "Detailed layout analysis skipped because the dataset does not fit "
                             "the safe in-memory analysis budget; lightweight preprocess reports "
                             "were written instead.\n";
            }
        }
        return {paths, result.nodeCount, result.edgeCount, false};
    } catch (...) {
        RemoveIfExists(graphTemporary);
        RemoveIfExists(rangesTemporary);
        RemoveIfExists(indexTemporary);
        RemoveIfExists(manifestTemporary);
        throw;
    }
}

} // namespace chmv::streaming::preprocess
