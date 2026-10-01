#include "streaming/analysis/DatasetLayoutAnalyzer.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace chmv::streaming::analysis {
namespace {

double Ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
               ? 0.0
               : static_cast<double>(numerator) / static_cast<double>(denominator);
}


constexpr std::uint32_t kCacheFormatVersion = 3;

constexpr std::array<const char*, 7> kReportFiles{
    "summary.txt",
    "graph_source_blocks.csv",
    "range_source_blocks.csv",
    "lod_density.csv",
    "lod_band_spatial_density.csv",
    "virtual_page_locality.csv",
    "runtime_graph_pages.csv",
};

std::string FileSignature(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        return {};
    }

    std::ostringstream output;
    output << path.filename().generic_string() << '|';
    output << std::filesystem::file_size(path) << '|';
    output << std::filesystem::last_write_time(path).time_since_epoch().count();
    return output.str();
}

std::string CacheManifestText(const std::filesystem::path& graphPath,
                              const std::filesystem::path& rangesPath,
                              const DatasetLayoutAnalysisConfig& config) {
    std::ostringstream output;
    output << "format_version=" << kCacheFormatVersion << '\n';
    output << "graph=" << FileSignature(graphPath) << '\n';
    output << "ranges=" << FileSignature(rangesPath) << '\n';
    output << "spatial_grid_size=" << config.spatialGridSize << '\n';
    output << "lod_band_width=" << config.lodBandWidth << '\n';
    output << "source_block_node_count=" << config.sourceBlockNodeCount << '\n';
    output << "source_block_edge_count=" << config.sourceBlockEdgeCount << '\n';
    output << "virtual_page_edge_count=" << config.virtualPageEdgeCount << '\n';
    return output.str();
}

std::string ReadTextFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}
void WriteSourceBlocks(const std::filesystem::path& path,
                       const std::vector<SourceBlockInfo>& blocks) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not write " + path.string());
    }
    output << "block_id,first_record,record_count,byte_offset\n";
    for (const auto& block : blocks) {
        output << block.blockId << ',' << block.firstRecord << ',' << block.recordCount << ','
               << block.byteOffset << '\n';
    }
}

} // namespace

std::filesystem::path DatasetLayoutAnalyzer::DefaultReportDirectory(
    const std::filesystem::path& graphPath) {
    return graphPath.parent_path() / "layout-analysis";
}

bool DatasetLayoutAnalyzer::HasValidCachedReport(
    const std::filesystem::path& graphPath, const std::filesystem::path& rangesPath,
    const DatasetLayoutAnalysisConfig& config,
    const std::filesystem::path& outputDirectory) {
    if (!std::filesystem::is_directory(outputDirectory)) {
        return false;
    }

    for (const auto* filename : kReportFiles) {
        if (!std::filesystem::is_regular_file(outputDirectory / filename)) {
            return false;
        }
    }

    const auto manifestPath = outputDirectory / "cache_manifest.txt";
    if (!std::filesystem::is_regular_file(manifestPath)) {
        return false;
    }

    return ReadTextFile(manifestPath) == CacheManifestText(graphPath, rangesPath, config);
}

std::string DatasetLayoutAnalyzer::ReadCachedSummary(
    const std::filesystem::path& outputDirectory) {
    const auto summary = ReadTextFile(outputDirectory / "summary.txt");
    if (summary.empty()) {
        throw std::runtime_error("could not read cached layout-analysis summary");
    }
    return summary;
}

std::string DatasetLayoutAnalyzer::SummaryText(const DatasetLayoutAnalysisReport& report) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(4);
    output << "CH_MapViewer dataset layout analysis\n";
    output << "Graph: " << report.graphPath.string() << '\n';
    output << "Ranges: " << report.rangesPath.string() << "\n\n";
    output << "Configuration\n";
    output << "  Spatial grid: " << report.config.spatialGridSize << 'x'
           << report.config.spatialGridSize << '\n';
    output << "  LOD band width: " << report.config.lodBandWidth << '\n';
    output << "  Source block nodes: " << report.config.sourceBlockNodeCount << '\n';
    output << "  Source block edges: " << report.config.sourceBlockEdgeCount << '\n';
    output << "  Virtual page target edges: " << report.config.virtualPageEdgeCount << "\n\n";

    output << "Dataset\n";
    output << "  Nodes: " << report.nodeCount << '\n';
    output << "  Edges: " << report.edgeCount << '\n';
    output << "  Shortcuts: " << report.shortcutCount << '\n';
    output << "  Drawable lifetime edges: " << report.drawableEdgeCount << '\n';
    output << "  Non-drawable (-1 -1): " << report.nonDrawableEdgeCount << '\n';
    output << "  Invalid ranges: " << report.invalidRangeCount << '\n';
    output << "  Duplicate range records: " << report.duplicateRangeRecordCount << '\n';
    output << "  Missing range records: " << report.missingRangeRecordCount << '\n';
    output << "  Node records already ID-ordered: "
           << report.nodeRecordSequentialRatio * 100.0 << "%\n";
    output << "  Range records already EdgeID-ordered: "
           << report.rangeRecordSequentialRatio * 100.0 << "%\n\n";

    output << "Source EdgeID spatial locality\n";
    output << "  Consecutive edges in same analysis cell: "
           << report.sourceOrderSameSpatialCellRatio * 100.0 << "%\n";
    output << "  Consecutive edges within 1-cell ring: "
           << report.sourceOrderWithinOneCellRatio * 100.0 << "%\n";
    output << "  Consecutive edges within 5-cell ring: "
           << report.sourceOrderWithinFiveCellsRatio * 100.0 << "%\n\n";

    output << "Source EdgeID lifetime locality\n";
    output << "  Consecutive drawable edges at same birth level: "
           << report.sourceOrderSameBirthLevelRatio * 100.0 << "%\n";
    output << "  Consecutive drawable edges in same LOD band: "
           << report.sourceOrderSameLodBandRatio * 100.0 << "%\n";
    output << "  Birth-level delta P50/P95/P99: " << report.sourceOrderBirthDeltaP50 << " / "
           << report.sourceOrderBirthDeltaP95 << " / " << report.sourceOrderBirthDeltaP99
           << "\n\n";

    output << "Shortcut hierarchy locality\n";
    output << "  Parent-child references: " << report.parentChildReferenceCount << '\n';
    output << "  Same source block: " << report.parentChildSameSourceBlockRatio * 100.0
           << "%\n";
    output << "  Within 1 source block: "
           << report.parentChildWithinOneSourceBlockRatio * 100.0 << "%\n";
    output << "  Within 5 source blocks: "
           << report.parentChildWithinFiveSourceBlocksRatio * 100.0 << "%\n";
    output << "  EdgeID distance P50/P95/P99 upper bounds: "
           << report.parentChildIdDistanceP50UpperBound << " / "
           << report.parentChildIdDistanceP95UpperBound << " / "
           << report.parentChildIdDistanceP99UpperBound << '\n';
    output << "  EdgeID distance max: " << report.parentChildIdDistanceMax << "\n\n";

    output << "Lifetime span\n";
    output << "  P50/P95/P99/max: " << report.lifetimeSpanP50 << " / "
           << report.lifetimeSpanP95 << " / " << report.lifetimeSpanP99 << " / "
           << report.lifetimeSpanMax << "\n\n";

    output << "Drawable shortcuts vs refinement children\n";
    output << "  Drawable shortcuts: " << report.drawableShortcutCount << '\n';
    output << "  Direct child references: " << report.drawableShortcutChildReferenceCount << '\n';
    output << "  Children that also have drawable lifetimes: "
           << report.drawableShortcutChildWithDrawableLifetimeCount << " ("
           << Ratio(report.drawableShortcutChildWithDrawableLifetimeCount,
                    report.drawableShortcutChildReferenceCount) *
                  100.0
           << "%)\n";
    output << "  Note: non-drawable children still matter for geometry refinement, so lifetime/root "
              "pages cannot replace the full graph backing store.\n\n";

    output << "Hypothetical lifetime+Morton virtual pages\n";
    output << "  Pages: " << report.virtualPageCount << '\n';
    output << "  Mean page fill: " << report.virtualPageMeanFill * 100.0 << "%\n";
    output << "  Source blocks/page P50/P95/P99/max: " << report.virtualPageSourceBlockP50
           << " / " << report.virtualPageSourceBlockP95 << " / "
           << report.virtualPageSourceBlockP99 << " / " << report.virtualPageSourceBlockMax
           << '\n';
    output << "  Direct-child source blocks/page P50/P95/P99/max: "
           << report.virtualPageDirectChildSourceBlockP50 << " / "
           << report.virtualPageDirectChildSourceBlockP95 << " / "
           << report.virtualPageDirectChildSourceBlockP99 << " / "
           << report.virtualPageDirectChildSourceBlockMax << '\n';
    output << "\nThese virtual pages are an analysis model: drawable roots are grouped by birth-level "
              "band, then Morton order, without duplicating source edge records.\n\n";
    output << "Runtime adaptive LOD-spatial pages\n";
    output << "  Pages: " << report.runtimeGraphPages.size() << '\n';
    if (!report.runtimeGraphPages.empty()) {
        std::uint32_t globalPages = 0;
        std::uint32_t maxSpatialLevel = 0;
        std::uint64_t sourceRefs = 0;
        for (const auto& page : report.runtimeGraphPages) {
            globalPages += page.spatialLevel == 0 ? 1u : 0u;
            maxSpatialLevel = std::max(maxSpatialLevel, page.spatialLevel);
            sourceRefs += page.sourceBlockRefCount;
        }
        output << "  Whole-dataset pages: " << globalPages << '\n';
        output << "  Maximum spatial level: " << maxSpatialLevel << '\n';
        output << "  Mean source blocks/page: "
               << static_cast<double>(sourceRefs) /
                      static_cast<double>(report.runtimeGraphPages.size())
               << '\n';
        output << "  Source blocks/page P50/P95/P99/max: "
               << report.runtimeGraphPageSourceBlockP50 << " / "
               << report.runtimeGraphPageSourceBlockP95 << " / "
               << report.runtimeGraphPageSourceBlockP99 << " / "
               << report.runtimeGraphPageSourceBlockMax << '\n';
        output << "  Direct-child source blocks/page P50/P95/P99/max: "
               << report.runtimeGraphPageChildSourceBlockP50 << " / "
               << report.runtimeGraphPageChildSourceBlockP95 << " / "
               << report.runtimeGraphPageChildSourceBlockP99 << " / "
               << report.runtimeGraphPageChildSourceBlockMax << '\n';
    }
    output << "  Runtime layout: adjacent birth/LOD levels are merged while small; overloaded groups are spatially subdivided.\n";
    output << "  Spatial ownership: each edge is stored once at the deepest required quadtree node that fully contains its shortcut geometry; boundary-crossing edges remain at an ancestor.\n";
    return output.str();
}

void DatasetLayoutAnalyzer::WriteReport(const DatasetLayoutAnalysisReport& report,
                                        const std::filesystem::path& outputDirectory) {
    std::filesystem::create_directories(outputDirectory);

    {
        const auto path = outputDirectory / "summary.txt";
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << SummaryText(report);
    }

    WriteSourceBlocks(outputDirectory / "graph_source_blocks.csv", report.graphSourceBlocks);
    WriteSourceBlocks(outputDirectory / "range_source_blocks.csv", report.rangeSourceBlocks);

    {
        const auto path = outputDirectory / "lod_density.csv";
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "level,birth_edges,death_edges,alive_edges\n";
        for (const auto& row : report.lodDensity) {
            output << row.level << ',' << row.birthCount << ',' << row.deathCount << ','
                   << row.aliveCount << '\n';
        }
    }

    {
        const auto path = outputDirectory / "lod_band_spatial_density.csv";
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "band_index,level_min,level_max,edge_count,occupied_cells,"
                  "mean_edges_per_occupied_cell,p95_edges_per_cell,max_edges_per_cell\n";
        output << std::fixed << std::setprecision(6);
        for (const auto& row : report.lodBandSpatialDensity) {
            output << row.bandIndex << ',' << row.levelMin << ',' << row.levelMax << ','
                   << row.edgeCount << ',' << row.occupiedCellCount << ','
                   << row.meanEdgesPerOccupiedCell << ',' << row.p95EdgesPerCell << ','
                   << row.maxEdgesPerCell << '\n';
        }
    }

    {
        const auto path = outputDirectory / "virtual_page_locality.csv";
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "page_id,band_index,level_min,level_max,edge_count,source_block_count,"
                  "direct_child_source_block_count,first_morton_cell,last_morton_cell\n";
        for (const auto& page : report.virtualPages) {
            output << page.pageId << ',' << page.bandIndex << ',' << page.levelMin << ','
                   << page.levelMax << ',' << page.edgeCount << ',' << page.sourceBlockCount
                   << ',' << page.directChildSourceBlockCount << ',' << page.firstMortonCell
                   << ',' << page.lastMortonCell << '\n';
        }
    }

    {
        const auto path = outputDirectory / "runtime_graph_pages.csv";
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << "page_id,lod_group_index,level_min,level_max,spatial_level,tile_x,tile_y,"
                  "sub_page,edge_count,source_block_count,child_source_block_count,"
                  "lod_child_page_count\n";
        for (const auto& page : report.runtimeGraphPages) {
            output << page.pageId << ',' << page.bandIndex << ',' << page.levelMin << ','
                   << page.levelMax << ',' << page.spatialLevel << ',' << page.tileX << ','
                   << page.tileY << ',' << page.subPage << ',' << page.edgeCount << ','
                   << page.sourceBlockRefCount << ',' << page.childSourceBlockRefCount << ','
                   << page.lodChildPageRefCount << '\n';
        }
    }

    {
        const auto path = outputDirectory / "cache_manifest.txt";
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("could not write " + path.string());
        }
        output << CacheManifestText(report.graphPath, report.rangesPath, report.config);
    }
}

} // namespace chmv::streaming::analysis
