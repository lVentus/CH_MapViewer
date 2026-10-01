#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace chmv::streaming::analysis {

enum class DatasetLayoutAnalysisStage {
    Nodes,
    Edges,
    Ranges,
    Layout,
};

struct DatasetLayoutAnalysisProgress {
    DatasetLayoutAnalysisStage stage = DatasetLayoutAnalysisStage::Nodes;
    std::uint64_t current = 0;
    std::uint64_t total = 0;
};

struct DatasetLayoutAnalysisConfig {
    // Legacy normalized grid used only by optional analysis reports. Runtime preprocessing
    // uses a fixed world-space grid so different map sizes keep the same spatial granularity.
    std::uint32_t spatialGridSize = 1024;
    double spatialCellSizeMeters = 4096.0;
    std::uint32_t lodBandWidth = 16;
    std::uint32_t sourceBlockNodeCount = 16384;
    std::uint32_t sourceBlockEdgeCount = 16384;
    std::uint32_t virtualPageEdgeCount = 16384;
};

struct SourceBlockInfo {
    std::uint32_t blockId = 0;
    std::uint64_t firstRecord = 0;
    std::uint32_t recordCount = 0;
    std::uint64_t byteOffset = 0;
};

struct StreamingEdgeBlockInfo {
    std::uint32_t blockId = 0;
    std::uint64_t firstEdgeId = 0;
    std::uint32_t edgeCount = 0;
    std::uint64_t graphByteOffset = 0;
    std::uint64_t rangeByteOffset = 0;

    float minLatitude = 0.0f;
    float minLongitude = 0.0f;
    float maxLatitude = 0.0f;
    float maxLongitude = 0.0f;

    std::uint32_t minCellX = 0;
    std::uint32_t minCellY = 0;
    std::uint32_t maxCellX = 0;
    std::uint32_t maxCellY = 0;

    std::uint32_t drawableEdgeCount = 0;
    std::uint32_t shortcutCount = 0;
    std::uint32_t nodeBlockRefOffset = 0;
    std::uint32_t nodeBlockRefCount = 0;
    std::uint32_t childBlockRefOffset = 0;
    std::uint32_t childBlockRefCount = 0;
};

struct LodDensityInfo {
    std::uint32_t level = 0;
    std::uint64_t birthCount = 0;
    std::uint64_t deathCount = 0;
    std::uint64_t aliveCount = 0;
};

struct LodBandSpatialInfo {
    std::uint32_t bandIndex = 0;
    std::uint32_t levelMin = 0;
    std::uint32_t levelMax = 0;
    std::uint64_t edgeCount = 0;
    std::uint32_t occupiedCellCount = 0;
    double meanEdgesPerOccupiedCell = 0.0;
    std::uint32_t p95EdgesPerCell = 0;
    std::uint32_t maxEdgesPerCell = 0;
};

// Phase-0 analysis model retained for thesis comparisons. These pages are formed by
// sorting a whole LOD band by Morton order and slicing fixed-size chunks. They are
// intentionally not the runtime paging format.
struct VirtualPageInfo {
    std::uint32_t pageId = 0;
    std::uint32_t bandIndex = 0;
    std::uint32_t levelMin = 0;
    std::uint32_t levelMax = 0;
    std::uint32_t edgeCount = 0;
    std::uint32_t sourceBlockCount = 0;
    std::uint32_t directChildSourceBlockCount = 0;
    std::uint32_t firstMortonCell = 0;
    std::uint32_t lastMortonCell = 0;
};

// Runtime physical-page metadata. The preprocessed out-of-core layout uses a fixed-size
// world-space base grid; long shortcut geometry is assigned directly to a coarser power-of-two
// cell without recursive subdivision. Spatial cells are only lookup keys: neighboring Morton
// cells are packed into dense physical pages, and .chidx stores exact cell -> page references.
struct RuntimeGraphPageInfo {
    std::uint32_t pageId = 0;
    std::uint32_t bandIndex = 0;
    std::uint32_t levelMin = 0;
    std::uint32_t levelMax = 0;
    std::uint32_t spatialLevel = 0;
    std::uint32_t tileX = 0;
    std::uint32_t tileY = 0;
    std::uint32_t subPage = 0;
    std::uint32_t edgeCount = 0;

    std::uint32_t edgeIdOffset = 0;
    std::uint32_t sourceBlockRefOffset = 0;
    std::uint32_t sourceBlockRefCount = 0;
    std::uint32_t childSourceBlockRefOffset = 0;
    std::uint32_t childSourceBlockRefCount = 0;
    std::uint32_t lodChildPageRefOffset = 0;
    std::uint32_t lodChildPageRefCount = 0;
    std::uint32_t lodMaskOffset = 0;
};

struct DatasetLayoutAnalysisReport {
    DatasetLayoutAnalysisConfig config;
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;

    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    std::uint32_t maxLevel = 0;
    std::uint32_t lodMaskWordsPerBlock = 0;
    double minLatitude = 0.0;
    double minLongitude = 0.0;
    double maxLatitude = 0.0;
    double maxLongitude = 0.0;
    std::uint64_t shortcutCount = 0;
    std::uint64_t drawableEdgeCount = 0;
    std::uint64_t nonDrawableEdgeCount = 0;
    std::uint64_t invalidRangeCount = 0;
    std::uint64_t duplicateRangeRecordCount = 0;
    std::uint64_t missingRangeRecordCount = 0;

    double nodeRecordSequentialRatio = 0.0;
    double rangeRecordSequentialRatio = 0.0;

    double sourceOrderSameSpatialCellRatio = 0.0;
    double sourceOrderWithinOneCellRatio = 0.0;
    double sourceOrderWithinFiveCellsRatio = 0.0;

    double sourceOrderSameBirthLevelRatio = 0.0;
    double sourceOrderSameLodBandRatio = 0.0;
    std::uint32_t sourceOrderBirthDeltaP50 = 0;
    std::uint32_t sourceOrderBirthDeltaP95 = 0;
    std::uint32_t sourceOrderBirthDeltaP99 = 0;

    std::uint64_t parentChildReferenceCount = 0;
    double parentChildSameSourceBlockRatio = 0.0;
    double parentChildWithinOneSourceBlockRatio = 0.0;
    double parentChildWithinFiveSourceBlocksRatio = 0.0;
    std::uint64_t parentChildIdDistanceP50UpperBound = 0;
    std::uint64_t parentChildIdDistanceP95UpperBound = 0;
    std::uint64_t parentChildIdDistanceP99UpperBound = 0;
    std::uint64_t parentChildIdDistanceMax = 0;

    std::uint64_t drawableShortcutCount = 0;
    std::uint64_t drawableShortcutChildReferenceCount = 0;
    std::uint64_t drawableShortcutChildWithDrawableLifetimeCount = 0;

    std::uint32_t lifetimeSpanP50 = 0;
    std::uint32_t lifetimeSpanP95 = 0;
    std::uint32_t lifetimeSpanP99 = 0;
    std::uint32_t lifetimeSpanMax = 0;

    std::uint32_t virtualPageCount = 0;
    double virtualPageMeanFill = 0.0;
    std::uint32_t virtualPageSourceBlockP50 = 0;
    std::uint32_t virtualPageSourceBlockP95 = 0;
    std::uint32_t virtualPageSourceBlockP99 = 0;
    std::uint32_t virtualPageSourceBlockMax = 0;
    std::uint32_t virtualPageDirectChildSourceBlockP50 = 0;
    std::uint32_t virtualPageDirectChildSourceBlockP95 = 0;
    std::uint32_t virtualPageDirectChildSourceBlockP99 = 0;
    std::uint32_t virtualPageDirectChildSourceBlockMax = 0;

    std::uint32_t runtimeGraphPageSourceBlockP50 = 0;
    std::uint32_t runtimeGraphPageSourceBlockP95 = 0;
    std::uint32_t runtimeGraphPageSourceBlockP99 = 0;
    std::uint32_t runtimeGraphPageSourceBlockMax = 0;
    std::uint32_t runtimeGraphPageChildSourceBlockP50 = 0;
    std::uint32_t runtimeGraphPageChildSourceBlockP95 = 0;
    std::uint32_t runtimeGraphPageChildSourceBlockP99 = 0;
    std::uint32_t runtimeGraphPageChildSourceBlockMax = 0;

    std::vector<SourceBlockInfo> nodeSourceBlocks;
    std::vector<SourceBlockInfo> graphSourceBlocks;
    std::vector<SourceBlockInfo> rangeSourceBlocks;
    std::vector<StreamingEdgeBlockInfo> streamingEdgeBlocks;
    std::vector<std::uint32_t> edgeBlockNodeRefs;
    std::vector<std::uint32_t> edgeBlockChildRefs;
    std::vector<std::uint64_t> edgeBlockAliveMasks;
    std::vector<LodDensityInfo> lodDensity;
    std::vector<LodBandSpatialInfo> lodBandSpatialDensity;
    std::vector<VirtualPageInfo> virtualPages;

    // Runtime LOD/fixed-grid page data used by .chidx.
    std::vector<RuntimeGraphPageInfo> runtimeGraphPages;
    std::vector<std::uint32_t> runtimeGraphPageEdgeIds;
    std::vector<std::uint32_t> runtimeGraphPageSourceBlockRefs;
    std::vector<std::uint32_t> runtimeGraphPageChildSourceBlockRefs;
    std::vector<std::uint32_t> runtimeGraphPageLodChildRefs;
    std::vector<std::uint64_t> runtimeGraphPageAliveMasks;

    // Transient full-geometry cell bounds used by preprocessing. Packed as
    // minX|minY|maxX|maxY in four 16-bit lanes; not serialized into .chidx.
    std::vector<std::uint64_t> edgeSpatialBounds;
};

class DatasetLayoutAnalyzer {
public:
    using ProgressCallback = std::function<void(const DatasetLayoutAnalysisProgress&)>;

    [[nodiscard]] static DatasetLayoutAnalysisReport Analyze(
        const std::filesystem::path& graphPath,
        const std::filesystem::path& rangesPath,
        const DatasetLayoutAnalysisConfig& config = {},
        ProgressCallback progressCallback = {});

    [[nodiscard]] static std::filesystem::path DefaultReportDirectory(
        const std::filesystem::path& graphPath);

    [[nodiscard]] static bool HasValidCachedReport(
        const std::filesystem::path& graphPath,
        const std::filesystem::path& rangesPath,
        const DatasetLayoutAnalysisConfig& config,
        const std::filesystem::path& outputDirectory);

    [[nodiscard]] static std::string ReadCachedSummary(
        const std::filesystem::path& outputDirectory);

    [[nodiscard]] static std::string SummaryText(
        const DatasetLayoutAnalysisReport& report);

    static void WriteReport(const DatasetLayoutAnalysisReport& report,
                            const std::filesystem::path& outputDirectory);
};

} // namespace chmv::streaming::analysis
