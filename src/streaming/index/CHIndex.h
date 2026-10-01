#pragma once

#include "streaming/analysis/DatasetLayoutAnalyzer.h"
#include "data/ch/CHTypes.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace chmv::streaming::index {

// Runtime/GPU residency granularity is deliberately smaller than the 16K source/storage block.
// RootPayload remains one dense contiguous binary stream; CHIDX pages are lightweight logical
// tiles used for spatial demand, RAM caching and GPU residency.
inline constexpr std::uint32_t kRuntimeRootTileTargetEdges = 1024u;

inline constexpr std::uint32_t kPackedSpatialCoordinateMask = 0x3fffu;
inline constexpr std::uint8_t kRootRoadStyleMarker = 0x80u;
inline constexpr std::uint8_t kRootRoadStyleMarkerMask = 0xc0u;

[[nodiscard]] inline constexpr std::uint8_t RootRoadStyleMetadataByte(
    std::uint32_t boundsLo, std::uint32_t boundsHi) {
    return static_cast<std::uint8_t>(
        ((boundsLo >> 14u) & 0x3u) |
        (((boundsLo >> 30u) & 0x3u) << 2u) |
        (((boundsHi >> 14u) & 0x3u) << 4u) |
        (((boundsHi >> 30u) & 0x3u) << 6u));
}

[[nodiscard]] inline constexpr bool HasRootRoadStyleMetadata(
    std::uint32_t boundsLo, std::uint32_t boundsHi) {
    return (RootRoadStyleMetadataByte(boundsLo, boundsHi) & kRootRoadStyleMarkerMask) ==
           kRootRoadStyleMarker;
}

[[nodiscard]] inline constexpr std::uint32_t DecodeRootRoadStyle(
    std::uint32_t boundsLo, std::uint32_t boundsHi) {
    const auto metadata = RootRoadStyleMetadataByte(boundsLo, boundsHi);
    return (metadata & kRootRoadStyleMarkerMask) == kRootRoadStyleMarker
               ? static_cast<std::uint32_t>(metadata & 0x3fu)
               : 0u;
}

inline constexpr void EncodeRootRoadStyle(std::uint32_t& boundsLo, std::uint32_t& boundsHi,
                                          std::uint32_t styleType) {
    const auto metadata = static_cast<std::uint8_t>(
        kRootRoadStyleMarker | (styleType & 0x3fu));
    boundsLo &= 0x3fff3fffu;
    boundsHi &= 0x3fff3fffu;
    boundsLo |= (static_cast<std::uint32_t>(metadata & 0x03u) << 14u);
    boundsLo |= (static_cast<std::uint32_t>((metadata >> 2u) & 0x03u) << 30u);
    boundsHi |= (static_cast<std::uint32_t>((metadata >> 4u) & 0x03u) << 14u);
    boundsHi |= (static_cast<std::uint32_t>((metadata >> 6u) & 0x03u) << 30u);
}

struct CHIndexSourceBlock {
    std::uint64_t firstRecord = 0;
    std::uint32_t recordCount = 0;
    std::uint64_t byteOffset = 0;
};

struct CHIndexSpatialPageRef {
    // Packed fixed-grid cell key: x | (y << 16) | (spatialLevel << 32).
    // Multiple entries may share the same key when one spatial cell spans two packed pages.
    std::uint64_t key = 0;
    std::uint32_t pageId = 0;
};

struct CHIndexGraphPage {
    std::uint32_t pageId = 0;
    std::uint32_t bandIndex = 0;
    std::uint32_t levelMin = 0;
    std::uint32_t levelMax = 0;
    std::uint32_t spatialLevel = 0;
    std::uint32_t tileX = 0;
    std::uint32_t tileY = 0;
    std::uint32_t subPage = 0;
    std::uint32_t edgeCount = 0;

    std::uint32_t sourceBlockRefOffset = 0;
    std::uint32_t sourceBlockRefCount = 0;
    std::uint32_t childSourceBlockRefOffset = 0;
    std::uint32_t childSourceBlockRefCount = 0;
    std::uint32_t lodChildPageRefOffset = 0;
    std::uint32_t lodChildPageRefCount = 0;
    std::uint32_t lodMaskOffset = 0;

    std::uint64_t membershipByteOffset = 0;
    std::uint32_t membershipByteSize = 0;
    // CHIDX v6+: first record in the packed RootPayload section. Drawable physical pages are
    // dense in this record space, so one page is one contiguous binary read.
    std::uint64_t rootPayloadRecordOffset = 0;
};

// Binary root record persisted directly in CHIDX v6. The layout intentionally matches the
// persistent GPU root record so runtime page loads no longer have to parse .sch/.ranges or fetch
// endpoint node blocks merely to draw the range-only root geometry.
struct CHIndexRootRecord {
    std::uint32_t globalEdgeId = 0;
    std::int32_t birthLevel = -1;
    std::int32_t deathLevel = -1;
    std::uint32_t boundsLo = 0;
    std::uint32_t boundsHi = 0;
    std::uint32_t childA = 0xffffffffu;
    std::uint32_t childB = 0xffffffffu;
    float geometryError = 0.0f;
    float sourceX = 0.0f;
    float sourceY = 0.0f;
    float targetX = 0.0f;
    float targetY = 0.0f;
};


struct CHIndexBuildData {
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    std::uint32_t maxLevel = 0;
    std::uint32_t spatialGridSize = 0;
    double spatialCellSizeMeters = 0.0;
    std::uint32_t nodeBlockRecordCount = 0;
    std::uint32_t edgeBlockRecordCount = 0;
    std::uint32_t lodBandWidth = 0;
    std::uint32_t lodMaskWordsPerPage = 0;
    double minLatitude = 0.0;
    double minLongitude = 0.0;
    double maxLatitude = 0.0;
    double maxLongitude = 0.0;

    std::vector<CHIndexSourceBlock> nodeBlocks;
    std::vector<CHIndexSourceBlock> graphEdgeBlocks;
    std::vector<CHIndexSourceBlock> rangeEdgeBlocks;
    std::vector<CHIndexGraphPage> graphPages;
    std::vector<CHIndexSpatialPageRef> spatialPageRefs;
    std::vector<std::uint32_t> pageSourceBlockRefs;
    std::vector<std::uint32_t> pageChildSourceBlockRefs;
    std::vector<std::uint32_t> pageLodChildRefs;
    std::vector<std::uint64_t> pageAliveMasks;
    std::filesystem::path membershipBytesPath;
    std::uint64_t membershipByteCount = 0;
    std::filesystem::path edgeSpatialBoundsPath;
    std::uint64_t edgeSpatialBoundsCount = 0;
    std::filesystem::path rootPayloadPath;
    std::uint64_t rootPayloadRecordCount = 0;
};

struct CHIndexData {
    std::filesystem::path indexPath;
    std::uint32_t formatVersion = 0;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    std::uint32_t maxLevel = 0;
    std::uint32_t spatialGridSize = 0;
    double spatialCellSizeMeters = 0.0;
    std::uint32_t nodeBlockRecordCount = 0;
    std::uint32_t edgeBlockRecordCount = 0;
    std::uint32_t lodBandWidth = 0;
    std::uint32_t lodMaskWordsPerPage = 0;

    double minLatitude = 0.0;
    double minLongitude = 0.0;
    double maxLatitude = 0.0;
    double maxLongitude = 0.0;

    std::uint64_t membershipSectionOffset = 0;
    std::uint64_t edgeSpatialBoundsSectionOffset = 0;
    std::uint64_t edgeSpatialBoundsCount = 0;
    std::uint64_t rootPayloadSectionOffset = 0;
    std::uint64_t rootPayloadRecordCount = 0;
    std::vector<std::uint64_t> edgeSpatialBounds;
    std::vector<CHIndexSourceBlock> nodeBlocks;
    std::vector<CHIndexSourceBlock> graphEdgeBlocks;
    std::vector<CHIndexSourceBlock> rangeEdgeBlocks;
    std::vector<CHIndexGraphPage> graphPages;
    std::vector<CHIndexSpatialPageRef> spatialPageRefs;
    std::vector<std::uint32_t> pageSourceBlockRefs;
    std::vector<std::uint32_t> pageChildSourceBlockRefs;
    std::vector<std::uint32_t> pageLodChildRefs;
    std::vector<std::uint64_t> pageAliveMasks;

    [[nodiscard]] bool GraphPageHasAliveEdges(std::uint32_t pageId,
                                               std::uint32_t level) const;
    [[nodiscard]] std::span<const std::uint32_t> SourceBlockRefs(
        const CHIndexGraphPage& page) const;
    [[nodiscard]] std::span<const std::uint32_t> ChildSourceBlockRefs(
        const CHIndexGraphPage& page) const;
    [[nodiscard]] std::span<const std::uint32_t> LodChildPageRefs(
        const CHIndexGraphPage& page) const;
    [[nodiscard]] bool HasRootPayload() const {
        return rootPayloadSectionOffset != 0 && rootPayloadRecordCount != 0;
    }
};

class CHIndex {
public:
    [[nodiscard]] static std::filesystem::path DefaultPath(
        const std::filesystem::path& graphPath);

    [[nodiscard]] static bool IsValid(
        const std::filesystem::path& indexPath,
        const std::filesystem::path& graphPath,
        const std::filesystem::path& rangesPath,
        const analysis::DatasetLayoutAnalysisConfig& config);

    static void Write(const analysis::DatasetLayoutAnalysisReport& report,
                      const std::filesystem::path& indexPath);

    static void Write(const CHIndexBuildData& data,
                      const std::filesystem::path& indexPath);

    [[nodiscard]] static CHIndexData Load(const std::filesystem::path& indexPath);
    [[nodiscard]] static std::vector<std::uint32_t> ReadGraphPageEdgeIds(
        const CHIndexData& index, std::uint32_t pageId);
    [[nodiscard]] static std::vector<CHIndexRootRecord> ReadGraphPageRootRecords(
        const CHIndexData& index, std::uint32_t pageId);
    [[nodiscard]] static std::vector<std::uint64_t> ReadEdgeSpatialBounds(
        const CHIndexData& index, std::uint32_t firstEdgeId, std::uint32_t count);
};

} // namespace chmv::streaming::index
