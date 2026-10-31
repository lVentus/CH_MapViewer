#pragma once

// On-disk layout of the self-contained LOD page file (<name>.chlod).
//
// The file has two layers:
//   * CH layer: the complete contraction hierarchy (nodes, edges, children, ranges), stored as
//     plain columns in source order. Nothing is lost, so the file is a valid CH dataset even
//     without the .sch/.ranges it was built from.
//   * Render layer: one record per hierarchy context (normally one per edge). A record is drawn
//     iff the current level R lies in its level interval and the screen tolerance lies in its
//     geometry-error interval. Records are packed into pages that the GPU consumes unchanged:
//     disk bytes == RAM bytes == GPU bytes.
//
// Pages are sorted by quality layer, then zoom octave, then Morton order. This grouping only
// affects locality; which pages a view needs is decided by the per-page level/error/bounds
// metadata, so any screen size, LOD curve or tolerance gives the exact DFS result.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace chmv::streaming::lod {

static_assert(std::endian::native == std::endian::little,
              "LOD page files are little-endian and mapped directly");

inline constexpr std::array<char, 8> kFileMagic{'C', 'H', 'L', 'O', 'D', 'P', 'G', 'S'};
inline constexpr std::uint32_t kFormatVersion = 1;
inline constexpr std::uint64_t kHeaderBytes = 4096;
inline constexpr std::uint64_t kSectionAlignment = 4096;

// Node coordinates in the render layer: normalized projected coordinates times 2^29, as int32.
// Normalized coordinates stay within about +-1, so differences never overflow int32.
inline constexpr double kFixedCoordinateScale = 536870912.0;
// Latitude/longitude in the CH layer: integer 1e-7 degrees (OSM's own resolution).
inline constexpr double kDegreeScale = 1e7;

// Physical calibration shared with the camera (MapCamera2D / Application).
inline constexpr double kMetersPerProjectedDegree = 111319.49079327357;

// Page geometry. One workgroup per page, one thread per record.
inline constexpr std::uint32_t kPageRecordCapacity = 256;
inline constexpr std::uint32_t kPageVertexCapacity = 2 * kPageRecordCapacity;
inline constexpr std::uint32_t kPageHeaderWords = 4;
inline constexpr std::uint32_t kRecordWords = 4;
inline constexpr std::uint32_t kVertexWords = 2;
inline constexpr std::uint32_t kPagesPerGroup = 64;

[[nodiscard]] inline constexpr std::uint32_t PageWordCount(std::uint32_t recordCount,
                                                           std::uint32_t vertexCount) {
    const auto words = kPageHeaderWords + recordCount * kRecordWords + vertexCount * kVertexWords +
                       (recordCount + 3u) / 4u;
    return (words + 3u) & ~3u;
}

// Every GPU slot can hold the largest page; the size is a multiple of four words so records
// can be fetched as uvec4.
inline constexpr std::uint32_t kPageSlotWords =
    PageWordCount(kPageRecordCapacity, kPageVertexCapacity);
static_assert(kPageSlotWords % 4u == 0u);

// Record word layout (one uvec4 per record):
//   x: local vertex index A | B << 16
//   y: level interval as root          (lo | hi << 16, empty = kEmptyLevelRange)
//   z: level interval below its parent (lo | hi << 16, empty = kEmptyLevelRange)
//   w: error code | parent error code << 16
// Drawn iff errorCode <= T and (R in y, or R in z and T < parentCode).
inline constexpr std::uint32_t kEmptyLevelRange = 0x0000ffffu;
inline constexpr std::uint16_t kMaxStoredLevel = 0xfffeu;

[[nodiscard]] inline constexpr std::uint32_t PackLevelRange(std::uint16_t low, std::uint16_t high) {
    return static_cast<std::uint32_t>(low) | (static_cast<std::uint32_t>(high) << 16u);
}

// Geometry errors are compared as monotone 16-bit codes: 1024 steps per octave over 64
// octaves of normalized units. Errors round up and tolerances round down, so the quantized
// cut is a valid DFS cut and never coarser than the requested tolerance.
inline constexpr std::uint16_t kErrorCodeInfinite = 0xffffu;
inline constexpr std::uint16_t kErrorCodeMax = 0xfffeu;
inline constexpr double kErrorCodeStepsPerOctave = 1024.0;
inline constexpr double kErrorCodeBiasOctaves = 40.0;

[[nodiscard]] inline std::uint16_t EncodeGeometryError(double error) {
    if (!(error > 0.0)) {
        return 0;
    }
    if (!std::isfinite(error)) {
        return kErrorCodeInfinite;
    }
    const auto code =
        std::ceil((std::log2(error) + kErrorCodeBiasOctaves) * kErrorCodeStepsPerOctave);
    return static_cast<std::uint16_t>(std::clamp(code, 1.0, static_cast<double>(kErrorCodeMax)));
}

[[nodiscard]] inline std::uint16_t EncodeErrorTolerance(double tolerance) {
    if (!(tolerance > 0.0)) {
        return 0;
    }
    if (!std::isfinite(tolerance)) {
        return kErrorCodeMax;
    }
    const auto code =
        std::floor((std::log2(tolerance) + kErrorCodeBiasOctaves) * kErrorCodeStepsPerOctave);
    return static_cast<std::uint16_t>(std::clamp(code, 0.0, static_cast<double>(kErrorCodeMax)));
}

[[nodiscard]] inline constexpr bool RecordDrawn(std::uint32_t levelWord,
                                                std::uint32_t childLevelWord,
                                                std::uint32_t errorWord, std::uint32_t level,
                                                std::uint32_t threshold) {
    const auto errorCode = errorWord & 0xffffu;
    const auto parentCode = errorWord >> 16u;
    const bool asRoot = level >= (levelWord & 0xffffu) && level <= (levelWord >> 16u);
    const bool asChild = level >= (childLevelWord & 0xffffu) && level <= (childLevelWord >> 16u) &&
                         threshold < parentCode;
    return errorCode <= threshold && (asRoot || asChild);
}

enum class Section : std::uint32_t {
    NodeOsmId,     // uint64
    NodeLatitude,  // int32, 1e-7 degrees
    NodeLongitude, // int32, 1e-7 degrees
    NodeElevation, // float
    NodeLevel,     // uint32
    EdgeSource,    // uint32
    EdgeTarget,    // uint32
    EdgeWeight,    // uint32
    EdgeType,      // int16
    EdgeMaxSpeed,  // int16
    EdgeChildA,    // uint32, 0xffffffff = none
    EdgeChildB,    // uint32
    EdgeBirth,     // int16, -1 = not drawable
    EdgeDeath,     // int16
    EdgeError,     // float, exact recursive geometry error in normalized units
    EdgeCoverLow,  // int16, hull of the levels at which the edge is reachable, -1 = never
    EdgeCoverHigh, // int16
    RecordEdge,    // uint32 per record, page order: links render records to CH edges
    PageDirectory, // PageEntry per page
    PageGroups,    // PageGroup per kPagesPerGroup consecutive pages
    Pages,         // page blobs, 16-byte aligned
    Count,
};

inline constexpr std::size_t kSectionCount = static_cast<std::size_t>(Section::Count);

struct SectionEntry {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct FileHeader {
    std::array<char, 8> magic{};
    std::uint32_t version = 0;
    std::uint32_t headerBytes = 0;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    std::uint64_t recordCount = 0;
    std::uint64_t pageCount = 0;
    std::uint64_t groupCount = 0;
    std::uint64_t coverGapEdges = 0;
    std::uint64_t extraContextRecords = 0;
    std::int32_t maxLevel = 0;
    std::uint32_t maxHierarchyHeight = 0;
    double centerLatitude = 0.0;
    double centerLongitude = 0.0;
    double longitudeScale = 1.0;
    double normalization = 1.0;
    double minLatitude = 0.0;
    double maxLatitude = 0.0;
    double minLongitude = 0.0;
    double maxLongitude = 0.0;
    std::uint64_t sourceGraphBytes = 0;
    std::uint64_t sourceRangesBytes = 0;
    std::int64_t sourceGraphWriteTime = 0;
    std::int64_t sourceRangesWriteTime = 0;
    std::uint32_t referenceFramebufferHeight = 0;
    std::uint32_t pageSlotWords = 0;
    std::array<SectionEntry, kSectionCount> sections{};

    [[nodiscard]] const SectionEntry& At(Section section) const {
        return sections[static_cast<std::size_t>(section)];
    }
    [[nodiscard]] SectionEntry& At(Section section) {
        return sections[static_cast<std::size_t>(section)];
    }
    [[nodiscard]] double MetersPerNormalizedUnit() const {
        return kMetersPerProjectedDegree / std::max(normalization, 1e-12);
    }
};

static_assert(std::is_trivially_copyable_v<FileHeader>);
static_assert(sizeof(FileHeader) <= kHeaderBytes);

// Per-page (and per-group) summary of which tolerances can draw anything, kept separately for
// kLevelBands equal parts of the page's level range. Without it a page mixing roots of low
// levels (any tolerance) with children of high levels (tiny tolerances only) would match
// almost every query. Codes are error codes >> 8 (quarter octaves); both bounds are inclusive
// and rounded outward, so the test stays conservative.
inline constexpr std::uint32_t kLevelBands = 16;

struct LevelBands {
    std::array<std::uint8_t, kLevelBands> minError{};
    std::array<std::uint8_t, kLevelBands> maxParent{};
};

[[nodiscard]] inline constexpr std::uint32_t LevelBand(std::uint32_t level, std::uint32_t minLevel,
                                                       std::uint32_t maxLevel) {
    return (level - minLevel) * kLevelBands / (maxLevel - minLevel + 1u);
}

[[nodiscard]] inline constexpr std::uint8_t BandCode(std::uint32_t errorCode) {
    return static_cast<std::uint8_t>(errorCode >> 8u);
}

struct PageEntry {
    std::uint64_t fileOffset = 0;
    std::uint32_t wordCount = 0;
    std::uint32_t firstRecord = 0;
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    std::uint16_t minLevel = 0;
    std::uint16_t maxLevel = 0;
    std::uint16_t minErrorCode = 0;
    std::uint16_t maxParentCode = 0;
    std::uint16_t recordCount = 0;
    std::uint16_t vertexCount = 0;
    std::uint8_t layer = 0;
    std::uint8_t octave = 0;
    std::uint16_t reserved = 0;
    LevelBands bands;
};

static_assert(sizeof(PageEntry) == 80);
static_assert(std::is_trivially_copyable_v<PageEntry>);

struct PageGroup {
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    std::uint16_t minLevel = 0;
    std::uint16_t maxLevel = 0;
    std::uint16_t minErrorCode = 0;
    std::uint16_t maxParentCode = 0;
    std::uint32_t firstPage = 0;
    std::uint32_t pageCount = 0;
    LevelBands bands;
};

static_assert(sizeof(PageGroup) == 64);
static_assert(std::is_trivially_copyable_v<PageGroup>);

} // namespace chmv::streaming::lod
