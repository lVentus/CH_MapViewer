#pragma once

// On-disk layout of the self-contained LOD page file (<name>.chlod).
//
// The file has two layers:
//   * CH layer: the complete contraction hierarchy (nodes, edges, children, ranges), stored as
//     columns in source order. Nothing is lost, so the file is a valid CH dataset even without
//     the .sch/.ranges it was built from.
//   * Render layer: one record per hierarchy context (normally one per edge). A record is drawn
//     iff the current level R lies in its level interval and the screen tolerance lies in its
//     geometry-error interval. Records are packed into pages that the GPU consumes unchanged:
//     a decoded page is exactly what goes into a GPU slot.
//
// Everything is stored zstd-compressed (see "Compression" below): columns in blocks, pages and
// children blocks one frame each, so a single page can still be read and decoded on its own.
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
// 4: everything compressed. A version 3 file is the same file uncompressed (it reads as
// Compression::None) and is compressed on load; files of an older version are rebuilt.
inline constexpr std::uint32_t kFormatVersion = 4;
inline constexpr std::uint32_t kUncompressedFormatVersion = 3;
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

// Geometry error bounds in fixed coordinate units, for culling with integers: at least the error
// a code stands for, rounded up, and at most kMaxErrorBound (more than the whole map, and small
// enough that a coordinate plus a bound fits in int32). Both the builder and the GPU kernels
// compute them from the same 1024 mantissas, so they agree to the unit.
inline constexpr std::uint32_t kMaxErrorBound = (1u << 30u) - 1u;
inline constexpr std::uint32_t kErrorBoundMantissas = 1024;

// 2^(step / 1024) in 1.31 fixed point, rounded up.
[[nodiscard]] inline std::uint32_t ErrorBoundMantissa(std::uint32_t step) {
    return static_cast<std::uint32_t>(std::ceil(
        std::exp2(static_cast<double>(step) / kErrorCodeStepsPerOctave) * 2147483648.0));
}

// code = 1024 * (log2(error) + 40), so the bound is mantissa * 2^(code / 1024 - 42) units.
[[nodiscard]] inline constexpr std::uint32_t ScaleErrorBound(std::uint32_t code,
                                                             std::uint32_t mantissa) {
    if (code == 0u) {
        return 0u;
    }
    const auto octave = code >> 10u;
    if (octave >= 41u) {
        return kMaxErrorBound;
    }
    const auto shift = 42u - octave;
    if (shift >= 32u) {
        return 1u;
    }
    const auto rest = (mantissa & ((1u << shift) - 1u)) != 0u ? 1u : 0u;
    return std::min((mantissa >> shift) + rest, kMaxErrorBound);
}

[[nodiscard]] inline std::uint32_t ErrorBound(std::uint32_t code) {
    return ScaleErrorBound(code, ErrorBoundMantissa(code & (kErrorBoundMantissas - 1u)));
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

// Data after the main sections for expansion methods that walk the hierarchy instead of testing
// intervals; only read when such a method is used.
enum class Extension : std::uint32_t {
    PageChildren,   // per page: kChildMaskWords mask of the records with children, then two child
                    // references per set bit, in record order
    VisitDirectory, // PageEntry per page, over the records a traversal visits (each edge once):
                    // levels and bands of every context, box grown by the largest error;
                    // fileOffset/wordCount locate the page's children block
    VisitGroups,    // PageGroup per kPagesPerGroup pages of the visit directory
    Count,
};

inline constexpr std::size_t kExtensionCount = static_cast<std::size_t>(Extension::Count);
// Sections and extensions numbered together, sections first.
inline constexpr std::size_t kStoredCount = kSectionCount + kExtensionCount;

[[nodiscard]] inline constexpr std::size_t StoredIndex(Section section) {
    return static_cast<std::size_t>(section);
}

[[nodiscard]] inline constexpr std::size_t StoredIndex(Extension extension) {
    return kSectionCount + static_cast<std::size_t>(extension);
}

inline constexpr std::uint32_t kChildMaskWords = kPageRecordCapacity / 32u;
inline constexpr std::uint32_t kChildSlotWords = kChildMaskWords + 2u * kPageRecordCapacity;
// A child reference is page << 8 | record; none for children no traversal can reach.
inline constexpr std::uint32_t kNoChildRef = 0xffffffffu;

struct SectionEntry {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

// Compression. The builder first writes every section uncompressed (compression = None) and
// checks that file; the final file stores the same bytes compressed:
//   * Pages and PageChildren: one zstd frame per page (per children block), back to back in
//     page order. The frame sizes are in pageFrames / childFrames (uint32 per page, stored
//     like a column); kFrameShuffled marks a frame whose words were byte-shuffled first.
//   * Every other section: a BlockTable, its StoredBlocks, then the blocks, one zstd frame each.
//     Byte shuffling stores byte k of every element before byte k + 1, which lets zstd see
//     the slowly changing high bytes of a column as long runs.
// Every frame carries zstd's checksum, so damage shows up when it is decoded. sections[] and
// extensions[] keep describing the uncompressed layout (the directories' fileOffset values
// refer to it), stored[] says where each section lies in this file, and the hashes are those of
// the uncompressed bytes.
enum class Compression : std::uint32_t {
    None = 0,
    Zstd = 1,
};

// Set in a frame size: the frame holds the words byte-shuffled (element size 4).
inline constexpr std::uint32_t kFrameShuffled = 0x80000000u;

struct BlockTable {
    // Uncompressed bytes per block; the last block may be shorter.
    std::uint32_t blockBytes = 0;
    std::uint32_t blockCount = 0;
};

struct StoredBlock {
    std::uint32_t storedBytes = 0;
    // 0: compressed as it is; otherwise the element size the block was byte-shuffled with.
    std::uint32_t shuffle = 0;
};

static_assert(sizeof(BlockTable) == 8 && sizeof(StoredBlock) == 8);

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
    // Hash of every section as written (StreamHash), so a copy can be checked later.
    std::array<std::uint64_t, kSectionCount> sectionHashes{};
    std::array<SectionEntry, kExtensionCount> extensions{};
    std::array<std::uint64_t, kExtensionCount> extensionHashes{};
    Compression compression = Compression::None;
    std::uint32_t reserved = 0;
    std::array<SectionEntry, kStoredCount> stored{};
    SectionEntry pageFrames{};
    SectionEntry childFrames{};

    [[nodiscard]] const SectionEntry& At(Section section) const {
        return sections[static_cast<std::size_t>(section)];
    }
    [[nodiscard]] SectionEntry& At(Section section) {
        return sections[static_cast<std::size_t>(section)];
    }
    [[nodiscard]] const SectionEntry& At(Extension extension) const {
        return extensions[static_cast<std::size_t>(extension)];
    }
    [[nodiscard]] SectionEntry& At(Extension extension) {
        return extensions[static_cast<std::size_t>(extension)];
    }
    // Uncompressed layout of a section or extension, by StoredIndex.
    [[nodiscard]] const SectionEntry& Layout(std::size_t index) const {
        return index < kSectionCount ? sections[index] : extensions[index - kSectionCount];
    }
    [[nodiscard]] std::uint64_t Hash(std::size_t index) const {
        return index < kSectionCount ? sectionHashes[index]
                                     : extensionHashes[index - kSectionCount];
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
