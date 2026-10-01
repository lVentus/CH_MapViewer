#include "streaming/index/CHIndex.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace chmv::streaming::index {
namespace {

constexpr std::array<char, 8> kMagicV5{'C', 'H', 'M', 'V', 'I', 'D', 'X', '5'};
constexpr std::array<char, 8> kMagicV6{'C', 'H', 'M', 'V', 'I', 'D', 'X', '6'};
constexpr std::array<char, 8> kMagicV7{'C', 'H', 'M', 'V', 'I', 'D', 'X', '7'};
constexpr std::uint32_t kFormatVersionV5 = 5;
constexpr std::uint32_t kFormatVersionV6 = 6;
constexpr std::uint32_t kFormatVersion = 7;
constexpr std::uint32_t kEndianTag = 0x01020304u;

struct FileStamp {
    std::uint64_t size = 0;
    std::int64_t writeTime = 0;
};

struct Header {
    std::uint32_t version = kFormatVersion;
    std::uint32_t endianTag = kEndianTag;
    FileStamp graph;
    FileStamp ranges;

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

    std::uint64_t nodeBlockCount = 0;
    std::uint64_t graphEdgeBlockCount = 0;
    std::uint64_t rangeEdgeBlockCount = 0;
    std::uint64_t graphPageCount = 0;
    std::uint64_t spatialPageRefCount = 0;
    std::uint64_t sourceBlockRefCount = 0;
    std::uint64_t childSourceBlockRefCount = 0;
    std::uint64_t lodChildPageRefCount = 0;
    std::uint64_t pageMaskWordCount = 0;
    std::uint64_t membershipByteCount = 0;
    std::uint64_t edgeSpatialBoundsCount = 0;
    std::uint64_t rootPayloadRecordCount = 0;

    std::uint64_t nodeBlocksOffset = 0;
    std::uint64_t graphEdgeBlocksOffset = 0;
    std::uint64_t rangeEdgeBlocksOffset = 0;
    std::uint64_t graphPagesOffset = 0;
    std::uint64_t spatialPageRefsOffset = 0;
    std::uint64_t sourceBlockRefsOffset = 0;
    std::uint64_t childSourceBlockRefsOffset = 0;
    std::uint64_t lodChildPageRefsOffset = 0;
    std::uint64_t pageMasksOffset = 0;
    std::uint64_t membershipOffset = 0;
    std::uint64_t edgeSpatialBoundsOffset = 0;
    std::uint64_t rootPayloadOffset = 0;
};

constexpr std::uint64_t kHeaderSerializedSize = 320;
constexpr std::uint64_t kSourceBlockSerializedSize = 8 + 4 + 8;
constexpr std::uint64_t kGraphPageSerializedSize = 84;
constexpr std::uint64_t kSpatialPageRefSerializedSize = 12;
constexpr std::uint64_t kRootRecordSerializedSize = 48;

static_assert(sizeof(CHIndexRootRecord) == kRootRecordSerializedSize);

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

template <typename T>
void WriteValue(std::ostream& output, T value) {
    static_assert(std::is_arithmetic_v<T>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed to write CH index");
    }
}

template <typename T>
T ReadValue(std::istream& input) {
    static_assert(std::is_arithmetic_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated CH index");
    }
    return value;
}

void WriteHeader(std::ostream& output, const Header& header) {
    const auto& magic = header.version >= kFormatVersion ? kMagicV7 : kMagicV6;
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    WriteValue(output, header.version);
    WriteValue(output, header.endianTag);
    WriteValue(output, header.graph.size);
    WriteValue(output, header.graph.writeTime);
    WriteValue(output, header.ranges.size);
    WriteValue(output, header.ranges.writeTime);
    WriteValue(output, header.nodeCount);
    WriteValue(output, header.edgeCount);
    WriteValue(output, header.maxLevel);
    WriteValue(output, header.spatialGridSize);
    WriteValue(output, header.spatialCellSizeMeters);
    WriteValue(output, header.nodeBlockRecordCount);
    WriteValue(output, header.edgeBlockRecordCount);
    WriteValue(output, header.lodBandWidth);
    WriteValue(output, header.lodMaskWordsPerPage);
    WriteValue(output, header.minLatitude);
    WriteValue(output, header.minLongitude);
    WriteValue(output, header.maxLatitude);
    WriteValue(output, header.maxLongitude);
    WriteValue(output, header.nodeBlockCount);
    WriteValue(output, header.graphEdgeBlockCount);
    WriteValue(output, header.rangeEdgeBlockCount);
    WriteValue(output, header.graphPageCount);
    WriteValue(output, header.spatialPageRefCount);
    WriteValue(output, header.sourceBlockRefCount);
    WriteValue(output, header.childSourceBlockRefCount);
    WriteValue(output, header.lodChildPageRefCount);
    WriteValue(output, header.pageMaskWordCount);
    WriteValue(output, header.membershipByteCount);
    WriteValue(output, header.edgeSpatialBoundsCount);
    WriteValue(output, header.rootPayloadRecordCount);
    WriteValue(output, header.nodeBlocksOffset);
    WriteValue(output, header.graphEdgeBlocksOffset);
    WriteValue(output, header.rangeEdgeBlocksOffset);
    WriteValue(output, header.graphPagesOffset);
    WriteValue(output, header.spatialPageRefsOffset);
    WriteValue(output, header.sourceBlockRefsOffset);
    WriteValue(output, header.childSourceBlockRefsOffset);
    WriteValue(output, header.lodChildPageRefsOffset);
    WriteValue(output, header.pageMasksOffset);
    WriteValue(output, header.membershipOffset);
    WriteValue(output, header.edgeSpatialBoundsOffset);
    WriteValue(output, header.rootPayloadOffset);
}

Header ReadHeader(std::istream& input) {
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || (magic != kMagicV5 && magic != kMagicV6 && magic != kMagicV7)) {
        throw std::runtime_error("invalid CH index magic");
    }

    Header header;
    header.version = ReadValue<std::uint32_t>(input);
    header.endianTag = ReadValue<std::uint32_t>(input);
    if (header.version != kFormatVersionV5 && header.version != kFormatVersionV6 &&
        header.version != kFormatVersion) {
        throw std::runtime_error("unsupported CH index version");
    }
    if ((header.version == kFormatVersionV5 && magic != kMagicV5) ||
        (header.version == kFormatVersionV6 && magic != kMagicV6) ||
        (header.version == kFormatVersion && magic != kMagicV7)) {
        throw std::runtime_error("CH index magic/version mismatch");
    }
    if (header.endianTag != kEndianTag || std::endian::native != std::endian::little) {
        throw std::runtime_error("CH index currently requires little-endian storage");
    }

    header.graph.size = ReadValue<std::uint64_t>(input);
    header.graph.writeTime = ReadValue<std::int64_t>(input);
    header.ranges.size = ReadValue<std::uint64_t>(input);
    header.ranges.writeTime = ReadValue<std::int64_t>(input);
    header.nodeCount = ReadValue<std::uint64_t>(input);
    header.edgeCount = ReadValue<std::uint64_t>(input);
    header.maxLevel = ReadValue<std::uint32_t>(input);
    header.spatialGridSize = ReadValue<std::uint32_t>(input);
    header.spatialCellSizeMeters = ReadValue<double>(input);
    header.nodeBlockRecordCount = ReadValue<std::uint32_t>(input);
    header.edgeBlockRecordCount = ReadValue<std::uint32_t>(input);
    header.lodBandWidth = ReadValue<std::uint32_t>(input);
    header.lodMaskWordsPerPage = ReadValue<std::uint32_t>(input);
    header.minLatitude = ReadValue<double>(input);
    header.minLongitude = ReadValue<double>(input);
    header.maxLatitude = ReadValue<double>(input);
    header.maxLongitude = ReadValue<double>(input);
    header.nodeBlockCount = ReadValue<std::uint64_t>(input);
    header.graphEdgeBlockCount = ReadValue<std::uint64_t>(input);
    header.rangeEdgeBlockCount = ReadValue<std::uint64_t>(input);
    header.graphPageCount = ReadValue<std::uint64_t>(input);
    header.spatialPageRefCount = ReadValue<std::uint64_t>(input);
    header.sourceBlockRefCount = ReadValue<std::uint64_t>(input);
    header.childSourceBlockRefCount = ReadValue<std::uint64_t>(input);
    header.lodChildPageRefCount = ReadValue<std::uint64_t>(input);
    header.pageMaskWordCount = ReadValue<std::uint64_t>(input);
    header.membershipByteCount = ReadValue<std::uint64_t>(input);
    header.edgeSpatialBoundsCount = ReadValue<std::uint64_t>(input);
    if (header.version >= kFormatVersionV6) {
        header.rootPayloadRecordCount = ReadValue<std::uint64_t>(input);
    }
    header.nodeBlocksOffset = ReadValue<std::uint64_t>(input);
    header.graphEdgeBlocksOffset = ReadValue<std::uint64_t>(input);
    header.rangeEdgeBlocksOffset = ReadValue<std::uint64_t>(input);
    header.graphPagesOffset = ReadValue<std::uint64_t>(input);
    header.spatialPageRefsOffset = ReadValue<std::uint64_t>(input);
    header.sourceBlockRefsOffset = ReadValue<std::uint64_t>(input);
    header.childSourceBlockRefsOffset = ReadValue<std::uint64_t>(input);
    header.lodChildPageRefsOffset = ReadValue<std::uint64_t>(input);
    header.pageMasksOffset = ReadValue<std::uint64_t>(input);
    header.membershipOffset = ReadValue<std::uint64_t>(input);
    header.edgeSpatialBoundsOffset = ReadValue<std::uint64_t>(input);
    if (header.version >= kFormatVersionV6) {
        header.rootPayloadOffset = ReadValue<std::uint64_t>(input);
    }
    return header;
}

void WriteSourceBlock(std::ostream& output, const CHIndexSourceBlock& block) {
    WriteValue(output, block.firstRecord);
    WriteValue(output, block.recordCount);
    WriteValue(output, block.byteOffset);
}

CHIndexSourceBlock ReadSourceBlock(std::istream& input) {
    CHIndexSourceBlock block;
    block.firstRecord = ReadValue<std::uint64_t>(input);
    block.recordCount = ReadValue<std::uint32_t>(input);
    block.byteOffset = ReadValue<std::uint64_t>(input);
    return block;
}

void WriteGraphPage(std::ostream& output, const CHIndexGraphPage& page) {
    WriteValue(output, page.pageId);
    WriteValue(output, page.bandIndex);
    WriteValue(output, page.levelMin);
    WriteValue(output, page.levelMax);
    WriteValue(output, page.spatialLevel);
    WriteValue(output, page.tileX);
    WriteValue(output, page.tileY);
    WriteValue(output, page.subPage);
    WriteValue(output, page.edgeCount);
    WriteValue(output, page.sourceBlockRefOffset);
    WriteValue(output, page.sourceBlockRefCount);
    WriteValue(output, page.childSourceBlockRefOffset);
    WriteValue(output, page.childSourceBlockRefCount);
    WriteValue(output, page.lodChildPageRefOffset);
    WriteValue(output, page.lodChildPageRefCount);
    WriteValue(output, page.lodMaskOffset);
    WriteValue(output, page.membershipByteOffset);
    WriteValue(output, page.membershipByteSize);
    WriteValue(output, page.rootPayloadRecordOffset);
}

CHIndexGraphPage ReadGraphPage(std::istream& input, std::uint32_t formatVersion) {
    CHIndexGraphPage page;
    page.pageId = ReadValue<std::uint32_t>(input);
    page.bandIndex = ReadValue<std::uint32_t>(input);
    page.levelMin = ReadValue<std::uint32_t>(input);
    page.levelMax = ReadValue<std::uint32_t>(input);
    page.spatialLevel = ReadValue<std::uint32_t>(input);
    page.tileX = ReadValue<std::uint32_t>(input);
    page.tileY = ReadValue<std::uint32_t>(input);
    page.subPage = ReadValue<std::uint32_t>(input);
    page.edgeCount = ReadValue<std::uint32_t>(input);
    page.sourceBlockRefOffset = ReadValue<std::uint32_t>(input);
    page.sourceBlockRefCount = ReadValue<std::uint32_t>(input);
    page.childSourceBlockRefOffset = ReadValue<std::uint32_t>(input);
    page.childSourceBlockRefCount = ReadValue<std::uint32_t>(input);
    page.lodChildPageRefOffset = ReadValue<std::uint32_t>(input);
    page.lodChildPageRefCount = ReadValue<std::uint32_t>(input);
    page.lodMaskOffset = ReadValue<std::uint32_t>(input);
    page.membershipByteOffset = ReadValue<std::uint64_t>(input);
    page.membershipByteSize = ReadValue<std::uint32_t>(input);
    if (formatVersion >= kFormatVersionV6) {
        page.rootPayloadRecordOffset = ReadValue<std::uint64_t>(input);
    }
    return page;
}

CHIndexRootRecord ReadRootRecord(std::istream& input) {
    CHIndexRootRecord record;
    record.globalEdgeId = ReadValue<std::uint32_t>(input);
    record.birthLevel = ReadValue<std::int32_t>(input);
    record.deathLevel = ReadValue<std::int32_t>(input);
    record.boundsLo = ReadValue<std::uint32_t>(input);
    record.boundsHi = ReadValue<std::uint32_t>(input);
    record.childA = ReadValue<std::uint32_t>(input);
    record.childB = ReadValue<std::uint32_t>(input);
    record.geometryError = ReadValue<float>(input);
    record.sourceX = ReadValue<float>(input);
    record.sourceY = ReadValue<float>(input);
    record.targetX = ReadValue<float>(input);
    record.targetY = ReadValue<float>(input);
    return record;
}

void WriteSpatialPageRef(std::ostream& output, const CHIndexSpatialPageRef& ref) {
    WriteValue(output, ref.key);
    WriteValue(output, ref.pageId);
}

CHIndexSpatialPageRef ReadSpatialPageRef(std::istream& input) {
    CHIndexSpatialPageRef ref;
    ref.key = ReadValue<std::uint64_t>(input);
    ref.pageId = ReadValue<std::uint32_t>(input);
    return ref;
}

std::uint64_t SpatialPageKey(std::uint32_t level, std::uint32_t x, std::uint32_t y) {
    return static_cast<std::uint64_t>(x) |
           (static_cast<std::uint64_t>(y) << 16u) |
           (static_cast<std::uint64_t>(level) << 32u);
}


void Seek(std::istream& input, std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("CH index offset exceeds stream range");
    }
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) {
        throw std::runtime_error("invalid CH index offset");
    }
}

template <typename T>
std::vector<T> ReadScalarArray(std::istream& input, std::uint64_t offset,
                               std::uint64_t count) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("CH index array is too large");
    }
    Seek(input, offset);
    std::vector<T> values(static_cast<std::size_t>(count));
    for (auto& value : values) {
        value = ReadValue<T>(input);
    }
    return values;
}

void EncodeVarUInt32(std::uint32_t value, std::vector<std::uint8_t>& output) {
    while (value >= 0x80u) {
        output.push_back(static_cast<std::uint8_t>((value & 0x7fu) | 0x80u));
        value >>= 7u;
    }
    output.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t DecodeVarUInt32(std::span<const std::uint8_t> bytes, std::size_t& position) {
    std::uint32_t value = 0;
    std::uint32_t shift = 0;
    while (position < bytes.size() && shift <= 28u) {
        const auto byte = bytes[position++];
        value |= static_cast<std::uint32_t>(byte & 0x7fu) << shift;
        if ((byte & 0x80u) == 0) {
            return value;
        }
        shift += 7u;
    }
    throw std::runtime_error("invalid CH index varint");
}

std::vector<std::uint8_t> EncodeMembership(
    const analysis::DatasetLayoutAnalysisReport& report,
    std::vector<CHIndexGraphPage>& pages) {
    std::vector<std::uint8_t> bytes;
    for (const auto& sourcePage : report.runtimeGraphPages) {
        auto& page = pages[sourcePage.pageId];
        page.membershipByteOffset = bytes.size();

        const auto begin = report.runtimeGraphPageEdgeIds.begin() + sourcePage.edgeIdOffset;
        const auto end = begin + sourcePage.edgeCount;
        std::uint32_t previous = 0;
        bool first = true;
        for (auto it = begin; it != end; ++it) {
            const auto value = first ? *it : *it - previous;
            EncodeVarUInt32(value, bytes);
            previous = *it;
            first = false;
        }
        page.membershipByteSize =
            static_cast<std::uint32_t>(bytes.size() - page.membershipByteOffset);
    }
    return bytes;
}

Header BuildHeader(const analysis::DatasetLayoutAnalysisReport& report,
                   std::uint64_t membershipByteCount) {
    Header header;
    header.graph = GetFileStamp(report.graphPath);
    header.ranges = GetFileStamp(report.rangesPath);
    header.nodeCount = report.nodeCount;
    header.edgeCount = report.edgeCount;
    header.maxLevel = report.maxLevel;
    header.spatialGridSize = report.config.spatialGridSize;
    header.spatialCellSizeMeters = report.config.spatialCellSizeMeters;
    header.nodeBlockRecordCount = report.config.sourceBlockNodeCount;
    header.edgeBlockRecordCount = report.config.sourceBlockEdgeCount;
    header.lodBandWidth = report.config.lodBandWidth;
    header.lodMaskWordsPerPage = report.lodMaskWordsPerBlock;
    header.minLatitude = report.minLatitude;
    header.minLongitude = report.minLongitude;
    header.maxLatitude = report.maxLatitude;
    header.maxLongitude = report.maxLongitude;
    header.nodeBlockCount = report.nodeSourceBlocks.size();
    header.graphEdgeBlockCount = report.graphSourceBlocks.size();
    header.rangeEdgeBlockCount = report.rangeSourceBlocks.size();
    header.graphPageCount = report.runtimeGraphPages.size();
    header.spatialPageRefCount = report.runtimeGraphPages.size();
    header.sourceBlockRefCount = report.runtimeGraphPageSourceBlockRefs.size();
    header.childSourceBlockRefCount = report.runtimeGraphPageChildSourceBlockRefs.size();
    header.lodChildPageRefCount = report.runtimeGraphPageLodChildRefs.size();
    header.pageMaskWordCount = report.runtimeGraphPageAliveMasks.size();
    header.membershipByteCount = membershipByteCount;
    header.edgeSpatialBoundsCount = report.edgeSpatialBounds.size();
    header.rootPayloadRecordCount = 0;

    header.nodeBlocksOffset = kHeaderSerializedSize;
    header.graphEdgeBlocksOffset =
        header.nodeBlocksOffset + header.nodeBlockCount * kSourceBlockSerializedSize;
    header.rangeEdgeBlocksOffset =
        header.graphEdgeBlocksOffset + header.graphEdgeBlockCount * kSourceBlockSerializedSize;
    header.graphPagesOffset =
        header.rangeEdgeBlocksOffset + header.rangeEdgeBlockCount * kSourceBlockSerializedSize;
    header.spatialPageRefsOffset =
        header.graphPagesOffset + header.graphPageCount * kGraphPageSerializedSize;
    header.sourceBlockRefsOffset =
        header.spatialPageRefsOffset +
        header.spatialPageRefCount * kSpatialPageRefSerializedSize;
    header.childSourceBlockRefsOffset =
        header.sourceBlockRefsOffset + header.sourceBlockRefCount * sizeof(std::uint32_t);
    header.lodChildPageRefsOffset =
        header.childSourceBlockRefsOffset +
        header.childSourceBlockRefCount * sizeof(std::uint32_t);
    header.pageMasksOffset =
        header.lodChildPageRefsOffset + header.lodChildPageRefCount * sizeof(std::uint32_t);
    header.membershipOffset =
        header.pageMasksOffset + header.pageMaskWordCount * sizeof(std::uint64_t);
    header.edgeSpatialBoundsOffset = header.membershipOffset + header.membershipByteCount;
    header.rootPayloadOffset = 0;
    return header;
}
Header BuildHeader(const CHIndexBuildData& data) {
    Header header;
    header.graph = GetFileStamp(data.graphPath);
    header.ranges = GetFileStamp(data.rangesPath);
    header.nodeCount = data.nodeCount;
    header.edgeCount = data.edgeCount;
    header.maxLevel = data.maxLevel;
    header.spatialGridSize = data.spatialGridSize;
    header.spatialCellSizeMeters = data.spatialCellSizeMeters;
    header.nodeBlockRecordCount = data.nodeBlockRecordCount;
    header.edgeBlockRecordCount = data.edgeBlockRecordCount;
    header.lodBandWidth = data.lodBandWidth;
    header.lodMaskWordsPerPage = data.lodMaskWordsPerPage;
    header.minLatitude = data.minLatitude;
    header.minLongitude = data.minLongitude;
    header.maxLatitude = data.maxLatitude;
    header.maxLongitude = data.maxLongitude;
    header.nodeBlockCount = data.nodeBlocks.size();
    header.graphEdgeBlockCount = data.graphEdgeBlocks.size();
    header.rangeEdgeBlockCount = data.rangeEdgeBlocks.size();
    header.graphPageCount = data.graphPages.size();
    header.spatialPageRefCount = data.spatialPageRefs.size();
    header.sourceBlockRefCount = data.pageSourceBlockRefs.size();
    header.childSourceBlockRefCount = data.pageChildSourceBlockRefs.size();
    header.lodChildPageRefCount = data.pageLodChildRefs.size();
    header.pageMaskWordCount = data.pageAliveMasks.size();
    header.membershipByteCount = data.membershipByteCount;
    header.edgeSpatialBoundsCount = data.edgeSpatialBoundsCount;
    header.rootPayloadRecordCount = data.rootPayloadRecordCount;

    header.nodeBlocksOffset = kHeaderSerializedSize;
    header.graphEdgeBlocksOffset =
        header.nodeBlocksOffset + header.nodeBlockCount * kSourceBlockSerializedSize;
    header.rangeEdgeBlocksOffset =
        header.graphEdgeBlocksOffset + header.graphEdgeBlockCount * kSourceBlockSerializedSize;
    header.graphPagesOffset =
        header.rangeEdgeBlocksOffset + header.rangeEdgeBlockCount * kSourceBlockSerializedSize;
    header.spatialPageRefsOffset =
        header.graphPagesOffset + header.graphPageCount * kGraphPageSerializedSize;
    header.sourceBlockRefsOffset =
        header.spatialPageRefsOffset +
        header.spatialPageRefCount * kSpatialPageRefSerializedSize;
    header.childSourceBlockRefsOffset =
        header.sourceBlockRefsOffset + header.sourceBlockRefCount * sizeof(std::uint32_t);
    header.lodChildPageRefsOffset =
        header.childSourceBlockRefsOffset +
        header.childSourceBlockRefCount * sizeof(std::uint32_t);
    header.pageMasksOffset =
        header.lodChildPageRefsOffset + header.lodChildPageRefCount * sizeof(std::uint32_t);
    header.membershipOffset =
        header.pageMasksOffset + header.pageMaskWordCount * sizeof(std::uint64_t);
    header.edgeSpatialBoundsOffset = header.membershipOffset + header.membershipByteCount;
    header.rootPayloadOffset = header.rootPayloadRecordCount == 0
                                   ? 0
                                   : header.edgeSpatialBoundsOffset +
                                         header.edgeSpatialBoundsCount * sizeof(std::uint64_t);
    return header;
}

bool MatchesSourceLayout(const Header& header, const std::filesystem::path& graphPath,
                         const std::filesystem::path& rangesPath,
                         const analysis::DatasetLayoutAnalysisConfig& config) {
    const auto graph = GetFileStamp(graphPath);
    const auto ranges = GetFileStamp(rangesPath);
    return graph.size != 0 && ranges.size != 0 &&
           header.graph.size == graph.size && header.graph.writeTime == graph.writeTime &&
           header.ranges.size == ranges.size && header.ranges.writeTime == ranges.writeTime &&
           std::abs(header.spatialCellSizeMeters - config.spatialCellSizeMeters) <=
               std::max(1e-9, std::abs(config.spatialCellSizeMeters) * 1e-12) &&
           header.nodeBlockRecordCount == config.sourceBlockNodeCount &&
           header.edgeBlockRecordCount == config.sourceBlockEdgeCount &&
           header.lodBandWidth == config.lodBandWidth;
}

bool Matches(const Header& header, const std::filesystem::path& graphPath,
             const std::filesystem::path& rangesPath,
             const analysis::DatasetLayoutAnalysisConfig& config) {
    return header.version == kFormatVersion &&
           MatchesSourceLayout(header, graphPath, rangesPath, config);
}

template <typename T>
std::span<const T> Subspan(const std::vector<T>& values, std::uint32_t offset,
                           std::uint32_t count) {
    const auto begin = static_cast<std::size_t>(offset);
    const auto size = static_cast<std::size_t>(count);
    if (begin > values.size() || size > values.size() - begin) {
        throw std::runtime_error("CH index reference range is invalid");
    }
    return std::span<const T>(values.data() + begin, size);
}

} // namespace

bool CHIndexData::GraphPageHasAliveEdges(std::uint32_t pageId,
                                         std::uint32_t level) const {
    if (pageId >= graphPages.size() || level > maxLevel || lodMaskWordsPerPage == 0) {
        return false;
    }
    const auto& page = graphPages[pageId];
    const auto word = level / 64u;
    const auto bit = level % 64u;
    const auto index = static_cast<std::size_t>(page.lodMaskOffset) + word;
    return index < pageAliveMasks.size() &&
           (pageAliveMasks[index] & (std::uint64_t{1} << bit)) != 0;
}

std::span<const std::uint32_t> CHIndexData::SourceBlockRefs(
    const CHIndexGraphPage& page) const {
    return Subspan(pageSourceBlockRefs, page.sourceBlockRefOffset, page.sourceBlockRefCount);
}

std::span<const std::uint32_t> CHIndexData::ChildSourceBlockRefs(
    const CHIndexGraphPage& page) const {
    return Subspan(pageChildSourceBlockRefs, page.childSourceBlockRefOffset,
                   page.childSourceBlockRefCount);
}

std::span<const std::uint32_t> CHIndexData::LodChildPageRefs(
    const CHIndexGraphPage& page) const {
    return Subspan(pageLodChildRefs, page.lodChildPageRefOffset, page.lodChildPageRefCount);
}

std::filesystem::path CHIndex::DefaultPath(const std::filesystem::path& graphPath) {
    if (graphPath.extension() == ".sch") {
        auto path = graphPath;
        path.replace_extension(".chidx");
        return path;
    }
    return std::filesystem::path(graphPath.string() + ".chidx");
}

bool CHIndex::IsValid(const std::filesystem::path& indexPath,
                      const std::filesystem::path& graphPath,
                      const std::filesystem::path& rangesPath,
                      const analysis::DatasetLayoutAnalysisConfig& config) {
    try {
        if (!std::filesystem::is_regular_file(indexPath)) {
            return false;
        }
        std::ifstream input(indexPath, std::ios::binary);
        if (!input) {
            return false;
        }
        return Matches(ReadHeader(input), graphPath, rangesPath, config);
    } catch (...) {
        return false;
    }
}

void CHIndex::Write(const analysis::DatasetLayoutAnalysisReport& report,
                    const std::filesystem::path& indexPath) {
    if (report.nodeSourceBlocks.empty() || report.graphSourceBlocks.empty() ||
        report.rangeSourceBlocks.empty() || report.runtimeGraphPages.empty()) {
        throw std::runtime_error("layout report does not contain runtime page metadata");
    }

    std::vector<CHIndexGraphPage> pages(report.runtimeGraphPages.size());
    for (const auto& source : report.runtimeGraphPages) {
        auto& page = pages[source.pageId];
        page.pageId = source.pageId;
        page.bandIndex = source.bandIndex;
        page.levelMin = source.levelMin;
        page.levelMax = source.levelMax;
        page.spatialLevel = source.spatialLevel;
        page.tileX = source.tileX;
        page.tileY = source.tileY;
        page.subPage = source.subPage;
        page.edgeCount = source.edgeCount;
        page.sourceBlockRefOffset = source.sourceBlockRefOffset;
        page.sourceBlockRefCount = source.sourceBlockRefCount;
        page.childSourceBlockRefOffset = source.childSourceBlockRefOffset;
        page.childSourceBlockRefCount = source.childSourceBlockRefCount;
        page.lodChildPageRefOffset = source.lodChildPageRefOffset;
        page.lodChildPageRefCount = source.lodChildPageRefCount;
        page.lodMaskOffset = source.lodMaskOffset;
    }

    std::vector<CHIndexSpatialPageRef> spatialPageRefs;
    spatialPageRefs.reserve(pages.size());
    for (const auto& page : pages) {
        spatialPageRefs.push_back(
            {SpatialPageKey(page.spatialLevel, page.tileX, page.tileY), page.pageId});
    }
    std::sort(spatialPageRefs.begin(), spatialPageRefs.end(), [](const auto& a, const auto& b) {
        return a.key != b.key ? a.key < b.key : a.pageId < b.pageId;
    });

    const auto membershipBytes = EncodeMembership(report, pages);
    const auto header = BuildHeader(report, membershipBytes.size());
    const auto temporaryPath = std::filesystem::path(indexPath.string() + ".tmp");
    std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not write CH index: " + temporaryPath.string());
    }

    WriteHeader(output, header);
    for (const auto& source : report.nodeSourceBlocks) {
        WriteSourceBlock(output, {source.firstRecord, source.recordCount, source.byteOffset});
    }
    for (const auto& source : report.graphSourceBlocks) {
        WriteSourceBlock(output, {source.firstRecord, source.recordCount, source.byteOffset});
    }
    for (const auto& source : report.rangeSourceBlocks) {
        WriteSourceBlock(output, {source.firstRecord, source.recordCount, source.byteOffset});
    }
    for (const auto& page : pages) {
        WriteGraphPage(output, page);
    }
    for (const auto& ref : spatialPageRefs) {
        WriteSpatialPageRef(output, ref);
    }
    for (const auto value : report.runtimeGraphPageSourceBlockRefs) {
        WriteValue(output, value);
    }
    for (const auto value : report.runtimeGraphPageChildSourceBlockRefs) {
        WriteValue(output, value);
    }
    for (const auto value : report.runtimeGraphPageLodChildRefs) {
        WriteValue(output, value);
    }
    for (const auto value : report.runtimeGraphPageAliveMasks) {
        WriteValue(output, value);
    }
    if (!membershipBytes.empty()) {
        output.write(reinterpret_cast<const char*>(membershipBytes.data()),
                     static_cast<std::streamsize>(membershipBytes.size()));
    }
    for (const auto value : report.edgeSpatialBounds) {
        WriteValue(output, value);
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize CH index");
    }

    std::error_code error;
    std::filesystem::remove(indexPath, error);
    error.clear();
    std::filesystem::rename(temporaryPath, indexPath, error);
    if (error) {
        std::filesystem::remove(temporaryPath);
        throw std::runtime_error("could not replace CH index: " + error.message());
    }
}

void CHIndex::Write(const CHIndexBuildData& data,
                    const std::filesystem::path& indexPath) {
    if (data.nodeBlocks.empty() || data.graphEdgeBlocks.empty() ||
        data.rangeEdgeBlocks.empty() || data.graphPages.empty() ||
        data.spatialPageRefs.empty()) {
        throw std::runtime_error("external CH index build data is incomplete");
    }
    if (!std::filesystem::is_regular_file(data.membershipBytesPath)) {
        throw std::runtime_error("external CH index membership stream is missing");
    }
    if (!std::filesystem::is_regular_file(data.edgeSpatialBoundsPath) ||
        data.edgeSpatialBoundsCount != data.edgeCount) {
        throw std::runtime_error("external CH index edge-spatial-bounds stream is missing or invalid");
    }
    if (data.rootPayloadRecordCount != 0 &&
        !std::filesystem::is_regular_file(data.rootPayloadPath)) {
        throw std::runtime_error("external CH index root-payload stream is missing");
    }

    const auto header = BuildHeader(data);
    const auto temporaryPath = std::filesystem::path(indexPath.string() + ".tmp");
    std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not write CH index: " + temporaryPath.string());
    }

    WriteHeader(output, header);
    for (const auto& block : data.nodeBlocks) {
        WriteSourceBlock(output, block);
    }
    for (const auto& block : data.graphEdgeBlocks) {
        WriteSourceBlock(output, block);
    }
    for (const auto& block : data.rangeEdgeBlocks) {
        WriteSourceBlock(output, block);
    }
    for (const auto& page : data.graphPages) {
        WriteGraphPage(output, page);
    }
    for (const auto& ref : data.spatialPageRefs) {
        WriteSpatialPageRef(output, ref);
    }
    for (const auto value : data.pageSourceBlockRefs) {
        WriteValue(output, value);
    }
    for (const auto value : data.pageChildSourceBlockRefs) {
        WriteValue(output, value);
    }
    for (const auto value : data.pageLodChildRefs) {
        WriteValue(output, value);
    }
    for (const auto value : data.pageAliveMasks) {
        WriteValue(output, value);
    }

    std::ifstream membership(data.membershipBytesPath, std::ios::binary);
    if (!membership) {
        throw std::runtime_error("could not read CH index membership stream");
    }
    std::vector<char> copyBuffer(1u << 20u);
    std::uint64_t copied = 0;
    while (membership && copied < data.membershipByteCount) {
        const auto remaining = data.membershipByteCount - copied;
        const auto count = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, copyBuffer.size()));
        membership.read(copyBuffer.data(), count);
        const auto got = membership.gcount();
        if (got <= 0) {
            break;
        }
        output.write(copyBuffer.data(), got);
        copied += static_cast<std::uint64_t>(got);
    }
    if (copied != data.membershipByteCount) {
        throw std::runtime_error("truncated CH index membership stream");
    }

    std::ifstream spatialBounds(data.edgeSpatialBoundsPath, std::ios::binary);
    if (!spatialBounds) {
        throw std::runtime_error("could not read CH index edge-spatial-bounds stream");
    }
    const auto expectedBoundsBytes = data.edgeSpatialBoundsCount * sizeof(std::uint64_t);
    copied = 0;
    while (spatialBounds && copied < expectedBoundsBytes) {
        const auto remaining = expectedBoundsBytes - copied;
        const auto count = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, copyBuffer.size()));
        spatialBounds.read(copyBuffer.data(), count);
        const auto got = spatialBounds.gcount();
        if (got <= 0) {
            break;
        }
        output.write(copyBuffer.data(), got);
        copied += static_cast<std::uint64_t>(got);
    }
    if (copied != expectedBoundsBytes) {
        throw std::runtime_error("truncated CH index edge-spatial-bounds stream");
    }

    if (data.rootPayloadRecordCount != 0) {
        std::ifstream rootPayload(data.rootPayloadPath, std::ios::binary);
        if (!rootPayload) {
            throw std::runtime_error("could not read CH index root-payload stream");
        }
        const auto expectedRootBytes =
            data.rootPayloadRecordCount * kRootRecordSerializedSize;
        copied = 0;
        while (rootPayload && copied < expectedRootBytes) {
            const auto remaining = expectedRootBytes - copied;
            const auto count = static_cast<std::streamsize>(
                std::min<std::uint64_t>(remaining, copyBuffer.size()));
            rootPayload.read(copyBuffer.data(), count);
            const auto got = rootPayload.gcount();
            if (got <= 0) {
                break;
            }
            output.write(copyBuffer.data(), got);
            copied += static_cast<std::uint64_t>(got);
        }
        if (copied != expectedRootBytes) {
            throw std::runtime_error("truncated CH index root-payload stream");
        }
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize CH index");
    }

    std::error_code error;
    std::filesystem::remove(indexPath, error);
    error.clear();
    std::filesystem::rename(temporaryPath, indexPath, error);
    if (error) {
        std::filesystem::remove(temporaryPath);
        throw std::runtime_error("could not replace CH index: " + error.message());
    }
}

CHIndexData CHIndex::Load(const std::filesystem::path& indexPath) {
    std::ifstream input(indexPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open CH index: " + indexPath.string());
    }
    const auto header = ReadHeader(input);

    CHIndexData result;
    result.indexPath = indexPath;
    result.formatVersion = header.version;
    result.nodeCount = header.nodeCount;
    result.edgeCount = header.edgeCount;
    result.maxLevel = header.maxLevel;
    result.spatialGridSize = header.spatialGridSize;
    result.spatialCellSizeMeters = header.spatialCellSizeMeters;
    result.nodeBlockRecordCount = header.nodeBlockRecordCount;
    result.edgeBlockRecordCount = header.edgeBlockRecordCount;
    result.lodBandWidth = header.lodBandWidth;
    result.lodMaskWordsPerPage = header.lodMaskWordsPerPage;
    result.minLatitude = header.minLatitude;
    result.minLongitude = header.minLongitude;
    result.maxLatitude = header.maxLatitude;
    result.maxLongitude = header.maxLongitude;
    result.membershipSectionOffset = header.membershipOffset;
    result.edgeSpatialBoundsSectionOffset = header.edgeSpatialBoundsOffset;
    result.edgeSpatialBoundsCount = header.edgeSpatialBoundsCount;
    result.rootPayloadSectionOffset = header.rootPayloadOffset;
    result.rootPayloadRecordCount = header.rootPayloadRecordCount;

    const auto checkCount = [](std::uint64_t count) {
        if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("CH index count exceeds addressable memory");
        }
    };
    checkCount(header.nodeBlockCount);
    checkCount(header.graphEdgeBlockCount);
    checkCount(header.rangeEdgeBlockCount);
    checkCount(header.graphPageCount);
    checkCount(header.spatialPageRefCount);

    Seek(input, header.nodeBlocksOffset);
    result.nodeBlocks.reserve(static_cast<std::size_t>(header.nodeBlockCount));
    for (std::uint64_t i = 0; i < header.nodeBlockCount; ++i) {
        result.nodeBlocks.push_back(ReadSourceBlock(input));
    }
    Seek(input, header.graphEdgeBlocksOffset);
    result.graphEdgeBlocks.reserve(static_cast<std::size_t>(header.graphEdgeBlockCount));
    for (std::uint64_t i = 0; i < header.graphEdgeBlockCount; ++i) {
        result.graphEdgeBlocks.push_back(ReadSourceBlock(input));
    }
    Seek(input, header.rangeEdgeBlocksOffset);
    result.rangeEdgeBlocks.reserve(static_cast<std::size_t>(header.rangeEdgeBlockCount));
    for (std::uint64_t i = 0; i < header.rangeEdgeBlockCount; ++i) {
        result.rangeEdgeBlocks.push_back(ReadSourceBlock(input));
    }
    Seek(input, header.graphPagesOffset);
    result.graphPages.reserve(static_cast<std::size_t>(header.graphPageCount));
    for (std::uint64_t i = 0; i < header.graphPageCount; ++i) {
        result.graphPages.push_back(ReadGraphPage(input, header.version));
    }
    Seek(input, header.spatialPageRefsOffset);
    result.spatialPageRefs.reserve(static_cast<std::size_t>(header.spatialPageRefCount));
    for (std::uint64_t i = 0; i < header.spatialPageRefCount; ++i) {
        result.spatialPageRefs.push_back(ReadSpatialPageRef(input));
    }

    result.pageSourceBlockRefs = ReadScalarArray<std::uint32_t>(
        input, header.sourceBlockRefsOffset, header.sourceBlockRefCount);
    result.pageChildSourceBlockRefs = ReadScalarArray<std::uint32_t>(
        input, header.childSourceBlockRefsOffset, header.childSourceBlockRefCount);
    result.pageLodChildRefs = ReadScalarArray<std::uint32_t>(
        input, header.lodChildPageRefsOffset, header.lodChildPageRefCount);
    result.pageAliveMasks = ReadScalarArray<std::uint64_t>(
        input, header.pageMasksOffset, header.pageMaskWordCount);
    if (header.edgeSpatialBoundsCount != header.edgeCount) {
        throw std::runtime_error("CH index edge-spatial-bounds count does not match edge count");
    }
    // v5 had no binary root payload, so its runtime still needs the full bounds vector to build
    // root records from text source pages. v6 keeps the billion-edge bounds table on disk and
    // reads only the contiguous ranges needed by backing/refinement work.
    if (header.version == kFormatVersionV5 || header.rootPayloadRecordCount == 0) {
        result.edgeSpatialBounds = ReadScalarArray<std::uint64_t>(
            input, header.edgeSpatialBoundsOffset, header.edgeSpatialBoundsCount);
    }
    if (header.version >= kFormatVersionV6 && header.rootPayloadRecordCount != 0) {
        if (header.rootPayloadOffset == 0) {
            throw std::runtime_error("CH index root payload count has no section offset");
        }
        for (const auto& page : result.graphPages) {
            if (page.rootPayloadRecordOffset > header.rootPayloadRecordCount ||
                page.edgeCount > header.rootPayloadRecordCount - page.rootPayloadRecordOffset) {
                throw std::runtime_error("CH index page root-payload range is invalid");
            }
        }
    }
    return result;
}

std::vector<std::uint32_t> CHIndex::ReadGraphPageEdgeIds(
    const CHIndexData& index, std::uint32_t pageId) {
    if (pageId >= index.graphPages.size()) {
        throw std::out_of_range("graph page id out of range");
    }
    const auto& page = index.graphPages[pageId];
    if (index.HasRootPayload()) {
        const auto records = ReadGraphPageRootRecords(index, pageId);
        std::vector<std::uint32_t> edgeIds;
        edgeIds.reserve(records.size());
        for (const auto& record : records) {
            edgeIds.push_back(record.globalEdgeId);
        }
        return edgeIds;
    }
    std::ifstream input(index.indexPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open CH index: " + index.indexPath.string());
    }
    Seek(input, index.membershipSectionOffset + page.membershipByteOffset);
    std::vector<std::uint8_t> bytes(page.membershipByteSize);
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw std::runtime_error("truncated CH index page membership");
        }
    }

    std::vector<std::uint32_t> edgeIds;
    edgeIds.reserve(page.edgeCount);
    std::size_t position = 0;
    std::uint32_t previous = 0;
    for (std::uint32_t i = 0; i < page.edgeCount; ++i) {
        const auto encoded = DecodeVarUInt32(bytes, position);
        const auto edgeId = i == 0 ? encoded : previous + encoded;
        edgeIds.push_back(edgeId);
        previous = edgeId;
    }
    if (position != bytes.size()) {
        throw std::runtime_error("CH index page membership has trailing data");
    }
    return edgeIds;
}

std::vector<CHIndexRootRecord> CHIndex::ReadGraphPageRootRecords(
    const CHIndexData& index, std::uint32_t pageId) {
    if (!index.HasRootPayload()) {
        return {};
    }
    if (pageId >= index.graphPages.size()) {
        throw std::out_of_range("graph page id out of range");
    }
    const auto& page = index.graphPages[pageId];
    if (page.rootPayloadRecordOffset > index.rootPayloadRecordCount ||
        page.edgeCount > index.rootPayloadRecordCount - page.rootPayloadRecordOffset) {
        throw std::runtime_error("CH index page root-payload range is invalid");
    }

    std::ifstream input(index.indexPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open CH index: " + index.indexPath.string());
    }
    Seek(input, index.rootPayloadSectionOffset +
                    page.rootPayloadRecordOffset * kRootRecordSerializedSize);
    std::vector<CHIndexRootRecord> records(page.edgeCount);
    if (!records.empty()) {
        input.read(reinterpret_cast<char*>(records.data()),
                   static_cast<std::streamsize>(records.size() * kRootRecordSerializedSize));
        if (!input) {
            throw std::runtime_error("truncated CH index root payload page");
        }
    }
    return records;
}

std::vector<std::uint64_t> CHIndex::ReadEdgeSpatialBounds(
    const CHIndexData& index, std::uint32_t firstEdgeId, std::uint32_t count) {
    if (static_cast<std::uint64_t>(firstEdgeId) + count > index.edgeSpatialBoundsCount) {
        throw std::out_of_range("edge spatial-bounds range is out of range");
    }
    if (!index.edgeSpatialBounds.empty()) {
        const auto begin = index.edgeSpatialBounds.begin() + firstEdgeId;
        return std::vector<std::uint64_t>(begin, begin + count);
    }
    if (count == 0) {
        return {};
    }
    std::ifstream input(index.indexPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open CH index: " + index.indexPath.string());
    }
    return ReadScalarArray<std::uint64_t>(
        input,
        index.edgeSpatialBoundsSectionOffset +
            static_cast<std::uint64_t>(firstEdgeId) * sizeof(std::uint64_t),
        count);
}

} // namespace chmv::streaming::index
