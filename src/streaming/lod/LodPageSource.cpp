#include "streaming/lod/LodPageSource.h"

#include "data/ch/CHProjection.h"
#include "data/ch/CHTypes.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace chmv::streaming::lod {
namespace {

constexpr std::uint32_t kNoEdge = 0xffffffffu;
constexpr std::uint64_t kProgressStep = 1u << 20;

// Whitespace-separated tokens with '#' line comments, read in large chunks. Same token rules
// as CHLoader, so both readers see identical values.
class SourceTokens {
public:
    explicit SourceTokens(const std::filesystem::path& path)
        : name_(path.filename()), input_(path, std::ios::binary), buffer_(4u << 20u) {
        if (!input_) {
            throw std::runtime_error("could not open " + path.string());
        }
    }

    [[nodiscard]] std::string_view Next() {
        for (;;) {
            while (position_ < end_ && IsSpace(buffer_[position_])) {
                ++position_;
            }
            if (position_ == end_) {
                if (!Refill(end_)) {
                    throw std::runtime_error(Name() + ": unexpected end of file");
                }
                continue;
            }
            if (buffer_[position_] == '#') {
                SkipLine();
                continue;
            }
            break;
        }

        auto begin = position_;
        for (;;) {
            while (position_ < end_ && !IsSpace(buffer_[position_])) {
                ++position_;
            }
            if (position_ < end_ || end_ - begin == buffer_.size()) {
                break;
            }
            const auto kept = end_ - begin;
            const bool more = Refill(begin);
            begin = 0;
            position_ = kept;
            if (!more) {
                break;
            }
        }
        return {buffer_.data() + begin, position_ - begin};
    }

    // Converted to narrow text only when a message needs it.
    [[nodiscard]] std::string Name() const { return name_.string(); }

private:
    static bool IsSpace(char ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'; }

    // Keeps [keepFrom, end_) at the front of the buffer and appends fresh input after it.
    bool Refill(std::size_t keepFrom) {
        const auto kept = end_ - keepFrom;
        if (kept != 0u && keepFrom != 0u) {
            std::memmove(buffer_.data(), buffer_.data() + keepFrom, kept);
        }
        input_.read(buffer_.data() + kept, static_cast<std::streamsize>(buffer_.size() - kept));
        const auto got = static_cast<std::size_t>(input_.gcount());
        end_ = kept + got;
        position_ = 0;
        return got != 0u;
    }

    void SkipLine() {
        for (;;) {
            while (position_ < end_ && buffer_[position_] != '\n') {
                ++position_;
            }
            if (position_ < end_ || !Refill(end_)) {
                return;
            }
        }
    }

    std::filesystem::path name_;
    std::ifstream input_;
    std::vector<char> buffer_;
    std::size_t position_ = 0;
    std::size_t end_ = 0;
};

[[noreturn]] void BadToken(const SourceTokens& tokens, std::string_view token, const char* what) {
    throw std::runtime_error(tokens.Name() + ": invalid " + what + " '" + std::string(token) + "'");
}

template <typename T>
[[nodiscard]] T ParseNumber(const SourceTokens& tokens, std::string_view token, const char* what) {
    T value{};
    const auto* end = token.data() + token.size();
    const auto [ptr, ec] = std::from_chars(token.data(), end, value);
    if (ec != std::errc{} || ptr != end) {
        BadToken(tokens, token, what);
    }
    return value;
}

template <typename T>
[[nodiscard]] T ParseRanged(const SourceTokens& tokens, std::string_view token, const char* what) {
    const auto value = ParseNumber<std::int64_t>(tokens, token, what);
    if (value < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
        value > static_cast<std::int64_t>(std::numeric_limits<T>::max())) {
        BadToken(tokens, token, what);
    }
    return static_cast<T>(value);
}

// Decimal degrees with at most seven significant decimals, as exact 1e-7 integers.
[[nodiscard]] std::int32_t ParseDegrees(const SourceTokens& tokens, std::string_view token) {
    std::size_t i = 0;
    bool negative = false;
    if (i < token.size() && (token[i] == '-' || token[i] == '+')) {
        negative = token[i] == '-';
        ++i;
    }
    std::int64_t whole = 0;
    std::int64_t fraction = 0;
    int fractionDigits = 0;
    bool anyDigit = false;
    bool afterPoint = false;
    for (; i < token.size(); ++i) {
        const auto ch = token[i];
        if (ch == '.' && !afterPoint) {
            afterPoint = true;
            continue;
        }
        if (ch < '0' || ch > '9') {
            BadToken(tokens, token, "coordinate");
        }
        anyDigit = true;
        const auto digit = static_cast<std::int64_t>(ch - '0');
        if (!afterPoint) {
            whole = whole * 10 + digit;
            if (whole > 180) {
                BadToken(tokens, token, "coordinate");
            }
        } else if (fractionDigits < 7) {
            fraction = fraction * 10 + digit;
            ++fractionDigits;
        } else if (digit != 0) {
            throw std::runtime_error(tokens.Name() + ": coordinate '" + std::string(token) +
                                     "' has more than 7 decimals and cannot be stored exactly");
        }
    }
    if (!anyDigit) {
        BadToken(tokens, token, "coordinate");
    }
    for (; fractionDigits < 7; ++fractionDigits) {
        fraction *= 10;
    }
    const auto value = whole * 10000000 + fraction;
    return static_cast<std::int32_t>(negative ? -value : value);
}

[[nodiscard]] std::uint32_t ParseWeight(const SourceTokens& tokens, std::string_view token) {
    const auto value = ParseNumber<double>(tokens, token, "edge weight");
    if (!(value >= 0.0) || value > 4294967295.0 || std::floor(value) != value) {
        throw std::runtime_error(tokens.Name() + ": edge weight '" + std::string(token) +
                                 "' is not a non-negative integer");
    }
    return static_cast<std::uint32_t>(value);
}

template <typename T>
void WriteSection(LodPageOutput& output, const SectionEntry& section, StreamHash& hash,
                  const std::vector<T>& values) {
    output.WriteAt(section.offset, values.data(), values.size() * sizeof(T));
    hash.Update(values.data(), values.size() * sizeof(T));
}

void Report(const LodPageBuilder::ProgressCallback& progress, BuildStage stage,
            std::uint64_t current, std::uint64_t total) {
    if (progress && (current % kProgressStep == 0u || current == total)) {
        progress({stage, current, total});
    }
}

void PlaceCHSections(FileHeader& header, std::uint64_t& offset) {
    const auto n = header.nodeCount;
    const auto m = header.edgeCount;
    const std::array<std::pair<Section, std::uint64_t>, 17> sizes{{
        {Section::NodeOsmId, n * 8u},
        {Section::NodeLatitude, n * 4u},
        {Section::NodeLongitude, n * 4u},
        {Section::NodeElevation, n * 4u},
        {Section::NodeLevel, n * 4u},
        {Section::EdgeSource, m * 4u},
        {Section::EdgeTarget, m * 4u},
        {Section::EdgeWeight, m * 4u},
        {Section::EdgeType, m * 2u},
        {Section::EdgeMaxSpeed, m * 2u},
        {Section::EdgeChildA, m * 4u},
        {Section::EdgeChildB, m * 4u},
        {Section::EdgeBirth, m * 2u},
        {Section::EdgeDeath, m * 2u},
        {Section::EdgeError, m * 4u},
        {Section::EdgeCoverLow, m * 2u},
        {Section::EdgeCoverHigh, m * 2u},
    }};
    offset = kHeaderBytes;
    for (const auto& [section, bytes] : sizes) {
        header.At(section) = {offset, bytes};
        offset = AlignSection(offset + bytes);
    }
}

[[nodiscard]] StreamHash& HashOf(SectionHashes& hashes, Section section) {
    return hashes[static_cast<std::size_t>(section)];
}

} // namespace

LodPageNodes ReadLodPageSource(const std::filesystem::path& graphPath,
                               const std::filesystem::path& rangesPath, FileHeader& header,
                               LodPageOutput& output, SectionHashes& hashes,
                               std::uint64_t& nextSectionOffset,
                               const SourceCountsCallback& checkCounts,
                               const LodPageBuilder::ProgressCallback& progress) {
    SourceTokens graph(graphPath);
    const auto nodeCount = ParseNumber<std::uint64_t>(graph, graph.Next(), "node count");
    const auto edgeCount = ParseNumber<std::uint64_t>(graph, graph.Next(), "edge count");
    if (nodeCount == 0u || nodeCount >= kNoEdge || edgeCount >= kNoEdge) {
        throw std::runtime_error(graph.Name() + ": node/edge counts must fit 32-bit ids");
    }
    if (edgeCount == 0u) {
        throw std::runtime_error(graph.Name() + ": graph has no edges");
    }
    if (checkCounts) {
        checkCounts(nodeCount, edgeCount);
    }

    header.nodeCount = nodeCount;
    header.edgeCount = edgeCount;
    PlaceCHSections(header, nextSectionOffset);

    LodPageNodes nodes;
    nodes.nodeCount = nodeCount;
    nodes.edgeCount = edgeCount;
    const auto n = static_cast<std::size_t>(nodeCount);
    const auto m = static_cast<std::size_t>(edgeCount);

    // Latitude goes to y and longitude to x; both become projected fixed point below.
    nodes.x.resize(n);
    nodes.y.resize(n);
    nodes.level.resize(n);
    {
        ColumnWriter<std::uint64_t> osmIds(output, header.At(Section::NodeOsmId),
                                           HashOf(hashes, Section::NodeOsmId));
        ColumnWriter<std::int32_t> latitudes(output, header.At(Section::NodeLatitude),
                                             HashOf(hashes, Section::NodeLatitude));
        ColumnWriter<std::int32_t> longitudes(output, header.At(Section::NodeLongitude),
                                              HashOf(hashes, Section::NodeLongitude));
        ColumnWriter<float> elevations(output, header.At(Section::NodeElevation),
                                       HashOf(hashes, Section::NodeElevation));
        ColumnWriter<std::uint32_t> levels(output, header.At(Section::NodeLevel),
                                           HashOf(hashes, Section::NodeLevel));
        for (std::size_t i = 0; i < n; ++i) {
            const auto id = ParseNumber<std::uint64_t>(graph, graph.Next(), "node id");
            if (id != i) {
                throw std::runtime_error(graph.Name() +
                                         ": node ids must be listed as 0..n-1 in order");
            }
            osmIds.Push(ParseNumber<std::uint64_t>(graph, graph.Next(), "OSM id"));
            const auto latitude = ParseDegrees(graph, graph.Next());
            const auto longitude = ParseDegrees(graph, graph.Next());
            latitudes.Push(latitude);
            longitudes.Push(longitude);
            nodes.y[i] = latitude;
            nodes.x[i] = longitude;
            elevations.Push(ParseNumber<float>(graph, graph.Next(), "elevation"));
            const auto level = ParseNumber<std::uint32_t>(graph, graph.Next(), "node level");
            if (level > 0xffffu) {
                throw std::runtime_error(graph.Name() + ": node level above 65535");
            }
            levels.Push(level);
            nodes.level[i] = static_cast<std::uint16_t>(level);
            nodes.maxNodeLevel = std::max(nodes.maxNodeLevel, level);
            Report(progress, BuildStage::ReadNodes, i + 1u, nodeCount);
        }
        osmIds.Flush();
        latitudes.Flush();
        longitudes.Flush();
        elevations.Flush();
        levels.Flush();
    }

    {
        ColumnWriter<std::uint32_t> sources(output, header.At(Section::EdgeSource),
                                            HashOf(hashes, Section::EdgeSource));
        ColumnWriter<std::uint32_t> targets(output, header.At(Section::EdgeTarget),
                                            HashOf(hashes, Section::EdgeTarget));
        ColumnWriter<std::uint32_t> weights(output, header.At(Section::EdgeWeight),
                                            HashOf(hashes, Section::EdgeWeight));
        ColumnWriter<std::int16_t> types(output, header.At(Section::EdgeType),
                                         HashOf(hashes, Section::EdgeType));
        ColumnWriter<std::int16_t> maxSpeeds(output, header.At(Section::EdgeMaxSpeed),
                                             HashOf(hashes, Section::EdgeMaxSpeed));
        ColumnWriter<std::uint32_t> childrenA(output, header.At(Section::EdgeChildA),
                                              HashOf(hashes, Section::EdgeChildA));
        ColumnWriter<std::uint32_t> childrenB(output, header.At(Section::EdgeChildB),
                                              HashOf(hashes, Section::EdgeChildB));
        const auto readChild = [&]() {
            const auto value = ParseNumber<std::int64_t>(graph, graph.Next(), "child edge");
            if (value >= static_cast<std::int64_t>(edgeCount)) {
                throw std::runtime_error(graph.Name() + ": child edge out of range");
            }
            return value < 0 ? kNoEdge : static_cast<std::uint32_t>(value);
        };
        for (std::size_t e = 0; e < m; ++e) {
            const auto source = ParseNumber<std::uint32_t>(graph, graph.Next(), "edge source");
            const auto target = ParseNumber<std::uint32_t>(graph, graph.Next(), "edge target");
            if (source >= nodeCount || target >= nodeCount) {
                throw std::runtime_error(graph.Name() + ": edge endpoint out of range");
            }
            sources.Push(source);
            targets.Push(target);
            weights.Push(ParseWeight(graph, graph.Next()));
            types.Push(ParseRanged<std::int16_t>(graph, graph.Next(), "edge type"));
            maxSpeeds.Push(ParseRanged<std::int16_t>(graph, graph.Next(), "max speed"));
            const auto childA = readChild();
            const auto childB = readChild();
            if ((childA == kNoEdge) != (childB == kNoEdge)) {
                throw std::runtime_error(graph.Name() + ": edge " + std::to_string(e) +
                                         " has only one child");
            }
            childrenA.Push(childA);
            childrenB.Push(childB);
            Report(progress, BuildStage::ReadEdges, e + 1u, edgeCount);
        }
        sources.Flush();
        targets.Flush();
        weights.Flush();
        types.Flush();
        maxSpeeds.Flush();
        childrenA.Flush();
        childrenB.Flush();
    }

    {
        // The ranges file may list edges in any order, so both columns are assembled in memory.
        std::vector<std::int16_t> birth(m, -1);
        std::vector<std::int16_t> death(m, -1);
        SourceTokens ranges(rangesPath);
        for (std::size_t i = 0; i < m; ++i) {
            const auto edge = ParseNumber<std::uint32_t>(ranges, ranges.Next(), "range edge id");
            if (edge >= edgeCount) {
                throw std::runtime_error(ranges.Name() + ": range edge id out of range");
            }
            birth[edge] = ParseRanged<std::int16_t>(ranges, ranges.Next(), "birth");
            death[edge] = ParseRanged<std::int16_t>(ranges, ranges.Next(), "death");
            Report(progress, BuildStage::ReadRanges, i + 1u, edgeCount);
        }
        WriteSection(output, header.At(Section::EdgeBirth), HashOf(hashes, Section::EdgeBirth),
                     birth);
        WriteSection(output, header.At(Section::EdgeDeath), HashOf(hashes, Section::EdgeDeath),
                     death);
        for (const auto value : birth) {
            nodes.maxRangeLevel = std::max<std::int32_t>(nodes.maxRangeLevel, value);
        }
    }

    auto minLatitude = std::numeric_limits<std::int32_t>::max();
    auto maxLatitude = std::numeric_limits<std::int32_t>::min();
    auto minLongitude = minLatitude;
    auto maxLongitude = maxLatitude;
    for (std::size_t i = 0; i < n; ++i) {
        minLatitude = std::min(minLatitude, nodes.y[i]);
        maxLatitude = std::max(maxLatitude, nodes.y[i]);
        minLongitude = std::min(minLongitude, nodes.x[i]);
        maxLongitude = std::max(maxLongitude, nodes.x[i]);
    }

    // Same projection as the full-resident loader: ComputeCHProjection only looks at bounds.
    std::array<data::CHNode, 2> corners{};
    corners[0].latitude = minLatitude / kDegreeScale;
    corners[0].longitude = minLongitude / kDegreeScale;
    corners[1].latitude = maxLatitude / kDegreeScale;
    corners[1].longitude = maxLongitude / kDegreeScale;
    const auto projection = data::ComputeCHProjection(corners);
    header.centerLatitude = projection.centerLatitude;
    header.centerLongitude = projection.centerLongitude;
    header.longitudeScale = projection.longitudeScale;
    header.normalization = projection.normalization;
    header.minLatitude = corners[0].latitude;
    header.maxLatitude = corners[1].latitude;
    header.minLongitude = corners[0].longitude;
    header.maxLongitude = corners[1].longitude;

    constexpr double kLimit = 1073741823.0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto point = projection.Project(nodes.y[i] / kDegreeScale, nodes.x[i] / kDegreeScale);
        const auto x = std::round(point.x * kFixedCoordinateScale);
        const auto y = std::round(point.y * kFixedCoordinateScale);
        if (std::abs(x) > kLimit || std::abs(y) > kLimit) {
            throw std::runtime_error("projected coordinate outside the fixed-point range");
        }
        nodes.x[i] = static_cast<std::int32_t>(x);
        nodes.y[i] = static_cast<std::int32_t>(y);
    }
    if (nodes.maxRangeLevel > kMaxStoredLevel) {
        throw std::runtime_error("CH level above 65534 is not supported");
    }
    return nodes;
}

} // namespace chmv::streaming::lod
