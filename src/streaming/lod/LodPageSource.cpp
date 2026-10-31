#include "streaming/lod/LodPageSource.h"

#include "data/ch/CHProjection.h"
#include "data/ch/CHTypes.h"
#include "streaming/runtime/GraphPageStreamer.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
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
        : name_(path.filename().string()), input_(path, std::ios::binary), buffer_(4u << 20u) {
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
                    throw std::runtime_error(name_ + ": unexpected end of file");
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

    [[nodiscard]] const std::string& Name() const { return name_; }

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

    std::string name_;
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

template <typename T> class ColumnWriter {
public:
    ColumnWriter(LodPageOutput& output, const SectionEntry& section)
        : output_(&output), offset_(section.offset), end_(section.offset + section.bytes) {
        buffer_.reserve(kBufferElements);
    }

    void Push(T value) {
        buffer_.push_back(value);
        if (buffer_.size() == kBufferElements) {
            Flush();
        }
    }

    void Flush() {
        if (buffer_.empty()) {
            return;
        }
        const auto bytes = static_cast<std::uint64_t>(buffer_.size() * sizeof(T));
        if (offset_ + bytes > end_) {
            throw std::logic_error("LOD page column overflow");
        }
        output_->WriteAt(offset_, buffer_.data(), bytes);
        offset_ += bytes;
        buffer_.clear();
    }

private:
    static constexpr std::size_t kBufferElements = 1u << 16u;
    LodPageOutput* output_;
    std::uint64_t offset_;
    std::uint64_t end_;
    std::vector<T> buffer_;
};

template <typename T>
void WriteColumn(LodPageOutput& output, const SectionEntry& section, const std::vector<T>& values) {
    output.WriteAt(section.offset, values.data(), values.size() * sizeof(T));
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

} // namespace

LodPageOutput::LodPageOutput(const std::filesystem::path& path)
    : path_(path), stream_(path, std::ios::binary | std::ios::out | std::ios::trunc) {
    if (!stream_) {
        throw std::runtime_error("could not create " + path.string());
    }
}

void LodPageOutput::WriteAt(std::uint64_t offset, const void* data, std::uint64_t bytes) {
    stream_.seekp(static_cast<std::streamoff>(offset));
    stream_.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream_) {
        throw std::runtime_error("write failed: " + path_.string() + " (disk full?)");
    }
}

void LodPageOutput::Close() {
    stream_.close();
    if (stream_.fail()) {
        throw std::runtime_error("could not finish writing " + path_.string());
    }
}

LodPageHierarchy ReadLodPageSource(const std::filesystem::path& graphPath,
                                   const std::filesystem::path& rangesPath, FileHeader& header,
                                   LodPageOutput& output, std::uint64_t& nextSectionOffset,
                                   const LodPageBuilder::ProgressCallback& progress) {
    SourceTokens graph(graphPath);
    const auto nodeCount = ParseNumber<std::uint64_t>(graph, graph.Next(), "node count");
    const auto edgeCount = ParseNumber<std::uint64_t>(graph, graph.Next(), "edge count");
    if (nodeCount >= kNoEdge || edgeCount >= kNoEdge) {
        throw std::runtime_error("graph is too large for 32-bit node/edge ids");
    }
    const auto needed = LodPageBuilder::EstimateMemoryBytes(nodeCount, edgeCount);
    const auto available = runtime::QuerySystemMemoryInfo().availablePhysicalBytes;
    if (available != 0u && needed > available) {
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        std::ostringstream message;
        message << std::fixed << std::setprecision(1)
                << "building LOD pages for this graph needs about "
                << static_cast<double>(needed) / kGiB << " GiB of free RAM, only "
                << static_cast<double>(available) / kGiB
                << " GiB available (the external-memory build is not implemented yet)";
        throw std::runtime_error(message.str());
    }

    header.nodeCount = nodeCount;
    header.edgeCount = edgeCount;
    PlaceCHSections(header, nextSectionOffset);

    LodPageHierarchy hierarchy;
    hierarchy.nodeCount = nodeCount;
    hierarchy.edgeCount = edgeCount;
    const auto n = static_cast<std::size_t>(nodeCount);
    const auto m = static_cast<std::size_t>(edgeCount);

    // Latitude goes to nodeY and longitude to nodeX; both become projected fixed point below.
    hierarchy.nodeX.resize(n);
    hierarchy.nodeY.resize(n);
    {
        ColumnWriter<std::uint64_t> osmIds(output, header.At(Section::NodeOsmId));
        ColumnWriter<std::int32_t> latitudes(output, header.At(Section::NodeLatitude));
        ColumnWriter<std::int32_t> longitudes(output, header.At(Section::NodeLongitude));
        ColumnWriter<float> elevations(output, header.At(Section::NodeElevation));
        ColumnWriter<std::uint32_t> levels(output, header.At(Section::NodeLevel));
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
            hierarchy.nodeY[i] = latitude;
            hierarchy.nodeX[i] = longitude;
            elevations.Push(ParseNumber<float>(graph, graph.Next(), "elevation"));
            levels.Push(ParseNumber<std::uint32_t>(graph, graph.Next(), "node level"));
            Report(progress, BuildStage::ReadNodes, i + 1u, nodeCount);
        }
        osmIds.Flush();
        latitudes.Flush();
        longitudes.Flush();
        elevations.Flush();
        levels.Flush();
    }

    hierarchy.source.resize(m);
    hierarchy.target.resize(m);
    hierarchy.childA.resize(m);
    hierarchy.childB.resize(m);
    hierarchy.type.resize(m);
    {
        ColumnWriter<std::uint32_t> weights(output, header.At(Section::EdgeWeight));
        ColumnWriter<std::int16_t> maxSpeeds(output, header.At(Section::EdgeMaxSpeed));
        for (std::size_t e = 0; e < m; ++e) {
            const auto source = ParseNumber<std::uint32_t>(graph, graph.Next(), "edge source");
            const auto target = ParseNumber<std::uint32_t>(graph, graph.Next(), "edge target");
            if (source >= nodeCount || target >= nodeCount) {
                throw std::runtime_error(graph.Name() + ": edge endpoint out of range");
            }
            hierarchy.source[e] = source;
            hierarchy.target[e] = target;
            weights.Push(ParseWeight(graph, graph.Next()));
            hierarchy.type[e] = ParseRanged<std::int16_t>(graph, graph.Next(), "edge type");
            maxSpeeds.Push(ParseRanged<std::int16_t>(graph, graph.Next(), "max speed"));
            for (auto* child : {&hierarchy.childA, &hierarchy.childB}) {
                const auto value = ParseNumber<std::int64_t>(graph, graph.Next(), "child edge");
                if (value >= static_cast<std::int64_t>(edgeCount)) {
                    throw std::runtime_error(graph.Name() + ": child edge out of range");
                }
                (*child)[e] = value < 0 ? kNoEdge : static_cast<std::uint32_t>(value);
            }
            if ((hierarchy.childA[e] == kNoEdge) != (hierarchy.childB[e] == kNoEdge)) {
                throw std::runtime_error(graph.Name() + ": edge " + std::to_string(e) +
                                         " has only one child");
            }
            Report(progress, BuildStage::ReadEdges, e + 1u, edgeCount);
        }
        weights.Flush();
        maxSpeeds.Flush();
    }
    WriteColumn(output, header.At(Section::EdgeSource), hierarchy.source);
    WriteColumn(output, header.At(Section::EdgeTarget), hierarchy.target);
    WriteColumn(output, header.At(Section::EdgeType), hierarchy.type);
    WriteColumn(output, header.At(Section::EdgeChildA), hierarchy.childA);
    WriteColumn(output, header.At(Section::EdgeChildB), hierarchy.childB);

    hierarchy.birth.assign(m, -1);
    hierarchy.death.assign(m, -1);
    {
        SourceTokens ranges(rangesPath);
        for (std::size_t i = 0; i < m; ++i) {
            const auto edge = ParseNumber<std::uint32_t>(ranges, ranges.Next(), "range edge id");
            if (edge >= edgeCount) {
                throw std::runtime_error(ranges.Name() + ": range edge id out of range");
            }
            hierarchy.birth[edge] = ParseRanged<std::int16_t>(ranges, ranges.Next(), "birth");
            hierarchy.death[edge] = ParseRanged<std::int16_t>(ranges, ranges.Next(), "death");
            Report(progress, BuildStage::ReadRanges, i + 1u, edgeCount);
        }
    }
    WriteColumn(output, header.At(Section::EdgeBirth), hierarchy.birth);
    WriteColumn(output, header.At(Section::EdgeDeath), hierarchy.death);
    for (const auto birth : hierarchy.birth) {
        hierarchy.maxLevel = std::max<std::int32_t>(hierarchy.maxLevel, birth);
    }
    if (hierarchy.maxLevel > kMaxStoredLevel) {
        throw std::runtime_error("CH level above 65534 is not supported");
    }

    if (n == 0u) {
        throw std::runtime_error(graph.Name() + ": graph has no nodes");
    }
    auto minLatitude = std::numeric_limits<std::int32_t>::max();
    auto maxLatitude = std::numeric_limits<std::int32_t>::min();
    auto minLongitude = minLatitude;
    auto maxLongitude = maxLatitude;
    for (std::size_t i = 0; i < n; ++i) {
        minLatitude = std::min(minLatitude, hierarchy.nodeY[i]);
        maxLatitude = std::max(maxLatitude, hierarchy.nodeY[i]);
        minLongitude = std::min(minLongitude, hierarchy.nodeX[i]);
        maxLongitude = std::max(maxLongitude, hierarchy.nodeX[i]);
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
        const auto point = projection.Project(hierarchy.nodeY[i] / kDegreeScale,
                                              hierarchy.nodeX[i] / kDegreeScale);
        const auto x = std::round(point.x * kFixedCoordinateScale);
        const auto y = std::round(point.y * kFixedCoordinateScale);
        if (std::abs(x) > kLimit || std::abs(y) > kLimit) {
            throw std::runtime_error("projected coordinate outside the fixed-point range");
        }
        hierarchy.nodeX[i] = static_cast<std::int32_t>(x);
        hierarchy.nodeY[i] = static_cast<std::int32_t>(y);
    }
    return hierarchy;
}

} // namespace chmv::streaming::lod
