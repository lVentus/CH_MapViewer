#include "streaming/lod/LodPageSource.h"

#include "data/ch/CHProjection.h"
#include "data/ch/CHTypes.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace chmv::streaming::lod {
namespace {

constexpr std::uint32_t kNoEdge = 0xffffffffu;

[[nodiscard]] bool IsSpace(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
}

// Calls visit(token, offset) for the tokens of text[begin, end) until it returns false. A '#'
// at the start of a token comments out the rest of the line.
template <typename Visit>
void ScanTokens(const char* text, std::size_t begin, std::size_t end, Visit&& visit) {
    auto i = begin;
    while (i < end) {
        if (IsSpace(text[i])) {
            ++i;
            continue;
        }
        if (text[i] == '#') {
            while (i < end && text[i] != '\n') {
                ++i;
            }
            continue;
        }
        const auto start = i;
        while (i < end && !IsSpace(text[i])) {
            ++i;
        }
        if (!visit(std::string_view(text + start, i - start), start)) {
            return;
        }
    }
}

// Whitespace-separated tokens with '#' line comments, same token rules as CHLoader, so both
// readers see identical values. The text is read in large blocks cut at line ends; worker
// threads first count the tokens of their piece of a block and then parse them knowing their
// position, so records may be laid out across lines in any way.
class RecordParser {
public:
    explicit RecordParser(const std::filesystem::path& path)
        : path_(path), input_(path, std::ios::binary), buffer_(2u * kBlockBytes) {
        if (!input_) {
            throw std::runtime_error("could not open " + path.string());
        }
    }

    RecordParser(const RecordParser&) = delete;
    RecordParser& operator=(const RecordParser&) = delete;

    // Converted to narrow text only when a message needs it.
    [[nodiscard]] std::string Name() const { return path_.filename().string(); }

    // Parses the next `count` records of `fields` tokens each, block by block: begin(first,
    // records) on the calling thread, field(record, index, token) on worker threads for every
    // token of those records, then end(first, records) on the calling thread.
    template <typename Begin, typename Field, typename End>
    void Parse(std::uint64_t count, std::uint32_t fields, Begin&& begin, Field&& field, End&& end) {
        const auto needed = count * fields;
        std::uint64_t done = 0;
        while (done < needed) {
            Fill();
            const auto regionEnd = LinesEnd();
            CountTokens(regionEnd);
            std::uint64_t total = 0;
            for (auto& piece : pieces_) {
                piece.first = total;
                total += piece.tokens;
            }
            const auto take = total >= needed - done ? needed - done : total / fields * fields;
            if (take == 0u) {
                if (eof_) {
                    throw std::runtime_error(Name() + ": unexpected end of file");
                }
                FinishReadAhead();
                if (!eof_ && start_ == 0u && size_ == buffer_.size()) {
                    // A single token or comment longer than half the buffer.
                    buffer_.resize(buffer_.size() * 2u);
                }
                continue;
            }

            auto cut = regionEnd;
            if (take < total) {
                const auto last = std::find_if(pieces_.begin(), pieces_.end(), [&](const Piece& p) {
                    return take < p.first + p.tokens;
                });
                cut = TokenStart(*last, take - last->first);
                last->end = cut;
                last->tokens = take - last->first;
                pieces_.erase(last + 1, pieces_.end());
            }

            const auto firstRecord = done / fields;
            begin(firstRecord, take / fields);
            const auto* text = buffer_.data();
            RunParallel(pieces_.size(), [&](std::size_t task, unsigned) {
                const auto& piece = pieces_[task];
                auto record = (done + piece.first) / fields;
                auto index = static_cast<std::uint32_t>((done + piece.first) % fields);
                auto left = piece.tokens;
                ScanTokens(text, piece.begin, piece.end, [&](std::string_view token, std::size_t) {
                    field(record, index, token);
                    if (++index == fields) {
                        index = 0;
                        ++record;
                    }
                    return --left != 0u;
                });
            });
            end(firstRecord, take / fields);
            done += take;
            start_ = cut;
        }
    }

private:
    static constexpr std::size_t kBlockBytes = 32u << 20u;

    struct Piece {
        std::size_t begin = 0;
        std::size_t end = 0;
        std::uint64_t tokens = 0;
        std::uint64_t first = 0;
    };

    // Moves the unparsed rest to the front, tops the first half of the buffer up from the file
    // and starts reading the next block behind it while this one is parsed.
    void Fill() {
        FinishReadAhead();
        if (start_ != 0u) {
            std::memmove(buffer_.data(), buffer_.data() + start_, size_ - start_);
            size_ -= start_;
            start_ = 0;
        }
        const auto block = buffer_.size() / 2u;
        while (!eof_ && size_ < block) {
            Received(Read(size_, block - size_), block - size_);
        }
        if (!eof_) {
            readAheadBytes_ = buffer_.size() - size_;
            readAhead_ =
                std::async(std::launch::async,
                           [this, at = size_, bytes = readAheadBytes_] { return Read(at, bytes); });
        }
    }

    void FinishReadAhead() {
        if (readAhead_.valid()) {
            Received(readAhead_.get(), readAheadBytes_);
        }
    }

    // Only touches the stream and buffer_[at, at + bytes); may run on another thread.
    [[nodiscard]] std::size_t Read(std::size_t at, std::size_t bytes) {
        input_.read(buffer_.data() + at, static_cast<std::streamsize>(bytes));
        if (input_.bad()) {
            throw std::runtime_error("read failed: " + Name());
        }
        return static_cast<std::size_t>(input_.gcount());
    }

    void Received(std::size_t got, std::size_t asked) {
        size_ += got;
        eof_ = got < asked;
    }

    // End of the last complete line in the buffer (all of it at the end of the file). Without
    // any line end, the start of the last token or comment, which may go on in the next block.
    [[nodiscard]] std::size_t LinesEnd() const {
        if (eof_) {
            return size_;
        }
        auto end = size_;
        while (end > start_ && buffer_[end - 1u] != '\n') {
            --end;
        }
        if (end != start_) {
            return end;
        }
        auto open = start_;
        auto i = start_;
        while (i < size_) {
            if (IsSpace(buffer_[i])) {
                open = ++i;
                continue;
            }
            const auto comment = buffer_[i] == '#';
            while (i < size_ && (comment ? buffer_[i] != '\n' : !IsSpace(buffer_[i]))) {
                ++i;
            }
        }
        return open;
    }

    // Splits [start_, end) into pieces at line ends and counts their tokens in parallel.
    void CountTokens(std::size_t end) {
        pieces_.clear();
        const auto length = end - start_;
        const auto count =
            std::max<std::size_t>(1u, std::min<std::size_t>(WorkerCount() * 2u, length / 4096u));
        auto from = start_;
        for (std::size_t i = 1; i <= count && from < end; ++i) {
            auto to = i == count ? end : start_ + length * i / count;
            while (to < end && buffer_[to - 1u] != '\n') {
                ++to;
            }
            if (to > from) {
                pieces_.push_back({from, to, 0u, 0u});
                from = to;
            }
        }
        const auto* text = buffer_.data();
        RunParallel(pieces_.size(), [&](std::size_t task, unsigned) {
            auto& piece = pieces_[task];
            ScanTokens(text, piece.begin, piece.end, [&](std::string_view, std::size_t) {
                ++piece.tokens;
                return true;
            });
        });
    }

    [[nodiscard]] std::size_t TokenStart(const Piece& piece, std::uint64_t index) const {
        auto at = piece.end;
        ScanTokens(buffer_.data(), piece.begin, piece.end,
                   [&](std::string_view, std::size_t offset) {
                       if (index-- == 0u) {
                           at = offset;
                           return false;
                       }
                       return true;
                   });
        return at;
    }

    std::filesystem::path path_;
    std::ifstream input_;
    std::vector<char> buffer_;
    std::size_t start_ = 0; // [start_, size_) is read but not parsed yet
    std::size_t size_ = 0;
    bool eof_ = false;
    std::vector<Piece> pieces_;
    std::size_t readAheadBytes_ = 0;
    // Last, so a read still running finishes before the buffer goes away.
    std::future<std::size_t> readAhead_;
};

[[noreturn]] void BadToken(const RecordParser& tokens, std::string_view token, const char* what) {
    throw std::runtime_error(tokens.Name() + ": invalid " + what + " '" + std::string(token) + "'");
}

template <typename T>
[[nodiscard]] T ParseNumber(const RecordParser& tokens, std::string_view token, const char* what) {
    T value{};
    const auto* end = token.data() + token.size();
    const auto [ptr, ec] = std::from_chars(token.data(), end, value);
    if (ec != std::errc{} || ptr != end) {
        BadToken(tokens, token, what);
    }
    return value;
}

template <typename T>
[[nodiscard]] T ParseRanged(const RecordParser& tokens, std::string_view token, const char* what) {
    const auto value = ParseNumber<std::int64_t>(tokens, token, what);
    if (value < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
        value > static_cast<std::int64_t>(std::numeric_limits<T>::max())) {
        BadToken(tokens, token, what);
    }
    return static_cast<T>(value);
}

// Decimal degrees in any notation (EUR is written from floats: 50.98522568, -4.89999984e-05),
// as the nearest double. With at most 15 significant digits that double gives the source
// decimal back exactly; longer coordinates are rejected rather than rounded.
[[nodiscard]] double ParseDegrees(const RecordParser& tokens, std::string_view token) {
    const auto value = ParseNumber<double>(tokens, token, "coordinate");
    if (!(std::abs(value) <= 180.0)) {
        BadToken(tokens, token, "coordinate");
    }
    // From the first non-zero digit to the last one, before any exponent.
    int digits = 0;
    int zeros = 0;
    for (const auto ch : token) {
        if (ch == 'e' || ch == 'E') {
            break;
        }
        if (ch == '0') {
            zeros += digits != 0 ? 1 : 0;
        } else if (ch >= '1' && ch <= '9') {
            digits += zeros + 1;
            zeros = 0;
        }
    }
    if (digits > std::numeric_limits<double>::digits10) {
        throw std::runtime_error(tokens.Name() + ": coordinate '" + std::string(token) +
                                 "' has more than 15 significant digits and cannot be stored "
                                 "exactly");
    }
    return value;
}

[[nodiscard]] std::uint32_t ParseWeight(const RecordParser& tokens, std::string_view token) {
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

void PlaceCHSections(FileHeader& header, std::uint64_t& offset) {
    const auto n = header.nodeCount;
    const auto m = header.edgeCount;
    const std::array<std::pair<Section, std::uint64_t>, 17> sizes{{
        {Section::NodeOsmId, n * 8u},
        {Section::NodeLatitude, n * 8u},
        {Section::NodeLongitude, n * 8u},
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
    // Freed before the ranges file is read.
    std::optional<RecordParser> graphFile(std::in_place, graphPath);
    auto& graph = *graphFile;
    std::uint64_t nodeCount = 0;
    std::uint64_t edgeCount = 0;
    graph.Parse(
        1u, 2u, [](std::uint64_t, std::uint64_t) {},
        [&](std::uint64_t, std::uint32_t field, std::string_view token) {
            if (field == 0u) {
                nodeCount = ParseNumber<std::uint64_t>(graph, token, "node count");
            } else {
                edgeCount = ParseNumber<std::uint64_t>(graph, token, "edge count");
            }
        },
        [](std::uint64_t, std::uint64_t) {});
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
    const auto report = [&](BuildStage stage, std::uint64_t current, std::uint64_t total) {
        if (progress) {
            progress({stage, current, total});
        }
    };

    // Kept until the projection below turns them into nodes.x/y.
    std::vector<double> latitude(n);
    std::vector<double> longitude(n);
    nodes.level.resize(n);
    {
        ColumnWriter<std::uint64_t> osmIds(output, header.At(Section::NodeOsmId),
                                           HashOf(hashes, Section::NodeOsmId));
        ColumnWriter<double> latitudes(output, header.At(Section::NodeLatitude),
                                       HashOf(hashes, Section::NodeLatitude));
        ColumnWriter<double> longitudes(output, header.At(Section::NodeLongitude),
                                        HashOf(hashes, Section::NodeLongitude));
        ColumnWriter<float> elevations(output, header.At(Section::NodeElevation),
                                       HashOf(hashes, Section::NodeElevation));
        ColumnWriter<std::uint32_t> levels(output, header.At(Section::NodeLevel),
                                           HashOf(hashes, Section::NodeLevel));
        std::vector<std::uint64_t> osm;
        std::vector<float> elevation;
        std::vector<std::uint32_t> level;
        std::uint64_t base = 0;
        graph.Parse(
            nodeCount, 6u,
            [&](std::uint64_t first, std::uint64_t records) {
                base = first;
                osm.resize(static_cast<std::size_t>(records));
                elevation.resize(static_cast<std::size_t>(records));
                level.resize(static_cast<std::size_t>(records));
            },
            [&](std::uint64_t record, std::uint32_t field, std::string_view token) {
                const auto i = static_cast<std::size_t>(record);
                const auto local = static_cast<std::size_t>(record - base);
                switch (field) {
                case 0u:
                    if (ParseNumber<std::uint64_t>(graph, token, "node id") != record) {
                        throw std::runtime_error(graph.Name() +
                                                 ": node ids must be listed as 0..n-1 in order");
                    }
                    break;
                case 1u:
                    osm[local] = ParseNumber<std::uint64_t>(graph, token, "OSM id");
                    break;
                case 2u:
                    latitude[i] = ParseDegrees(graph, token);
                    break;
                case 3u:
                    longitude[i] = ParseDegrees(graph, token);
                    break;
                case 4u:
                    elevation[local] = ParseNumber<float>(graph, token, "elevation");
                    break;
                default:
                    level[local] = ParseNumber<std::uint32_t>(graph, token, "node level");
                    if (level[local] > 0xffffu) {
                        throw std::runtime_error(graph.Name() + ": node level above 65535");
                    }
                    nodes.level[i] = static_cast<std::uint16_t>(level[local]);
                    break;
                }
            },
            [&](std::uint64_t first, std::uint64_t records) {
                const auto from = static_cast<std::size_t>(first);
                const auto count = static_cast<std::size_t>(records);
                RunTogether({
                    [&] { osmIds.Append(osm); },
                    [&] {
                        latitudes.Append(std::span<const double>(latitude).subspan(from, count));
                    },
                    [&] {
                        longitudes.Append(std::span<const double>(longitude).subspan(from, count));
                    },
                    [&] { elevations.Append(elevation); },
                    [&] { levels.Append(level); },
                });
                nodes.maxNodeLevel =
                    std::max(nodes.maxNodeLevel, *std::max_element(level.begin(), level.end()));
                report(BuildStage::ReadNodes, first + records, nodeCount);
            });
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
        std::vector<std::uint32_t> source;
        std::vector<std::uint32_t> target;
        std::vector<std::uint32_t> weight;
        std::vector<std::int16_t> type;
        std::vector<std::int16_t> maxSpeed;
        std::vector<std::uint32_t> childA;
        std::vector<std::uint32_t> childB;
        std::uint64_t base = 0;
        const auto parseEndpoint = [&](std::string_view token, const char* what) {
            const auto node = ParseNumber<std::uint32_t>(graph, token, what);
            if (node >= nodeCount) {
                throw std::runtime_error(graph.Name() + ": edge endpoint out of range");
            }
            return node;
        };
        const auto parseChild = [&](std::string_view token) {
            const auto value = ParseNumber<std::int64_t>(graph, token, "child edge");
            if (value >= static_cast<std::int64_t>(edgeCount)) {
                throw std::runtime_error(graph.Name() + ": child edge out of range");
            }
            return value < 0 ? kNoEdge : static_cast<std::uint32_t>(value);
        };
        graph.Parse(
            edgeCount, 7u,
            [&](std::uint64_t first, std::uint64_t records) {
                base = first;
                const auto count = static_cast<std::size_t>(records);
                for (auto* column : {&source, &target, &weight, &childA, &childB}) {
                    column->resize(count);
                }
                type.resize(count);
                maxSpeed.resize(count);
            },
            [&](std::uint64_t record, std::uint32_t field, std::string_view token) {
                const auto local = static_cast<std::size_t>(record - base);
                switch (field) {
                case 0u:
                    source[local] = parseEndpoint(token, "edge source");
                    break;
                case 1u:
                    target[local] = parseEndpoint(token, "edge target");
                    break;
                case 2u:
                    weight[local] = ParseWeight(graph, token);
                    break;
                case 3u:
                    type[local] = ParseRanged<std::int16_t>(graph, token, "edge type");
                    break;
                case 4u:
                    maxSpeed[local] = ParseRanged<std::int16_t>(graph, token, "max speed");
                    break;
                case 5u:
                    childA[local] = parseChild(token);
                    break;
                default:
                    childB[local] = parseChild(token);
                    break;
                }
            },
            [&](std::uint64_t first, std::uint64_t records) {
                for (std::size_t i = 0; i < childA.size(); ++i) {
                    if ((childA[i] == kNoEdge) != (childB[i] == kNoEdge)) {
                        throw std::runtime_error(graph.Name() + ": edge " +
                                                 std::to_string(first + i) + " has only one child");
                    }
                }
                RunTogether({
                    [&] { sources.Append(source); },
                    [&] { targets.Append(target); },
                    [&] { weights.Append(weight); },
                    [&] { types.Append(type); },
                    [&] { maxSpeeds.Append(maxSpeed); },
                    [&] { childrenA.Append(childA); },
                    [&] { childrenB.Append(childB); },
                });
                report(BuildStage::ReadEdges, first + records, edgeCount);
            });
        sources.Flush();
        targets.Flush();
        weights.Flush();
        types.Flush();
        maxSpeeds.Flush();
        childrenA.Flush();
        childrenB.Flush();
    }
    graphFile.reset();

    {
        // The ranges file may list edges in any order, so both columns are assembled in memory;
        // entries are applied in file order, a later one for the same edge wins.
        std::vector<std::int16_t> birth(m, -1);
        std::vector<std::int16_t> death(m, -1);
        RecordParser ranges(rangesPath);
        std::vector<std::uint32_t> edge;
        std::vector<std::int16_t> born;
        std::vector<std::int16_t> died;
        std::uint64_t base = 0;
        ranges.Parse(
            edgeCount, 3u,
            [&](std::uint64_t first, std::uint64_t records) {
                base = first;
                edge.resize(static_cast<std::size_t>(records));
                born.resize(static_cast<std::size_t>(records));
                died.resize(static_cast<std::size_t>(records));
            },
            [&](std::uint64_t record, std::uint32_t field, std::string_view token) {
                const auto local = static_cast<std::size_t>(record - base);
                if (field == 0u) {
                    edge[local] = ParseNumber<std::uint32_t>(ranges, token, "range edge id");
                    if (edge[local] >= edgeCount) {
                        throw std::runtime_error(ranges.Name() + ": range edge id out of range");
                    }
                } else if (field == 1u) {
                    born[local] = ParseRanged<std::int16_t>(ranges, token, "birth");
                } else {
                    died[local] = ParseRanged<std::int16_t>(ranges, token, "death");
                }
            },
            [&](std::uint64_t first, std::uint64_t records) {
                for (std::size_t i = 0; i < edge.size(); ++i) {
                    birth[edge[i]] = born[i];
                    death[edge[i]] = died[i];
                }
                report(BuildStage::ReadRanges, first + records, edgeCount);
            });
        WriteSection(output, header.At(Section::EdgeBirth), HashOf(hashes, Section::EdgeBirth),
                     birth);
        WriteSection(output, header.At(Section::EdgeDeath), HashOf(hashes, Section::EdgeDeath),
                     death);
        for (const auto value : birth) {
            nodes.maxRangeLevel = std::max<std::int32_t>(nodes.maxRangeLevel, value);
        }
    }

    struct Bounds {
        double minLatitude = std::numeric_limits<double>::max();
        double maxLatitude = std::numeric_limits<double>::lowest();
        double minLongitude = std::numeric_limits<double>::max();
        double maxLongitude = std::numeric_limits<double>::lowest();
    };
    const auto pieces = SplitRange(0u, nodeCount, 1u << 16u);
    std::vector<Bounds> pieceBounds(pieces.size());
    RunParallel(pieces.size(), [&](std::size_t task, unsigned) {
        auto& bounds = pieceBounds[task];
        for (auto i = pieces[task].first; i < pieces[task].second; ++i) {
            bounds.minLatitude = std::min(bounds.minLatitude, latitude[i]);
            bounds.maxLatitude = std::max(bounds.maxLatitude, latitude[i]);
            bounds.minLongitude = std::min(bounds.minLongitude, longitude[i]);
            bounds.maxLongitude = std::max(bounds.maxLongitude, longitude[i]);
        }
    });
    Bounds bounds;
    for (const auto& piece : pieceBounds) {
        bounds.minLatitude = std::min(bounds.minLatitude, piece.minLatitude);
        bounds.maxLatitude = std::max(bounds.maxLatitude, piece.maxLatitude);
        bounds.minLongitude = std::min(bounds.minLongitude, piece.minLongitude);
        bounds.maxLongitude = std::max(bounds.maxLongitude, piece.maxLongitude);
    }

    // Same projection as the full-resident loader: ComputeCHProjection only looks at bounds.
    std::array<data::CHNode, 2> corners{};
    corners[0].latitude = bounds.minLatitude;
    corners[0].longitude = bounds.minLongitude;
    corners[1].latitude = bounds.maxLatitude;
    corners[1].longitude = bounds.maxLongitude;
    const auto projection = data::ComputeCHProjection(corners);
    header.centerLatitude = projection.centerLatitude;
    header.centerLongitude = projection.centerLongitude;
    header.longitudeScale = projection.longitudeScale;
    header.normalization = projection.normalization;
    header.minLatitude = corners[0].latitude;
    header.maxLatitude = corners[1].latitude;
    header.minLongitude = corners[0].longitude;
    header.maxLongitude = corners[1].longitude;

    nodes.x.resize(n);
    nodes.y.resize(n);
    RunParallel(pieces.size(), [&](std::size_t task, unsigned) {
        constexpr double kLimit = 1073741823.0;
        for (auto i = pieces[task].first; i < pieces[task].second; ++i) {
            const auto point = projection.Project(latitude[i], longitude[i]);
            const auto x = std::round(point.x * kFixedCoordinateScale);
            const auto y = std::round(point.y * kFixedCoordinateScale);
            if (std::abs(x) > kLimit || std::abs(y) > kLimit) {
                throw std::runtime_error("projected coordinate outside the fixed-point range");
            }
            nodes.x[i] = static_cast<std::int32_t>(x);
            nodes.y[i] = static_cast<std::int32_t>(y);
        }
    });
    if (nodes.maxRangeLevel > kMaxStoredLevel) {
        throw std::runtime_error("CH level above 65534 is not supported");
    }
    return nodes;
}

} // namespace chmv::streaming::lod
