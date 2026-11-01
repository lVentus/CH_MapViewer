#include "streaming/lod/LodPageBuilder.h"

#include "data/ch/CHTypes.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPageHierarchy.h"
#include "streaming/lod/LodPageIo.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/lod/LodPageSource.h"
#include "streaming/runtime/GraphPageStreamer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chmv::streaming::lod {
namespace {

// Layout reference only: pages are grouped by how a 4K screen would first need them.
constexpr std::uint32_t kReferenceFramebufferHeight = 2160;
constexpr int kQualityLayers = 5; // 1, 2, 4, 8 px on the reference curve, then everything else
constexpr int kCatchAllLayer = kQualityLayers - 1;
constexpr int kMaxOctave = 31;
constexpr double kFarZoom = 1048576.0;
constexpr std::uint64_t kPageWriteChunkBytes = 16ull << 20u;
constexpr std::uint8_t kNoRecord = 0xffu;
// Sort partitions are cut along (layer, octave, top Morton bits) cells.
constexpr std::uint32_t kCellMortonBits = 12;
constexpr std::uint64_t kProgressMask = (1u << 20u) - 1u;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
// The page pass rereads the edges once per sort partition. The memory estimate leaves room for
// about kTargetPagePasses of them; far more than that means the budget is too small to finish.
constexpr std::uint32_t kTargetPagePasses = 8;
constexpr std::uint32_t kMaxPagePasses = 64;
// Disk still needed once the records are known: record ids, pages and directory.
constexpr std::uint64_t kDiskBytesPerRecord = 40;
// Hash map entry plus vector allocation for an edge with further contexts.
constexpr std::uint64_t kExtraContextBytes = 128;

template <typename T> void Release(std::vector<T>& values) {
    std::vector<T>().swap(values);
}

[[nodiscard]] std::uint64_t Mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

[[nodiscard]] StreamHash& HashOf(SectionHashes& hashes, Section section) {
    return hashes[static_cast<std::size_t>(section)];
}

struct LevelInterval {
    std::int32_t low = 1;
    std::int32_t high = 0;

    [[nodiscard]] bool Empty() const { return low > high; }
    [[nodiscard]] bool Overlaps(const LevelInterval& other) const {
        return !Empty() && !other.Empty() && low <= other.high && other.low <= high;
    }
};

[[nodiscard]] double DecodeGeometryError(std::uint16_t code) {
    if (code == 0u) {
        return 0.0;
    }
    if (code == kErrorCodeInfinite) {
        return std::numeric_limits<double>::infinity();
    }
    return std::exp2(static_cast<double>(code) / kErrorCodeStepsPerOctave - kErrorCodeBiasOctaves);
}

// ---------------------------------------------------------------------------------------------
// Layout. The default zoom curve at a 4K reference screen decides in which quality layer and
// zoom octave a record is first needed; this only groups records that are used together.

class ReferenceCurve {
public:
    ReferenceCurve(double metersPerUnit, std::int32_t maxLevel) {
        renderer::MapCamera2D camera;
        camera.SetMetersPerNormalizedUnit(metersPerUnit);
        camera.Reset();
        overviewZoom_ = camera.ZoomFactor();
        pixelFactor_ = 2.0 * static_cast<double>(camera.ZoomFactor()) /
                       (static_cast<double>(camera.ViewScale()) * kReferenceFramebufferHeight);
        lod_.SetLevelRange(0, maxLevel);
        lod_.SetOverviewZoom(camera.ZoomFactor());
        crossings_.resize(static_cast<std::size_t>(maxLevel) + 2u);
        for (std::int32_t k = -1; k <= maxLevel; ++k) {
            crossings_[static_cast<std::size_t>(k + 1)] = ZoomBelow(k + 0.5);
        }
    }

    [[nodiscard]] double OverviewZoom() const { return overviewZoom_; }
    // One reference pixel in normalized units at the given zoom.
    [[nodiscard]] double PixelSize(double zoom) const { return pixelFactor_ / zoom; }
    [[nodiscard]] std::int32_t LevelAt(double zoom) const {
        return static_cast<std::int32_t>(std::lround(Level(zoom)));
    }
    // Zoom range (first, last) on which the rounded level lies in [low, high].
    [[nodiscard]] std::pair<double, double> ZoomRange(std::int32_t low, std::int32_t high) const {
        return {crossings_[static_cast<std::size_t>(high + 1)],
                crossings_[static_cast<std::size_t>(low)]};
    }

private:
    [[nodiscard]] double Level(double zoom) const {
        return lod_.ContinuousLevel(static_cast<float>(zoom));
    }

    // Smallest zoom at which the continuous level drops below `level`.
    [[nodiscard]] double ZoomBelow(double level) const {
        if (Level(overviewZoom_) < level) {
            return 0.0;
        }
        if (Level(kFarZoom) >= level) {
            return std::numeric_limits<double>::infinity();
        }
        auto low = std::log2(overviewZoom_);
        auto high = std::log2(kFarZoom);
        for (int i = 0; i < 64; ++i) {
            const auto middle = 0.5 * (low + high);
            if (Level(std::exp2(middle)) < level) {
                high = middle;
            } else {
                low = middle;
            }
        }
        return std::exp2(high);
    }

    renderer::LODController lod_;
    double overviewZoom_ = 1.0;
    double pixelFactor_ = 1.0;
    std::vector<double> crossings_;
};

struct Placement {
    int layer = kCatchAllLayer;
    int octave = 0;
};

[[nodiscard]] Placement Place(const ReferenceCurve& curve, LevelInterval levels, double error,
                              double parentError) {
    constexpr auto kInfinity = std::numeric_limits<double>::infinity();
    const auto [first, last] = curve.ZoomRange(levels.low, levels.high);
    const auto octaveOf = [&](double zoom) {
        const auto relative = std::max(zoom, curve.OverviewZoom()) / curve.OverviewZoom();
        return std::clamp(static_cast<int>(std::floor(std::log2(relative))), 0, kMaxOctave);
    };
    if (!(first < last)) {
        return {kCatchAllLayer, 0};
    }
    // Needed at zoom z with tolerance q px iff error <= q * px(z) < parentError.
    const auto pixel = curve.PixelSize(1.0);
    const auto qLow = error * first / pixel;
    const auto qHigh =
        std::isinf(parentError) || std::isinf(last) ? kInfinity : parentError * last / pixel;
    if (!(qLow < qHigh)) {
        return {kCatchAllLayer, octaveOf(first)};
    }
    int layer = kCatchAllLayer;
    if (qLow <= 1.0 && qHigh > 1.0) {
        layer = 0;
    } else if (qLow > 1.0) {
        layer = std::min(kCatchAllLayer, static_cast<int>(std::ceil(std::log2(qLow))));
    }
    auto firstZoom = first;
    if (layer < kCatchAllLayer && !std::isinf(parentError)) {
        firstZoom = std::max(first, std::exp2(layer) * pixel / parentError);
    }
    return {layer, octaveOf(firstZoom)};
}

[[nodiscard]] std::uint32_t Part1By1(std::uint32_t x) {
    x &= 0xffffu;
    x = (x | (x << 8u)) & 0x00ff00ffu;
    x = (x | (x << 4u)) & 0x0f0f0f0fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

// ---------------------------------------------------------------------------------------------
// Records. One per DFS visit context: an edge's own alive interval together with its first
// context below a parent, plus one record for every further context.

struct Record {
    std::uint64_t key = 0; // layer << 37 | octave << 32 | Morton
    std::uint32_t edge = 0;
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    std::uint32_t rootLevels = kEmptyLevelRange;
    std::uint32_t childLevels = kEmptyLevelRange;
    std::uint32_t errors = 0;
    std::uint8_t style = 0;

    [[nodiscard]] std::uint8_t LayerOctave() const { return static_cast<std::uint8_t>(key >> 32u); }
    [[nodiscard]] int Layer() const { return static_cast<int>(key >> 37u); }
    [[nodiscard]] int Octave() const { return static_cast<int>((key >> 32u) & 31u); }
};

[[nodiscard]] std::uint32_t PackLevels(LevelInterval levels) {
    return levels.Empty() ? kEmptyLevelRange
                          : PackLevelRange(static_cast<std::uint16_t>(levels.low),
                                           static_cast<std::uint16_t>(levels.high));
}

[[nodiscard]] std::uint64_t CellOf(std::uint8_t layerOctave, std::uint32_t morton) {
    return (std::uint64_t{layerOctave} << kCellMortonBits) | (morton >> (32u - kCellMortonBits));
}

[[nodiscard]] std::uint64_t CellOf(const Record& record) {
    return CellOf(record.LayerOctave(), static_cast<std::uint32_t>(record.key));
}

// What a record draws (edge, style, both end positions), summed over all records so the order
// of the records does not matter.
[[nodiscard]] std::uint64_t GeometryHash(std::uint32_t edge, std::uint8_t style, std::int32_t x0,
                                         std::int32_t y0, std::int32_t x1, std::int32_t y1) {
    const auto pack = [](std::int32_t x, std::int32_t y) {
        return (std::uint64_t{static_cast<std::uint32_t>(x)} << 32u) |
               static_cast<std::uint32_t>(y);
    };
    auto h = Mix64((std::uint64_t{edge} << 8u) | style);
    h = Mix64(h ^ pack(x0, y0));
    return Mix64(h ^ pack(x1, y1));
}

[[nodiscard]] std::uint64_t GeometryHash(const LodPageNodes& nodes, const Record& record) {
    return GeometryHash(record.edge, record.style, nodes.x[record.source], nodes.y[record.source],
                        nodes.x[record.target], nodes.y[record.target]);
}

// One CH edge as the record passes read it, in edge order.
struct EdgeRow {
    std::uint32_t edge = 0;
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    std::int16_t type = 0;
    std::int16_t birth = -1;
    std::int16_t death = -1;
    float error = 0.0f;
};

class EdgeRows {
public:
    EdgeRows(const std::filesystem::path& file, const FileHeader& header)
        : sources_(file, header.At(Section::EdgeSource)),
          targets_(file, header.At(Section::EdgeTarget)),
          types_(file, header.At(Section::EdgeType)), births_(file, header.At(Section::EdgeBirth)),
          deaths_(file, header.At(Section::EdgeDeath)),
          errors_(file, header.At(Section::EdgeError)) {}

    [[nodiscard]] EdgeRow Next(std::uint32_t edge) {
        EdgeRow row;
        row.edge = edge;
        row.source = sources_.Next();
        row.target = targets_.Next();
        row.type = types_.Next();
        row.birth = births_.Next();
        row.death = deaths_.Next();
        row.error = errors_.Next();
        return row;
    }

private:
    ValueReader<std::uint32_t> sources_;
    ValueReader<std::uint32_t> targets_;
    ValueReader<std::int16_t> types_;
    ValueReader<std::int16_t> births_;
    ValueReader<std::int16_t> deaths_;
    ValueReader<float> errors_;
};

[[nodiscard]] LevelInterval RootInterval(const EdgeRow& row) {
    return row.birth >= 0 && row.death >= 0 && row.death <= row.birth
               ? LevelInterval{row.death, row.birth}
               : LevelInterval{};
}

class RecordMaker {
public:
    RecordMaker(const LodPageNodes& nodes, const ContextTable& contexts,
                const ReferenceCurve& curve)
        : nodes_(nodes), contexts_(contexts), curve_(curve) {
        minX_ = *std::min_element(nodes.x.begin(), nodes.x.end());
        minY_ = *std::min_element(nodes.y.begin(), nodes.y.end());
        const auto maxX = *std::max_element(nodes.x.begin(), nodes.x.end());
        const auto maxY = *std::max_element(nodes.y.begin(), nodes.y.end());
        scaleX_ = 65535.0 / std::max(1.0, static_cast<double>(maxX) - minX_);
        scaleY_ = 65535.0 / std::max(1.0, static_cast<double>(maxY) - minY_);
    }

    [[nodiscard]] std::uint32_t Morton(std::uint32_t source, std::uint32_t target) const {
        const auto cx = 0.5 * (static_cast<double>(nodes_.x[source]) + nodes_.x[target]);
        const auto cy = 0.5 * (static_cast<double>(nodes_.y[source]) + nodes_.y[target]);
        const auto mx =
            static_cast<std::uint32_t>(std::clamp((cx - minX_) * scaleX_, 0.0, 65535.0));
        const auto my =
            static_cast<std::uint32_t>(std::clamp((cy - minY_) * scaleY_, 0.0, 65535.0));
        return Part1By1(mx) | (Part1By1(my) << 1u);
    }

    // Main record of the edge (false if it has none). With `extras`, also appends a record for
    // every further context. Keys are placed only when `place` is set; otherwise the caller
    // supplies them.
    bool Make(const EdgeRow& row, bool place, Record& main, std::vector<Record>* extras) const {
        const auto errorCode = EncodeGeometryError(row.error);
        const auto root = RootInterval(row);
        LevelInterval child;
        std::uint16_t childCode = kErrorCodeInfinite;
        LevelContext moved;
        bool hasMoved = false;
        if (contexts_.HasPrimary(row.edge)) {
            child = {contexts_.low[row.edge], contexts_.high[row.edge]};
            childCode = contexts_.parentCode[row.edge];
            // Alive as a root and reached below a parent at the same level: DFS visits it
            // twice, so the second visit becomes its own record.
            if (child.Overlaps(root)) {
                moved = {contexts_.low[row.edge], contexts_.high[row.edge], childCode};
                hasMoved = true;
                child = {};
            }
        }
        if (!child.Empty() && errorCode >= childCode) {
            child = {}; // empty error interval: never the cut, not even after rounding
        }

        const bool hasMain = !root.Empty() || !child.Empty();
        if (hasMain) {
            Fill(row, root, child, errorCode, child.Empty() ? kErrorCodeInfinite : childCode, main);
            if (place) {
                main.key = Key(row, root, child, childCode);
            }
        }
        if (extras != nullptr) {
            const auto addExtra = [&](const LevelContext& context) {
                if (errorCode >= context.parentCode) {
                    return;
                }
                const LevelInterval levels{context.low, context.high};
                Record record;
                Fill(row, {}, levels, errorCode, context.parentCode, record);
                record.key = Key(row, {}, levels, context.parentCode);
                extras->push_back(record);
            };
            if (hasMoved) {
                addExtra(moved);
            }
            if (const auto it = contexts_.extra.find(row.edge); it != contexts_.extra.end()) {
                for (const auto& context : it->second) {
                    addExtra(context);
                }
            }
        }
        return hasMain;
    }

private:
    void Fill(const EdgeRow& row, LevelInterval root, LevelInterval child, std::uint16_t errorCode,
              std::uint16_t parentCode, Record& record) const {
        record.edge = row.edge;
        record.source = row.source;
        record.target = row.target;
        record.rootLevels = PackLevels(root);
        record.childLevels = PackLevels(child);
        record.errors =
            static_cast<std::uint32_t>(errorCode) | (static_cast<std::uint32_t>(parentCode) << 16u);
        record.style = static_cast<std::uint8_t>(data::RoadStyleIndexFromType(row.type));
    }

    [[nodiscard]] std::uint64_t Key(const EdgeRow& row, LevelInterval root, LevelInterval child,
                                    std::uint16_t childCode) const {
        Placement placement;
        if (!root.Empty()) {
            placement = Place(curve_, root, row.error, std::numeric_limits<double>::infinity());
        }
        if (!child.Empty()) {
            const auto below = Place(curve_, child, row.error, DecodeGeometryError(childCode));
            if (root.Empty() || below.layer < placement.layer ||
                (below.layer == placement.layer && below.octave < placement.octave)) {
                placement = below;
            }
        }
        return (static_cast<std::uint64_t>(placement.layer) << 37u) |
               (static_cast<std::uint64_t>(placement.octave) << 32u) |
               Morton(row.source, row.target);
    }

    const LodPageNodes& nodes_;
    const ContextTable& contexts_;
    const ReferenceCurve& curve_;
    double minX_ = 0.0;
    double minY_ = 0.0;
    double scaleX_ = 1.0;
    double scaleY_ = 1.0;
};

// ---------------------------------------------------------------------------------------------
// Page packing. A page never mixes layers or octaves and holds at most 256 records.

[[nodiscard]] LevelBands EmptyBands() {
    LevelBands bands;
    bands.minError.fill(0xffu);
    bands.maxParent.fill(0u);
    return bands;
}

void AddToBands(LevelBands& bands, std::uint32_t minLevel, std::uint32_t maxLevel,
                std::uint32_t levels, std::uint32_t errorCode, std::uint32_t parentCode) {
    if (levels == kEmptyLevelRange) {
        return;
    }
    const auto first = LevelBand(levels & 0xffffu, minLevel, maxLevel);
    const auto last = LevelBand(levels >> 16u, minLevel, maxLevel);
    for (auto band = first; band <= last; ++band) {
        bands.minError[band] = std::min(bands.minError[band], BandCode(errorCode));
        bands.maxParent[band] = std::max(bands.maxParent[band], BandCode(parentCode));
    }
}

// Group bands cover the group's own level range; each level takes the band summary of every
// page that contains it, which keeps the group test conservative.
void MergeGroupBands(PageGroup& group, const PageEntry& page) {
    for (std::uint32_t level = page.minLevel; level <= page.maxLevel; ++level) {
        const auto from = LevelBand(level, page.minLevel, page.maxLevel);
        const auto to = LevelBand(level, group.minLevel, group.maxLevel);
        group.bands.minError[to] = std::min(group.bands.minError[to], page.bands.minError[from]);
        group.bands.maxParent[to] = std::max(group.bands.maxParent[to], page.bands.maxParent[from]);
    }
}

class PageWriter {
public:
    PageWriter(const LodPageNodes& nodes, LodPageOutput& output, const FileHeader& header,
               SectionHashes& hashes)
        : nodes_(nodes), output_(output), pagesOffset_(header.At(Section::Pages).offset),
          pageHash_(HashOf(hashes, Section::Pages)),
          recordEdges_(output, header.At(Section::RecordEdge),
                       HashOf(hashes, Section::RecordEdge)) {
        local_.reserve(kPageVertexCapacity * 2u);
        vertices_.reserve(kPageVertexCapacity);
        open_.reserve(kPageRecordCapacity);
    }

    // Records must arrive in key order across calls. A page left short at the end of one call
    // is filled up by the next, so the layout does not depend on how the records were split.
    void Write(std::span<const Record> records) {
        std::size_t begin = 0;
        while (!open_.empty() && begin < records.size()) {
            if (open_.size() == kPageRecordCapacity ||
                (records[begin].key >> 32u) != (open_.front().key >> 32u)) {
                WritePage(open_);
                open_.clear();
                break;
            }
            open_.push_back(records[begin++]);
        }
        while (begin < records.size()) {
            const auto bucket = records[begin].key >> 32u;
            auto end = begin + 1u;
            while (end < records.size() && end - begin < kPageRecordCapacity &&
                   (records[end].key >> 32u) == bucket) {
                ++end;
            }
            if (end == records.size() && end - begin < kPageRecordCapacity) {
                const auto rest = records.subspan(begin);
                open_.assign(rest.begin(), rest.end());
                break;
            }
            WritePage(records.subspan(begin, end - begin));
            begin = end;
        }
    }

    void Finish() {
        if (!open_.empty()) {
            WritePage(open_);
            open_.clear();
        }
        Flush();
        recordEdges_.Flush();
    }

    [[nodiscard]] std::vector<PageEntry>& Pages() { return pages_; }
    [[nodiscard]] std::uint64_t PageBytes() const { return pageBytes_; }
    [[nodiscard]] std::uint64_t RecordCount() const { return recordCount_; }

private:
    void WritePage(std::span<const Record> records) {
        PageEntry entry;
        entry.fileOffset = pagesOffset_ + pageBytes_ + pending_.size() * 4u;
        entry.firstRecord = static_cast<std::uint32_t>(recordCount_);
        entry.recordCount = static_cast<std::uint16_t>(records.size());
        entry.layer = static_cast<std::uint8_t>(records.front().Layer());
        entry.octave = static_cast<std::uint8_t>(records.front().Octave());

        local_.clear();
        vertices_.clear();
        const auto localIndex = [&](std::uint32_t node) {
            const auto [it, inserted] =
                local_.try_emplace(node, static_cast<std::uint32_t>(vertices_.size()));
            if (inserted) {
                vertices_.push_back(node);
            }
            return it->second;
        };

        const auto recordCount = static_cast<std::uint32_t>(records.size());
        std::array<std::uint32_t, kPageRecordCapacity> endpoints{};
        for (std::uint32_t i = 0; i < recordCount; ++i) {
            const auto from = localIndex(records[i].source);
            const auto to = localIndex(records[i].target);
            endpoints[i] = from | (to << 16u);
        }
        const auto vertexCount = static_cast<std::uint32_t>(vertices_.size());
        entry.vertexCount = static_cast<std::uint16_t>(vertexCount);
        entry.wordCount = PageWordCount(recordCount, vertexCount);

        const auto base = pending_.size();
        pending_.resize(base + entry.wordCount, 0u);
        auto* words = pending_.data() + base;
        words[0] = recordCount | (vertexCount << 16u);
        words[1] = static_cast<std::uint32_t>(pages_.size());
        words[2] = static_cast<std::uint32_t>(entry.layer) |
                   (static_cast<std::uint32_t>(entry.octave) << 8u);

        std::uint32_t minLevel = 0xffffu;
        std::uint32_t maxLevel = 0u;
        std::uint32_t minErrorCode = 0xffffu;
        std::uint32_t maxParentCode = 0u;
        auto* recordWords = words + kPageHeaderWords;
        auto* vertexWords = recordWords + recordCount * kRecordWords;
        auto* typeBytes = reinterpret_cast<std::uint8_t*>(vertexWords + vertexCount * kVertexWords);
        for (std::uint32_t i = 0; i < recordCount; ++i) {
            const auto& record = records[i];
            recordWords[i * 4u + 0u] = endpoints[i];
            recordWords[i * 4u + 1u] = record.rootLevels;
            recordWords[i * 4u + 2u] = record.childLevels;
            recordWords[i * 4u + 3u] = record.errors;
            typeBytes[i] = record.style;
            recordEdges_.Push(record.edge);
            for (const auto levels : {record.rootLevels, record.childLevels}) {
                if (levels != kEmptyLevelRange) {
                    minLevel = std::min(minLevel, levels & 0xffffu);
                    maxLevel = std::max(maxLevel, levels >> 16u);
                }
            }
            minErrorCode = std::min(minErrorCode, record.errors & 0xffffu);
            maxParentCode = std::max(maxParentCode, record.rootLevels != kEmptyLevelRange
                                                        ? std::uint32_t{kErrorCodeInfinite}
                                                        : record.errors >> 16u);
        }

        auto minX = std::numeric_limits<std::int32_t>::max();
        auto minY = minX;
        auto maxX = std::numeric_limits<std::int32_t>::min();
        auto maxY = maxX;
        for (std::uint32_t v = 0; v < vertexCount; ++v) {
            const auto node = vertices_[v];
            vertexWords[v * 2u + 0u] = static_cast<std::uint32_t>(nodes_.x[node]);
            vertexWords[v * 2u + 1u] = static_cast<std::uint32_t>(nodes_.y[node]);
            minX = std::min(minX, nodes_.x[node]);
            minY = std::min(minY, nodes_.y[node]);
            maxX = std::max(maxX, nodes_.x[node]);
            maxY = std::max(maxY, nodes_.y[node]);
        }
        entry.minX = OutwardFloat(minX, -1.0f);
        entry.minY = OutwardFloat(minY, -1.0f);
        entry.maxX = OutwardFloat(maxX, 1.0f);
        entry.maxY = OutwardFloat(maxY, 1.0f);
        entry.minLevel = static_cast<std::uint16_t>(minLevel);
        entry.maxLevel = static_cast<std::uint16_t>(maxLevel);
        entry.minErrorCode = static_cast<std::uint16_t>(minErrorCode);
        entry.maxParentCode = static_cast<std::uint16_t>(maxParentCode);
        entry.bands = EmptyBands();
        for (const auto& record : records) {
            AddToBands(entry.bands, minLevel, maxLevel, record.rootLevels, record.errors & 0xffffu,
                       kErrorCodeInfinite);
            AddToBands(entry.bands, minLevel, maxLevel, record.childLevels, record.errors & 0xffffu,
                       record.errors >> 16u);
        }
        pages_.push_back(entry);
        recordCount_ += recordCount;

        if (pending_.size() * 4u >= kPageWriteChunkBytes) {
            Flush();
        }
    }

    void Flush() {
        if (pending_.empty()) {
            return;
        }
        const auto bytes = pending_.size() * 4u;
        output_.WriteAt(pagesOffset_ + pageBytes_, pending_.data(), bytes);
        pageHash_.Update(pending_.data(), bytes);
        pageBytes_ += bytes;
        pending_.clear();
    }

    [[nodiscard]] static float OutwardFloat(std::int32_t fixed, float direction) {
        const auto value = static_cast<float>(static_cast<double>(fixed) / kFixedCoordinateScale);
        return std::nextafter(value, direction * std::numeric_limits<float>::infinity());
    }

    const LodPageNodes& nodes_;
    LodPageOutput& output_;
    std::uint64_t pagesOffset_;
    StreamHash& pageHash_;
    ColumnWriter<std::uint32_t> recordEdges_;
    std::unordered_map<std::uint32_t, std::uint32_t> local_;
    std::vector<std::uint32_t> vertices_;
    std::vector<std::uint32_t> pending_;
    std::vector<Record> open_;
    std::vector<PageEntry> pages_;
    std::uint64_t pageBytes_ = 0;
    std::uint64_t recordCount_ = 0;
};

[[nodiscard]] std::vector<PageGroup> BuildGroups(std::span<const PageEntry> pages) {
    std::vector<PageGroup> groups;
    for (std::size_t first = 0; first < pages.size(); first += kPagesPerGroup) {
        const auto last = std::min(pages.size(), first + kPagesPerGroup);
        PageGroup group;
        group.minX = std::numeric_limits<float>::infinity();
        group.minY = group.minX;
        group.maxX = -std::numeric_limits<float>::infinity();
        group.maxY = group.maxX;
        group.minLevel = std::uint16_t{0xffff};
        group.minErrorCode = std::uint16_t{0xffff};
        group.firstPage = static_cast<std::uint32_t>(first);
        group.pageCount = static_cast<std::uint32_t>(last - first);
        for (auto p = first; p < last; ++p) {
            const auto& page = pages[p];
            group.minX = std::min(group.minX, page.minX);
            group.minY = std::min(group.minY, page.minY);
            group.maxX = std::max(group.maxX, page.maxX);
            group.maxY = std::max(group.maxY, page.maxY);
            group.minLevel = std::min(group.minLevel, page.minLevel);
            group.maxLevel = std::max(group.maxLevel, page.maxLevel);
            group.minErrorCode = std::min(group.minErrorCode, page.minErrorCode);
            group.maxParentCode = std::max(group.maxParentCode, page.maxParentCode);
        }
        group.bands = EmptyBands();
        for (auto p = first; p < last; ++p) {
            MergeGroupBands(group, pages[p]);
        }
        groups.push_back(group);
    }
    return groups;
}

// What the first record pass learns: for every edge its main record's (layer, octave), the
// rare further records in full, and how records spread over sort cells.
struct RecordPlan {
    std::vector<std::uint8_t> layerOctave;
    std::vector<Record> extras;
    std::vector<std::uint64_t> cells;
    std::uint64_t recordCount = 0;
    std::uint64_t coverGapEdges = 0;
    std::uint64_t geometry = 0; // sum of GeometryHash over all records
};

[[nodiscard]] RecordPlan PlanRecords(const std::filesystem::path& file, FileHeader& header,
                                     const LodPageNodes& nodes, const RecordMaker& maker,
                                     const ContextTable& contexts, LodPageOutput& output,
                                     SectionHashes& hashes,
                                     const LodPageBuilder::ProgressCallback& progress) {
    const auto m = header.edgeCount;
    RecordPlan plan;
    plan.layerOctave.assign(static_cast<std::size_t>(m), kNoRecord);
    plan.cells.assign(std::size_t{256} << kCellMortonBits, 0u);
    StageProgress report(progress, BuildStage::LayoutPages);

    // Hull of the levels an edge is reached at, for the CH-layer cover columns.
    ColumnWriter<std::int16_t> coverLow(output, header.At(Section::EdgeCoverLow),
                                        HashOf(hashes, Section::EdgeCoverLow));
    ColumnWriter<std::int16_t> coverHigh(output, header.At(Section::EdgeCoverHigh),
                                         HashOf(hashes, Section::EdgeCoverHigh));
    std::vector<LevelInterval> cover;
    EdgeRows rows(file, header);
    for (std::uint64_t e = 0; e < m; ++e) {
        const auto edge = static_cast<std::uint32_t>(e);
        const auto row = rows.Next(edge);
        const auto extrasBefore = plan.extras.size();
        Record main;
        if (maker.Make(row, true, main, &plan.extras)) {
            plan.layerOctave[edge] = main.LayerOctave();
            ++plan.cells[CellOf(main)];
            ++plan.recordCount;
            plan.geometry += GeometryHash(nodes, main);
        }
        for (auto i = extrasBefore; i < plan.extras.size(); ++i) {
            plan.geometry += GeometryHash(nodes, plan.extras[i]);
        }

        cover.clear();
        if (const auto root = RootInterval(row); !root.Empty()) {
            cover.push_back(root);
        }
        if (contexts.HasPrimary(edge)) {
            cover.push_back({contexts.low[edge], contexts.high[edge]});
        }
        if (const auto it = contexts.extra.find(edge); it != contexts.extra.end()) {
            for (const auto& context : it->second) {
                cover.push_back({context.low, context.high});
            }
        }
        std::sort(cover.begin(), cover.end(),
                  [](const LevelInterval& a, const LevelInterval& b) { return a.low < b.low; });
        std::size_t pieces = cover.empty() ? 0u : 1u;
        auto low = cover.empty() ? -1 : cover.front().low;
        auto high = cover.empty() ? -1 : cover.front().high;
        for (std::size_t i = 1; i < cover.size(); ++i) {
            if (cover[i].low > high + 1) {
                ++pieces;
            }
            low = std::min(low, cover[i].low);
            high = std::max(high, cover[i].high);
        }
        plan.coverGapEdges += pieces > 1u ? 1u : 0u;
        coverLow.Push(static_cast<std::int16_t>(low));
        coverHigh.Push(static_cast<std::int16_t>(high));
        if ((e & kProgressMask) == 0u) {
            report.Report(0u, 1u, e, m);
        }
    }
    coverLow.Flush();
    coverHigh.Flush();
    for (const auto& record : plan.extras) {
        ++plan.cells[CellOf(record)];
    }
    plan.recordCount += plan.extras.size();
    plan.extras.shrink_to_fit();
    return plan;
}

using Partitions = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

// Ranges of cells whose records fit the sort budget together.
[[nodiscard]] Partitions PlanPartitions(const RecordPlan& plan, std::uint64_t sortBudget) {
    const auto capacity = std::max<std::uint64_t>(1u, sortBudget / sizeof(Record));
    Partitions partitions;
    std::uint64_t begin = 0;
    std::uint64_t filled = 0;
    for (std::uint64_t cell = 0; cell < plan.cells.size(); ++cell) {
        if (filled != 0u && filled + plan.cells[cell] > capacity) {
            partitions.emplace_back(begin, cell);
            begin = cell;
            filled = 0;
        }
        filled += plan.cells[cell];
    }
    partitions.emplace_back(begin, plan.cells.size());
    return partitions;
}

// Sorts and packs the records partition by partition, so the records never all sit in memory
// at once.
void WriteRecordPages(const std::filesystem::path& file, const FileHeader& header,
                      const RecordMaker& maker, const RecordPlan& plan,
                      const Partitions& partitions, PageWriter& writer,
                      const LodPageBuilder::ProgressCallback& progress) {
    const auto m = header.edgeCount;
    StageProgress report(progress, BuildStage::WritePages);
    const auto parts = static_cast<std::uint32_t>(partitions.size());
    std::vector<Record> records;
    for (std::uint32_t part = 0; part < parts; ++part) {
        const auto [first, last] = partitions[part];
        std::uint64_t expected = 0;
        for (auto cell = first; cell < last; ++cell) {
            expected += plan.cells[cell];
        }
        Release(records);
        records.reserve(static_cast<std::size_t>(expected));

        EdgeRows rows(file, header);
        for (std::uint64_t e = 0; e < m; ++e) {
            const auto edge = static_cast<std::uint32_t>(e);
            const auto row = rows.Next(edge);
            const auto layerOctave = plan.layerOctave[edge];
            if (layerOctave != kNoRecord) {
                const auto morton = maker.Morton(row.source, row.target);
                const auto cell = CellOf(layerOctave, morton);
                if (cell >= first && cell < last) {
                    Record record;
                    maker.Make(row, false, record, nullptr);
                    record.key = (std::uint64_t{layerOctave} << 32u) | morton;
                    records.push_back(record);
                }
            }
            if ((e & kProgressMask) == 0u) {
                report.Report(part, parts, e, m);
            }
        }
        for (const auto& record : plan.extras) {
            const auto cell = CellOf(record);
            if (cell >= first && cell < last) {
                records.push_back(record);
            }
        }
        if (records.size() != expected) {
            throw std::logic_error("LOD page partition lost records");
        }
        // Full order, so equal keys of one edge's records land the same way on every machine.
        std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
            return std::tie(a.key, a.edge, a.rootLevels, a.childLevels, a.errors) <
                   std::tie(b.key, b.edge, b.rootLevels, b.childLevels, b.errors);
        });
        writer.Write(records);
    }
    writer.Finish();
}

// ---------------------------------------------------------------------------------------------
// Verification, all streaming: every section reads back to the hash computed while writing,
// and a few reference cuts drawn from the pages match plain DFS over the hierarchy, counted
// with multiplicity.

struct Cut {
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
};

[[nodiscard]] std::uint64_t EdgeHash(std::uint32_t edge) {
    return Mix64(edge);
}

struct CutSum {
    std::uint64_t hash = 0;
    std::uint64_t count = 0;

    void Add(std::uint32_t edge, std::uint64_t times) {
        hash += EdgeHash(edge) * times;
        count += times;
    }
    [[nodiscard]] bool operator==(const CutSum&) const = default;
};

[[nodiscard]] std::vector<CutSum> DfsCuts(const std::filesystem::path& path,
                                          const FileHeader& header, LevelChunks& chunks,
                                          std::span<const Cut> cuts, StageProgress& report) {
    const auto m = static_cast<std::size_t>(header.edgeCount);
    const auto k = cuts.size();
    // Visit counts of all cuts side by side, one byte per edge and cut.
    std::vector<std::uint8_t> visits(m * k, 0u);
    {
        ValueReader<std::int16_t> births(path, header.At(Section::EdgeBirth));
        ValueReader<std::int16_t> deaths(path, header.At(Section::EdgeDeath));
        for (std::size_t e = 0; e < m; ++e) {
            const auto birth = births.Next();
            const auto death = deaths.Next();
            for (std::size_t c = 0; c < k; ++c) {
                const auto level = static_cast<std::int32_t>(cuts[c].level);
                if (birth >= 0 && death >= 0 && death <= level && level <= birth) {
                    visits[e * k + c] = 1u;
                }
            }
        }
    }

    std::vector<CutSum> sums(k);
    const auto visit = [](std::uint8_t& slot, std::uint8_t times) {
        if (slot + times > 255) {
            throw std::runtime_error("LOD page verification: an edge is reached more than 255 "
                                     "times at one level");
        }
        slot = static_cast<std::uint8_t>(slot + times);
    };
    // Parents before children; an expanded shortcut hands its visits down.
    std::uint64_t done = 0;
    chunks.ForEach(false, false, [&](std::span<LevelRecord> records) {
        for (const auto& record : records) {
            const auto code = EncodeGeometryError(record.value);
            auto* counts = visits.data() + std::size_t{record.edge} * k;
            for (std::size_t c = 0; c < k; ++c) {
                const auto times = counts[c];
                if (times == 0u) {
                    continue;
                }
                counts[c] = 0u;
                if (code > cuts[c].threshold) {
                    visit(visits[std::size_t{record.childA} * k + c], times);
                    visit(visits[std::size_t{record.childB} * k + c], times);
                } else {
                    sums[c].Add(record.edge, times);
                }
            }
        }
        done += records.size();
        report.Report(1u, 3u, done, chunks.Count());
    });
    // Whatever is still counted now is an original edge that was reached.
    for (std::size_t e = 0; e < m; ++e) {
        for (std::size_t c = 0; c < k; ++c) {
            if (const auto times = visits[e * k + c]; times != 0u) {
                sums[c].Add(static_cast<std::uint32_t>(e), times);
            }
        }
    }
    return sums;
}

struct PageSums {
    std::vector<CutSum> cuts;
    std::uint64_t geometry = 0;
};

[[nodiscard]] PageSums PageCuts(const LodPageFile& file, std::span<const Cut> cuts,
                                StageProgress& report) {
    PageSums sums;
    sums.cuts.resize(cuts.size());
    std::vector<LodPageQuery> everywhere(cuts.size());
    for (std::size_t k = 0; k < cuts.size(); ++k) {
        everywhere[k].minX = everywhere[k].minY = -std::numeric_limits<float>::max();
        everywhere[k].maxX = everywhere[k].maxY = std::numeric_limits<float>::max();
        everywhere[k].level = cuts[k].level;
        everywhere[k].threshold = cuts[k].threshold;
    }

    const auto& header = file.Header();
    ValueReader<std::uint32_t> recordEdges(file.Path(), header.At(Section::RecordEdge));
    std::ifstream input(file.Path(), std::ios::binary);
    std::vector<std::uint32_t> words(kPageSlotWords);
    std::vector<bool> drawn(cuts.size());
    const auto pages = file.Pages();
    std::uint64_t done = 0;
    for (const auto& group : file.Groups()) {
        for (auto p = group.firstPage; p < group.firstPage + group.pageCount; ++p) {
            const auto& page = pages[p];
            input.seekg(static_cast<std::streamoff>(page.fileOffset));
            input.read(reinterpret_cast<char*>(words.data()),
                       static_cast<std::streamsize>(page.wordCount * 4u));
            const auto counts =
                std::uint32_t{page.recordCount} | (std::uint32_t{page.vertexCount} << 16u);
            if (!input || words[0] != counts) {
                throw std::runtime_error("LOD page verification: could not read page " +
                                         std::to_string(p));
            }
            const auto* vertices =
                words.data() + kPageHeaderWords + page.recordCount * kRecordWords;
            const auto* types =
                reinterpret_cast<const std::uint8_t*>(vertices + page.vertexCount * kVertexWords);
            const auto position = [&](std::uint32_t local, std::uint32_t axis) {
                return static_cast<std::int32_t>(vertices[local * kVertexWords + axis]);
            };
            std::fill(drawn.begin(), drawn.end(), false);
            for (std::uint32_t i = 0; i < page.recordCount; ++i) {
                const auto edge = recordEdges.Next();
                const auto* record = words.data() + kPageHeaderWords + i * kRecordWords;
                const auto from = record[0] & 0xffffu;
                const auto to = record[0] >> 16u;
                if (from >= page.vertexCount || to >= page.vertexCount) {
                    throw std::runtime_error("LOD page verification failed: page " +
                                             std::to_string(p) + " points past its vertices");
                }
                sums.geometry +=
                    GeometryHash(edge, types[i], position(from, 0u), position(from, 1u),
                                 position(to, 0u), position(to, 1u));
                for (std::size_t k = 0; k < cuts.size(); ++k) {
                    if (RecordDrawn(record[1], record[2], record[3], cuts[k].level,
                                    cuts[k].threshold)) {
                        sums.cuts[k].Add(edge, 1u);
                        drawn[k] = true;
                    }
                }
            }
            for (std::size_t k = 0; k < cuts.size(); ++k) {
                if (drawn[k] && (!MayDraw(page, everywhere[k]) || !MayDraw(group, everywhere[k]))) {
                    throw std::runtime_error("LOD page verification failed: metadata of page " +
                                             std::to_string(p) + " hides a drawn record");
                }
            }
        }
        done += group.pageCount;
        report.Report(2u, 3u, done, pages.size());
    }
    return sums;
}

std::uint32_t Verify(const std::filesystem::path& path, const ReferenceCurve& curve,
                     LevelChunks& chunks, std::uint64_t geometry,
                     const LodPageBuilder::ProgressCallback& progress) {
    StageProgress report(progress, BuildStage::Verify);
    const auto file = LodPageFile::Open(path);
    const auto& header = file->Header();
    std::uint64_t total = 0;
    for (const auto& section : header.sections) {
        total += section.bytes;
    }
    std::uint64_t read = 0;
    const auto onRead = [&](std::uint64_t bytes) {
        read += bytes;
        report.Report(0u, 3u, read, total);
    };
    for (std::size_t s = 0; s < kSectionCount; ++s) {
        const auto& section = header.sections[s];
        if (HashFileRange(path, section.offset, section.bytes, onRead) != header.sectionHashes[s]) {
            throw std::runtime_error("LOD page verification failed: section " + std::to_string(s) +
                                     " does not read back as written");
        }
    }

    // Overview, regional and street-level views at 1 px, plus a coarse tolerance.
    std::vector<Cut> cuts;
    const auto overview = curve.OverviewZoom();
    for (const auto& [zoomOctaves, pixels] : std::array<std::pair<double, double>, 4>{
             {{0.0, 1.0}, {4.0, 1.0}, {4.0, 8.0}, {12.0, 1.0}}}) {
        const auto zoom = overview * std::exp2(zoomOctaves);
        cuts.push_back({static_cast<std::uint32_t>(std::max(0, curve.LevelAt(zoom))),
                        EncodeErrorTolerance(pixels * curve.PixelSize(zoom))});
    }
    const auto expected = DfsCuts(path, header, chunks, cuts, report);
    const auto actual = PageCuts(*file, cuts, report);
    if (actual.geometry != geometry) {
        throw std::runtime_error("LOD page verification failed: page vertices or styles differ "
                                 "from the records they were packed from");
    }
    for (std::size_t k = 0; k < cuts.size(); ++k) {
        if (!(expected[k] == actual.cuts[k])) {
            throw std::runtime_error("LOD page verification failed at level " +
                                     std::to_string(cuts[k].level) + ": DFS cut has " +
                                     std::to_string(expected[k].count) + " edges, pages give " +
                                     std::to_string(actual.cuts[k].count));
        }
    }
    return static_cast<std::uint32_t>(cuts.size());
}

[[nodiscard]] std::string Gigabytes(std::uint64_t bytes) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / kGiB << " GiB";
    return text.str();
}

[[nodiscard]] std::filesystem::path DirectoryOf(const std::filesystem::path& file) {
    auto directory = file.parent_path();
    return directory.empty() ? std::filesystem::path(".") : directory;
}

[[nodiscard]] std::uint64_t FreeDiskSpace(const std::filesystem::path& file) {
    return std::filesystem::space(DirectoryOf(file)).available;
}

[[noreturn]] void ThrowNotEnoughMemory(std::uint64_t needed, std::uint64_t budget) {
    throw std::runtime_error("building LOD pages for this graph needs about " + Gigabytes(needed) +
                             " of free RAM, " + Gigabytes(budget) + " can be used right now");
}

[[noreturn]] void ThrowNotEnoughDisk(std::uint64_t needed, std::uint64_t available) {
    throw std::runtime_error("building LOD pages for this graph needs about " + Gigabytes(needed) +
                             " of free disk space next to it, " + Gigabytes(available) +
                             " are free");
}

// Scratch files of a build that was killed (<output>.tmp.levels, <output>.tmp.split<k>); they
// would otherwise count against the disk check and stay forever.
void RemoveStaleScratch(const std::filesystem::path& temporaryPath) {
    const auto prefix = temporaryPath.filename().native();
    std::vector<std::filesystem::path> stale;
    std::error_code error;
    for (std::filesystem::directory_iterator it(DirectoryOf(temporaryPath), error), end;
         !error && it != end; it.increment(error)) {
        const auto name = it->path().filename().native();
        if (name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0 &&
            name[prefix.size()] == '.') {
            stale.push_back(it->path());
        }
    }
    for (const auto& path : stale) {
        std::filesystem::remove(path, error);
    }
}

} // namespace

std::uint64_t LodPageBuilder::EstimateMemoryBytes(std::uint64_t nodeCount,
                                                  std::uint64_t edgeCount) {
    // The fullest stage holds node positions and levels (10 B per node), the level contexts
    // and each edge's record cell (7 B per edge) and a sort buffer that packs the pages in
    // about kTargetPagePasses passes (about one record per edge).
    return nodeCount * 10u + edgeCount * (7u + sizeof(Record) / kTargetPagePasses) +
           (512ull << 20u);
}

std::uint64_t LodPageBuilder::EstimateDiskBytes(std::uint64_t nodeCount, std::uint64_t edgeCount) {
    // CH layer and derived columns (24 B per node, 36 B per edge), records and pages (about
    // 30 B per edge), the shortcut level scratch file (20 B per shortcut, about half the edges)
    // and some slack.
    return nodeCount * 24u + edgeCount * 80u;
}

BuildResult LodPageBuilder::Build(const std::filesystem::path& graphPath,
                                  const std::filesystem::path& rangesPath,
                                  const std::filesystem::path& outputPath,
                                  const ProgressCallback& progress,
                                  std::uint64_t memoryBudgetBytes) {
    const auto begin = std::chrono::steady_clock::now();
    auto temporaryPath = outputPath;
    temporaryPath += ".tmp";
    RemoveStaleScratch(temporaryPath);
    TemporaryFile temporary(temporaryPath);

    const auto memory = runtime::QuerySystemMemoryInfo();
    const auto budget = memoryBudgetBytes != 0u ? memoryBudgetBytes
                        : memory.availablePhysicalBytes != 0u
                            ? memory.availablePhysicalBytes / 4u * 3u
                            : 4ull << 30u;

    FileHeader header;
    header.magic = kFileMagic;
    header.version = kFormatVersion;
    header.headerBytes = static_cast<std::uint32_t>(kHeaderBytes);
    header.pageSlotWords = kPageSlotWords;
    header.referenceFramebufferHeight = kReferenceFramebufferHeight;
    const auto graphIdentity = QuerySourceIdentity(graphPath);
    const auto rangesIdentity = QuerySourceIdentity(rangesPath);
    header.sourceGraphBytes = graphIdentity.bytes;
    header.sourceGraphWriteTime = graphIdentity.writeTime;
    header.sourceRangesBytes = rangesIdentity.bytes;
    header.sourceRangesWriteTime = rangesIdentity.writeTime;

    const auto checkResources = [&](std::uint64_t nodes, std::uint64_t edges) {
        if (const auto needed = EstimateMemoryBytes(nodes, edges); needed > budget) {
            ThrowNotEnoughMemory(needed, budget);
        }
        const auto disk = EstimateDiskBytes(nodes, edges);
        if (const auto available = FreeDiskSpace(outputPath); available < disk) {
            ThrowNotEnoughDisk(disk, available);
        }
    };

    SectionHashes hashes{};
    LodPageOutput output(temporary.Path());
    std::uint64_t offset = 0;
    const auto nodes = ReadLodPageSource(graphPath, rangesPath, header, output, hashes, offset,
                                         checkResources, progress);
    output.Flush();
    header.maxLevel = nodes.maxRangeLevel;
    const auto m = header.edgeCount;

    // Shortcuts by level, geometry errors upwards, level contexts downwards.
    auto levelsPath = temporary.Path();
    levelsPath += ".levels";
    LevelChunks chunks(levelsPath, nodes.maxNodeLevel + 1u);
    const auto shortcuts = BuildLevelChunks(temporary.Path(), header, nodes, budget,
                                            temporary.Path(), chunks, progress);
    {
        auto errors = ComputeErrors(chunks, shortcuts, m, progress);
        header.maxHierarchyHeight = errors.maxHeight;
        output.WriteAt(header.At(Section::EdgeError).offset, errors.error.data(),
                       errors.error.size() * sizeof(float));
        HashOf(hashes, Section::EdgeError)
            .Update(errors.error.data(), errors.error.size() * sizeof(float));
    }
    // Records read the CH columns back while the page section is appended.
    output.Flush();

    const ReferenceCurve curve(header.MetersPerNormalizedUnit(), nodes.maxRangeLevel);
    BuildResult result;
    std::uint64_t geometry = 0;
    {
        const auto contexts = ComputeContexts(chunks, m, progress);
        const RecordMaker maker(nodes, contexts, curve);
        auto plan =
            PlanRecords(temporary.Path(), header, nodes, maker, contexts, output, hashes, progress);
        if (plan.recordCount > 0xffffffffull) {
            throw std::runtime_error("too many render records for 32-bit record ids");
        }
        geometry = plan.geometry;
        if (const auto available = FreeDiskSpace(outputPath);
            available < plan.recordCount * kDiskBytesPerRecord) {
            ThrowNotEnoughDisk(plan.recordCount * kDiskBytesPerRecord, available);
        }
        header.recordCount = plan.recordCount;
        header.coverGapEdges = plan.coverGapEdges;
        header.extraContextRecords = plan.extras.size();
        header.At(Section::RecordEdge) = {offset, header.recordCount * 4u};
        header.At(Section::Pages) = {AlignSection(offset + header.recordCount * 4u), 0u};

        const auto fixed = nodes.Bytes() + m * 7u + plan.extras.size() * sizeof(Record) +
                           contexts.extra.size() * kExtraContextBytes +
                           header.recordCount / kPageRecordCapacity * sizeof(PageEntry) * 2u +
                           (256ull << 20u);
        const auto sortBudget =
            std::max<std::uint64_t>(budget > fixed ? budget - fixed : 0u, 64ull << 20u);
        const auto partitions = PlanPartitions(plan, sortBudget);
        if (partitions.size() > kMaxPagePasses) {
            ThrowNotEnoughMemory(fixed + header.recordCount * sizeof(Record) / kTargetPagePasses,
                                 budget);
        }
        result.pagePasses = static_cast<std::uint32_t>(partitions.size());
        PageWriter writer(nodes, output, header, hashes);
        WriteRecordPages(temporary.Path(), header, maker, plan, partitions, writer, progress);
        if (writer.RecordCount() != header.recordCount) {
            throw std::logic_error("LOD page record count changed between passes");
        }

        auto& pages = writer.Pages();
        const auto groups = BuildGroups(pages);
        header.pageCount = pages.size();
        header.groupCount = groups.size();
        header.At(Section::Pages).bytes = writer.PageBytes();
        offset = AlignSection(header.At(Section::Pages).offset + writer.PageBytes());
        header.At(Section::PageDirectory) = {offset, header.pageCount * sizeof(PageEntry)};
        output.WriteAt(offset, pages.data(), header.pageCount * sizeof(PageEntry));
        HashOf(hashes, Section::PageDirectory)
            .Update(pages.data(), header.pageCount * sizeof(PageEntry));
        offset = AlignSection(offset + header.pageCount * sizeof(PageEntry));
        header.At(Section::PageGroups) = {offset, header.groupCount * sizeof(PageGroup)};
        output.WriteAt(offset, groups.data(), header.groupCount * sizeof(PageGroup));
        HashOf(hashes, Section::PageGroups)
            .Update(groups.data(), header.groupCount * sizeof(PageGroup));
    }

    for (std::size_t s = 0; s < kSectionCount; ++s) {
        header.sectionHashes[s] = hashes[s].Finish();
    }
    std::vector<std::byte> headerBytes(kHeaderBytes);
    std::memcpy(headerBytes.data(), &header, sizeof(FileHeader));
    output.WriteAt(0, headerBytes.data(), headerBytes.size());
    output.Close();

    result.verifiedCuts = Verify(temporary.Path(), curve, chunks, geometry, progress);
    temporary.MoveTo(outputPath);

    result.path = outputPath;
    result.nodeCount = header.nodeCount;
    result.edgeCount = header.edgeCount;
    result.recordCount = header.recordCount;
    result.pageCount = header.pageCount;
    result.coverGapEdges = header.coverGapEdges;
    result.extraContextRecords = header.extraContextRecords;
    result.memoryBudgetBytes = budget;
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return result;
}

} // namespace chmv::streaming::lod
