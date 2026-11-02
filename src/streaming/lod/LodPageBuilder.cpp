#include "streaming/lod/LodPageBuilder.h"

#include "data/SystemMemory.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "streaming/lod/LodPageCompression.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPageHierarchy.h"
#include "streaming/lod/LodPageIo.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/lod/LodPageSource.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
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
constexpr std::uint8_t kNoRecord = 0xffu;
// Sort partitions are cut along (layer, octave, top Morton bits) cells.
constexpr std::uint32_t kCellMortonBits = 12;
constexpr std::uint64_t kProgressMask = (1u << 20u) - 1u;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
// The page pass rereads the edges once per sort partition. The memory estimate leaves room for
// about kTargetPagePasses of them; far more than that means the budget is too small to finish.
constexpr std::uint32_t kTargetPagePasses = 8;
constexpr std::uint32_t kMaxPagePasses = 64;
// Disk still needed once the records are known: record ids, pages, directories, the children
// section and the representative scratch file (52 B), then the compressed copy of the whole
// file (about 28 B).
constexpr std::uint64_t kDiskBytesPerRecord = 80;
// Hash map entry plus vector allocation for an edge with further contexts.
constexpr std::uint64_t kExtraContextBytes = 128;
// Readers and scan buffers of one worker thread.
constexpr std::uint64_t kWorkerBytes = 8ull << 20u;
// Smallest edge range one thread scans on its own, and edges per round of a scan whose results
// are written in edge order.
constexpr std::uint64_t kMinScanEdges = 1u << 16u;
constexpr std::uint64_t kWaveEdges = 1u << 23u;
// Pages read back and checked per round of the verification.
constexpr std::uint64_t kVerifyBlockBytes = 64u << 20u;

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
    // The one record of its edge a traversal visits (and whose children it follows).
    bool representative = false;

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
[[nodiscard]] std::uint64_t GeometryHash(std::uint32_t edge, std::uint8_t style, std::int32_t fromX,
                                         std::int32_t fromY, std::int32_t toX, std::int32_t toY) {
    const auto pack = [](std::int32_t x, std::int32_t y) {
        return (std::uint64_t{static_cast<std::uint32_t>(x)} << 32u) |
               static_cast<std::uint32_t>(y);
    };
    auto h = Mix64((std::uint64_t{edge} << 8u) | style);
    h = Mix64(h ^ pack(fromX, fromY));
    return Mix64(h ^ pack(toX, toY));
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
    std::int16_t birth = -1;
    std::int16_t death = -1;
    float error = 0.0f;
};

// The CH columns the record passes read, with one set of readers per worker thread.
class EdgeRows {
public:
    EdgeRows(const std::filesystem::path& file, const FileHeader& header)
        : sources_(file, header.At(Section::EdgeSource)),
          targets_(file, header.At(Section::EdgeTarget)),
          births_(file, header.At(Section::EdgeBirth)),
          deaths_(file, header.At(Section::EdgeDeath)),
          errors_(file, header.At(Section::EdgeError)) {}

    // Edges [range.first, range.second), read in order by one worker.
    class Range {
    public:
        [[nodiscard]] EdgeRow Next(std::uint32_t edge) {
            EdgeRow row;
            row.edge = edge;
            row.source = sources_.Next();
            row.target = targets_.Next();
            row.birth = births_.Next();
            row.death = deaths_.Next();
            row.error = errors_.Next();
            return row;
        }

    private:
        friend class EdgeRows;
        Range(ValueReader<std::uint32_t>& sources, ValueReader<std::uint32_t>& targets,
              ValueReader<std::int16_t>& births, ValueReader<std::int16_t>& deaths,
              ValueReader<float>& errors)
            : sources_(sources), targets_(targets), births_(births), deaths_(deaths),
              errors_(errors) {}

        ValueReader<std::uint32_t>& sources_;
        ValueReader<std::uint32_t>& targets_;
        ValueReader<std::int16_t>& births_;
        ValueReader<std::int16_t>& deaths_;
        ValueReader<float>& errors_;
    };

    [[nodiscard]] Range At(unsigned worker, IndexRange range) {
        return Range(sources_.At(worker, range), targets_.At(worker, range),
                     births_.At(worker, range), deaths_.At(worker, range),
                     errors_.At(worker, range));
    }

private:
    ColumnReaders<std::uint32_t> sources_;
    ColumnReaders<std::uint32_t> targets_;
    ColumnReaders<std::int16_t> births_;
    ColumnReaders<std::int16_t> deaths_;
    ColumnReaders<float> errors_;
};

[[nodiscard]] LevelInterval RootInterval(const EdgeRow& row) {
    return row.birth >= 0 && row.death >= 0 && row.death <= row.birth
               ? LevelInterval{row.death, row.birth}
               : LevelInterval{};
}

class RecordMaker {
public:
    RecordMaker(const LodPageNodes& nodes, const ContextTable& contexts,
                const ReferenceCurve& curve, const EdgeMask& shortcuts,
                std::span<const std::uint8_t> styles)
        : nodes_(nodes), contexts_(contexts), curve_(curve), shortcuts_(shortcuts),
          styles_(styles) {
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
    // every further context, and for a shortcut that has no record at all but can still be
    // walked through, one that is never drawn. Keys are placed only when `place` is set;
    // otherwise the caller supplies them.
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
            main.representative = true;
            if (place) {
                main.key = Key(row, root, child, childCode);
            }
        }
        if (extras != nullptr) {
            // Without a main record the first further record represents the edge.
            bool representative = !hasMain;
            const auto addExtra = [&](const LevelContext& context) {
                if (errorCode >= context.parentCode) {
                    return;
                }
                const LevelInterval levels{context.low, context.high};
                Record record;
                Fill(row, {}, levels, errorCode, context.parentCode, record);
                record.key = Key(row, {}, levels, context.parentCode);
                record.representative = representative;
                representative = false;
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
            // Reached only below parents of the same error code, so always expanded there.
            if (representative && shortcuts_.Test(row.edge) && Expandable(row.edge)) {
                Record record;
                Fill(row, {}, {}, errorCode, errorCode, record);
                record.key = (std::uint64_t{kCatchAllLayer} << 37u) | Morton(row.source, row.target);
                record.representative = true;
                extras->push_back(record);
            }
        }
        return hasMain;
    }

private:
    // Some parent of the edge can be expanded (has a non-zero error code).
    [[nodiscard]] bool Expandable(std::uint32_t edge) const {
        if (contexts_.HasPrimary(edge) && contexts_.parentCode[edge] != 0u) {
            return true;
        }
        if (const auto it = contexts_.extra.find(edge); it != contexts_.extra.end()) {
            for (const auto& context : it->second) {
                if (context.parentCode != 0u) {
                    return true;
                }
            }
        }
        return false;
    }

    void Fill(const EdgeRow& row, LevelInterval root, LevelInterval child, std::uint16_t errorCode,
              std::uint16_t parentCode, Record& record) const {
        record.edge = row.edge;
        record.source = row.source;
        record.target = row.target;
        record.rootLevels = PackLevels(root);
        record.childLevels = PackLevels(child);
        record.errors =
            static_cast<std::uint32_t>(errorCode) | (static_cast<std::uint32_t>(parentCode) << 16u);
        record.style = styles_[row.edge];
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
    const EdgeMask& shortcuts_;
    std::span<const std::uint8_t> styles_;
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

// Where an edge's representative record ended up: page << 8 | record.
struct RepresentativeLocation {
    std::uint32_t edge = 0;
    std::uint32_t location = 0;
};

using ChildMask = std::array<std::uint32_t, kChildMaskWords>;

class PageWriter {
public:
    // Representative locations go to `representatives` in page order.
    PageWriter(const LodPageNodes& nodes, const ContextTable& contexts, const EdgeMask& shortcuts,
               LodPageOutput& output, const FileHeader& header, SectionHashes& hashes,
               const std::filesystem::path& representatives)
        : nodes_(nodes), contexts_(contexts), shortcuts_(shortcuts), output_(output),
          pagesOffset_(header.At(Section::Pages).offset), pageHash_(HashOf(hashes, Section::Pages)),
          recordEdges_(output, header.At(Section::RecordEdge), HashOf(hashes, Section::RecordEdge)),
          representatives_(representatives) {
        open_.reserve(kPageRecordCapacity);
    }

    // Records must arrive in key order across calls. A page left short at the end of one call
    // is filled up by the next, so the layout does not depend on how the records were split.
    void Write(std::span<const Record> records) {
        spans_.clear();
        std::size_t begin = 0;
        while (!open_.empty() && begin < records.size()) {
            if (open_.size() == kPageRecordCapacity ||
                (records[begin].key >> 32u) != (open_.front().key >> 32u)) {
                closed_ = open_;
                open_.clear();
                spans_.emplace_back(closed_);
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
            spans_.push_back(records.subspan(begin, end - begin));
            begin = end;
        }
        Pack();
    }

    void Finish() {
        spans_.clear();
        if (!open_.empty()) {
            closed_ = open_;
            open_.clear();
            spans_.emplace_back(closed_);
        }
        Pack();
        recordEdges_.Flush();
        representatives_.Close();
    }

    [[nodiscard]] std::vector<PageEntry>& Pages() { return pages_; }
    // Visit entries without their children block location yet, and the children masks.
    [[nodiscard]] std::vector<PageEntry>& VisitPages() { return visits_; }
    [[nodiscard]] std::vector<ChildMask>& ChildMasks() { return masks_; }
    [[nodiscard]] std::uint64_t RepresentativeCount() const { return representatives_.Count(); }
    [[nodiscard]] std::uint64_t PageBytes() const { return pageBytes_; }
    [[nodiscard]] std::uint64_t RecordCount() const { return recordCount_; }

private:
    struct LocalSlot {
        std::uint32_t node = 0;
        std::uint32_t index = 0;
        std::uint32_t stamp = 0;
    };
    // At least twice the most vertices a page can have.
    static constexpr std::uint32_t kLocalSlotBits = 10;
    static constexpr std::uint32_t kLocalSlots = 1u << kLocalSlotBits;
    static_assert(kLocalSlots >= 2u * kPageVertexCapacity);
    // Pages built in parallel before they are written out in order.
    static constexpr std::size_t kPagesPerRound = 2048;

    // Page-local vertex index in order of first use: open addressing over the node id, slots of
    // earlier pages are told apart by their stamp.
    struct LocalVertices {
        std::vector<LocalSlot> slots = std::vector<LocalSlot>(kLocalSlots);
        std::uint32_t stamp = 0;
        std::vector<std::uint32_t> nodes;

        void Reset() {
            nodes.clear();
            if (++stamp == 0u) {
                std::fill(slots.begin(), slots.end(), LocalSlot{});
                stamp = 1u;
            }
        }
        [[nodiscard]] std::uint32_t Index(std::uint32_t node) {
            auto slot = (node * 0x9e3779b1u) >> (32u - kLocalSlotBits);
            while (slots[slot].stamp == stamp) {
                if (slots[slot].node == node) {
                    return slots[slot].index;
                }
                slot = (slot + 1u) & (kLocalSlots - 1u);
            }
            const auto index = static_cast<std::uint32_t>(nodes.size());
            slots[slot] = {node, index, stamp};
            nodes.push_back(node);
            return index;
        }
    };

    // Consecutive pages built by one thread; offsets are relative to `words` until written.
    struct PageRun {
        std::vector<std::uint32_t> words;
        std::vector<PageEntry> entries;
        std::vector<std::uint32_t> edges;
        std::vector<PageEntry> visits;
        std::vector<ChildMask> masks;
        std::vector<RepresentativeLocation> representatives;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> reach;
    };

    void Pack() {
        for (std::size_t first = 0; first < spans_.size(); first += kPagesPerRound) {
            const auto last = std::min(spans_.size(), first + kPagesPerRound);
            const auto pieces = SplitRange(first, last, 64u);
            runs_.resize(pieces.size());
            RunParallel(pieces.size(), [&](std::size_t task, unsigned) {
                auto& run = runs_[task];
                run.words.clear();
                run.entries.clear();
                run.edges.clear();
                run.visits.clear();
                run.masks.clear();
                run.representatives.clear();
                LocalVertices local;
                auto firstRecord = recordCount_;
                for (auto i = first; i < pieces[task].first; ++i) {
                    firstRecord += spans_[i].size();
                }
                for (auto i = pieces[task].first; i < pieces[task].second; ++i) {
                    BuildPage(spans_[i], static_cast<std::uint32_t>(pages_.size() + i - first),
                              firstRecord, local, run);
                    firstRecord += spans_[i].size();
                }
            });
            const auto roundOffset = pagesOffset_ + pageBytes_;
            for (auto& run : runs_) {
                for (auto& entry : run.entries) {
                    entry.fileOffset += pagesOffset_ + pageBytes_;
                    pages_.push_back(entry);
                }
                pageBytes_ += run.words.size() * 4u;
                recordCount_ += run.edges.size();
                visits_.insert(visits_.end(), run.visits.begin(), run.visits.end());
                masks_.insert(masks_.end(), run.masks.begin(), run.masks.end());
            }
            // Pages, record ids and representative locations of the round go out in order, each
            // on its own thread.
            RunTogether({
                [&] {
                    auto offset = roundOffset;
                    for (const auto& run : runs_) {
                        const auto bytes = run.words.size() * 4u;
                        output_.WriteAt(offset, run.words.data(), bytes);
                        pageHash_.Update(run.words.data(), bytes);
                        offset += bytes;
                    }
                },
                [&] {
                    for (const auto& run : runs_) {
                        recordEdges_.Append(run.edges);
                    }
                },
                [&] {
                    for (const auto& run : runs_) {
                        representatives_.Append(run.representatives);
                    }
                },
            });
        }
    }

    void BuildPage(std::span<const Record> records, std::uint32_t pageIndex,
                   std::uint64_t firstRecord, LocalVertices& local, PageRun& run) const {
        PageEntry entry;
        entry.fileOffset = run.words.size() * 4u;
        entry.firstRecord = static_cast<std::uint32_t>(firstRecord);
        entry.recordCount = static_cast<std::uint16_t>(records.size());
        entry.layer = static_cast<std::uint8_t>(records.front().Layer());
        entry.octave = static_cast<std::uint8_t>(records.front().Octave());

        local.Reset();
        const auto recordCount = static_cast<std::uint32_t>(records.size());
        std::array<std::uint32_t, kPageRecordCapacity> endpoints{};
        for (std::uint32_t i = 0; i < recordCount; ++i) {
            const auto from = local.Index(records[i].source);
            const auto to = local.Index(records[i].target);
            endpoints[i] = from | (to << 16u);
        }
        const auto vertexCount = static_cast<std::uint32_t>(local.nodes.size());
        entry.vertexCount = static_cast<std::uint16_t>(vertexCount);
        entry.wordCount = PageWordCount(recordCount, vertexCount);

        const auto base = run.words.size();
        run.words.resize(base + entry.wordCount, 0u);
        auto* words = run.words.data() + base;
        words[0] = recordCount | (vertexCount << 16u);
        words[1] = pageIndex;
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
            run.edges.push_back(record.edge);
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
            const auto node = local.nodes[v];
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
        run.entries.push_back(entry);
        AddVisit(records, pageIndex, entry, run);
    }

    // What a traversal needs from the page: where its representative records are visited
    // (under every context of their edges), how far their geometry can reach beyond their chords,
    // and which of them have children.
    void AddVisit(std::span<const Record> records, std::uint32_t pageIndex, const PageEntry& entry,
                  PageRun& run) const {
        PageEntry visit;
        visit.firstRecord = entry.firstRecord;
        visit.recordCount = entry.recordCount;
        visit.layer = entry.layer;
        visit.octave = entry.octave;
        ChildMask mask{};
        auto& reach = run.reach;
        reach.clear();
        double grow = 0.0;
        for (std::uint32_t i = 0; i < records.size(); ++i) {
            const auto& record = records[i];
            if (!record.representative) {
                continue;
            }
            const auto edge = record.edge;
            run.representatives.push_back({edge, (pageIndex << 8u) | i});
            if (shortcuts_.Test(edge)) {
                mask[i >> 5u] |= 1u << (i & 31u);
            }
            grow = std::max(grow, static_cast<double>(ErrorBound(record.errors & 0xffffu)) /
                                      kFixedCoordinateScale);
            if (record.rootLevels != kEmptyLevelRange) {
                reach.emplace_back(record.rootLevels, kErrorCodeInfinite);
            }
            if (contexts_.HasPrimary(edge)) {
                reach.emplace_back(PackLevels({contexts_.low[edge], contexts_.high[edge]}),
                                   contexts_.parentCode[edge]);
            }
            if (const auto it = contexts_.extra.find(edge); it != contexts_.extra.end()) {
                for (const auto& context : it->second) {
                    reach.emplace_back(PackLevels({context.low, context.high}),
                                       context.parentCode);
                }
            }
        }

        std::uint32_t minLevel = 0xffffu;
        std::uint32_t maxLevel = 0u;
        std::uint32_t maxParentCode = 0u;
        for (const auto& [levels, parentCode] : reach) {
            minLevel = std::min(minLevel, levels & 0xffffu);
            maxLevel = std::max(maxLevel, levels >> 16u);
            maxParentCode = std::max(maxParentCode, parentCode);
        }
        visit.minLevel = static_cast<std::uint16_t>(minLevel);
        visit.maxLevel = static_cast<std::uint16_t>(maxLevel);
        visit.minErrorCode = 0;
        visit.maxParentCode = static_cast<std::uint16_t>(maxParentCode);
        visit.bands = EmptyBands();
        for (const auto& [levels, parentCode] : reach) {
            AddToBands(visit.bands, minLevel, maxLevel, levels, 0u, parentCode);
        }
        const auto outward = [](double value, float direction) {
            return std::nextafter(static_cast<float>(value),
                                  direction * std::numeric_limits<float>::infinity());
        };
        visit.minX = outward(static_cast<double>(entry.minX) - grow, -1.0f);
        visit.minY = outward(static_cast<double>(entry.minY) - grow, -1.0f);
        visit.maxX = outward(static_cast<double>(entry.maxX) + grow, 1.0f);
        visit.maxY = outward(static_cast<double>(entry.maxY) + grow, 1.0f);
        run.visits.push_back(visit);
        run.masks.push_back(mask);
    }

    [[nodiscard]] static float OutwardFloat(std::int32_t fixed, float direction) {
        const auto value = static_cast<float>(static_cast<double>(fixed) / kFixedCoordinateScale);
        return std::nextafter(value, direction * std::numeric_limits<float>::infinity());
    }

    const LodPageNodes& nodes_;
    const ContextTable& contexts_;
    const EdgeMask& shortcuts_;
    LodPageOutput& output_;
    std::uint64_t pagesOffset_;
    StreamHash& pageHash_;
    ColumnWriter<std::uint32_t> recordEdges_;
    ScratchWriter<RepresentativeLocation> representatives_;
    std::vector<PageEntry> visits_;
    std::vector<ChildMask> masks_;
    std::vector<Record> open_;
    std::vector<Record> closed_;
    std::vector<std::span<const Record>> spans_;
    std::vector<PageRun> runs_;
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
    std::uint64_t visitOnlyRecords = 0; // among the extras: never drawn, only walked through
    std::uint64_t geometry = 0;         // sum of GeometryHash over all records
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

    // What one scan piece learns; pieces are merged in edge order.
    struct Piece {
        std::vector<Record> extras;
        std::uint64_t recordCount = 0;
        std::uint64_t coverGapEdges = 0;
        std::uint64_t visitOnlyRecords = 0;
        std::uint64_t geometry = 0;
    };

    // Hull of the levels an edge is reached at, for the CH-layer cover columns.
    ColumnWriter<std::int16_t> coverLow(output, header.At(Section::EdgeCoverLow),
                                        HashOf(hashes, Section::EdgeCoverLow));
    ColumnWriter<std::int16_t> coverHigh(output, header.At(Section::EdgeCoverHigh),
                                         HashOf(hashes, Section::EdgeCoverHigh));
    std::vector<std::int16_t> lows;
    std::vector<std::int16_t> highs;
    EdgeRows edgeRows(file, header);
    report.Report(0u, 1u, 0u, m);
    for (std::uint64_t wave = 0; wave < m; wave += kWaveEdges) {
        const auto waveEnd = std::min(m, wave + kWaveEdges);
        lows.resize(static_cast<std::size_t>(waveEnd - wave));
        highs.resize(lows.size());
        const auto scans = SplitRange(wave, waveEnd, kMinScanEdges);
        std::vector<Piece> pieces(scans.size());
        RunParallel(scans.size(), [&](std::size_t task, unsigned worker) {
            auto& piece = pieces[task];
            std::vector<LevelInterval> cover;
            auto rows = edgeRows.At(worker, scans[task]);
            for (auto e = scans[task].first; e < scans[task].second; ++e) {
                const auto edge = static_cast<std::uint32_t>(e);
                const auto row = rows.Next(edge);
                const auto extrasBefore = piece.extras.size();
                Record main;
                if (maker.Make(row, true, main, &piece.extras)) {
                    plan.layerOctave[edge] = main.LayerOctave();
                    std::atomic_ref(plan.cells[CellOf(main)])
                        .fetch_add(1u, std::memory_order_relaxed);
                    ++piece.recordCount;
                    piece.geometry += GeometryHash(nodes, main);
                }
                for (auto i = extrasBefore; i < piece.extras.size(); ++i) {
                    const auto& extra = piece.extras[i];
                    piece.geometry += GeometryHash(nodes, extra);
                    if (extra.rootLevels == kEmptyLevelRange &&
                        extra.childLevels == kEmptyLevelRange) {
                        ++piece.visitOnlyRecords;
                    }
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
                std::sort(
                    cover.begin(), cover.end(),
                    [](const LevelInterval& a, const LevelInterval& b) { return a.low < b.low; });
                std::size_t parts = cover.empty() ? 0u : 1u;
                auto low = cover.empty() ? -1 : cover.front().low;
                auto high = cover.empty() ? -1 : cover.front().high;
                for (std::size_t i = 1; i < cover.size(); ++i) {
                    if (cover[i].low > high + 1) {
                        ++parts;
                    }
                    low = std::min(low, cover[i].low);
                    high = std::max(high, cover[i].high);
                }
                piece.coverGapEdges += parts > 1u ? 1u : 0u;
                lows[static_cast<std::size_t>(e - wave)] = static_cast<std::int16_t>(low);
                highs[static_cast<std::size_t>(e - wave)] = static_cast<std::int16_t>(high);
            }
        });
        RunTogether({[&] { coverLow.Append(lows); }, [&] { coverHigh.Append(highs); }});
        for (auto& piece : pieces) {
            plan.extras.insert(plan.extras.end(), piece.extras.begin(), piece.extras.end());
            plan.recordCount += piece.recordCount;
            plan.coverGapEdges += piece.coverGapEdges;
            plan.visitOnlyRecords += piece.visitOnlyRecords;
            plan.geometry += piece.geometry;
        }
        report.Report(0u, 1u, waveEnd, m);
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
    const auto scans = SplitRange(0u, m, kMinScanEdges);
    EdgeRows edgeRows(file, header);
    std::vector<Record> records;
    std::vector<std::uint64_t> begins;
    std::vector<std::uint64_t> cursors;
    report.Report(0u, parts, 0u, m);
    for (std::uint32_t part = 0; part < parts; ++part) {
        // A cell is a prefix of the record key, so the records sit in key order once every
        // cell has its own slots and each cell is sorted on its own.
        const auto first = partitions[part].first;
        const auto last = partitions[part].second;
        begins.assign(static_cast<std::size_t>(last - first) + 1u, 0u);
        for (auto cell = first; cell < last; ++cell) {
            begins[cell - first + 1u] = begins[cell - first] + plan.cells[cell];
        }
        const auto expected = begins.back();
        cursors.assign(begins.begin(), begins.end() - 1);
        Release(records);
        records.resize(static_cast<std::size_t>(expected));

        std::atomic<std::uint64_t> scanned{0};
        RunParallel(
            scans.size(),
            [&](std::size_t task, unsigned worker) {
                const auto range = scans[task];
                auto rows = edgeRows.At(worker, range);
                for (auto e = range.first; e < range.second; ++e) {
                    const auto edge = static_cast<std::uint32_t>(e);
                    const auto row = rows.Next(edge);
                    const auto layerOctave = plan.layerOctave[edge];
                    if (layerOctave != kNoRecord) {
                        const auto morton = maker.Morton(row.source, row.target);
                        const auto cell = CellOf(layerOctave, morton);
                        if (cell >= first && cell < last) {
                            const auto slot = std::atomic_ref(cursors[cell - first])
                                                  .fetch_add(1u, std::memory_order_relaxed);
                            if (slot >= begins[cell - first + 1u]) {
                                throw std::logic_error(
                                    "LOD page cell got more records than planned");
                            }
                            auto& record = records[static_cast<std::size_t>(slot)];
                            maker.Make(row, false, record, nullptr);
                            record.key = (std::uint64_t{layerOctave} << 32u) | morton;
                        }
                    }
                    if ((e & kProgressMask) == kProgressMask) {
                        scanned += kProgressMask + 1u;
                    }
                }
            },
            [&] { report.Report(part, parts, scanned, m); });
        for (const auto& record : plan.extras) {
            if (const auto cell = CellOf(record); cell >= first && cell < last) {
                records[static_cast<std::size_t>(cursors[cell - first]++)] = record;
            }
        }
        for (std::size_t c = 0; c < cursors.size(); ++c) {
            if (cursors[c] != begins[c + 1u]) {
                throw std::logic_error("LOD page partition lost records");
            }
        }

        // Full order, so equal keys of one edge's records land the same way on every machine.
        const auto byKey = [](const Record& a, const Record& b) {
            return std::tie(a.key, a.edge, a.rootLevels, a.childLevels, a.errors,
                            a.representative) < std::tie(b.key, b.edge, b.rootLevels,
                                                         b.childLevels, b.errors, b.representative);
        };
        std::vector<IndexRange> sorts;
        const auto share = expected / (std::uint64_t{WorkerCount()} * 8u) + 1u;
        for (std::uint64_t from = 0, c = 0; c < cursors.size(); ++c) {
            if (begins[c + 1u] - begins[from] >= share || c + 1u == cursors.size()) {
                sorts.emplace_back(from, c + 1u);
                from = c + 1u;
            }
        }
        RunParallel(sorts.size(), [&](std::size_t task, unsigned) {
            for (auto c = sorts[task].first; c < sorts[task].second; ++c) {
                const auto begin = records.begin() + static_cast<std::ptrdiff_t>(begins[c]);
                const auto end = records.begin() + static_cast<std::ptrdiff_t>(begins[c + 1u]);
                std::sort(begin, end, byKey);
            }
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

// Reads the birth and death sections back while it uses them; `hashes` gets their hashes.
[[nodiscard]] std::vector<CutSum> DfsCuts(const std::filesystem::path& path,
                                          const FileHeader& header, LevelChunks& chunks,
                                          std::span<const Cut> cuts, SectionHashes& hashes,
                                          StageProgress& report) {
    const auto m = static_cast<std::size_t>(header.edgeCount);
    const auto k = cuts.size();
    // Visit counts of all cuts side by side, one byte per edge and cut.
    std::vector<std::uint8_t> visits(m * k, 0u);
    {
        ValueReader<std::int16_t> births(path, header.At(Section::EdgeBirth));
        ValueReader<std::int16_t> deaths(path, header.At(Section::EdgeDeath));
        births.Hash(HashOf(hashes, Section::EdgeBirth));
        deaths.Hash(HashOf(hashes, Section::EdgeDeath));
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

// Reads every page and the record ids back, block by block, hashing both sections; the pages
// of a block are checked on worker threads.
[[nodiscard]] PageSums PageCuts(const LodPageFile& file, std::span<const Cut> cuts,
                                SectionHashes& hashes, StageProgress& report) {
    std::vector<LodPageQuery> everywhere(cuts.size());
    for (std::size_t k = 0; k < cuts.size(); ++k) {
        everywhere[k].minX = everywhere[k].minY = -std::numeric_limits<float>::max();
        everywhere[k].maxX = everywhere[k].maxY = std::numeric_limits<float>::max();
        everywhere[k].level = cuts[k].level;
        everywhere[k].threshold = cuts[k].threshold;
    }

    const auto& header = file.Header();
    const auto pages = file.Pages();
    const auto groups = file.Groups();
    ValueReader<std::uint32_t> recordEdges(file.Path(), header.At(Section::RecordEdge));
    recordEdges.Hash(HashOf(hashes, Section::RecordEdge));
    auto& pageHash = HashOf(hashes, Section::Pages);
    std::ifstream input(file.Path(), std::ios::binary);
    auto nextOffset = header.At(Section::Pages).offset;
    std::uint64_t nextRecord = 0;
    std::vector<std::uint32_t> words;
    std::vector<std::uint32_t> edges;

    PageSums sums;
    sums.cuts.resize(cuts.size());
    for (std::size_t firstGroup = 0; firstGroup < groups.size();) {
        // Whole groups, about kVerifyBlockBytes of pages.
        auto lastGroup = firstGroup;
        std::uint64_t bytes = 0;
        std::uint64_t records = 0;
        while (lastGroup < groups.size() &&
               (lastGroup == firstGroup || bytes < kVerifyBlockBytes)) {
            const auto& group = groups[lastGroup++];
            for (auto p = group.firstPage; p < group.firstPage + group.pageCount; ++p) {
                if (pages[p].fileOffset != nextOffset + bytes ||
                    pages[p].firstRecord != nextRecord + records) {
                    throw std::runtime_error("LOD page verification: page " + std::to_string(p) +
                                             " is not where the directory says");
                }
                bytes += pages[p].wordCount * 4u;
                records += pages[p].recordCount;
            }
        }
        words.resize(static_cast<std::size_t>(bytes / 4u));
        input.seekg(static_cast<std::streamoff>(nextOffset));
        input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes));
        if (!input) {
            throw std::runtime_error("LOD page verification: could not read the pages");
        }
        pageHash.Update(words.data(), bytes);
        edges.resize(static_cast<std::size_t>(records));
        for (auto& edge : edges) {
            edge = recordEdges.Next();
        }

        const auto blockOffset = nextOffset;
        const auto blockRecord = nextRecord;
        const auto tasks = SplitRange(firstGroup, lastGroup, 1u);
        std::vector<PageSums> parts(tasks.size());
        RunParallel(tasks.size(), [&](std::size_t task, unsigned) {
            auto& part = parts[task];
            part.cuts.resize(cuts.size());
            std::vector<bool> drawn(cuts.size());
            for (auto g = tasks[task].first; g < tasks[task].second; ++g) {
                const auto& group = groups[g];
                for (auto p = group.firstPage; p < group.firstPage + group.pageCount; ++p) {
                    const auto& page = pages[p];
                    const auto* pageWords = words.data() + (page.fileOffset - blockOffset) / 4u;
                    const auto counts =
                        std::uint32_t{page.recordCount} | (std::uint32_t{page.vertexCount} << 16u);
                    if (pageWords[0] != counts) {
                        throw std::runtime_error("LOD page verification: page " +
                                                 std::to_string(p) + " does not match its entry");
                    }
                    const auto* vertices =
                        pageWords + kPageHeaderWords + page.recordCount * kRecordWords;
                    const auto* types = reinterpret_cast<const std::uint8_t*>(
                        vertices + page.vertexCount * kVertexWords);
                    const auto position = [&](std::uint32_t local, std::uint32_t axis) {
                        return static_cast<std::int32_t>(vertices[local * kVertexWords + axis]);
                    };
                    std::fill(drawn.begin(), drawn.end(), false);
                    for (std::uint32_t i = 0; i < page.recordCount; ++i) {
                        const auto edge = edges[page.firstRecord - blockRecord + i];
                        const auto* record = pageWords + kPageHeaderWords + i * kRecordWords;
                        const auto from = record[0] & 0xffffu;
                        const auto to = record[0] >> 16u;
                        if (from >= page.vertexCount || to >= page.vertexCount) {
                            throw std::runtime_error("LOD page verification failed: page " +
                                                     std::to_string(p) +
                                                     " points past its vertices");
                        }
                        part.geometry +=
                            GeometryHash(edge, types[i], position(from, 0u), position(from, 1u),
                                         position(to, 0u), position(to, 1u));
                        for (std::size_t k = 0; k < cuts.size(); ++k) {
                            if (RecordDrawn(record[1], record[2], record[3], cuts[k].level,
                                            cuts[k].threshold)) {
                                part.cuts[k].Add(edge, 1u);
                                drawn[k] = true;
                            }
                        }
                    }
                    for (std::size_t k = 0; k < cuts.size(); ++k) {
                        if (drawn[k] &&
                            (!MayDraw(page, everywhere[k]) || !MayDraw(group, everywhere[k]))) {
                            throw std::runtime_error(
                                "LOD page verification failed: metadata of page " +
                                std::to_string(p) + " hides a drawn record");
                        }
                    }
                }
            }
        });
        for (const auto& part : parts) {
            for (std::size_t k = 0; k < cuts.size(); ++k) {
                sums.cuts[k].hash += part.cuts[k].hash;
                sums.cuts[k].count += part.cuts[k].count;
            }
            sums.geometry += part.geometry;
        }
        nextOffset += bytes;
        nextRecord += records;
        firstGroup = lastGroup;
        report.Report(2u, 3u, firstGroup, groups.size());
    }
    if (nextOffset != header.At(Section::Pages).offset + header.At(Section::Pages).bytes ||
        nextRecord != header.recordCount) {
        throw std::runtime_error("LOD page verification: the directory does not cover the pages");
    }
    return sums;
}

// ---------------------------------------------------------------------------------------------
// Hierarchy extension: for every representative record of a shortcut, where the representative
// records of its two children are, so a traversal can walk the hierarchy page by page.

// Word offsets of every page's children block (and the total at the end).
[[nodiscard]] std::vector<std::uint64_t> ChildBlockOffsets(std::span<const ChildMask> masks) {
    std::vector<std::uint64_t> offsets(masks.size() + 1u, 0u);
    for (std::size_t p = 0; p < masks.size(); ++p) {
        std::uint32_t children = 0;
        for (const auto word : masks[p]) {
            children += static_cast<std::uint32_t>(std::popcount(word));
        }
        offsets[p + 1u] = offsets[p] + kChildMaskWords + 2u * children;
    }
    return offsets;
}

// Writes the children section, in page ranges that fit the memory left: the representative
// locations (which arrived in page order) are spread out per edge, then the child columns are
// read in edge order and every shortcut's two child locations go to its slot.
void WriteChildren(const std::filesystem::path& file, const FileHeader& header,
                   std::span<const ChildMask> masks, std::span<const std::uint64_t> offsets,
                   const std::filesystem::path& representatives, std::uint64_t representativeCount,
                   std::uint64_t memoryLeft, LodPageOutput& output, std::uint64_t sectionOffset,
                   StreamHash& hash, const LodPageBuilder::ProgressCallback& progress) {
    const auto m = header.edgeCount;
    StageProgress report(progress, BuildStage::WriteHierarchy);
    report.Report(0u, 1u, 0u, 1u);
    std::vector<std::uint32_t> location(static_cast<std::size_t>(m), kNoChildRef);
    {
        ValueReader<RepresentativeLocation> reader(representatives, 0, representativeCount);
        for (std::uint64_t i = 0; i < representativeCount; ++i) {
            const auto item = reader.Next();
            location[item.edge] = item.location;
        }
    }

    const auto capacityWords =
        std::max<std::uint64_t>(memoryLeft > location.size() * 4u
                                    ? (memoryLeft - location.size() * 4u) / 4u
                                    : 0u,
                                kChildSlotWords);
    const auto scans = SplitRange(0u, m, kMinScanEdges);
    ColumnReaders<std::uint32_t> childAReaders(file, header.At(Section::EdgeChildA));
    ColumnReaders<std::uint32_t> childBReaders(file, header.At(Section::EdgeChildB));
    std::vector<std::uint32_t> words;
    const auto pageCount = static_cast<std::uint32_t>(masks.size());
    for (std::uint32_t first = 0; first < pageCount;) {
        auto last = first + 1u;
        while (last < pageCount && offsets[last + 1u] - offsets[first] <= capacityWords) {
            ++last;
        }
        const auto base = offsets[first];
        words.assign(static_cast<std::size_t>(offsets[last] - base), kNoChildRef);
        for (auto p = first; p < last; ++p) {
            std::copy(masks[p].begin(), masks[p].end(),
                      words.begin() + static_cast<std::ptrdiff_t>(offsets[p] - base));
        }
        RunParallel(scans.size(), [&](std::size_t task, unsigned worker) {
            auto& childrenA = childAReaders.At(worker, scans[task]);
            auto& childrenB = childBReaders.At(worker, scans[task]);
            for (auto e = scans[task].first; e < scans[task].second; ++e) {
                const auto childA = childrenA.Next();
                const auto childB = childrenB.Next();
                const auto here = location[static_cast<std::size_t>(e)];
                const auto page = here >> 8u;
                if (childA == kNoChildRef || here == kNoChildRef || page < first ||
                    page >= last) {
                    continue;
                }
                const auto local = here & 0xffu;
                const auto& mask = masks[page];
                std::uint32_t rank = 0;
                for (std::uint32_t w = 0; w < (local >> 5u); ++w) {
                    rank += static_cast<std::uint32_t>(std::popcount(mask[w]));
                }
                rank += static_cast<std::uint32_t>(
                    std::popcount(mask[local >> 5u] & ((1u << (local & 31u)) - 1u)));
                const auto slot = offsets[page] - base + kChildMaskWords + 2u * rank;
                words[static_cast<std::size_t>(slot)] = location[childA];
                words[static_cast<std::size_t>(slot) + 1u] = location[childB];
            }
        });
        output.WriteAt(sectionOffset + base * 4u, words.data(), words.size() * 4u);
        hash.Update(words.data(), words.size() * 4u);
        report.Report(0u, 1u, last, pageCount);
        first = last;
    }
}

// Children blocks point at the representative records of the child edges: checked on a spread
// of pages by reading the edges back from the file.
void CheckChildren(const LodPageFile& file) {
    const auto& header = file.Header();
    const auto pages = file.Pages();
    const auto visits = file.VisitPages();
    std::ifstream input(file.Path(), std::ios::binary);
    const auto readWord = [&](std::uint64_t offset) {
        std::uint32_t value = 0;
        input.seekg(static_cast<std::streamoff>(offset));
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input) {
            throw std::runtime_error("LOD page verification: could not read the hierarchy");
        }
        return value;
    };
    const auto recordEdge = [&](std::uint64_t record) {
        return readWord(header.At(Section::RecordEdge).offset + record * 4u);
    };
    const auto step = std::max<std::size_t>(1u, pages.size() / 256u);
    std::vector<std::uint32_t> block;
    for (std::size_t p = 0; p < pages.size(); p += step) {
        block.resize(visits[p].wordCount);
        input.seekg(static_cast<std::streamoff>(visits[p].fileOffset));
        input.read(reinterpret_cast<char*>(block.data()),
                   static_cast<std::streamsize>(block.size() * 4u));
        if (!input) {
            throw std::runtime_error("LOD page verification: could not read the hierarchy");
        }
        std::uint32_t next = kChildMaskWords;
        for (std::uint32_t i = 0; i < pages[p].recordCount; ++i) {
            if ((block[i >> 5u] >> (i & 31u) & 1u) == 0u) {
                continue;
            }
            const auto edge = recordEdge(pages[p].firstRecord + i);
            const std::array<std::uint32_t, 2> children{
                readWord(header.At(Section::EdgeChildA).offset + std::uint64_t{edge} * 4u),
                readWord(header.At(Section::EdgeChildB).offset + std::uint64_t{edge} * 4u)};
            for (const auto child : children) {
                const auto ref = block[next++];
                if (ref == kNoChildRef) {
                    continue;
                }
                const auto page = ref >> 8u;
                if (page >= pages.size() || (ref & 0xffu) >= pages[page].recordCount ||
                    recordEdge(pages[page].firstRecord + (ref & 0xffu)) != child) {
                    throw std::runtime_error("LOD page verification failed: a child reference of "
                                             "page " +
                                             std::to_string(p) + " points at the wrong record");
                }
            }
        }
    }
}

// The reference cuts walked through the hierarchy from the alive roots, all in memory.
[[nodiscard]] std::vector<CutSum> TraversalCuts(const LodPageFile& file,
                                                std::span<const Cut> cuts) {
    const auto pages = file.Pages();
    const auto visits = file.VisitPages();
    const auto words = file.ReadPageWords({});
    const auto children = file.ReadChildrenWords({});
    const auto recordEdge = file.ReadSection<std::uint32_t>(Section::RecordEdge);
    const auto record = [&](std::uint32_t ref) {
        return words.data() + file.PageWordOffset(ref >> 8u) + kPageHeaderWords +
               (ref & 0xffu) * kRecordWords;
    };
    std::vector<CutSum> sums(cuts.size());
    std::vector<std::uint32_t> stack;
    for (std::size_t k = 0; k < cuts.size(); ++k) {
        const auto level = cuts[k].level;
        const auto threshold = cuts[k].threshold;
        LodPageQuery everywhere;
        everywhere.minX = everywhere.minY = -std::numeric_limits<float>::max();
        everywhere.maxX = everywhere.maxY = std::numeric_limits<float>::max();
        everywhere.level = level;
        everywhere.threshold = threshold;
        for (std::uint32_t p = 0; p < pages.size(); ++p) {
            for (std::uint32_t i = 0; i < pages[p].recordCount; ++i) {
                const auto root = (p << 8u) | i;
                const auto levels = record(root)[1];
                if (levels == kEmptyLevelRange || level < (levels & 0xffffu) ||
                    level > (levels >> 16u)) {
                    continue;
                }
                stack.push_back(root);
                while (!stack.empty()) {
                    const auto ref = stack.back();
                    stack.pop_back();
                    const auto page = ref >> 8u;
                    const auto local = ref & 0xffu;
                    // The planner must find every page a traversal reaches.
                    if (!MayDraw(visits[page], everywhere)) {
                        throw std::runtime_error("LOD page verification failed: the visit "
                                                 "directory misses page " +
                                                 std::to_string(page));
                    }
                    const auto* block = children.data() + file.ChildrenWordOffset(page);
                    const bool hasChildren = (block[local >> 5u] >> (local & 31u) & 1u) != 0u;
                    if (hasChildren && (record(ref)[3] & 0xffffu) > threshold) {
                        std::uint32_t rank = 0;
                        for (std::uint32_t b = 0; b < local; ++b) {
                            rank += block[b >> 5u] >> (b & 31u) & 1u;
                        }
                        for (std::uint32_t c = 0; c < 2u; ++c) {
                            const auto child = block[kChildMaskWords + 2u * rank + c];
                            if (child == kNoChildRef) {
                                throw std::runtime_error("LOD page verification failed: an "
                                                         "expanded shortcut has no child");
                            }
                            stack.push_back(child);
                        }
                    } else {
                        sums[k].Add(recordEdge[pages[page].firstRecord + local], 1u);
                    }
                }
            }
        }
    }
    return sums;
}

std::uint32_t Verify(const std::filesystem::path& path, const ReferenceCurve& curve,
                     LevelChunks& chunks, std::uint64_t geometry, std::uint64_t memoryBudget,
                     const LodPageBuilder::ProgressCallback& progress) {
    StageProgress report(progress, BuildStage::Verify);
    const auto file = LodPageFile::Open(path);
    const auto& header = file->Header();

    // Sections the cut checks read anyway are hashed there; the rest is read back here, every
    // section on its own thread.
    const auto readLater = [](std::size_t s) {
        return s == static_cast<std::size_t>(Section::EdgeBirth) ||
               s == static_cast<std::size_t>(Section::EdgeDeath) ||
               s == static_cast<std::size_t>(Section::RecordEdge) ||
               s == static_cast<std::size_t>(Section::Pages);
    };
    std::vector<std::pair<const SectionEntry*, std::uint64_t*>> readBacks;
    std::vector<std::uint64_t> readBack(kSectionCount + kExtensionCount, 0u);
    std::uint64_t total = 0;
    for (std::size_t s = 0; s < kSectionCount; ++s) {
        if (!readLater(s)) {
            readBacks.emplace_back(&header.sections[s], &readBack[s]);
            total += header.sections[s].bytes;
        }
    }
    for (std::size_t e = 0; e < kExtensionCount; ++e) {
        readBacks.emplace_back(&header.extensions[e], &readBack[kSectionCount + e]);
        total += header.extensions[e].bytes;
    }
    SectionHashes hashes{};
    std::atomic<std::uint64_t> readBytes{0};
    RunParallel(
        readBacks.size(),
        [&](std::size_t task, unsigned) {
            const auto& section = *readBacks[task].first;
            *readBacks[task].second =
                HashFileRange(path, section.offset, section.bytes,
                              [&](std::uint64_t bytes) { readBytes += bytes; });
        },
        [&] { report.Report(0u, 3u, readBytes, total); });

    // Overview, regional and street-level views at 1 px, plus a coarse tolerance.
    std::vector<Cut> cuts;
    const auto overview = curve.OverviewZoom();
    for (const auto& [zoomOctaves, pixels] : std::array<std::pair<double, double>, 4>{
             {{0.0, 1.0}, {4.0, 1.0}, {4.0, 8.0}, {12.0, 1.0}}}) {
        const auto zoom = overview * std::exp2(zoomOctaves);
        cuts.push_back({static_cast<std::uint32_t>(std::max(0, curve.LevelAt(zoom))),
                        EncodeErrorTolerance(pixels * curve.PixelSize(zoom))});
    }
    const auto expected = DfsCuts(path, header, chunks, cuts, hashes, report);
    const auto actual = PageCuts(*file, cuts, hashes, report);
    for (std::size_t s = 0; s < kSectionCount; ++s) {
        const auto hash = readLater(s) ? hashes[s].Finish() : readBack[s];
        if (hash != header.sectionHashes[s]) {
            throw std::runtime_error("LOD page verification failed: section " + std::to_string(s) +
                                     " does not read back as written");
        }
    }
    for (std::size_t e = 0; e < kExtensionCount; ++e) {
        if (readBack[kSectionCount + e] != header.extensionHashes[e]) {
            throw std::runtime_error("LOD page verification failed: extension " +
                                     std::to_string(e) + " does not read back as written");
        }
    }
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

    // The hierarchy: always a sample of its references; the reference cuts walked through it
    // as well when pages and children fit comfortably in memory.
    CheckChildren(*file);
    const auto bytes = header.At(Section::Pages).bytes + header.At(Extension::PageChildren).bytes +
                       header.At(Section::RecordEdge).bytes;
    if (bytes <= memoryBudget / 4u) {
        const auto walked = TraversalCuts(*file, cuts);
        for (std::size_t k = 0; k < cuts.size(); ++k) {
            if (!(expected[k] == walked[k])) {
                throw std::runtime_error(
                    "LOD page verification failed at level " + std::to_string(cuts[k].level) +
                    ": DFS cut has " + std::to_string(expected[k].count) +
                    " edges, the hierarchy walk gives " + std::to_string(walked[k].count));
            }
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
    // The fullest stage holds node positions and levels (10 B per node), the level contexts,
    // each edge's record cell and road style (8 B per edge) and a sort buffer that packs the
    // pages in about kTargetPagePasses passes (about one record per edge).
    return nodeCount * 10u + edgeCount * (8u + sizeof(Record) / kTargetPagePasses) +
           WorkerCount() * kWorkerBytes + (512ull << 20u);
}

std::uint64_t LodPageBuilder::EstimateDiskBytes(std::uint64_t nodeCount, std::uint64_t edgeCount) {
    // The uncompressed file (24 B per node, about 66 B per edge) next to its compressed copy
    // (about 40 % of it), with some slack; the scratch files of the earlier stages (shortcut
    // levels, hierarchy) need less than the copy.
    return nodeCount * 32u + edgeCount * 100u;
}

namespace {

// The uncompressed file, checked against DFS reference cuts.
[[nodiscard]] BuildResult BuildUncompressed(const std::filesystem::path& graphPath,
                                            const std::filesystem::path& rangesPath,
                                            const std::filesystem::path& outputPath,
                                            const TemporaryFile& temporary, std::uint64_t budget,
                                            const LodPageBuilder::ProgressCallback& progress) {
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
        if (const auto needed = LodPageBuilder::EstimateMemoryBytes(nodes, edges);
            needed > budget) {
            ThrowNotEnoughMemory(needed, budget);
        }
        const auto disk = LodPageBuilder::EstimateDiskBytes(nodes, edges);
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
    auto representativesPath = temporary.Path();
    representativesPath += ".reps";
    TemporaryFile representatives(representativesPath);
    {
        auto errors = ComputeErrors(chunks, shortcuts, m, progress);
        header.maxHierarchyHeight = errors.maxHeight;
        output.WriteAt(header.At(Section::EdgeError).offset, errors.error.data(),
                       errors.error.size() * sizeof(float));
        HashOf(hashes, Section::EdgeError)
            .Update(errors.error.data(), errors.error.size() * sizeof(float));
    }
    const auto styles = ComputeStyles(temporary.Path(), header, nodes, chunks, shortcuts, progress);
    // Records read the CH columns back while the page section is appended.
    output.Flush();

    const ReferenceCurve curve(header.MetersPerNormalizedUnit(), nodes.maxRangeLevel);
    BuildResult result;
    std::uint64_t geometry = 0;
    std::vector<PageEntry> visits;
    std::vector<ChildMask> masks;
    std::uint64_t representativeCount = 0;
    {
        const auto contexts = ComputeContexts(chunks, m, progress);
        const RecordMaker maker(nodes, contexts, curve, shortcuts, styles);
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
        header.extraContextRecords = plan.extras.size() - plan.visitOnlyRecords;
        result.visitOnlyRecords = plan.visitOnlyRecords;
        header.At(Section::RecordEdge) = {offset, header.recordCount * 4u};
        header.At(Section::Pages) = {AlignSection(offset + header.recordCount * 4u), 0u};

        const auto fixed = nodes.Bytes() + m * 8u + plan.extras.size() * sizeof(Record) +
                           contexts.extra.size() * kExtraContextBytes +
                           header.recordCount / kPageRecordCapacity *
                               (sizeof(PageEntry) * 3u + sizeof(ChildMask)) +
                           WorkerCount() * kWorkerBytes + (256ull << 20u);
        const auto sortBudget =
            std::max<std::uint64_t>(budget > fixed ? budget - fixed : 0u, 64ull << 20u);
        const auto partitions = PlanPartitions(plan, sortBudget);
        if (partitions.size() > kMaxPagePasses) {
            ThrowNotEnoughMemory(fixed + header.recordCount * sizeof(Record) / kTargetPagePasses,
                                 budget);
        }
        result.pagePasses = static_cast<std::uint32_t>(partitions.size());
        PageWriter writer(nodes, contexts, shortcuts, output, header, hashes,
                          representatives.Path());
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
        offset = AlignSection(offset + header.groupCount * sizeof(PageGroup));
        visits = std::move(writer.VisitPages());
        masks = std::move(writer.ChildMasks());
        representativeCount = writer.RepresentativeCount();
    }

    // The hierarchy extension: children blocks, then the visit directory pointing at them.
    std::array<StreamHash, kExtensionCount> extensionHashes{};
    {
        const auto hashOf = [&](Extension extension) -> StreamHash& {
            return extensionHashes[static_cast<std::size_t>(extension)];
        };
        const auto blocks = ChildBlockOffsets(masks);
        header.At(Extension::PageChildren) = {offset, blocks.back() * 4u};
        const auto reserved = nodes.Bytes() + visits.size() * (sizeof(PageEntry) * 2u) +
                              masks.size() * sizeof(ChildMask) + blocks.size() * 8u +
                              WorkerCount() * kWorkerBytes + (256ull << 20u);
        WriteChildren(temporary.Path(), header, masks, blocks, representatives.Path(),
                      representativeCount, budget > reserved ? budget - reserved : 0u, output,
                      offset, hashOf(Extension::PageChildren), progress);
        for (std::size_t p = 0; p < visits.size(); ++p) {
            visits[p].fileOffset = offset + blocks[p] * 4u;
            visits[p].wordCount = static_cast<std::uint32_t>(blocks[p + 1u] - blocks[p]);
        }
        offset = AlignSection(offset + blocks.back() * 4u);
        header.At(Extension::VisitDirectory) = {offset, visits.size() * sizeof(PageEntry)};
        output.WriteAt(offset, visits.data(), visits.size() * sizeof(PageEntry));
        hashOf(Extension::VisitDirectory).Update(visits.data(), visits.size() * sizeof(PageEntry));
        const auto visitGroups = BuildGroups(visits);
        offset = AlignSection(offset + visits.size() * sizeof(PageEntry));
        header.At(Extension::VisitGroups) = {offset, visitGroups.size() * sizeof(PageGroup)};
        output.WriteAt(offset, visitGroups.data(), visitGroups.size() * sizeof(PageGroup));
        hashOf(Extension::VisitGroups)
            .Update(visitGroups.data(), visitGroups.size() * sizeof(PageGroup));
    }

    for (std::size_t s = 0; s < kSectionCount; ++s) {
        header.sectionHashes[s] = hashes[s].Finish();
    }
    for (std::size_t e = 0; e < kExtensionCount; ++e) {
        header.extensionHashes[e] = extensionHashes[e].Finish();
    }
    std::vector<std::byte> headerBytes(kHeaderBytes);
    std::memcpy(headerBytes.data(), &header, sizeof(FileHeader));
    output.WriteAt(0, headerBytes.data(), headerBytes.size());
    output.Close();

    result.verifiedCuts = Verify(temporary.Path(), curve, chunks, geometry, budget, progress);
    result.nodeCount = header.nodeCount;
    result.edgeCount = header.edgeCount;
    result.recordCount = header.recordCount;
    result.pageCount = header.pageCount;
    result.coverGapEdges = header.coverGapEdges;
    result.extraContextRecords = header.extraContextRecords;
    result.memoryBudgetBytes = budget;
    return result;
}

// The compressed file must decode to exactly the bytes the uncompressed one was checked with.
void CheckCompressed(const std::filesystem::path& path, const FileHeader& checked,
                     const std::function<void(std::uint64_t, std::uint64_t)>& report) {
    const auto file = LodPageFile::Open(path);
    const auto& header = file->Header();
    std::uint64_t total = 0;
    for (const auto& stored : header.stored) {
        total += stored.bytes;
    }
    std::uint64_t done = 0;
    for (std::size_t index = 0; index < kStoredCount; ++index) {
        const auto hash = file->HashSection(index, [&](std::uint64_t bytes) {
            done += bytes;
            report(done, total);
        });
        if (hash != checked.Hash(index) ||
            header.Layout(index).bytes != checked.Layout(index).bytes) {
            throw std::runtime_error("LOD page compression failed: section " +
                                     std::to_string(index) + " does not decode as written");
        }
    }
    report(total, total);
}

// Compresses the checked uncompressed file `raw` into `compressed` and decodes it again.
void WriteCompressed(const std::filesystem::path& raw, const std::filesystem::path& compressed,
                     const LodPageBuilder::ProgressCallback& progress) {
    StageProgress report(progress, BuildStage::Compress);
    const auto file = LodPageFile::Open(raw);
    if (file->Compressed()) {
        throw std::logic_error("LOD page compression: the file is already compressed");
    }
    // Compressing takes about three quarters of the stage, decoding it again the rest.
    CompressPageFile(*file, compressed, [&](std::uint64_t done, std::uint64_t total) {
        report.Report(0u, 1u, done * 3u, total * 4u);
    });
    CheckCompressed(compressed, file->Header(), [&](std::uint64_t done, std::uint64_t total) {
        report.Report(0u, 1u, total * 3u + done, total * 4u);
    });
}

} // namespace

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

    const auto memory = data::QuerySystemMemoryInfo();
    const auto budget = memoryBudgetBytes != 0u ? memoryBudgetBytes
                        : memory.availablePhysicalBytes != 0u
                            ? memory.availablePhysicalBytes / 4u * 3u
                            : 4ull << 30u;
    auto result = BuildUncompressed(graphPath, rangesPath, outputPath, temporary, budget, progress);

    // The scratch files are gone; the checked file is compressed next to it.
    auto compressedPath = temporaryPath;
    compressedPath += ".zst";
    TemporaryFile compressed(compressedPath);
    WriteCompressed(temporary.Path(), compressed.Path(), progress);
    result.uncompressedBytes = std::filesystem::file_size(temporary.Path());
    result.fileBytes = std::filesystem::file_size(compressed.Path());
    compressed.MoveTo(outputPath);

    result.path = outputPath;
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return result;
}

void LodPageBuilder::Compress(const std::filesystem::path& path, const ProgressCallback& progress) {
    auto compressedPath = path;
    compressedPath += ".tmp";
    RemoveStaleScratch(compressedPath);
    compressedPath += ".zst";
    // The copy is about 40 % of the file.
    const auto needed = std::filesystem::file_size(path) / 2u;
    if (const auto available = FreeDiskSpace(path); available < needed) {
        ThrowNotEnoughDisk(needed, available);
    }
    TemporaryFile compressed(compressedPath);
    WriteCompressed(path, compressed.Path(), progress);
    compressed.MoveTo(path);
}

} // namespace chmv::streaming::lod
