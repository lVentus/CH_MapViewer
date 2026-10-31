#include "streaming/lod/LodPageBuilder.h"

#include "data/ch/CHTypes.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/lod/LodPageSource.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chmv::streaming::lod {
namespace {

constexpr std::uint32_t kNoEdge = 0xffffffffu;
constexpr std::uint64_t kProgressStep = 1u << 20;
// Layout reference only: pages are grouped by how a 4K screen would first need them.
constexpr std::uint32_t kReferenceFramebufferHeight = 2160;
constexpr int kQualityLayers = 5; // 1, 2, 4, 8 px on the reference curve, then everything else
constexpr int kCatchAllLayer = kQualityLayers - 1;
constexpr int kMaxOctave = 31;
constexpr double kFarZoom = 1048576.0;
constexpr std::uint64_t kPageWriteChunkBytes = 16ull << 20u;

void Report(const LodPageBuilder::ProgressCallback& progress, BuildStage stage,
            std::uint64_t current, std::uint64_t total) {
    if (progress && (current % kProgressStep == 0u || current == total)) {
        progress({stage, current, total});
    }
}

template <typename T> void Release(std::vector<T>& values) {
    std::vector<T>().swap(values);
}

// ---------------------------------------------------------------------------------------------
// Exact recursive geometry error: distance of the split node to the shortcut's straight
// segment plus the larger error of the two children. Monotone by construction.

struct ErrorTable {
    std::vector<float> error;
    std::vector<std::uint8_t> height;
    std::uint32_t maxHeight = 0;
};

[[nodiscard]] std::uint32_t SplitNode(const LodPageHierarchy& h, std::uint32_t edge) {
    const auto a = h.childA[edge];
    const auto b = h.childB[edge];
    std::uint32_t split = kNoEdge;
    for (const auto ca : {h.source[a], h.target[a]}) {
        for (const auto cb : {h.source[b], h.target[b]}) {
            if (ca == cb && (split == kNoEdge || (ca != h.source[edge] && ca != h.target[edge]))) {
                split = ca;
            }
        }
    }
    if (split == kNoEdge) {
        throw std::runtime_error("children of shortcut " + std::to_string(edge) +
                                 " do not share a node");
    }
    return split;
}

[[nodiscard]] double DistanceToSegment(const LodPageHierarchy& h, std::uint32_t point,
                                       std::uint32_t from, std::uint32_t to) {
    const double px = h.nodeX[point];
    const double py = h.nodeY[point];
    const double ax = h.nodeX[from];
    const double ay = h.nodeY[from];
    const double dx = h.nodeX[to] - ax;
    const double dy = h.nodeY[to] - ay;
    const double lengthSquared = dx * dx + dy * dy;
    const double t = lengthSquared > 0.0
                         ? std::clamp(((px - ax) * dx + (py - ay) * dy) / lengthSquared, 0.0, 1.0)
                         : 0.0;
    return std::hypot(px - (ax + t * dx), py - (ay + t * dy)) / kFixedCoordinateScale;
}

[[nodiscard]] ErrorTable ComputeErrors(const LodPageHierarchy& h,
                                       const LodPageBuilder::ProgressCallback& progress) {
    const auto m = static_cast<std::uint32_t>(h.edgeCount);
    ErrorTable table;
    table.error.assign(m, 0.0f);
    table.height.assign(m, 0);

    // 0 = not visited, 1 = on the stack, 2 = done. Original edges are done from the start.
    std::vector<std::uint8_t> state(m, 0);
    std::uint64_t shortcuts = 0;
    for (std::uint32_t e = 0; e < m; ++e) {
        if (h.IsShortcut(e)) {
            ++shortcuts;
        } else {
            state[e] = 2;
        }
    }

    std::uint64_t done = 0;
    std::vector<std::uint32_t> stack;
    for (std::uint32_t root = 0; root < m; ++root) {
        if (state[root] == 2) {
            continue;
        }
        stack.push_back(root);
        while (!stack.empty()) {
            const auto e = stack.back();
            if (state[e] == 2) {
                stack.pop_back();
                continue;
            }
            const auto a = h.childA[e];
            const auto b = h.childB[e];
            if (state[e] == 0) {
                state[e] = 1;
                for (const auto child : {a, b}) {
                    if (state[child] == 1) {
                        throw std::runtime_error("shortcut " + std::to_string(e) +
                                                 " contains itself (cyclic children)");
                    }
                    if (state[child] == 0) {
                        stack.push_back(child);
                    }
                }
                continue;
            }

            const auto split = SplitNode(h, e);
            const double childError = std::max(table.error[a], table.error[b]);
            table.error[e] = static_cast<float>(
                DistanceToSegment(h, split, h.source[e], h.target[e]) + childError);
            const auto height = 1u + std::max(table.height[a], table.height[b]);
            if (height > 250u) {
                throw std::runtime_error("shortcut hierarchy deeper than 250 levels");
            }
            table.height[e] = static_cast<std::uint8_t>(height);
            table.maxHeight = std::max(table.maxHeight, height);
            state[e] = 2;
            stack.pop_back();
            Report(progress, BuildStage::ComputeHierarchy, ++done, shortcuts * 2u);
        }
    }
    return table;
}

// ---------------------------------------------------------------------------------------------
// Level contexts. An edge is drawn either as an alive root ([death, birth]) or below a parent
// whose own reachable levels ("cover") contain R. cover(e) = alive(e) U cover(parents), so it is
// computed parents first, i.e. by decreasing hierarchy height.

struct LevelInterval {
    std::int32_t low = 1;
    std::int32_t high = 0;

    [[nodiscard]] bool Empty() const { return low > high; }
    // Directly next to each other without sharing a level: the union counts every level once.
    [[nodiscard]] bool Adjoins(const LevelInterval& other) const {
        return !Empty() && !other.Empty() && (high + 1 == other.low || other.high + 1 == low);
    }
    [[nodiscard]] bool Overlaps(const LevelInterval& other) const {
        return !Empty() && !other.Empty() && low <= other.high && other.low <= high;
    }
};

// cover(e) is a multiset: when two parents (or an alive root and a parent) reach an edge at the
// same level, DFS visits it - and everything below it - twice. Intervals are therefore only
// joined when they adjoin; overlapping ones stay separate so each visit keeps its own context.
class CoverSets {
public:
    explicit CoverSets(std::size_t edgeCount) : low_(edgeCount, 1), high_(edgeCount, 0) {}

    void Add(std::uint32_t edge, LevelInterval interval) {
        const LevelInterval primary{low_[edge], high_[edge]};
        if (const auto it = overflow_.find(edge); it != overflow_.end()) {
            auto& list = it->second;
            list.push_back(primary);
            list.push_back(interval);
            Normalize(list);
            Store(edge, list.front());
            list.erase(list.begin());
            if (list.empty()) {
                overflow_.erase(it);
            }
            return;
        }
        if (primary.Empty()) {
            Store(edge, interval);
        } else if (primary.Adjoins(interval)) {
            Store(edge,
                  {std::min(primary.low, interval.low), std::max(primary.high, interval.high)});
        } else {
            overflow_[edge].push_back(interval);
        }
    }

    void Collect(std::uint32_t edge, std::vector<LevelInterval>& out) const {
        out.clear();
        if (low_[edge] <= high_[edge]) {
            out.push_back({low_[edge], high_[edge]});
        }
        if (const auto it = overflow_.find(edge); it != overflow_.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
        }
    }

    // Edges whose cover is not one interval (a gap, or several visits at the same level).
    [[nodiscard]] std::uint64_t GapEdges() const { return overflow_.size(); }

private:
    static void Normalize(std::vector<LevelInterval>& list) {
        std::erase_if(list, [](const LevelInterval& interval) { return interval.Empty(); });
        std::sort(list.begin(), list.end(),
                  [](const LevelInterval& a, const LevelInterval& b) { return a.low < b.low; });
        std::size_t out = 0;
        for (std::size_t i = 1; i < list.size(); ++i) {
            if (list[out].Adjoins(list[i])) {
                list[out].high = std::max(list[out].high, list[i].high);
            } else {
                list[++out] = list[i];
            }
        }
        list.resize(list.empty() ? 0u : out + 1u);
    }

    void Store(std::uint32_t edge, LevelInterval interval) {
        low_[edge] = static_cast<std::int16_t>(interval.low);
        high_[edge] = static_cast<std::int16_t>(interval.high);
    }

    std::vector<std::int16_t> low_;
    std::vector<std::int16_t> high_;
    std::unordered_map<std::uint32_t, std::vector<LevelInterval>> overflow_;
};

struct ExtraContext {
    std::uint32_t edge = 0;
    LevelInterval levels;
    float parentError = 0.0f;
};

struct ContextTable {
    // Primary context below a parent. Further contexts of the same edge (a second parent, a gap
    // in the parent's cover, or an overlap with the edge's own root interval) become extra
    // records so that every DFS visit has exactly one record.
    std::vector<std::int16_t> low;
    std::vector<std::int16_t> high;
    std::vector<float> parentError;
    std::vector<ExtraContext> extra;
    std::vector<std::int16_t> coverLow;
    std::vector<std::int16_t> coverHigh;
    std::uint64_t gapEdges = 0;

    [[nodiscard]] LevelInterval Primary(std::uint32_t edge) const {
        return {low[edge], high[edge]};
    }
};

void AddContext(ContextTable& table, std::uint32_t edge, LevelInterval levels, float parentError) {
    const auto primary = table.Primary(edge);
    if (primary.Empty()) {
        table.low[edge] = static_cast<std::int16_t>(levels.low);
        table.high[edge] = static_cast<std::int16_t>(levels.high);
        table.parentError[edge] = parentError;
    } else if (table.parentError[edge] == parentError &&
               (levels.low == primary.high + 1 || levels.high + 1 == primary.low)) {
        table.low[edge] = static_cast<std::int16_t>(std::min(primary.low, levels.low));
        table.high[edge] = static_cast<std::int16_t>(std::max(primary.high, levels.high));
    } else {
        table.extra.push_back({edge, levels, parentError});
    }
}

[[nodiscard]] LevelInterval RootInterval(const LodPageHierarchy& h, std::uint32_t edge) {
    return h.HasRootInterval(edge) ? LevelInterval{h.death[edge], h.birth[edge]} : LevelInterval{};
}

[[nodiscard]] ContextTable ComputeContexts(const LodPageHierarchy& h, const ErrorTable& errors,
                                           const LodPageBuilder::ProgressCallback& progress) {
    const auto m = static_cast<std::uint32_t>(h.edgeCount);
    ContextTable table;
    table.low.assign(m, 1);
    table.high.assign(m, 0);
    table.parentError.assign(m, 0.0f);

    CoverSets cover(m);
    for (std::uint32_t e = 0; e < m; ++e) {
        if (const auto root = RootInterval(h, e); !root.Empty()) {
            cover.Add(e, root);
        }
    }

    // Shortcuts by decreasing height: every parent is final before its children are visited.
    std::vector<std::uint64_t> firstOfHeight(errors.maxHeight + 2u, 0);
    for (std::uint32_t e = 0; e < m; ++e) {
        if (h.IsShortcut(e)) {
            ++firstOfHeight[errors.maxHeight - errors.height[e] + 1u];
        }
    }
    for (std::size_t i = 1; i < firstOfHeight.size(); ++i) {
        firstOfHeight[i] += firstOfHeight[i - 1];
    }
    std::vector<std::uint32_t> order(static_cast<std::size_t>(firstOfHeight.back()));
    for (std::uint32_t e = 0; e < m; ++e) {
        if (h.IsShortcut(e)) {
            order[firstOfHeight[errors.maxHeight - errors.height[e]]++] = e;
        }
    }

    std::vector<LevelInterval> intervals;
    std::uint64_t done = 0;
    for (const auto parent : order) {
        cover.Collect(parent, intervals);
        for (const auto& levels : intervals) {
            for (const auto child : {h.childA[parent], h.childB[parent]}) {
                AddContext(table, child, levels, errors.error[parent]);
                cover.Add(child, levels);
            }
        }
        Report(progress, BuildStage::ComputeHierarchy, order.size() + ++done, order.size() * 2u);
    }

    // An edge that is an alive root and, at the same level, also reachable below a parent is
    // visited twice by DFS; keep that second visit as its own record.
    for (std::uint32_t e = 0; e < m; ++e) {
        const auto primary = table.Primary(e);
        if (!primary.Empty() && primary.Overlaps(RootInterval(h, e))) {
            table.extra.push_back({e, primary, table.parentError[e]});
            table.low[e] = 1;
            table.high[e] = 0;
        }
    }

    table.coverLow.assign(m, -1);
    table.coverHigh.assign(m, -1);
    for (std::uint32_t e = 0; e < m; ++e) {
        cover.Collect(e, intervals);
        if (!intervals.empty()) {
            auto low = intervals.front().low;
            auto high = intervals.front().high;
            for (const auto& levels : intervals) {
                low = std::min(low, levels.low);
                high = std::max(high, levels.high);
            }
            table.coverLow[e] = static_cast<std::int16_t>(low);
            table.coverHigh[e] = static_cast<std::int16_t>(high);
        }
    }
    table.gapEdges = cover.GapEdges();
    return table;
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

struct Record {
    std::uint64_t key = 0;
    std::uint32_t edge = 0;
    std::uint32_t rootLevels = kEmptyLevelRange;
    std::uint32_t childLevels = kEmptyLevelRange;
    std::uint32_t errors = 0;

    [[nodiscard]] int Layer() const { return static_cast<int>(key >> 37u); }
    [[nodiscard]] int Octave() const { return static_cast<int>((key >> 32u) & 31u); }
};

[[nodiscard]] std::uint32_t PackLevels(LevelInterval levels) {
    return levels.Empty() ? kEmptyLevelRange
                          : PackLevelRange(static_cast<std::uint16_t>(levels.low),
                                           static_cast<std::uint16_t>(levels.high));
}

class RecordFactory {
public:
    RecordFactory(const LodPageHierarchy& h, const ErrorTable& errors, const ReferenceCurve& curve)
        : h_(h), errors_(errors), curve_(curve) {
        minX_ = *std::min_element(h.nodeX.begin(), h.nodeX.end());
        minY_ = *std::min_element(h.nodeY.begin(), h.nodeY.end());
        const auto maxX = *std::max_element(h.nodeX.begin(), h.nodeX.end());
        const auto maxY = *std::max_element(h.nodeY.begin(), h.nodeY.end());
        scaleX_ = 65535.0 / std::max(1.0, static_cast<double>(maxX) - minX_);
        scaleY_ = 65535.0 / std::max(1.0, static_cast<double>(maxY) - minY_);
    }

    // Returns false if neither context can ever be drawn.
    bool Make(std::uint32_t edge, LevelInterval root, LevelInterval child, float parentError,
              Record& record) const {
        const auto error = errors_.error[edge];
        const auto errorCode = EncodeGeometryError(error);
        auto parentCode = kErrorCodeInfinite;
        if (!child.Empty()) {
            parentCode = EncodeGeometryError(parentError);
            if (errorCode >= parentCode) {
                // Empty error interval: this context is never the cut, not even after rounding.
                child = {};
                parentCode = kErrorCodeInfinite;
            }
        }
        if (root.Empty() && child.Empty()) {
            return false;
        }

        Placement placement;
        if (!root.Empty()) {
            placement = Place(curve_, root, error, std::numeric_limits<double>::infinity());
        }
        if (!child.Empty()) {
            const auto below = Place(curve_, child, error, parentError);
            if (root.Empty() || below.layer < placement.layer ||
                (below.layer == placement.layer && below.octave < placement.octave)) {
                placement = below;
            }
        }

        const auto cx =
            0.5 * (static_cast<double>(h_.nodeX[h_.source[edge]]) + h_.nodeX[h_.target[edge]]);
        const auto cy =
            0.5 * (static_cast<double>(h_.nodeY[h_.source[edge]]) + h_.nodeY[h_.target[edge]]);
        const auto mx =
            static_cast<std::uint32_t>(std::clamp((cx - minX_) * scaleX_, 0.0, 65535.0));
        const auto my =
            static_cast<std::uint32_t>(std::clamp((cy - minY_) * scaleY_, 0.0, 65535.0));
        record.key = (static_cast<std::uint64_t>(placement.layer) << 37u) |
                     (static_cast<std::uint64_t>(placement.octave) << 32u) |
                     (Part1By1(mx) | (Part1By1(my) << 1u));
        record.edge = edge;
        record.rootLevels = PackLevels(root);
        record.childLevels = PackLevels(child);
        record.errors =
            static_cast<std::uint32_t>(errorCode) | (static_cast<std::uint32_t>(parentCode) << 16u);
        return true;
    }

private:
    const LodPageHierarchy& h_;
    const ErrorTable& errors_;
    const ReferenceCurve& curve_;
    double minX_ = 0.0;
    double minY_ = 0.0;
    double scaleX_ = 1.0;
    double scaleY_ = 1.0;
};

[[nodiscard]] std::vector<Record> BuildRecords(const LodPageHierarchy& h, const ErrorTable& errors,
                                               const ContextTable& contexts,
                                               const ReferenceCurve& curve,
                                               const LodPageBuilder::ProgressCallback& progress) {
    const auto m = static_cast<std::uint32_t>(h.edgeCount);
    const RecordFactory factory(h, errors, curve);
    std::vector<Record> records;
    std::uint64_t estimate = contexts.extra.size();
    for (std::uint32_t e = 0; e < m; ++e) {
        estimate += h.HasRootInterval(e) || !contexts.Primary(e).Empty() ? 1u : 0u;
    }
    records.reserve(static_cast<std::size_t>(estimate));

    Record record;
    for (std::uint32_t e = 0; e < m; ++e) {
        if (factory.Make(e, RootInterval(h, e), contexts.Primary(e), contexts.parentError[e],
                         record)) {
            records.push_back(record);
        }
        Report(progress, BuildStage::LayoutPages, e + 1u, m);
    }
    for (const auto& extra : contexts.extra) {
        if (factory.Make(extra.edge, {}, extra.levels, extra.parentError, record)) {
            records.push_back(record);
        }
    }
    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
        return a.key != b.key ? a.key < b.key : a.edge < b.edge;
    });
    return records;
}

// ---------------------------------------------------------------------------------------------
// Page packing. A page never mixes layers or octaves and holds at most 256 records.

struct PageOutput {
    std::vector<PageEntry> pages;
    std::vector<PageGroup> groups;
    std::vector<std::uint32_t> recordEdges;
    std::uint64_t pageBytes = 0;
};

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
    PageWriter(const LodPageHierarchy& h, LodPageOutput& output, std::uint64_t pagesOffset)
        : h_(h), output_(output), pagesOffset_(pagesOffset) {
        local_.reserve(kPageVertexCapacity * 2u);
        vertices_.reserve(kPageVertexCapacity);
    }

    void Write(std::span<const Record> records, PageOutput& out) {
        PageEntry entry;
        entry.fileOffset = pagesOffset_ + out.pageBytes + pending_.size() * 4u;
        entry.firstRecord = static_cast<std::uint32_t>(out.recordEdges.size());
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
            const auto edge = records[i].edge;
            const auto from = localIndex(h_.source[edge]);
            const auto to = localIndex(h_.target[edge]);
            endpoints[i] = from | (to << 16u);
        }
        const auto vertexCount = static_cast<std::uint32_t>(vertices_.size());
        entry.vertexCount = static_cast<std::uint16_t>(vertexCount);
        entry.wordCount = PageWordCount(recordCount, vertexCount);

        const auto base = pending_.size();
        pending_.resize(base + entry.wordCount, 0u);
        auto* words = pending_.data() + base;
        words[0] = recordCount | (vertexCount << 16u);
        words[1] = static_cast<std::uint32_t>(out.pages.size());
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
            typeBytes[i] =
                static_cast<std::uint8_t>(data::RoadStyleIndexFromType(h_.type[record.edge]));
            out.recordEdges.push_back(record.edge);

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
            vertexWords[v * 2u + 0u] = static_cast<std::uint32_t>(h_.nodeX[node]);
            vertexWords[v * 2u + 1u] = static_cast<std::uint32_t>(h_.nodeY[node]);
            minX = std::min(minX, h_.nodeX[node]);
            minY = std::min(minY, h_.nodeY[node]);
            maxX = std::max(maxX, h_.nodeX[node]);
            maxY = std::max(maxY, h_.nodeY[node]);
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
        out.pages.push_back(entry);

        if (pending_.size() * 4u >= kPageWriteChunkBytes) {
            Flush(out);
        }
    }

    void Flush(PageOutput& out) {
        if (pending_.empty()) {
            return;
        }
        output_.WriteAt(pagesOffset_ + out.pageBytes, pending_.data(), pending_.size() * 4u);
        out.pageBytes += pending_.size() * 4u;
        pending_.clear();
    }

private:
    [[nodiscard]] static float OutwardFloat(std::int32_t fixed, float direction) {
        const auto value = static_cast<float>(static_cast<double>(fixed) / kFixedCoordinateScale);
        return std::nextafter(value, direction * std::numeric_limits<float>::infinity());
    }

    const LodPageHierarchy& h_;
    LodPageOutput& output_;
    std::uint64_t pagesOffset_;
    std::unordered_map<std::uint32_t, std::uint32_t> local_;
    std::vector<std::uint32_t> vertices_;
    std::vector<std::uint32_t> pending_;
};

[[nodiscard]] PageOutput WritePages(const LodPageHierarchy& h, std::span<const Record> records,
                                    LodPageOutput& output, std::uint64_t pagesOffset,
                                    const LodPageBuilder::ProgressCallback& progress) {
    PageOutput out;
    out.recordEdges.reserve(records.size());
    PageWriter writer(h, output, pagesOffset);
    std::size_t begin = 0;
    while (begin < records.size()) {
        const auto bucket = records[begin].key >> 32u;
        auto end = begin + 1u;
        while (end < records.size() && end - begin < kPageRecordCapacity &&
               (records[end].key >> 32u) == bucket) {
            ++end;
        }
        writer.Write(records.subspan(begin, end - begin), out);
        begin = end;
        Report(progress, BuildStage::WritePages, begin, records.size());
    }
    writer.Flush(out);
    Report(progress, BuildStage::WritePages, records.size(), records.size());

    for (std::size_t first = 0; first < out.pages.size(); first += kPagesPerGroup) {
        const auto last = std::min(out.pages.size(), first + kPagesPerGroup);
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
            const auto& page = out.pages[p];
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
            MergeGroupBands(group, out.pages[p]);
        }
        out.groups.push_back(group);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Verification against the hierarchy itself, reading everything back from the written file.

template <typename T>
void RequireSame(const std::vector<T>& written, const std::vector<T>& expected, const char* name) {
    if (written != expected) {
        throw std::runtime_error(std::string("LOD page verification failed: CH column ") + name +
                                 " differs after reading it back");
    }
}

struct Cut {
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
};

void VerifyCut(const LodPageHierarchy& h, const ErrorTable& errors, const LodPageFile& file,
               std::span<const std::uint32_t> pageWords, std::span<const std::uint32_t> recordEdges,
               Cut cut) {
    std::vector<std::uint32_t> expected;
    std::vector<std::uint32_t> stack;
    const auto level = static_cast<std::int32_t>(cut.level);
    for (std::uint32_t root = 0; root < h.edgeCount; ++root) {
        if (!h.HasRootInterval(root) || level < h.death[root] || level > h.birth[root]) {
            continue;
        }
        stack.push_back(root);
        while (!stack.empty()) {
            const auto e = stack.back();
            stack.pop_back();
            if (h.IsShortcut(e) && EncodeGeometryError(errors.error[e]) > cut.threshold) {
                stack.push_back(h.childA[e]);
                stack.push_back(h.childB[e]);
            } else {
                expected.push_back(e);
            }
        }
    }

    LodPageQuery everywhere;
    everywhere.minX = everywhere.minY = -std::numeric_limits<float>::max();
    everywhere.maxX = everywhere.maxY = std::numeric_limits<float>::max();
    everywhere.level = cut.level;
    everywhere.threshold = cut.threshold;

    std::vector<std::uint32_t> actual;
    const auto pages = file.Pages();
    for (std::uint32_t p = 0; p < pages.size(); ++p) {
        const auto& page = pages[p];
        const auto* words = pageWords.data() + file.PageWordOffset(p);
        bool any = false;
        for (std::uint32_t i = 0; i < page.recordCount; ++i) {
            const auto* record = words + kPageHeaderWords + i * kRecordWords;
            if (RecordDrawn(record[1], record[2], record[3], cut.level, cut.threshold)) {
                actual.push_back(recordEdges[page.firstRecord + i]);
                any = true;
            }
        }
        if (any && !MayDraw(page, everywhere)) {
            throw std::runtime_error("LOD page verification failed: page " + std::to_string(p) +
                                     " metadata hides a drawn record");
        }
    }
    for (const auto& group : file.Groups()) {
        for (auto p = group.firstPage; p < group.firstPage + group.pageCount; ++p) {
            if (MayDraw(pages[p], everywhere) && !MayDraw(group, everywhere)) {
                throw std::runtime_error("LOD page verification failed: group metadata");
            }
        }
    }

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    if (expected != actual) {
        throw std::runtime_error("LOD page verification failed at level " +
                                 std::to_string(cut.level) + ": DFS cut has " +
                                 std::to_string(expected.size()) + " edges, pages give " +
                                 std::to_string(actual.size()));
    }
}

std::uint32_t Verify(const LodPageHierarchy& h, const ErrorTable& errors,
                     const ReferenceCurve& curve, const std::filesystem::path& path,
                     const LodPageBuilder::ProgressCallback& progress) {
    const auto file = LodPageFile::Open(path);
    RequireSame(file->ReadSection<std::uint32_t>(Section::EdgeSource), h.source, "source");
    RequireSame(file->ReadSection<std::uint32_t>(Section::EdgeTarget), h.target, "target");
    RequireSame(file->ReadSection<std::uint32_t>(Section::EdgeChildA), h.childA, "childA");
    RequireSame(file->ReadSection<std::uint32_t>(Section::EdgeChildB), h.childB, "childB");
    RequireSame(file->ReadSection<std::int16_t>(Section::EdgeBirth), h.birth, "birth");
    RequireSame(file->ReadSection<std::int16_t>(Section::EdgeDeath), h.death, "death");
    RequireSame(file->ReadSection<std::int16_t>(Section::EdgeType), h.type, "type");
    RequireSame(file->ReadSection<float>(Section::EdgeError), errors.error, "geometry error");

    const auto pageWords = file->ReadPageWords({});
    const auto recordEdges = file->ReadSection<std::uint32_t>(Section::RecordEdge);

    // Overview, regional and street-level views at 1 px, plus a coarse tolerance.
    std::vector<Cut> cuts;
    const auto overview = curve.OverviewZoom();
    for (const auto& [zoomOctaves, pixels] : std::array<std::pair<double, double>, 4>{
             {{0.0, 1.0}, {4.0, 1.0}, {4.0, 8.0}, {12.0, 1.0}}}) {
        const auto zoom = overview * std::exp2(zoomOctaves);
        cuts.push_back({static_cast<std::uint32_t>(std::max(0, curve.LevelAt(zoom))),
                        EncodeErrorTolerance(pixels * curve.PixelSize(zoom))});
    }
    for (std::size_t i = 0; i < cuts.size(); ++i) {
        VerifyCut(h, errors, *file, pageWords, recordEdges, cuts[i]);
        if (progress) {
            progress({BuildStage::Verify, i + 1u, cuts.size()});
        }
    }
    return static_cast<std::uint32_t>(cuts.size());
}

class TemporaryFile {
public:
    explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {}
    ~TemporaryFile() {
        if (!path_.empty()) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;

    [[nodiscard]] const std::filesystem::path& Path() const { return path_; }
    // rename replaces an existing file on every platform; on failure the old file survives.
    void MoveTo(const std::filesystem::path& destination) {
        std::filesystem::rename(path_, destination);
        path_.clear();
    }

private:
    std::filesystem::path path_;
};

} // namespace

std::uint64_t LodPageBuilder::EstimateMemoryBytes(std::uint64_t nodeCount,
                                                  std::uint64_t edgeCount) {
    // Hierarchy 22 B, errors 5 B, contexts and cover 14 B and records 24 B per edge while the
    // records are built; the final check holds the hierarchy, the errors and the whole page
    // section (about 21 B per record) at once. Plus coordinates and slack.
    return nodeCount * 8u + edgeCount * 80u + (256ull << 20u);
}

BuildResult LodPageBuilder::Build(const std::filesystem::path& graphPath,
                                  const std::filesystem::path& rangesPath,
                                  const std::filesystem::path& outputPath,
                                  const ProgressCallback& progress) {
    const auto begin = std::chrono::steady_clock::now();
    auto temporaryPath = outputPath;
    temporaryPath += ".tmp";
    TemporaryFile temporary(temporaryPath);

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

    LodPageOutput output(temporary.Path());
    std::uint64_t offset = 0;
    auto hierarchy = ReadLodPageSource(graphPath, rangesPath, header, output, offset, progress);
    header.maxLevel = hierarchy.maxLevel;

    const auto errors = ComputeErrors(hierarchy, progress);
    header.maxHierarchyHeight = errors.maxHeight;
    output.WriteAt(header.At(Section::EdgeError).offset, errors.error.data(),
                   errors.error.size() * sizeof(float));

    const ReferenceCurve curve(header.MetersPerNormalizedUnit(), hierarchy.maxLevel);
    std::vector<Record> records;
    {
        auto contexts = ComputeContexts(hierarchy, errors, progress);
        output.WriteAt(header.At(Section::EdgeCoverLow).offset, contexts.coverLow.data(),
                       contexts.coverLow.size() * sizeof(std::int16_t));
        output.WriteAt(header.At(Section::EdgeCoverHigh).offset, contexts.coverHigh.data(),
                       contexts.coverHigh.size() * sizeof(std::int16_t));
        header.coverGapEdges = contexts.gapEdges;
        header.extraContextRecords = contexts.extra.size();
        Release(contexts.coverLow);
        Release(contexts.coverHigh);
        records = BuildRecords(hierarchy, errors, contexts, curve, progress);
    }
    if (records.size() > 0xffffffffull) {
        throw std::runtime_error("too many render records for 32-bit record ids");
    }
    header.recordCount = records.size();

    header.At(Section::RecordEdge) = {offset, header.recordCount * 4u};
    const auto pagesOffset = AlignSection(offset + header.recordCount * 4u);
    auto pages = WritePages(hierarchy, records, output, pagesOffset, progress);
    Release(records);
    output.WriteAt(header.At(Section::RecordEdge).offset, pages.recordEdges.data(),
                   pages.recordEdges.size() * 4u);
    Release(pages.recordEdges);

    header.pageCount = pages.pages.size();
    header.groupCount = pages.groups.size();
    header.At(Section::Pages) = {pagesOffset, pages.pageBytes};
    offset = AlignSection(pagesOffset + pages.pageBytes);
    header.At(Section::PageDirectory) = {offset, header.pageCount * sizeof(PageEntry)};
    output.WriteAt(offset, pages.pages.data(), header.pageCount * sizeof(PageEntry));
    offset = AlignSection(offset + header.pageCount * sizeof(PageEntry));
    header.At(Section::PageGroups) = {offset, header.groupCount * sizeof(PageGroup)};
    output.WriteAt(offset, pages.groups.data(), header.groupCount * sizeof(PageGroup));

    std::vector<std::byte> headerBytes(kHeaderBytes);
    std::memcpy(headerBytes.data(), &header, sizeof(FileHeader));
    output.WriteAt(0, headerBytes.data(), headerBytes.size());
    output.Close();

    BuildResult result;
    result.verifiedCuts = Verify(hierarchy, errors, curve, temporary.Path(), progress);
    temporary.MoveTo(outputPath);

    result.path = outputPath;
    result.nodeCount = header.nodeCount;
    result.edgeCount = header.edgeCount;
    result.recordCount = header.recordCount;
    result.pageCount = header.pageCount;
    result.coverGapEdges = header.coverGapEdges;
    result.extraContextRecords = header.extraContextRecords;
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return result;
}

} // namespace chmv::streaming::lod
