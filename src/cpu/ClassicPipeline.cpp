#include "cpu/ClassicPipeline.h"

#include "cpu/CpuWorkQueue.h"
#include "cpu/CpuWorkers.h"
#include "data/ch/CHProjection.h"
#include "data/ch/CHTypes.h"
#include "streaming/lod/LodPageIo.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace chmv::cpu {
namespace {

namespace lod = streaming::lod;

using Clock = std::chrono::steady_clock;
constexpr std::size_t kEdgesPerTask = 1u << 16u;
constexpr std::size_t kItemsPerTask = 2048;
constexpr std::size_t kRootsPerTask = 256;
constexpr std::size_t kCellsPerTask = 64;
// Edges per finest grid cell aimed at, and the finest level at most (4^11 cells).
constexpr std::uint64_t kEdgesPerCell = 32;
constexpr std::uint32_t kMaxGridLevel = 11;

[[nodiscard]] double Milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

[[nodiscard]] std::size_t TaskCount(std::size_t items, std::size_t perTask) {
    return (items + perTask - 1u) / perTask;
}

[[nodiscard]] std::int64_t FloorDivide(std::int64_t value, std::int64_t divisor) {
    const auto quotient = value / divisor;
    return quotient * divisor > value ? quotient - 1 : quotient;
}

[[nodiscard]] std::vector<std::uint32_t> Gather(std::vector<std::vector<std::uint32_t>>& found) {
    std::size_t total = 0;
    for (const auto& part : found) {
        total += part.size();
    }
    std::vector<std::uint32_t> all;
    all.reserve(total);
    for (auto& part : found) {
        all.insert(all.end(), part.begin(), part.end());
        part.clear();
    }
    return all;
}

} // namespace

std::uint64_t ClassicGraph::EstimateBytes(const lod::FileHeader& header) {
    // Nodes: coordinates, and latitude/longitude while projecting. Edges: the CH columns kept
    // (23 B), birth order and grid (10 B), and one column being converted (or the road styles
    // being derived).
    return header.nodeCount * 24u + header.edgeCount * 37u + (64ull << 20u);
}

std::unique_ptr<ClassicGraph> ClassicGraph::Load(const lod::LodPageFile& file,
                                                 const std::stop_token& stop) {
    const auto check = [&] {
        if (stop.stop_requested()) {
            throw std::runtime_error("loading the whole CH was cancelled");
        }
    };
    std::unique_ptr<ClassicGraph> graph(new ClassicGraph());
    const auto& header = file.Header();
    const auto nodes = static_cast<std::size_t>(header.nodeCount);
    const auto edges = static_cast<std::size_t>(header.edgeCount);
    {
        // The projection the builder used, so coordinates match the pages to the unit.
        const auto latitude = file.ReadSection<double>(lod::Section::NodeLatitude);
        check();
        const auto longitude = file.ReadSection<double>(lod::Section::NodeLongitude);
        check();
        const data::CHProjection projection{header.centerLatitude, header.centerLongitude,
                                            header.longitudeScale, header.normalization};
        graph->x_.resize(nodes);
        graph->y_.resize(nodes);
        const auto ranges = lod::SplitRange(0u, nodes, kEdgesPerTask);
        lod::RunParallel(ranges.size(), [&](std::size_t task, unsigned) {
            for (auto i = ranges[task].first; i < ranges[task].second; ++i) {
                const auto point = projection.Project(latitude[i], longitude[i]);
                graph->x_[i] =
                    static_cast<std::int32_t>(std::round(point.x * lod::kFixedCoordinateScale));
                graph->y_[i] =
                    static_cast<std::int32_t>(std::round(point.y * lod::kFixedCoordinateScale));
            }
        });
    }
    check();
    graph->source_ = file.ReadSection<std::uint32_t>(lod::Section::EdgeSource);
    check();
    graph->target_ = file.ReadSection<std::uint32_t>(lod::Section::EdgeTarget);
    check();
    graph->childA_ = file.ReadSection<std::uint32_t>(lod::Section::EdgeChildA);
    check();
    graph->childB_ = file.ReadSection<std::uint32_t>(lod::Section::EdgeChildB);
    check();
    graph->birth_ = file.ReadSection<std::int16_t>(lod::Section::EdgeBirth);
    check();
    graph->death_ = file.ReadSection<std::int16_t>(lod::Section::EdgeDeath);
    check();
    const auto ranges = lod::SplitRange(0u, edges, kEdgesPerTask);
    {
        const auto error = file.ReadSection<float>(lod::Section::EdgeError);
        graph->errorCode_.resize(edges);
        lod::RunParallel(ranges.size(), [&](std::size_t task, unsigned) {
            for (auto e = ranges[task].first; e < ranges[task].second; ++e) {
                graph->errorCode_[e] = lod::EncodeGeometryError(error[e]);
            }
        });
    }
    check();
    {
        const auto type = file.ReadSection<std::int16_t>(lod::Section::EdgeType);
        graph->type_.resize(edges);
        lod::RunParallel(ranges.size(), [&](std::size_t task, unsigned) {
            for (auto e = ranges[task].first; e < ranges[task].second; ++e) {
                graph->type_[e] = static_cast<std::uint8_t>(data::RoadStyleIndexFromType(type[e]));
            }
        });
    }
    for (std::size_t e = 0; e < edges; ++e) {
        const auto inside = [&](std::uint32_t node) { return node < nodes; };
        const auto child = [&](std::uint32_t id) {
            return id == data::InvalidEdgeId || id < edges;
        };
        if (!inside(graph->source_[e]) || !inside(graph->target_[e]) || !child(graph->childA_[e]) ||
            !child(graph->childB_[e])) {
            throw std::runtime_error("CH layer of " + file.Path().filename().string() +
                                     " refers to nodes or edges that do not exist");
        }
    }
    check();
    graph->DeriveStyles();
    check();
    graph->BuildFilters();
    return graph;
}

// The road styles the page builder gives the records (EdgeStyle in LodPageFormat.h): originals
// from their type, shortcuts from their halves, children first.
void ClassicGraph::DeriveStyles() {
    const auto edges = type_.size();
    std::vector<float> length(edges, 0.0f);
    // 0: not yet, 1: waiting for its halves, 2: done.
    std::vector<std::uint8_t> state(edges, 0u);
    std::vector<std::uint32_t> stack;
    for (std::size_t first = 0; first < edges; ++first) {
        stack.push_back(static_cast<std::uint32_t>(first));
        while (!stack.empty()) {
            const auto e = stack.back();
            if (state[e] == 2u) {
                stack.pop_back();
                continue;
            }
            const auto a = childA_[e];
            const auto b = childB_[e];
            if (a == data::InvalidEdgeId) {
                const auto own = lod::OriginalStyle(type_[e], x_[source_[e]], y_[source_[e]],
                                                    x_[target_[e]], y_[target_[e]]);
                type_[e] = own.style;
                length[e] = own.length;
                state[e] = 2u;
                stack.pop_back();
            } else if (state[e] == 0u) {
                state[e] = 1u;
                for (const auto child : {a, b}) {
                    if (state[child] == 1u) {
                        throw std::runtime_error("CH layer contains a cycle of shortcuts");
                    }
                    if (state[child] == 0u) {
                        stack.push_back(child);
                    }
                }
            } else {
                const auto joined = lod::JoinStyles({type_[a], length[a]}, {type_[b], length[b]});
                type_[e] = joined.style;
                length[e] = joined.length;
                state[e] = 2u;
                stack.pop_back();
            }
        }
    }
}

std::uint64_t ClassicGraph::Bytes() const {
    return (x_.size() + y_.size()) * 4u +
           (source_.size() + target_.size() + childA_.size() + childB_.size()) * 4u +
           (birth_.size() + death_.size() + errorCode_.size()) * 2u + type_.size() +
           (byBirth_.size() + byBirthCount_.size()) * 4u + grid_.levelStart.size() * 8u +
           (grid_.cellStart.size() + grid_.entries.size()) * 4u + grid_.entryBirth.size() * 2u;
}

bool ClassicGraph::Alive(std::uint32_t edge, std::uint32_t level) const {
    const auto birth = birth_[edge];
    const auto death = death_[edge];
    return birth >= 0 && death >= 0 && death <= birth &&
           level >= static_cast<std::uint32_t>(death) && level <= static_cast<std::uint32_t>(birth);
}

std::array<std::int64_t, 4> ClassicGraph::GrownBox(std::uint32_t edge) const {
    const std::int64_t ax = x_[source_[edge]];
    const std::int64_t ay = y_[source_[edge]];
    const std::int64_t bx = x_[target_[edge]];
    const std::int64_t by = y_[target_[edge]];
    const std::int64_t bound = lod::ErrorBound(errorCode_[edge]);
    return {std::min(ax, bx) - bound, std::min(ay, by) - bound, std::max(ax, bx) + bound,
            std::max(ay, by) + bound};
}

// lod_page_traversal.glsl's nodeVisible on the CH edge itself.
bool ClassicGraph::Visible(std::uint32_t edge, const std::array<std::int32_t, 4>& cull) const {
    const auto box = GrownBox(edge);
    return box[2] >= cull[0] && box[0] <= cull[2] && box[3] >= cull[1] && box[1] <= cull[3];
}

void ClassicGraph::BuildFilters() {
    const auto edges = static_cast<std::uint32_t>(birth_.size());
    // Possible roots by birth level, highest first (a counting sort, stable by edge id).
    std::int32_t highest = -1;
    for (std::uint32_t e = 0; e < edges; ++e) {
        if (birth_[e] >= 0 && death_[e] >= 0 && death_[e] <= birth_[e]) {
            highest = std::max<std::int32_t>(highest, birth_[e]);
        }
    }
    byBirthCount_.assign(static_cast<std::size_t>(highest) + 2u, 0u);
    for (std::uint32_t e = 0; e < edges; ++e) {
        if (birth_[e] >= 0 && death_[e] >= 0 && death_[e] <= birth_[e]) {
            ++byBirthCount_[static_cast<std::size_t>(birth_[e])];
        }
    }
    for (auto level = static_cast<std::int64_t>(byBirthCount_.size()) - 2; level >= 0; --level) {
        byBirthCount_[static_cast<std::size_t>(level)] +=
            byBirthCount_[static_cast<std::size_t>(level) + 1u];
    }
    byBirth_.resize(byBirthCount_.empty() ? 0u : byBirthCount_[0]);
    std::vector<std::uint32_t> next(byBirthCount_.size(), 0u);
    for (std::size_t level = 0; level + 1u < byBirthCount_.size(); ++level) {
        next[level] = byBirthCount_[level + 1u];
    }
    for (std::uint32_t e = 0; e < edges; ++e) {
        if (birth_[e] >= 0 && death_[e] >= 0 && death_[e] <= birth_[e]) {
            byBirth_[next[static_cast<std::size_t>(birth_[e])]++] = e;
        }
    }

    // The grid over the node bounds; its size a power of two.
    auto& grid = grid_;
    if (x_.empty()) {
        return;
    }
    const auto [minX, maxX] = std::minmax_element(x_.begin(), x_.end());
    const auto [minY, maxY] = std::minmax_element(y_.begin(), y_.end());
    grid.originX = *minX;
    grid.originY = *minY;
    const auto extent = static_cast<std::uint64_t>(
        std::max<std::int64_t>(std::int64_t{*maxX} - *minX, std::int64_t{*maxY} - *minY) + 1);
    grid.shift = static_cast<std::uint32_t>(std::bit_width(std::bit_ceil(extent))) - 1u;
    std::uint32_t finest = 0;
    while (finest < kMaxGridLevel && finest < grid.shift &&
           (std::uint64_t{1} << (2u * (finest + 1u))) * kEdgesPerCell <= byBirth_.size()) {
        ++finest;
    }
    grid.levels = finest + 1u;
    grid.levelStart.resize(grid.levels + 1u);
    for (std::uint32_t level = 0; level < grid.levels; ++level) {
        grid.levelStart[level + 1u] = grid.levelStart[level] + (std::uint64_t{1} << (2u * level));
    }

    std::vector<std::uint32_t> cellOf(byBirth_.size());
    grid.cellStart.assign(static_cast<std::size_t>(grid.levelStart[grid.levels]) + 1u, 0u);
    for (std::size_t i = 0; i < byBirth_.size(); ++i) {
        const auto box = GrownBox(byBirth_[i]);
        const auto size = std::max(box[2] - box[0], box[3] - box[1]);
        auto level = finest;
        while (level > 0u && (std::int64_t{1} << (grid.shift - level)) < size) {
            --level;
        }
        const auto cells = (std::int64_t{1} << level) - 1;
        const auto cellShift = grid.shift - level;
        const auto cx =
            std::clamp(((box[0] + box[2]) / 2 - grid.originX) >> cellShift, std::int64_t{0}, cells);
        const auto cy =
            std::clamp(((box[1] + box[3]) / 2 - grid.originY) >> cellShift, std::int64_t{0}, cells);
        cellOf[i] = static_cast<std::uint32_t>(grid.levelStart[level] +
                                               static_cast<std::uint64_t>(cy << level) +
                                               static_cast<std::uint64_t>(cx));
        ++grid.cellStart[cellOf[i] + 1u];
    }
    for (std::size_t cell = 1; cell < grid.cellStart.size(); ++cell) {
        grid.cellStart[cell] += grid.cellStart[cell - 1u];
    }
    // Filled in birth order, so every cell stays ordered by birth.
    grid.entries.resize(byBirth_.size());
    grid.entryBirth.resize(byBirth_.size());
    std::vector<std::uint32_t> fill(grid.cellStart.begin(), grid.cellStart.end() - 1);
    for (std::size_t i = 0; i < byBirth_.size(); ++i) {
        const auto at = fill[cellOf[i]]++;
        grid.entries[at] = byBirth_[i];
        grid.entryBirth[at] = birth_[byBirth_[i]];
    }
}

CpuResult ClassicGraph::Run(CpuWorkers& workers, const lod::LodPageQuery& query, CpuFilter filter,
                            CpuExpansion expansion, bool cullSegments) const {
    CpuResult result;
    result.threads = workers.Threads();
    if (expansion == CpuExpansion::Intervals) {
        result.unavailable = "Dual interval needs the page records; the classic method has no "
                             "counterpart. Choose a DFS or BFS expansion.";
        return result;
    }
    const auto threads = workers.Threads();
    const auto level = query.level;
    const auto cull = lod::TraversalCullBox(query);
    std::vector<std::vector<std::uint32_t>> found(threads);

    // Roots.
    auto begin = Clock::now();
    std::vector<std::uint32_t> roots;
    if (filter == CpuFilter::TwoPass) {
        // Count per block of edges, then every block writes its roots to its own place.
        const auto blocks = TaskCount(birth_.size(), kEdgesPerTask);
        std::vector<std::size_t> offsets(blocks + 1u, 0u);
        workers.Run(blocks, [&](std::size_t block, unsigned) {
            const auto last = std::min(birth_.size(), (block + 1u) * kEdgesPerTask);
            std::size_t count = 0;
            for (auto e = block * kEdgesPerTask; e < last; ++e) {
                count += Alive(static_cast<std::uint32_t>(e), level) ? 1u : 0u;
            }
            offsets[block + 1u] = count;
        });
        for (std::size_t block = 0; block < blocks; ++block) {
            offsets[block + 1u] += offsets[block];
        }
        roots.resize(offsets.back());
        workers.Run(blocks, [&](std::size_t block, unsigned) {
            const auto last = std::min(birth_.size(), (block + 1u) * kEdgesPerTask);
            auto at = offsets[block];
            for (auto e = block * kEdgesPerTask; e < last; ++e) {
                if (Alive(static_cast<std::uint32_t>(e), level)) {
                    roots[at++] = static_cast<std::uint32_t>(e);
                }
            }
        });
    } else if (filter == CpuFilter::FullScan) {
        workers.Run(TaskCount(birth_.size(), kEdgesPerTask),
                    [&](std::size_t task, unsigned worker) {
                        const auto last = std::min(birth_.size(), (task + 1u) * kEdgesPerTask);
                        for (auto e = task * kEdgesPerTask; e < last; ++e) {
                            if (Alive(static_cast<std::uint32_t>(e), level)) {
                                found[worker].push_back(static_cast<std::uint32_t>(e));
                            }
                        }
                    });
    } else if (filter == CpuFilter::BirthOrdered) {
        const std::size_t count = level < byBirthCount_.size() ? byBirthCount_[level] : 0u;
        workers.Run(TaskCount(count, kEdgesPerTask), [&](std::size_t task, unsigned worker) {
            const auto last = std::min(count, (task + 1u) * kEdgesPerTask);
            for (auto i = task * kEdgesPerTask; i < last; ++i) {
                const auto e = byBirth_[i];
                if (static_cast<std::uint32_t>(death_[e]) <= level) {
                    found[worker].push_back(e);
                }
            }
        });
    } else if (grid_.levels != 0u) {
        // Cells whose edges can reach the cull box (a cell's edges stay within half a cell of
        // it; one more cell on each side keeps the integer bounds simple).
        std::vector<std::uint32_t> cells;
        for (std::uint32_t gridLevel = 0; gridLevel < grid_.levels; ++gridLevel) {
            const auto size = std::int64_t{1} << (grid_.shift - gridLevel);
            const auto last = (std::int64_t{1} << gridLevel) - 1;
            const auto x0 =
                std::clamp(FloorDivide(cull[0] - grid_.originX, size) - 2, std::int64_t{0}, last);
            const auto x1 =
                std::clamp(FloorDivide(cull[2] - grid_.originX, size) + 1, std::int64_t{0}, last);
            const auto y0 =
                std::clamp(FloorDivide(cull[1] - grid_.originY, size) - 2, std::int64_t{0}, last);
            const auto y1 =
                std::clamp(FloorDivide(cull[3] - grid_.originY, size) + 1, std::int64_t{0}, last);
            for (auto cy = y0; cy <= y1; ++cy) {
                for (auto cx = x0; cx <= x1; ++cx) {
                    cells.push_back(static_cast<std::uint32_t>(
                        grid_.levelStart[gridLevel] + static_cast<std::uint64_t>(cy << gridLevel) +
                        static_cast<std::uint64_t>(cx)));
                }
            }
        }
        const auto signedLevel = static_cast<std::int32_t>(std::min<std::uint32_t>(level, 0x7fffu));
        workers.Run(TaskCount(cells.size(), kCellsPerTask), [&](std::size_t task, unsigned worker) {
            const auto last = std::min(cells.size(), (task + 1u) * kCellsPerTask);
            for (auto i = task * kCellsPerTask; i < last; ++i) {
                const auto from = grid_.entryBirth.begin() + grid_.cellStart[cells[i]];
                const auto to = grid_.entryBirth.begin() + grid_.cellStart[cells[i] + 1u];
                const auto end = std::partition_point(
                    from, to, [&](std::int16_t birth) { return birth >= signedLevel; });
                for (auto at = from; at != end; ++at) {
                    const auto e =
                        grid_.entries[static_cast<std::size_t>(at - grid_.entryBirth.begin())];
                    if (static_cast<std::uint32_t>(death_[e]) <= level && Visible(e, cull)) {
                        found[worker].push_back(e);
                    }
                }
            }
        });
    }
    if (filter != CpuFilter::TwoPass) {
        roots = Gather(found);
    }
    result.candidates = roots.size();
    result.filterMs = Milliseconds(begin);

    // Expansion: draw an edge once it is accurate enough or original, else go to its children.
    begin = Clock::now();
    const auto threshold = query.threshold;
    const auto expands = [&](std::uint32_t e) {
        return errorCode_[e] > threshold && childA_[e] != data::InvalidEdgeId &&
               childB_[e] != data::InvalidEdgeId;
    };
    // An edge to draw, unless culling drops it.
    const auto draw = [&](std::uint32_t e, unsigned worker) {
        if (!cullSegments ||
            ChordInCullBox(x_[source_[e]], y_[source_[e]], x_[target_[e]], y_[target_[e]], cull)) {
            found[worker].push_back(e);
        }
    };
    std::vector<std::uint32_t> drawn;
    if (expansion == CpuExpansion::Roots) {
        workers.Run(TaskCount(roots.size(), kItemsPerTask), [&](std::size_t task, unsigned worker) {
            const auto last = std::min(roots.size(), (task + 1u) * kItemsPerTask);
            for (auto i = task * kItemsPerTask; i < last; ++i) {
                if (errorCode_[roots[i]] <= threshold && Visible(roots[i], cull)) {
                    draw(roots[i], worker);
                }
            }
        });
        drawn = Gather(found);
    } else if (expansion == CpuExpansion::Dfs) {
        std::vector<std::vector<std::uint32_t>> stacks(threads);
        workers.Run(TaskCount(roots.size(), kRootsPerTask), [&](std::size_t task, unsigned worker) {
            auto& stack = stacks[worker];
            const auto last = std::min(roots.size(), (task + 1u) * kRootsPerTask);
            for (auto i = task * kRootsPerTask; i < last; ++i) {
                stack.push_back(roots[i]);
                while (!stack.empty()) {
                    const auto e = stack.back();
                    stack.pop_back();
                    if (!Visible(e, cull)) {
                        continue;
                    }
                    if (!expands(e)) {
                        draw(e, worker);
                        continue;
                    }
                    stack.push_back(childB_[e]);
                    stack.push_back(childA_[e]);
                }
            }
        });
        drawn = Gather(found);
    } else if (expansion == CpuExpansion::Queue) {
        // Room for a few times the roots, at most 2^24 edges; the rest is walked by the worker
        // that finds it.
        constexpr std::size_t kMaxQueue = std::size_t{1} << 24u;
        RunWorkQueue(workers, roots, std::min(roots.size() * 4u + 4096u, kMaxQueue),
                     [&](std::uint32_t e, unsigned worker, std::array<std::uint32_t, 2>& next) {
                         if (!Visible(e, cull)) {
                             return 0;
                         }
                         if (!expands(e)) {
                             draw(e, worker);
                             return 0;
                         }
                         next = {childA_[e], childB_[e]};
                         return 2;
                     });
        drawn = Gather(found);
    } else {
        std::vector<std::vector<std::uint32_t>> next(threads);
        auto frontier = roots;
        while (!frontier.empty()) {
            workers.Run(
                TaskCount(frontier.size(), kItemsPerTask), [&](std::size_t task, unsigned worker) {
                    const auto last = std::min(frontier.size(), (task + 1u) * kItemsPerTask);
                    for (auto i = task * kItemsPerTask; i < last; ++i) {
                        const auto e = frontier[i];
                        if (!Visible(e, cull)) {
                            continue;
                        }
                        if (!expands(e)) {
                            draw(e, worker);
                            continue;
                        }
                        next[worker].push_back(childA_[e]);
                        next[worker].push_back(childB_[e]);
                    }
                });
            frontier = Gather(next);
        }
        drawn = Gather(found);
    }
    result.expandMs = Milliseconds(begin);

    begin = Clock::now();
    result.segments.resize(drawn.size());
    result.edges = std::move(drawn);
    workers.Run(TaskCount(result.edges.size(), kItemsPerTask), [&](std::size_t task, unsigned) {
        const auto last = std::min(result.edges.size(), (task + 1u) * kItemsPerTask);
        for (auto i = task * kItemsPerTask; i < last; ++i) {
            const auto e = result.edges[i];
            auto& segment = result.segments[i];
            segment.ax = x_[source_[e]];
            segment.ay = y_[source_[e]];
            segment.bx = x_[target_[e]];
            segment.by = y_[target_[e]];
            segment.type = type_[e];
        }
    });
    result.outputMs = Milliseconds(begin);
    return result;
}

} // namespace chmv::cpu
