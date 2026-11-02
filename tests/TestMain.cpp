#include "cpu/ClassicPipeline.h"
#include "cpu/CpuRuntime.h"
#include "cpu/CpuWorkers.h"
#include "cpu/PagePipeline.h"
#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHTypes.h"
#include "renderer/RoadStyle.h"
#include "renderer/ToleranceController.h"
#include "streaming/lod/LodPageBuilder.h"
#include "streaming/lod/LodPageCompression.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPageIo.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/lod/LodPageReader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Five nodes along a bent road: originals 0-3 at level 0, shortcuts 4 (0-2) and 5 (2-4) at
// level 1, shortcut 6 (0-4) above them. Extra lines add shortcuts or change ranges.
struct LodTestGraph {
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;

    LodTestGraph(const std::vector<std::string>& extraEdges, const std::string& ranges) {
        const auto base = std::filesystem::temp_directory_path() / "chmv_lod_test";
        graphPath = base.string() + ".sch";
        rangesPath = base.string() + ".sch.ranges";
        std::ofstream graph(graphPath);
        graph << "# bent road\n5\n" << 7u + extraEdges.size() << "\n";
        graph << "0 100 48.0000000 9.0000000 0 4\n";
        graph << "1 101 48.0100000 9.0100000 0 0\n";
        graph << "2 102 48.0000000 9.0200000 0 2\n";
        graph << "3 103 48.0100000 9.0300000 0 1\n";
        graph << "4 104 48.0000000 9.0400000 0 3\n";
        graph << "0 1 10 1 50 -1 -1\n1 2 10 1 50 -1 -1\n2 3 10.0 2 50 -1 -1\n";
        graph << "3 4 10 2 50 -1 -1\n0 2 20 1 50 0 1\n2 4 20 2 50 2 3\n0 4 40 1 50 4 5\n";
        for (const auto& edge : extraEdges) {
            graph << edge << '\n';
        }
        std::ofstream(rangesPath) << ranges;
    }
    ~LodTestGraph() {
        std::filesystem::remove(graphPath);
        std::filesystem::remove(rangesPath);
        std::filesystem::remove(chmv::streaming::lod::LodPageFile::DefaultPath(graphPath));
    }
};

// A straight road of `edges` original edges, all drawn at every level: enough records for
// several compressed blocks and page frames.
struct ChainGraph {
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;

    explicit ChainGraph(std::uint32_t edges) {
        const auto base = std::filesystem::temp_directory_path() / "chmv_chain_test";
        graphPath = base.string() + ".sch";
        rangesPath = base.string() + ".sch.ranges";
        std::ofstream graph(graphPath);
        graph << "# chain\n" << edges + 1u << "\n" << edges << "\n";
        for (std::uint32_t node = 0; node <= edges; ++node) {
            graph << node << ' ' << 1000u + node << " 48." << 1000000u + node * 37u << " 9."
                  << 1000000u + (node * node) % 9973u << " 0 0\n";
        }
        std::ofstream ranges(rangesPath);
        for (std::uint32_t edge = 0; edge < edges; ++edge) {
            graph << edge << ' ' << edge + 1u << ' ' << 1u + edge % 7u << " 1 50 -1 -1\n";
            ranges << edge << " 0 0\n";
        }
    }
    ~ChainGraph() {
        std::filesystem::remove(graphPath);
        std::filesystem::remove(rangesPath);
        std::filesystem::remove(chmv::streaming::lod::LodPageFile::DefaultPath(graphPath));
    }
};

// Many units of a bent shortcut S (0-2 via 1) below a shortcut P (0-3 via S and 2-3) whose chord
// is the same line, so both have the same error: S is never drawn, only walked through, and
// whole pages hold nothing but such records.
struct WalkOnlyGraph {
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;

    explicit WalkOnlyGraph(std::uint32_t units) {
        const auto base = std::filesystem::temp_directory_path() / "chmv_walk_only_test";
        graphPath = base.string() + ".sch";
        rangesPath = base.string() + ".sch.ranges";
        std::ofstream graph(graphPath);
        graph << units * 4u << '\n' << units * 5u << '\n' << std::fixed << std::setprecision(7);
        for (std::uint32_t node = 0; node < units * 4u; ++node) {
            const auto k = node % 4u;
            const std::uint32_t level = k == 1u ? 0u : k == 2u ? 1u : 2u;
            graph << node << ' ' << 1000u + node << ' ' << (k == 1u ? 48.0005 : 48.0) << ' '
                  << 9.0 + node * 0.001 << " 0 " << level << '\n';
        }
        std::ofstream ranges(rangesPath);
        for (std::uint32_t unit = 0; unit < units; ++unit) {
            const auto node = unit * 4u;
            const auto edge = unit * 5u;
            graph << node << ' ' << node + 1u << " 1 1 50 -1 -1\n";
            graph << node + 1u << ' ' << node + 2u << " 1 1 50 -1 -1\n";
            graph << node + 2u << ' ' << node + 3u << " 1 1 50 -1 -1\n";
            graph << node << ' ' << node + 2u << " 2 1 50 " << edge << ' ' << edge + 1u << '\n';
            graph << node << ' ' << node + 3u << " 3 1 50 " << edge + 3u << ' ' << edge + 2u
                  << '\n';
            ranges << edge << " 0 0\n"
                   << edge + 1u << " 0 0\n"
                   << edge + 2u << " 0 0\n"
                   << edge + 3u << " -1 -1\n"
                   << edge + 4u << " 2 1\n";
        }
    }
    ~WalkOnlyGraph() {
        std::filesystem::remove(graphPath);
        std::filesystem::remove(rangesPath);
        std::filesystem::remove(chmv::streaming::lod::LodPageFile::DefaultPath(graphPath));
    }
};

// Random roads: chains of 2^depth originals along random walks, each below a binary tree of
// shortcuts, some of them drawn one level longer. Many pages, real traversals, views anywhere.
struct RandomRoadsGraph {
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;

    RandomRoadsGraph(std::uint32_t seed, std::uint32_t chains, std::uint32_t depth) {
        const auto base = std::filesystem::temp_directory_path() / "chmv_random_roads_test";
        graphPath = base.string() + ".sch";
        rangesPath = base.string() + ".sch.ranges";
        std::mt19937 random(seed);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const std::uint32_t length = 1u << depth;
        std::ofstream graph(graphPath);
        graph << "# random roads\n"
              << chains * (length + 1u) << '\n'
              << chains * (2u * length - 1u) << '\n'
              << std::fixed << std::setprecision(7);
        for (std::uint32_t chain = 0; chain < chains; ++chain) {
            double lat = 48.0 + unit(random) * 0.5;
            double lon = 9.0 + unit(random) * 0.7;
            for (std::uint32_t i = 0; i <= length; ++i) {
                const auto node = chain * (length + 1u) + i;
                const auto level = i == 0u || i == length
                                       ? depth + 1u
                                       : static_cast<std::uint32_t>(std::countr_zero(i));
                graph << node << ' ' << 1000u + node << ' ' << lat << ' ' << lon << " 0 "
                      << level << '\n';
                lat += (unit(random) - 0.5) * 0.004;
                lon += (unit(random) - 0.3) * 0.004;
            }
        }
        std::ofstream ranges(rangesPath);
        std::uint32_t edge = 0;
        const auto write = [&](std::uint32_t from, std::uint32_t to, const std::string& children,
                               int birth, int death) {
            graph << from << ' ' << to << " 10 " << 1u + edge % 5u << " 50 " << children << '\n';
            ranges << edge << ' ' << birth << ' ' << death << '\n';
            return edge++;
        };
        for (std::uint32_t chain = 0; chain < chains; ++chain) {
            const auto first = chain * (length + 1u);
            // (edge, from, to) of the current level of the tree.
            std::vector<std::array<std::uint32_t, 3>> current;
            for (std::uint32_t i = 0; i < length; ++i) {
                current.push_back({write(first + i, first + i + 1u, "-1 -1", 0, 0), first + i,
                                   first + i + 1u});
            }
            for (std::uint32_t height = 1; height <= depth; ++height) {
                std::vector<std::array<std::uint32_t, 3>> next;
                for (std::size_t k = 0; k + 1u < current.size(); k += 2u) {
                    const auto& a = current[k];
                    const auto& b = current[k + 1u];
                    auto birth = static_cast<int>(height == depth ? depth + 2u : height);
                    if (height < depth && unit(random) < 0.15) {
                        birth = static_cast<int>(height) + 1;
                    }
                    next.push_back({write(a[1], b[2],
                                          std::to_string(a[0]) + ' ' + std::to_string(b[0]),
                                          birth, static_cast<int>(height)),
                                    a[1], b[2]});
                }
                current = std::move(next);
            }
        }
    }
    ~RandomRoadsGraph() {
        std::filesystem::remove(graphPath);
        std::filesystem::remove(rangesPath);
        std::filesystem::remove(chmv::streaming::lod::LodPageFile::DefaultPath(graphPath));
    }
};

// Edges drawn by the page file, as a sorted multiset.
std::vector<std::uint32_t> LodPageCut(const chmv::streaming::lod::LodPageFile& file,
                                      std::uint32_t level, std::uint32_t threshold) {
    namespace lod = chmv::streaming::lod;
    const auto words = file.ReadPageWords({});
    const auto recordEdges = file.ReadSection<std::uint32_t>(lod::Section::RecordEdge);
    std::vector<std::uint32_t> edges;
    for (std::uint32_t p = 0; p < file.Pages().size(); ++p) {
        const auto& page = file.Pages()[p];
        const auto* pageWords = words.data() + file.PageWordOffset(p);
        for (std::uint32_t i = 0; i < page.recordCount; ++i) {
            const auto* record = pageWords + lod::kPageHeaderWords + i * lod::kRecordWords;
            if (lod::RecordDrawn(record[1], record[2], record[3], level, threshold)) {
                edges.push_back(recordEdges[page.firstRecord + i]);
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

// The same cut by plain DFS over the CH layer of the file.
std::vector<std::uint32_t> DfsCut(const chmv::streaming::lod::LodPageFile& file,
                                  std::uint32_t level, std::uint32_t threshold) {
    namespace lod = chmv::streaming::lod;
    const auto childA = file.ReadSection<std::uint32_t>(lod::Section::EdgeChildA);
    const auto childB = file.ReadSection<std::uint32_t>(lod::Section::EdgeChildB);
    const auto birth = file.ReadSection<std::int16_t>(lod::Section::EdgeBirth);
    const auto death = file.ReadSection<std::int16_t>(lod::Section::EdgeDeath);
    const auto error = file.ReadSection<float>(lod::Section::EdgeError);
    std::vector<std::uint32_t> edges;
    std::vector<std::uint32_t> stack;
    const auto r = static_cast<std::int32_t>(level);
    for (std::uint32_t root = 0; root < childA.size(); ++root) {
        if (death[root] < 0 || birth[root] < 0 || r < death[root] || r > birth[root]) {
            continue;
        }
        stack.push_back(root);
        while (!stack.empty()) {
            const auto e = stack.back();
            stack.pop_back();
            if (childA[e] != chmv::data::InvalidEdgeId &&
                lod::EncodeGeometryError(error[e]) > threshold) {
                stack.push_back(childA[e]);
                stack.push_back(childB[e]);
            } else {
                edges.push_back(e);
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

void RequireLodCutsMatchDfs(const chmv::streaming::lod::LodPageFile& file, const char* message) {
    const auto codes = {std::uint32_t{0}, std::uint32_t{chmv::streaming::lod::kErrorCodeMax}};
    for (std::uint32_t level = 0; level <= 3; ++level) {
        for (const auto threshold : codes) {
            Require(LodPageCut(file, level, threshold) == DfsCut(file, level, threshold),
                    message);
        }
    }
}

// The same cut by walking the hierarchy section from the roots, as the traversal kernels do;
// every page reached must be found by the planner through the visit directory.
std::vector<std::uint32_t> TraversalCut(const chmv::streaming::lod::LodPageFile& file,
                                        std::uint32_t level, std::uint32_t threshold) {
    namespace lod = chmv::streaming::lod;
    const auto words = file.ReadPageWords({});
    const auto children = file.ReadChildrenWords({});
    const auto recordEdges = file.ReadSection<std::uint32_t>(lod::Section::RecordEdge);
    lod::LodPageQuery everywhere{-1e9f, -1e9f, 1e9f, 1e9f, level, threshold};
    std::vector<std::uint32_t> planned;
    lod::CollectLodPages(file.VisitGroups(), file.VisitPages(), everywhere, planned);
    const auto record = [&](std::uint32_t ref) {
        return words.data() + file.PageWordOffset(ref >> 8u) + lod::kPageHeaderWords +
               (ref & 0xffu) * lod::kRecordWords;
    };
    std::vector<std::uint32_t> edges;
    std::vector<std::uint32_t> stack;
    for (std::uint32_t p = 0; p < file.Pages().size(); ++p) {
        for (std::uint32_t i = 0; i < file.Pages()[p].recordCount; ++i) {
            const auto levels = record((p << 8u) | i)[1];
            if (level < (levels & 0xffffu) || level > (levels >> 16u)) {
                continue;
            }
            stack.push_back((p << 8u) | i);
            while (!stack.empty()) {
                const auto ref = stack.back();
                stack.pop_back();
                const auto page = ref >> 8u;
                const auto local = ref & 0xffu;
                Require(std::find(planned.begin(), planned.end(), page) != planned.end(),
                        "the visit directory misses a page the traversal reaches");
                const auto* block = children.data() + file.ChildrenWordOffset(page);
                const bool marked = ((block[local >> 5u] >> (local & 31u)) & 1u) != 0u;
                if (marked && (record(ref)[3] & 0xffffu) > threshold) {
                    std::uint32_t rank = 0;
                    for (std::uint32_t b = 0; b < local; ++b) {
                        rank += (block[b >> 5u] >> (b & 31u)) & 1u;
                    }
                    stack.push_back(block[lod::kChildMaskWords + 2u * rank]);
                    stack.push_back(block[lod::kChildMaskWords + 2u * rank + 1u]);
                } else {
                    edges.push_back(recordEdges[file.Pages()[page].firstRecord + local]);
                }
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

void RequireTraversalsMatchDfs(const chmv::streaming::lod::LodPageFile& file,
                               const char* message) {
    const auto codes = {std::uint32_t{0}, std::uint32_t{chmv::streaming::lod::kErrorCodeMax}};
    for (std::uint32_t level = 0; level <= 3; ++level) {
        for (const auto threshold : codes) {
            Require(TraversalCut(file, level, threshold) == DfsCut(file, level, threshold),
                    message);
        }
    }
}

// Roots drawn as they are: alive at the level and accurate enough.
std::vector<std::uint32_t> RootsCut(const chmv::streaming::lod::LodPageFile& file,
                                    std::uint32_t level, std::uint32_t threshold) {
    namespace lod = chmv::streaming::lod;
    const auto birth = file.ReadSection<std::int16_t>(lod::Section::EdgeBirth);
    const auto death = file.ReadSection<std::int16_t>(lod::Section::EdgeDeath);
    const auto error = file.ReadSection<float>(lod::Section::EdgeError);
    const auto r = static_cast<std::int32_t>(level);
    std::vector<std::uint32_t> edges;
    for (std::uint32_t e = 0; e < birth.size(); ++e) {
        if (death[e] >= 0 && birth[e] >= 0 && r >= death[e] && r <= birth[e] &&
            lod::EncodeGeometryError(error[e]) <= threshold) {
            edges.push_back(e);
        }
    }
    return edges;
}

// Every CPU filter and expansion, on pages read from the file and from RAM images, and the
// classic method on the whole CH, with one and with several threads: the same edges as DFS
// (everywhere), and the page and classic methods draw the same segments.
void RequireCpuPassesMatchDfs(const std::shared_ptr<const chmv::streaming::lod::LodPageFile>& file,
                              const char* message) {
    namespace lod = chmv::streaming::lod;
    namespace cpu = chmv::cpu;
    cpu::CpuPageStore diskStore(file, nullptr, nullptr);
    cpu::CpuPageStore ramStore(file,
                               std::make_shared<const std::vector<std::uint32_t>>(
                                   file->ReadPageWords({})),
                               std::make_shared<const std::vector<std::uint32_t>>(
                                   file->ReadChildrenWords({})));
    const std::stop_source stop;
    const auto classic = cpu::ClassicGraph::Load(*file, stop.get_token());
    const auto sorted = [](std::vector<std::uint32_t> edges) {
        std::sort(edges.begin(), edges.end());
        return edges;
    };
    const auto segments = [](const cpu::CpuResult& result) {
        std::vector<std::array<std::int64_t, 6>> keyed;
        for (std::size_t i = 0; i < result.segments.size(); ++i) {
            const auto& s = result.segments[i];
            keyed.push_back({result.edges[i], s.ax, s.ay, s.bx, s.by, s.type});
        }
        std::sort(keyed.begin(), keyed.end());
        return keyed;
    };
    for (const unsigned threads : {1u, 3u}) {
        cpu::CpuWorkers workers(threads);
        for (std::uint32_t level = 0; level <= 3u; ++level) {
            for (const std::uint32_t threshold :
                 {0u, 27000u, 30000u, std::uint32_t{lod::kErrorCodeMax}}) {
                const auto expected = DfsCut(*file, level, threshold);
                const auto roots = RootsCut(*file, level, threshold);
                lod::LodPageQuery query{-1e9f, -1e9f, 1e9f, 1e9f, level, threshold};
                for (const auto filter : {cpu::CpuFilter::FullScan, cpu::CpuFilter::BirthOrdered,
                                          cpu::CpuFilter::Culling}) {
                    for (const auto expansion :
                         {cpu::CpuExpansion::Roots, cpu::CpuExpansion::Intervals,
                          cpu::CpuExpansion::Dfs, cpu::CpuExpansion::Bfs}) {
                        const auto& want =
                            expansion == cpu::CpuExpansion::Roots ? roots : expected;
                        const auto fromDisk =
                            cpu::RunPagePass(diskStore, workers, query, filter, expansion);
                        const auto fromRam =
                            cpu::RunPagePass(ramStore, workers, query, filter, expansion);
                        Require(sorted(fromDisk.edges) == want && sorted(fromRam.edges) == want &&
                                    fromDisk.missingPages == fromRam.missingPages,
                                message);
                        const auto whole = classic->Run(workers, query, filter, expansion);
                        if (expansion == cpu::CpuExpansion::Intervals) {
                            Require(!whole.unavailable.empty(), message);
                        } else {
                            Require(sorted(whole.edges) == want &&
                                        segments(whole) == segments(fromDisk),
                                    message);
                        }
                    }
                }
            }
        }
    }
}

// Random views anywhere on random roads: every CPU pass draws, inside the view, exactly what a
// plain depth-first walk of the whole CH draws there; from disk and from RAM, with one and
// several threads, in mixed order (the store then holds pages fetched for other passes).
// A damaged page or children block is left out alike from disk and from RAM.
void TestCpuPartialViews() {
    namespace lod = chmv::streaming::lod;
    namespace cpu = chmv::cpu;
    constexpr std::uint32_t kDepth = 6;
    const RandomRoadsGraph source(7u, 40u, kDepth);
    const auto filePath = lod::LodPageFile::DefaultPath(source.graphPath);
    (void)lod::LodPageBuilder::Build(source.graphPath, source.rangesPath, filePath);
    const auto file = lod::LodPageFile::Open(filePath);
    Require(file->Pages().size() > 10u, "the random roads span many pages");
    const std::stop_source stop;
    const auto classic = cpu::ClassicGraph::Load(*file, stop.get_token());
    const auto ramStore = [](const std::shared_ptr<const lod::LodPageFile>& pages) {
        return std::make_unique<cpu::CpuPageStore>(
            pages, std::make_shared<const std::vector<std::uint32_t>>(pages->ReadPageWords({})),
            std::make_shared<const std::vector<std::uint32_t>>(pages->ReadChildrenWords({})));
    };
    cpu::CpuPageStore disk(file, nullptr, nullptr);
    const auto ram = ramStore(file);
    cpu::CpuWorkers one(1);
    cpu::CpuWorkers four(4);

    const lod::LodPageQuery everywhere{-1e9f, -1e9f, 1e9f, 1e9f, 0u, 0u};
    const auto whole = classic->Run(one, everywhere, cpu::CpuFilter::FullScan,
                                    cpu::CpuExpansion::Dfs);
    double minX = 1e9;
    double minY = 1e9;
    double maxX = -1e9;
    double maxY = -1e9;
    for (const auto& segment : whole.segments) {
        for (const auto& [x, y] :
             {std::pair{segment.ax, segment.ay}, std::pair{segment.bx, segment.by}}) {
            minX = std::min(minX, x / lod::kFixedCoordinateScale);
            maxX = std::max(maxX, x / lod::kFixedCoordinateScale);
            minY = std::min(minY, y / lod::kFixedCoordinateScale);
            maxY = std::max(maxY, y / lod::kFixedCoordinateScale);
        }
    }
    std::mt19937 random(11u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    constexpr std::array<std::uint32_t, 7> kThresholds{0u,     20000u, 24000u, 26000u,
                                                       28000u, 30000u, lod::kErrorCodeMax};
    std::vector<int> combinations(12);
    for (int k = 0; k < 12; ++k) {
        combinations[static_cast<std::size_t>(k)] = k;
    }
    for (std::uint32_t round = 0; round < 60u; ++round) {
        const auto level = static_cast<std::uint32_t>(random() % (kDepth + 3u));
        const auto threshold = kThresholds[random() % kThresholds.size()];
        const double width = (maxX - minX) * std::pow(10.0, -3.0 * unit(random));
        const double height = width * (0.3 + unit(random));
        const double x = minX + (maxX - minX) * (unit(random) * 1.2 - 0.1);
        const double y = minY + (maxY - minY) * (unit(random) * 1.2 - 0.1);
        const std::array<double, 4> view{x - width / 2.0, y - height / 2.0, x + width / 2.0,
                                         y + height / 2.0};
        const double margin = width * 0.01;
        const lod::LodPageQuery query{static_cast<float>(view[0] - margin),
                                      static_cast<float>(view[1] - margin),
                                      static_cast<float>(view[2] + margin),
                                      static_cast<float>(view[3] + margin), level, threshold};
        const lod::LodPageQuery all{-1e9f, -1e9f, 1e9f, 1e9f, level, threshold};
        const auto dfs = classic->Run(one, all, cpu::CpuFilter::FullScan, cpu::CpuExpansion::Dfs);
        const auto roots =
            classic->Run(one, all, cpu::CpuFilter::FullScan, cpu::CpuExpansion::Roots);
        auto dfsEdges = dfs.edges;
        std::sort(dfsEdges.begin(), dfsEdges.end());
        Require(dfsEdges == DfsCut(*file, level, threshold),
                "the unculled classic walk is the DFS cut");
        const auto wantDfs = cpu::EdgesInView(dfs.segments, dfs.edges, view);
        const auto wantRoots = cpu::EdgesInView(roots.segments, roots.edges, view);
        std::shuffle(combinations.begin(), combinations.end(), random);
        for (const int k : combinations) {
            const auto filter = static_cast<cpu::CpuFilter>(k / 4);
            const auto expansion = static_cast<cpu::CpuExpansion>(k % 4);
            const auto& want = expansion == cpu::CpuExpansion::Roots ? wantRoots : wantDfs;
            auto& workers = (round & 1u) != 0u ? four : one;
            for (auto* store : {&disk, ram.get()}) {
                const auto result = cpu::RunPagePass(*store, workers, query, filter, expansion);
                Require(result.unavailable.empty() &&
                            cpu::EdgesInView(result.segments, result.edges, view) == want,
                        "a page pass draws the view as the whole-CH walk does");
            }
            if (expansion != cpu::CpuExpansion::Intervals) {
                const auto result = classic->Run(workers, query, filter, expansion);
                Require(cpu::EdgesInView(result.segments, result.edges, view) == want,
                        "a classic pass draws the view as the whole-CH walk does");
            }
        }
    }

    // Segments per tolerance step: the step a pass draws is counted exactly (other steps may
    // need pages this pass did not plan).
    for (std::uint32_t level = 0; level <= kDepth + 2u; ++level) {
        constexpr std::uint32_t kBase = 24000u;
        for (int step = 0; step < chmv::renderer::kToleranceSteps; ++step) {
            const auto threshold =
                kBase + static_cast<std::uint32_t>(step) * chmv::renderer::kCodesPerToleranceStep;
            const lod::LodPageQuery all{-1e9f, -1e9f, 1e9f, 1e9f, level, threshold};
            const auto result = cpu::RunPagePass(disk, four, all, cpu::CpuFilter::FullScan,
                                                 cpu::CpuExpansion::Intervals, kBase);
            Require(result.stepCounts.has_value() &&
                        (*result.stepCounts)[static_cast<std::size_t>(step)] ==
                            result.segments.size(),
                    "the segments of every tolerance step are counted");
        }
    }

    // A damaged page, then a damaged children block of another page.
    auto damagedPath = filePath;
    damagedPath += ".damaged";
    const auto damage = [&](lod::StoredRange range) {
        std::filesystem::copy_file(filePath, damagedPath,
                                   std::filesystem::copy_options::overwrite_existing);
        std::fstream io(damagedPath, std::ios::in | std::ios::out | std::ios::binary);
        io.seekg(static_cast<std::streamoff>(range.offset + range.bytes / 2u));
        char byte = 0;
        io.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0x5a);
        io.seekp(static_cast<std::streamoff>(range.offset + range.bytes / 2u));
        io.write(&byte, 1);
    };
    const auto victim = static_cast<std::uint32_t>(file->Pages().size() / 2u);
    for (const bool children : {false, true}) {
        damage(children ? file->ChildrenStorage(victim + 1u) : file->PageStorage(victim));
        const auto damaged = lod::LodPageFile::Open(damagedPath);
        cpu::CpuPageStore damagedDisk(damaged, nullptr, nullptr);
        const auto damagedRam = ramStore(damaged);
        bool lost = false;
        for (std::uint32_t level = 0; level <= kDepth + 2u; ++level) {
            for (const auto threshold : kThresholds) {
                const lod::LodPageQuery all{-1e9f, -1e9f, 1e9f, 1e9f, level, threshold};
                for (int k = 0; k < 12; ++k) {
                    const auto filter = static_cast<cpu::CpuFilter>(k / 4);
                    const auto expansion = static_cast<cpu::CpuExpansion>(k % 4);
                    auto fromDisk = cpu::RunPagePass(damagedDisk, four, all, filter, expansion);
                    auto fromRam = cpu::RunPagePass(*damagedRam, four, all, filter, expansion);
                    const auto intact = cpu::RunPagePass(disk, four, all, filter, expansion);
                    std::sort(fromDisk.edges.begin(), fromDisk.edges.end());
                    std::sort(fromRam.edges.begin(), fromRam.edges.end());
                    Require(fromDisk.edges == fromRam.edges &&
                                fromDisk.edges.size() <= intact.edges.size(),
                            "a damaged page is left out alike from disk and from RAM");
                    lost = lost || fromDisk.edges.size() < intact.edges.size();
                }
            }
        }
        Require(lost, "a damaged page or children block loses roads");
    }
    std::filesystem::remove(damagedPath);
}

// A simulated machine for the tolerance controller: every step draws 10% fewer segments than the
// one before; drawing and expanding cost `perSegmentMs` each, a pass 1 ms more, the rest 2 ms.
// `hiddenBelow` makes the steps finer than it twice as slow as their counts say. Frames are
// measured one frame late, the part costs `costLag` frames late, and a finer step is drawn only
// `loadFrames` frames after it was asked for (its pages load meanwhile).
struct SimulatedMachine {
    double perSegmentMs = 0.0;
    int hiddenBelow = -1;
    bool counted = true;
    std::size_t costLag = 0;
    int loadFrames = 0;

    [[nodiscard]] std::array<std::uint64_t, chmv::renderer::kToleranceSteps> Counts() const {
        std::array<std::uint64_t, chmv::renderer::kToleranceSteps> counts{};
        for (std::size_t step = 0; step < counts.size(); ++step) {
            counts[step] = static_cast<std::uint64_t>(100000.0 * std::pow(0.9, step));
        }
        return counts;
    }
    [[nodiscard]] double Segments(int step) const {
        const auto segments = static_cast<double>(Counts()[static_cast<std::size_t>(step)]);
        return step < hiddenBelow ? segments * 2.0 : segments;
    }
    [[nodiscard]] double FrameMs(int step, bool pass) const {
        const auto segments = Segments(step);
        return 2.0 + segments * perSegmentMs + (pass ? 1.0 + segments * perSegmentMs : 0.0);
    }
};

struct SimulatedRun {
    int step = 0;
    std::uint32_t changes = 0;
    std::uint32_t slowFrames = 0;
    std::uint32_t refinesTo = 0;
};

// The viewer around the controller, at 60 Hz.
class SimulatedViewer {
public:
    SimulatedViewer(chmv::renderer::ToleranceController& controller, SimulatedMachine machine,
                    int step)
        : controller_(controller), machine_(machine), requested_(step), drawn_(step) {}

    // Runs `frames` frames, moving or not, and reports the last step asked for and what
    // happened; `watchedStep` counts how often that step was asked for when going finer.
    SimulatedRun Run(bool moving, int frames, int watchedStep = -1) {
        namespace renderer = chmv::renderer;
        SimulatedRun run;
        for (int frame = 0; frame < frames; ++frame) {
            renderer::ToleranceInput input;
            input.seconds = 1.0 / 60.0;
            input.budgetMs = 1000.0 / 60.0;
            input.moving = moving;
            input.drawnStep = drawn_;
            input.requestDrawn = drawn_ == requested_;
            const auto lag = std::min(machine_.costLag, drawnHistory_.size() - 1u);
            const auto measured =
                drawnHistory_.empty() ? drawn_ : drawnHistory_[drawnHistory_.size() - 1u - lag];
            const auto segments = machine_.Segments(measured);
            input.passFixedMs = 1.0;
            input.passScaledMs = segments * machine_.perSegmentMs;
            input.drawScaledMs = segments * machine_.perSegmentMs;
            if (machine_.counted) {
                input.counts = machine_.Counts();
            }
            input.sample = pending_;
            const auto next = controller_.Update(input);
            if (next != requested_) {
                ++run.changes;
                run.refinesTo += next == watchedStep && next < requested_ ? 1u : 0u;
                loading_ = next < drawn_ ? machine_.loadFrames : 0;
                requested_ = next;
            }
            bool pass = moving;
            if (drawn_ != requested_ && loading_-- <= 0) {
                drawn_ = requested_;
                pass = true;
            }
            const auto frameMs = machine_.FrameMs(drawn_, pass);
            run.slowFrames += frameMs > input.budgetMs ? 1u : 0u;
            pending_ = renderer::ToleranceSample{frameMs, pass, drawn_};
            drawnHistory_.push_back(drawn_);
        }
        run.step = requested_;
        return run;
    }

private:
    chmv::renderer::ToleranceController& controller_;
    SimulatedMachine machine_;
    int requested_ = 0;
    int drawn_ = 0;
    int loading_ = 0;
    std::vector<int> drawnHistory_;
    std::optional<chmv::renderer::ToleranceSample> pending_;
};

void TestToleranceController() {
    namespace renderer = chmv::renderer;
    Require(renderer::NearestStep(1.0f) == 0 && renderer::NearestStep(2.0f) == 4 &&
                renderer::NearestStep(8.0f) == 12 && renderer::NearestStep(100.0f) == 12 &&
                std::abs(renderer::StepPixels(4) - 2.0f) < 1e-5f,
            "tolerance steps are quarter octaves from 1 px to 8 px");
    Require(renderer::StepOfThreshold(1000u + 3u * 256u, 1000u) == 3 &&
                renderer::StepOfThreshold(1000u - 256u, 1000u) == -1,
            "a threshold maps back to its step");
    const auto predictive = [](int step) {
        renderer::ToleranceController controller;
        controller.SetMode(renderer::ToleranceMode::Predictive);
        controller.Reset(step);
        return controller;
    };

    {
        // Fast enough for 1 px: stays there, moving or not.
        auto controller = predictive(0);
        SimulatedMachine fast;
        fast.perSegmentMs = 1e-5;
        SimulatedViewer viewer(controller, fast, 0);
        const auto moving = viewer.Run(true, 120);
        const auto still = viewer.Run(false, 120);
        Require(moving.changes == 0u && still.changes == 0u && still.step == 0,
                "a fast machine keeps 1 px");
    }
    for (const bool counted : {true, false}) {
        for (const std::size_t lag : {std::size_t{0}, std::size_t{3}}) {
            // Too slow for 1 px while moving: coarser at once, finer again once the camera
            // stops, and then quiet; also when the part costs arrive late.
            auto controller = predictive(0);
            SimulatedMachine slow;
            slow.perSegmentMs = 13.5 / 100000.0;
            slow.counted = counted;
            slow.costLag = lag;
            SimulatedViewer viewer(controller, slow, 0);
            (void)viewer.Run(false, 60);
            const auto moving = viewer.Run(true, 240);
            const auto settled = viewer.Run(true, 60);
            Require(moving.step >= 8 && moving.step <= (counted ? 9 : 10) &&
                        settled.slowFrames == 0u,
                    "moving on a slow machine finds a step within the budget");
            Require(moving.slowFrames <= (counted ? 6u : 30u),
                    "the controller gets within the budget quickly");
            (void)viewer.Run(false, 240);
            const auto quiet = viewer.Run(false, 120);
            Require(quiet.step >= 1 && quiet.step <= 3 && quiet.changes == 0u &&
                        quiet.slowFrames == 0u,
                    "standing still refines and then stays put");
        }
    }
    for (const int load : {0, 40}) {
        // Steps below 4 are twice as slow as counted: tried rarely, not again and again, also
        // when finer steps take a while to load.
        auto controller = predictive(8);
        SimulatedMachine tricky;
        tricky.perSegmentMs = 10.0 / 100000.0;
        tricky.hiddenBelow = 4;
        tricky.loadFrames = load;
        SimulatedViewer viewer(controller, tricky, 8);
        const auto run = viewer.Run(false, 1200, 3);
        Require(run.refinesTo <= 5u && run.slowFrames <= 12u && run.step >= 3,
                "a step that was too slow is tried rarely");
    }
    {
        // Feedback alone also leaves 1 px on a slow machine and stays within the steps.
        renderer::ToleranceController controller;
        controller.SetMode(renderer::ToleranceMode::Feedback);
        controller.Reset(0);
        SimulatedMachine slow;
        slow.perSegmentMs = 13.5 / 100000.0;
        SimulatedViewer viewer(controller, slow, 0);
        const auto run = viewer.Run(true, 300);
        Require(run.step >= 5 && run.step < renderer::kToleranceSteps,
                "the feedback controller coarsens a slow machine");
    }
    {
        renderer::MotionCoarsening motion;
        const bool before = motion.Update(false, 0.1);
        const bool moving = motion.Update(true, 0.016);
        const bool just = motion.Update(false, 0.1);
        const bool later = motion.Update(false, 0.1);
        Require(!before && moving && just && !later,
                "coarser while moving, and for a moment after stopping");
    }
}

void TestErrorBounds() {
    namespace lod = chmv::streaming::lod;
    Require(lod::ErrorBound(0) == 0u, "error code 0 means no error at all");
    Require(lod::ErrorBound(lod::kErrorCodeInfinite) == lod::kMaxErrorBound,
            "an infinite error covers the whole map");
    std::uint32_t previous = 0;
    for (std::uint32_t code = 1; code <= lod::kErrorCodeMax; ++code) {
        const auto bound = lod::ErrorBound(code);
        const auto error = std::exp2(static_cast<double>(code) / lod::kErrorCodeStepsPerOctave -
                                     lod::kErrorCodeBiasOctaves) *
                           lod::kFixedCoordinateScale;
        Require(bound >= previous, "error bounds must grow with the code");
        Require(static_cast<double>(bound) >= std::min(error, double{lod::kMaxErrorBound}),
                "an error bound must not be below the error it stands for");
        Require(static_cast<double>(bound) <= error * 1.0001 + 1.0,
                "an error bound should round up by at most one unit");
        previous = bound;
    }
}

void TestCompression() {
    namespace lod = chmv::streaming::lod;
    // Slowly changing words, and a tail shorter than one element.
    std::vector<std::byte> input(10003);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<std::byte>((i / 4u) >> (i % 4u * 3u));
    }
    for (const std::uint32_t shuffle : {0u, 2u, 4u, 8u}) {
        std::vector<std::byte> frame;
        lod::AppendFrame(input, lod::kBlockLevel, shuffle, frame);
        std::vector<std::byte> output(input.size());
        Require(lod::DecodeFrame(frame, shuffle, output) && output == input,
                "a frame must decode to what was compressed");
        std::vector<std::byte> shorter(input.size() - 1u);
        Require(!lod::DecodeFrame(frame, shuffle, shorter), "a frame of another size is refused");
        auto damaged = frame;
        damaged[damaged.size() / 2u] ^= std::byte{0x5a};
        Require(!lod::DecodeFrame(damaged, shuffle, output), "a damaged frame is refused");
    }
    std::vector<std::byte> smaller;
    const auto shuffle = lod::AppendSmallerFrame(input, lod::kBlockLevel, 4u, smaller);
    std::vector<std::byte> output(input.size());
    Require(lod::DecodeFrame(smaller, shuffle, output) && output == input,
            "the smaller of both frames decodes as well");
}

// Every way of reading a compressed file gives the same bytes; damage is found, not drawn.
void TestCompressedPageFile() {
    namespace lod = chmv::streaming::lod;
    const ChainGraph source(20000);
    const auto filePath = lod::LodPageFile::DefaultPath(source.graphPath);
    const auto result = lod::LodPageBuilder::Build(source.graphPath, source.rangesPath, filePath);
    const auto file = lod::LodPageFile::Open(filePath);
    Require(file->Compressed() && result.fileBytes == std::filesystem::file_size(filePath) &&
                result.fileBytes < result.uncompressedBytes,
            "the page file is stored compressed");
    Require(file->Header().At(lod::Section::RecordEdge).bytes > lod::kSmallBlockBytes &&
                file->Pages().size() > 50u,
            "the chain spans several blocks and pages");
    RequireLodCutsMatchDfs(*file, "compressed chain differs from DFS");
    RequireCpuPassesMatchDfs(file, "CPU passes on the chain differ from DFS");
    {
        // The runtime: recomputes only when something changed, loads the whole CH for the
        // classic method in the background, says why it cannot run a combination.
        namespace cpu = chmv::cpu;
        cpu::CpuRuntime runtime(file, nullptr, nullptr);
        runtime.SetThreads(2);
        const lod::LodPageQuery query{-1e9f, -1e9f, 1e9f, 1e9f, 0u, 0u};
        cpu::CpuStrategy strategy{cpu::CpuFilter::BirthOrdered, cpu::CpuExpansion::Bfs,
                                  chmv::geometry::RefinementMode::Full};
        const auto expected = DfsCut(*file, 0u, 0u);
        const auto sortedEdges = [&] {
            auto edges = runtime.Result().edges;
            std::sort(edges.begin(), edges.end());
            return edges;
        };
        Require(runtime.Update(query, cpu::CpuMethod::Pages, strategy) && sortedEdges() == expected,
                "the CPU runtime draws the DFS cut");
        Require(!runtime.Update(query, cpu::CpuMethod::Pages, strategy),
                "an unchanged view is not computed again");
        (void)runtime.Update(query, cpu::CpuMethod::Classic, strategy);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (runtime.Classic().state == cpu::ClassicState::Loading &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Require(runtime.Classic().state == cpu::ClassicState::Ready,
                "the whole CH loads for the classic method");
        Require(runtime.Update(query, cpu::CpuMethod::Classic, strategy) &&
                    runtime.Result().unavailable.empty() && sortedEdges() == expected,
                "the classic method draws the DFS cut");
        strategy.expansion = cpu::CpuExpansion::Intervals;
        Require(runtime.Update(query, cpu::CpuMethod::Classic, strategy) &&
                    !runtime.Result().unavailable.empty() && runtime.Result().segments.empty(),
                "the classic method has no dual-interval counterpart");
        strategy.expansion.reset();
        Require(runtime.Update(query, cpu::CpuMethod::Pages, strategy) &&
                    !runtime.Result().unavailable.empty(),
                "a strategy without CPU version is reported");
        strategy.expansion = cpu::CpuExpansion::Dfs;
        const auto benchmark =
            runtime.Benchmark(query, cpu::CpuMethod::Pages, strategy, 5u, 0.0);
        Require(benchmark.unavailable.empty() && benchmark.iterations == 3u &&
                    benchmark.segments == expected.size() &&
                    benchmark.filterMinMs <= benchmark.filterMaxMs,
                "the CPU benchmark repeats the pass");
    }
    {
        // Workers run every task once and hand on the lowest failure.
        chmv::cpu::CpuWorkers workers(4);
        std::vector<int> runs(1000, 0);
        workers.Run(runs.size(), [&](std::size_t task, unsigned worker) {
            Require(worker < 4u, "worker index within the pool");
            ++runs[task];
        });
        Require(std::all_of(runs.begin(), runs.end(), [](int count) { return count == 1; }),
                "every task runs exactly once");
        bool failed = false;
        try {
            workers.Run(100, [](std::size_t task, unsigned) {
                if (task == 17u || task == 60u) {
                    throw std::runtime_error(task == 17u ? "17" : "60");
                }
            });
        } catch (const std::runtime_error& error) {
            failed = std::string(error.what()) == "17";
        }
        Require(failed, "the lowest failed task is reported");
    }

    const auto words = file->ReadPageWords({});
    const auto children = file->ReadChildrenWords({});
    const auto recordEdges = file->ReadSection<std::uint32_t>(lod::Section::RecordEdge);
    for (std::uint32_t p = 0; p < file->Pages().size(); ++p) {
        const auto& entry = file->Pages()[p];
        const auto page = file->ReadPage(p);
        Require(std::equal(page.begin(), page.end(),
                           words.begin() + static_cast<std::ptrdiff_t>(file->PageWordOffset(p))) &&
                    page.size() == entry.wordCount,
                "a single page reads as in the whole section");
        const auto edges = file->ReadRecordEdges(entry.firstRecord, entry.recordCount);
        Require(std::equal(edges.begin(), edges.end(),
                           recordEdges.begin() + static_cast<std::ptrdiff_t>(entry.firstRecord)),
                "record edges of a page read as in the whole section");
    }
    for (std::size_t index = 0; index < lod::kStoredCount; ++index) {
        Require(file->HashSection(index, {}) == file->Header().Hash(index),
                "every section decodes to its hash");
    }
    {
        lod::LodPageReader reader(file, true);
        std::vector<std::uint32_t> all(file->Pages().size());
        for (std::uint32_t p = 0; p < all.size(); ++p) {
            all[p] = p;
        }
        reader.SetDemand(all);
        std::vector<lod::LodPageData> read;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (read.size() < all.size() && std::chrono::steady_clock::now() < deadline) {
            reader.Take(read, ~std::uint64_t{0});
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Require(read.size() == all.size(), "the reader delivers every page");
        for (const auto& page : read) {
            const auto first =
                words.begin() + static_cast<std::ptrdiff_t>(file->PageWordOffset(page.page));
            const auto block =
                children.begin() + static_cast<std::ptrdiff_t>(file->ChildrenWordOffset(page.page));
            Require(page.words.size() == file->Pages()[page.page].wordCount &&
                        std::equal(page.words.begin(), page.words.end(), first) &&
                        page.children.size() == file->VisitPages()[page.page].wordCount &&
                        std::equal(page.children.begin(), page.children.end(), block),
                    "the reader decodes pages and children blocks exactly");
        }
    }

    // One damaged page frame: refused on its own, zero in the whole section.
    auto damagedPath = filePath;
    damagedPath += ".damaged";
    std::filesystem::copy_file(filePath, damagedPath,
                               std::filesystem::copy_options::overwrite_existing);
    const std::uint32_t victim = 3;
    {
        const auto range = file->PageStorage(victim);
        std::fstream damage(damagedPath, std::ios::in | std::ios::out | std::ios::binary);
        damage.seekg(static_cast<std::streamoff>(range.offset + range.bytes / 2u));
        char byte = 0;
        damage.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0x5a);
        damage.seekp(static_cast<std::streamoff>(range.offset + range.bytes / 2u));
        damage.write(&byte, 1);
    }
    {
        const auto damaged = lod::LodPageFile::Open(damagedPath);
        bool refused = false;
        try {
            (void)damaged->ReadPage(victim);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        Require(refused, "a damaged page is refused");
        const auto damagedWords = damaged->ReadPageWords({});
        const auto offset = static_cast<std::ptrdiff_t>(file->PageWordOffset(victim));
        const auto count = static_cast<std::ptrdiff_t>(file->Pages()[victim].wordCount);
        Require(
            std::all_of(damagedWords.begin() + offset, damagedWords.begin() + offset + count,
                        [](std::uint32_t word) { return word == 0u; }) &&
                std::equal(damagedWords.begin(), damagedWords.begin() + offset, words.begin()) &&
                std::equal(damagedWords.begin() + offset + count, damagedWords.end(),
                           words.begin() + offset + count),
            "only the damaged page is lost from the whole section");
        refused = false;
        try {
            (void)damaged->HashSection(lod::StoredIndex(lod::Section::Pages), {});
        } catch (const std::runtime_error&) {
            refused = true;
        }
        Require(refused, "checking a damaged page section fails");
    }
    std::filesystem::remove(damagedPath);
}

// Shortcuts whose own type is 0, as many datasets give them: a record draws with the style of
// the roads it stands for, the longer one where its halves differ, on the GPU pages and in the
// classic method alike.
void TestShortcutStyles() {
    namespace lod = chmv::streaming::lod;
    namespace cpu = chmv::cpu;
    const auto base = std::filesystem::temp_directory_path() / "chmv_shortcut_styles_test";
    const std::filesystem::path graphPath = base.string() + ".sch";
    const std::filesystem::path rangesPath = base.string() + ".sch.ranges";
    {
        // Edges 0 and 1 (primary) are short, edge 2 (residential) four times as long as both.
        std::ofstream graph(graphPath);
        graph << "# shortcut styles\n4\n5\n";
        graph << "0 100 48.0000000 9.0000000 0 2\n";
        graph << "1 101 48.0005000 9.0010000 0 0\n";
        graph << "2 102 48.0000000 9.0020000 0 1\n";
        graph << "3 103 48.0000000 9.0100000 0 2\n";
        graph << "0 1 10 3 50 -1 -1\n1 2 10 3 50 -1 -1\n2 3 40 12 50 -1 -1\n";
        graph << "0 2 20 0 50 0 1\n0 3 60 0 50 3 2\n";
        std::ofstream(rangesPath) << "0 0 0\n1 0 0\n2 1 0\n3 1 1\n4 2 2\n";
    }
    const auto filePath = lod::LodPageFile::DefaultPath(graphPath);
    (void)lod::LodPageBuilder::Build(graphPath, rangesPath, filePath);
    {
        const auto file = lod::LodPageFile::Open(filePath);
        const std::array<std::uint32_t, 5> expected{3u, 3u, 12u, 3u, 12u};
        const auto words = file->ReadPageWords({});
        const auto recordEdges = file->ReadSection<std::uint32_t>(lod::Section::RecordEdge);
        std::array<bool, 5> seen{};
        for (std::uint32_t p = 0; p < file->Pages().size(); ++p) {
            const auto& page = file->Pages()[p];
            const auto* pageWords = words.data() + file->PageWordOffset(p);
            const auto vertexCount = pageWords[0] >> 16u;
            const auto* typeBytes = reinterpret_cast<const std::uint8_t*>(
                pageWords + lod::kPageHeaderWords + page.recordCount * lod::kRecordWords +
                vertexCount * lod::kVertexWords);
            for (std::uint32_t i = 0; i < page.recordCount; ++i) {
                const auto edge = recordEdges[page.firstRecord + i];
                Require(typeBytes[i] == expected[edge],
                        "a shortcut draws with the style of the roads it stands for");
                seen[edge] = true;
            }
        }
        Require(std::all_of(seen.begin(), seen.end(), [](bool s) { return s; }),
                "every edge of the test graph has a record");
        // The classic method gives its segments the same styles.
        RequireCpuPassesMatchDfs(file, "CPU passes on the shortcut style graph differ");
        const std::stop_source stop;
        const auto classic = cpu::ClassicGraph::Load(*file, stop.get_token());
        cpu::CpuWorkers workers(1);
        const lod::LodPageQuery top{-1e9f, -1e9f, 1e9f, 1e9f, 2u, lod::kErrorCodeMax};
        const auto roots =
            classic->Run(workers, top, cpu::CpuFilter::FullScan, cpu::CpuExpansion::Roots);
        Require(roots.segments.size() == 1u && roots.edges[0] == 4u &&
                    roots.segments[0].type == 12u,
                "the classic method draws the top shortcut as residential");
    }
    std::filesystem::remove(graphPath);
    std::filesystem::remove(rangesPath);
    std::filesystem::remove(filePath);
}

void TestLodPageBuilder() {
    namespace lod = chmv::streaming::lod;
    {
        // Section hashes must not depend on how the writes were chunked.
        std::vector<std::uint8_t> bytes(1000);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::uint8_t>(i * 7u + 3u);
        }
        lod::StreamHash whole;
        whole.Update(bytes.data(), bytes.size());
        lod::StreamHash pieces;
        std::size_t offset = 0;
        for (const std::size_t size : {1u, 3u, 8u, 13u, 975u}) {
            pieces.Update(bytes.data() + offset, size);
            offset += size;
        }
        Require(whole.Finish() == pieces.Finish(), "stream hash depends on chunking");
    }
    {
        const LodTestGraph source({}, "0 0 0\n1 0 0\n2 0 0\n3 0 0\n4 1 1\n5 1 1\n6 2 2\n");
        const auto filePath = lod::LodPageFile::DefaultPath(source.graphPath);
        const auto result = lod::LodPageBuilder::Build(source.graphPath, source.rangesPath,
                                                       filePath);
        Require(result.recordCount == 7, "every edge should get exactly one record");
        Require(result.coverGapEdges == 0 && result.extraContextRecords == 0,
                "simple hierarchy needs no extra contexts");
        Require(lod::LodPageFile::MatchesSource(filePath, source.graphPath, source.rangesPath),
                "page file should match its source");

        const auto file = lod::LodPageFile::Open(filePath);
        Require(file->Header().nodeCount == 5 && file->Header().edgeCount == 7,
                "CH layer counts");
        Require(file->ReadSection<std::uint32_t>(lod::Section::EdgeWeight) ==
                    std::vector<std::uint32_t>{10, 10, 10, 10, 20, 20, 40},
                "edge weights must survive exactly");
        Require(file->ReadSection<std::int32_t>(lod::Section::NodeLatitude)[1] == 480100000,
                "latitudes are stored as exact 1e-7 degrees");

        const auto coarse = std::uint32_t{lod::kErrorCodeMax};
        Require(LodPageCut(*file, 2, coarse) == std::vector<std::uint32_t>{6},
                "level 2, coarse: one shortcut");
        Require(LodPageCut(*file, 1, coarse) == std::vector<std::uint32_t>{4, 5},
                "level 1, coarse: two shortcuts");
        Require(LodPageCut(*file, 2, 0) == std::vector<std::uint32_t>{0, 1, 2, 3},
                "level 2, exact: fully unfolded");
        RequireLodCutsMatchDfs(*file, "page cut differs from DFS");
        RequireTraversalsMatchDfs(*file, "traversal differs from DFS");
        RequireCpuPassesMatchDfs(file, "CPU passes differ from DFS");

        lod::LodPageQuery everywhere{-1e9f, -1e9f, 1e9f, 1e9f, 2, 0};
        std::vector<std::uint32_t> pages;
        lod::CollectLodPages(file->Groups(), file->Pages(), everywhere, pages);
        Require(!pages.empty(), "planner must find the pages of a drawn level");
    }
    {
        // Shortcut 4 is still alive at level 2 where its parent 6 is alive too: DFS reaches 4
        // (and everything below it) twice.
        const LodTestGraph source({}, "0 0 0\n1 0 0\n2 0 0\n3 0 0\n4 2 1\n5 1 1\n6 2 2\n");
        const auto filePath = lod::LodPageFile::DefaultPath(source.graphPath);
        const auto result = lod::LodPageBuilder::Build(source.graphPath, source.rangesPath,
                                                       filePath);
        Require(result.extraContextRecords != 0, "second visit needs its own records");
        const auto file = lod::LodPageFile::Open(filePath);
        RequireLodCutsMatchDfs(*file, "root + parent at the same level differs from DFS");
        RequireTraversalsMatchDfs(*file, "traversal with a revisited shortcut differs from DFS");
        RequireCpuPassesMatchDfs(file, "CPU passes differ from DFS");
    }
    {
        // Shortcut 7 (0-3 via 2) is a second parent of shortcut 4, alive at the same level as 6.
        const LodTestGraph source(
            {"0 3 30 1 50 4 2"},
            "0 0 0\n1 0 0\n2 0 0\n3 0 0\n4 1 1\n5 1 1\n6 2 2\n7 2 2\n");
        const auto filePath = lod::LodPageFile::DefaultPath(source.graphPath);
        (void)lod::LodPageBuilder::Build(source.graphPath, source.rangesPath, filePath);
        const auto file = lod::LodPageFile::Open(filePath);
        RequireLodCutsMatchDfs(*file, "two parents at the same level differ from DFS");
        RequireTraversalsMatchDfs(*file, "traversal with two parents differs from DFS");
        RequireCpuPassesMatchDfs(file, "CPU passes differ from DFS");
    }
    {
        const WalkOnlyGraph source(600);
        const auto filePath = lod::LodPageFile::DefaultPath(source.graphPath);
        const auto result = lod::LodPageBuilder::Build(source.graphPath, source.rangesPath,
                                                       filePath);
        Require(result.visitOnlyRecords == 600u, "every bent shortcut is only walked through");
        const auto file = lod::LodPageFile::Open(filePath);
        Require(std::any_of(file->Pages().begin(), file->Pages().end(),
                            [](const lod::PageEntry& page) {
                                return page.minLevel > page.maxLevel;
                            }),
                "some pages hold only records that are never drawn");
        RequireLodCutsMatchDfs(*file, "pages of walk-only records change the cut");
        RequireTraversalsMatchDfs(*file, "traversal through walk-only pages differs from DFS");
        RequireCpuPassesMatchDfs(file, "CPU passes differ from DFS");
    }
}

// The same file as the builder writes it before compressing.
void WriteUncompressed(const chmv::streaming::lod::LodPageFile& file,
                       const std::filesystem::path& path) {
    namespace lod = chmv::streaming::lod;
    auto header = file.Header();
    header.compression = lod::Compression::None;
    header.stored = {};
    header.pageFrames = {};
    header.childFrames = {};
    std::uint64_t end = lod::kHeaderBytes;
    for (std::size_t index = 0; index < lod::kStoredCount; ++index) {
        end = std::max(end, header.Layout(index).offset + header.Layout(index).bytes);
    }
    std::vector<char> bytes(static_cast<std::size_t>(end));
    std::memcpy(bytes.data(), &header, sizeof(header));
    for (std::size_t index = 0; index < lod::kStoredCount; ++index) {
        const auto& layout = header.Layout(index);
        file.ReadRange(index, 0u, bytes.data() + layout.offset, layout.bytes);
    }
    std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(end));
}

void TestAsyncDatasetLoader() {
    const auto root = std::filesystem::temp_directory_path() / "chmv_async_test_root";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto graphPath = root / "sample.sch";
    const auto rangePath = root / "sample.sch.ranges";
    {
        std::ofstream graph(graphPath);
        graph << "2\n1\n";
        graph << "0 100 48.0 9.0 0 1\n";
        graph << "1 101 48.1 9.1 0 0\n";
        graph << "0 1 1.0 1 50 -1 -1\n";
    }
    std::ofstream(rangePath) << "0 1 0\n";

    const auto pageFile = chmv::streaming::lod::LodPageFile::DefaultPath(graphPath);
    const auto finish = [](chmv::data::AsyncDatasetLoader& loader) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (loader.Snapshot().phase != chmv::data::DatasetLoadPhase::Ready &&
               loader.Snapshot().phase != chmv::data::DatasetLoadPhase::Failed &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return loader.Snapshot();
    };
    const auto load = [&](bool withSources = true) {
        chmv::data::AsyncDatasetLoader loader;
        Require(withSources ? loader.Start(graphPath, rangePath) : loader.StartPageFile(pageFile),
                "async loader did not start");
        const auto snapshot = finish(loader);
        Require(snapshot.phase == chmv::data::DatasetLoadPhase::Ready, "dataset did not load");
        Require(snapshot.progress == 1.0f, "async loader progress did not reach 100 percent");
        auto dataset = loader.TakeCompleted();
        Require(dataset.has_value() && dataset->file != nullptr, "loaded dataset missing");
        Require(dataset->file->Header().nodeCount == 2 && dataset->file->Header().edgeCount == 1,
                "loaded dataset counts");
        return std::filesystem::last_write_time(dataset->file->Path());
    };
    const auto built = load();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Require(load() == built, "an up-to-date page file is reused");

    // A file of an older format is rebuilt on load (opening it as it is would fail).
    {
        namespace lod = chmv::streaming::lod;
        std::fstream file(lod::LodPageFile::DefaultPath(graphPath),
                          std::ios::in | std::ios::out | std::ios::binary);
        const std::uint32_t older = lod::kFormatVersion - 1u;
        file.seekp(static_cast<std::streamoff>(offsetof(lod::FileHeader, version)));
        file.write(reinterpret_cast<const char*>(&older), sizeof(older));
    }
    (void)load();

    // Damage past the header is found on open: the file is rebuilt from its sources.
    {
        namespace lod = chmv::streaming::lod;
        const auto stored = lod::LodPageFile::Open(pageFile)
                                ->Header()
                                .stored[lod::StoredIndex(lod::Section::PageDirectory)];
        std::fstream file(pageFile, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(stored.offset + stored.bytes - 2u));
        file.put('\x5a');
    }
    (void)load();
    {
        namespace lod = chmv::streaming::lod;
        const auto file = lod::LodPageFile::Open(pageFile);
        const auto index = lod::StoredIndex(lod::Section::PageDirectory);
        Require(file->HashSection(index, {}) == file->Header().Hash(index),
                "a damaged page file is rebuilt");
    }

    // An uncompressed file (as the builder writes it before compressing) is compressed, not
    // rebuilt: the sources keep their size and time but could not be read any more.
    {
        namespace lod = chmv::streaming::lod;
        WriteUncompressed(*lod::LodPageFile::Open(pageFile), pageFile);
        Require(!lod::LodPageFile::Open(pageFile)->Compressed(), "the copy is uncompressed");
        const auto time = std::filesystem::last_write_time(graphPath);
        const auto size = std::filesystem::file_size(graphPath);
        std::ofstream(graphPath, std::ios::binary) << std::string(size, 'x');
        std::filesystem::last_write_time(graphPath, time);
    }
    const auto rebuilt = load();
    Require(chmv::streaming::lod::LodPageFile::Open(pageFile)->Compressed() &&
                chmv::streaming::lod::LodPageFile::Open(pageFile)->Header().version ==
                    chmv::streaming::lod::kFormatVersion,
            "an uncompressed page file is compressed on load");
    {
        const chmv::data::DatasetCatalog catalog(root);
        Require(catalog.Entries().size() == 1u && catalog.Entries()[0].HasSources() &&
                    catalog.Entries()[0].pageFilePath == pageFile,
                "a source pair is listed once, together with its page file");
    }

    // Without its sources the page file is listed and loaded on its own.
    std::filesystem::remove(graphPath);
    std::filesystem::remove(rangePath);
    {
        const chmv::data::DatasetCatalog catalog(root);
        Require(catalog.Entries().size() == 1u && !catalog.Entries()[0].HasSources() &&
                    catalog.Entries()[0].pageFilePath == pageFile,
                "a page file without its sources is listed by itself");
    }
    Require(load(false) == rebuilt, "a page file without its sources is used as it is");

    // An older one cannot be rebuilt then: the load fails and says what is missing.
    {
        namespace lod = chmv::streaming::lod;
        std::fstream file(pageFile, std::ios::in | std::ios::out | std::ios::binary);
        const std::uint32_t older = lod::kFormatVersion - 1u;
        file.seekp(static_cast<std::streamoff>(offsetof(lod::FileHeader, version)));
        file.write(reinterpret_cast<const char*>(&older), sizeof(older));
    }
    {
        chmv::data::AsyncDatasetLoader loader;
        Require(loader.StartPageFile(pageFile), "async loader did not start");
        const auto snapshot = finish(loader);
        Require(snapshot.phase == chmv::data::DatasetLoadPhase::Failed &&
                    snapshot.error.find(".sch/.ranges") != std::string::npos,
                "an outdated page file without sources fails with a reason");
    }
    std::filesystem::remove_all(root);
}

void TestRoadStyles() {
    namespace renderer = chmv::renderer;
    const auto shipped = renderer::LoadRoadStyles(CHMV_ROAD_STYLES);
    Require(shipped.classes.size() == 9u, "the shipped road styles have nine classes");
    const auto nameOf = [&](std::uint32_t type) {
        return shipped.classes[shipped.classOfType[type]].name;
    };
    Require(nameOf(1) == "motorway" && nameOf(2) == "motorway" && nameOf(10) == "trunk" &&
                nameOf(12) == "minor" && nameOf(22) == "path",
            "road types map to the classes of all.cfg");
    Require(nameOf(0) == "other" && nameOf(chmv::data::RoadStyleFallbackType) == "other",
            "unlisted types use the * class");
    Require(shipped.TypeWidths()[1] == 4.5f, "width of a class reaches its types");
    const auto depths = shipped.TypeDepths();
    Require(depths[1] < depths[9] && depths[9] < depths[3] && depths[3] < depths[12] &&
                depths[12] < depths[0] && depths[1] > 0.0f && depths[0] < 1.0f,
            "classes listed first lie on top, inside the depth range");

    const auto path = std::filesystem::temp_directory_path() / "chmv_road_styles_test.cfg";
    const auto parse = [&](const std::string& text) {
        std::ofstream(path) << text;
        return renderer::LoadRoadStyles(path);
    };
    const auto custom = parse("# comment\nbig #ff000080 3 1 2 # inline\nrest #00ff00 1 *\n");
    Require(custom.classes.size() == 2u && custom.classOfType[2] == 0u &&
                custom.classOfType[5] == 1u && custom.classes[0].color[3] < 0.51f,
            "colors with alpha and comments");
    for (const auto* broken : {"a #ff0000 1 1\n", "a #ff0000 1 1 *\nb #00ff00 1 1\n",
                               "a red 1 *\n", "a #ff0000 0 *\n", "a #ff0000 1\n",
                               "a #ff0000 1 63 *\n", "a #ff0000 1 * \nb #00ff00 1 *\n"}) {
        bool failed = false;
        try {
            (void)parse(broken);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Require(failed, "a broken road style file must be refused");
    }
    std::filesystem::remove(path);
}

} // namespace

int main() {
    try {
        TestErrorBounds();
        TestRoadStyles();
        TestToleranceController();
        TestCompression();
        TestLodPageBuilder();
        TestShortcutStyles();
        TestCompressedPageFile();
        TestCpuPartialViews();
        TestAsyncDatasetLoader();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
