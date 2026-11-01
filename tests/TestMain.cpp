#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHTypes.h"
#include "renderer/RoadStyle.h"
#include "streaming/lod/LodPageBuilder.h"
#include "streaming/lod/LodPageCompression.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPageIo.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/lod/LodPageReader.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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
    }
}

// The same file as the builder writes it before compressing, marked as a version 3 file.
void WriteUncompressed(const chmv::streaming::lod::LodPageFile& file,
                       const std::filesystem::path& path) {
    namespace lod = chmv::streaming::lod;
    auto header = file.Header();
    header.version = lod::kUncompressedFormatVersion;
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

    // A version 3 file (the same file uncompressed) is compressed, not rebuilt: the sources keep
    // their size and time but could not be read any more.
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
            "a version 3 file is compressed on load");
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
        TestCompression();
        TestLodPageBuilder();
        TestCompressedPageFile();
        TestAsyncDatasetLoader();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
