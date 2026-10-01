#include "benchmark/CorrectnessValidator.h"
#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHLoader.h"
#include "data/preprocessing/GeometryErrorPreprocessor.h"
#include "geometry/GeometryRefinement.h"
#include "pipeline/CPUReferencePipeline.h"
#include "reference/CPURangeFilter.h"
#include "reference/CPUReferenceUnfolder.h"
#include "streaming/analysis/DatasetLayoutAnalyzer.h"
#include "streaming/preprocess/DatasetPreprocessor.h"
#include "streaming/index/CHIndex.h"
#include "streaming/runtime/GraphPageStreamer.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

chmv::data::CHGraph LoadTestGraph() {
    const auto base = std::filesystem::temp_directory_path() / "chmv_test";
    const auto graphPath = base.string() + ".sch";
    const auto rangePath = base.string() + ".sch.ranges";

    {
        std::ofstream graph(graphPath);
        graph << "# small graph\n";
        graph << "3\n";
        graph << "3\n";
        graph << "0 100 48.0 9.0 0 2\n";
        graph << "1 101 48.1 9.2 0 1\n";
        graph << "2 102 48.2 9.2 0 0\n";
        graph << "0 1 1.0 1 50 -1 -1\n";
        graph << "1 2 1.0 1 50 -1 -1\n";
        graph << "0 2 2.0 1 50 0 1\n";
    }

    {
        std::ofstream ranges(rangePath);
        ranges << "0 2 0\n";
        ranges << "1 2 0\n";
        ranges << "2 2 1\n";
    }

    auto graph = chmv::data::CHLoader::Load(graphPath, rangePath);
    chmv::data::preprocessing::GeometryErrorPreprocessor::Run(graph);
    std::filesystem::remove(graphPath);
    std::filesystem::remove(rangePath);
    return graph;
}

void TestLoaderAndReferenceUnfolding() {
    const auto graph = LoadTestGraph();
    Require(graph.NodeCount() == 3, "node count mismatch");
    Require(graph.EdgeCount() == 3, "edge count mismatch");
    Require(graph.ShortcutCount() == 1, "shortcut count mismatch");
    Require(graph.Range(2).IsAlive(1), "range should be alive at level 1");
    Require(!graph.Range(2).IsAlive(0), "range should not be alive at level 0");

    const chmv::reference::CPUReferenceUnfolder unfolder(graph);
    const std::vector<std::uint32_t> roots{2};
    const auto output = unfolder.UnfoldFully(roots);

    Require(output.size() == 2, "unexpected unfolded edge count");
    Require(output[0] == 0 && output[1] == 1, "unexpected unfolding order");
}

void TestCPURangeFilterAndPipeline() {
    const auto graph = LoadTestGraph();

    chmv::reference::CPURangeFilter filter;
    std::vector<std::uint32_t> alive;
    const auto filterStats = filter.Filter(graph, 1, alive);
    Require(alive.size() == 3, "unexpected CPU range-filter result");
    Require(filterStats.scannedEdgeCount == 3, "unexpected ordered scan count");

    chmv::pipeline::CPUReferencePipeline pipeline;
    const chmv::geometry::RefinementParameters paperRefinement{};
    const auto paperResult = pipeline.Process(graph, 1, paperRefinement);
    Require(paperResult.stats.aliveEdgeCount == 3, "unexpected CPU pipeline alive count");
    Require(paperResult.edgeIds.size() == 3, "paper pipeline should render lifetime edges directly");
    Require(!paperResult.stats.cacheHit, "first CPU pipeline result must not be cached");

    const auto cachedPaper = pipeline.Process(graph, 1, paperRefinement);
    Require(cachedPaper.stats.cacheHit, "unchanged CPU paper result should be cached");

    const chmv::geometry::RefinementParameters fullRefinement{
        chmv::geometry::RefinementMode::Full, 0.0f, 1.0f};
    const auto refinedResult = pipeline.Process(graph, 1, fullRefinement);
    Require(refinedResult.edgeIds.size() == 4, "unexpected CPU geometry-refined edge count");
    Require(!refinedResult.stats.cacheHit, "changing geometry refinement must invalidate the cache");

    Require(graph.Edge(2).geometryError > 0.0f, "shortcut geometry error was not precomputed");
    const chmv::geometry::RefinementParameters adaptiveRefinement{
        chmv::geometry::RefinementMode::Adaptive, 100.0f, 0.01f};
    const auto adaptiveResult = pipeline.Process(graph, 1, adaptiveRefinement);
    Require(adaptiveResult.edgeIds.size() == 4, "adaptive refinement did not unfold visible error");
}

void TestOrderedRangeRetrieval() {
    chmv::data::CHGraph graph;
    graph.Edges().resize(4);
    graph.Ranges() = {
        {3, 0},
        {2, 1},
        {1, 1},
        {0, 0},
    };

    chmv::reference::CPURangeFilter filter;
    std::vector<std::uint32_t> output;
    const auto stats = filter.Filter(graph, 2, output);

    Require(stats.scannedEdgeCount == 2, "ordered retrieval scanned too many edges");
    Require(output.size() == 2, "ordered retrieval output count mismatch");
    Require(output[0] == 0 && output[1] == 1, "ordered retrieval output mismatch");
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
    {
        std::ofstream ranges(rangePath);
        ranges << "0 1 0\n";
    }

    chmv::data::AsyncDatasetLoader loader;
    Require(loader.Start(graphPath, rangePath), "async loader did not start");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (loader.Snapshot().phase != chmv::data::DatasetLoadPhase::Ready &&
           loader.Snapshot().phase != chmv::data::DatasetLoadPhase::Failed &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto snapshot = loader.Snapshot();
    Require(snapshot.phase == chmv::data::DatasetLoadPhase::Ready,
            "async loader did not complete");
    Require(snapshot.progress == 1.0f, "async loader progress did not reach 100 percent");

    auto graph = loader.TakeCompleted();
    Require(graph.has_value(), "async loader result missing");
    Require(graph->NodeCount() == 2 && graph->EdgeCount() == 1,
            "async loader graph mismatch");
    Require(chmv::streaming::preprocess::DatasetPreprocessor::HasValidPreprocess(graphPath, rangePath),
            "async loader should create and validate Preprocess automatically");

    chmv::data::AsyncDatasetLoader streamingLoader;
    Require(streamingLoader.StartStreaming(graphPath, rangePath),
            "streaming async loader did not start");
    const auto streamingDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (streamingLoader.Snapshot().phase != chmv::data::DatasetLoadPhase::PreprocessedForStreaming &&
           streamingLoader.Snapshot().phase != chmv::data::DatasetLoadPhase::Failed &&
           std::chrono::steady_clock::now() < streamingDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(streamingLoader.Snapshot().phase == chmv::data::DatasetLoadPhase::PreprocessedForStreaming,
            "streaming async loader did not expose the preprocessed runtime dataset");
    auto streamingDataset = streamingLoader.TakeStreamingCompleted();
    Require(streamingDataset.has_value(), "streaming async loader result missing");
    Require(streamingDataset->index.nodeCount == 2 && streamingDataset->index.edgeCount == 1,
            "streaming async loader index mismatch");

    std::filesystem::remove_all(root);
}


void TestDatasetLayoutAnalyzer() {
    const auto base = std::filesystem::temp_directory_path() / "chmv_layout_analysis_test";
    const auto graphPath = base.string() + ".sch";
    const auto rangePath = base.string() + ".sch.ranges";
    const auto outputPath = base.string() + ".output";

    {
        std::ofstream graph(graphPath);
        graph << "4\n4\n";
        graph << "0 100 48.0 9.0 0 3\n";
        graph << "1 101 48.0 9.1 0 2\n";
        graph << "2 102 48.1 9.1 0 1\n";
        graph << "3 103 48.1 9.2 0 0\n";
        graph << "0 1 1.0 1 50 -1 -1\n";
        graph << "1 2 1.0 1 50 -1 -1\n";
        graph << "2 3 1.0 1 50 -1 -1\n";
        graph << "0 2 2.0 1 50 0 1\n";
    }
    {
        std::ofstream ranges(rangePath);
        ranges << "0 3 0\n";
        ranges << "1 3 0\n";
        ranges << "2 -1 -1\n";
        ranges << "3 3 2\n";
    }

    chmv::streaming::analysis::DatasetLayoutAnalysisConfig config;
    config.spatialGridSize = 16;
    config.lodBandWidth = 2;
    config.sourceBlockNodeCount = 2;
    config.sourceBlockEdgeCount = 2;
    config.virtualPageEdgeCount = 2;

    const auto report = chmv::streaming::analysis::DatasetLayoutAnalyzer::Analyze(
        graphPath, rangePath, config);
    Require(report.nodeCount == 4 && report.edgeCount == 4,
            "layout analyzer dataset counts mismatch");
    Require(report.shortcutCount == 1, "layout analyzer shortcut count mismatch");
    Require(report.drawableEdgeCount == 3 && report.nonDrawableEdgeCount == 1,
            "layout analyzer drawable counts mismatch");
    Require(report.nodeRecordSequentialRatio == 1.0,
            "layout analyzer node order detection mismatch");
    Require(report.rangeRecordSequentialRatio == 1.0,
            "layout analyzer range order detection mismatch");
    Require(report.nodeSourceBlocks.size() == 2,
            "layout analyzer node source block count mismatch");
    Require(report.graphSourceBlocks.size() == 2,
            "layout analyzer source block count mismatch");
    Require(report.streamingEdgeBlocks.size() == 2,
            "layout analyzer streaming block count mismatch");
    Require(report.lodMaskWordsPerBlock == 1,
            "layout analyzer LOD mask width mismatch");
    Require(report.virtualPageCount == 2,
            "layout analyzer virtual page count mismatch");
    Require(report.runtimeGraphPages.size() == 2,
            "runtime graph page count mismatch");
    Require(report.runtimeGraphPageEdgeIds.size() == 3,
            "runtime graph page membership mismatch");
    Require(report.drawableShortcutCount == 1,
            "layout analyzer drawable shortcut count mismatch");
    Require(report.drawableShortcutChildReferenceCount == 2,
            "layout analyzer child reference count mismatch");

    chmv::streaming::analysis::DatasetLayoutAnalyzer::WriteReport(report, outputPath);
    Require(std::filesystem::exists(std::filesystem::path(outputPath) / "summary.txt"),
            "layout analyzer summary was not written");
    Require(std::filesystem::exists(
                std::filesystem::path(outputPath) / "virtual_page_locality.csv"),
            "layout analyzer virtual-page report was not written");
    Require(std::filesystem::exists(
                std::filesystem::path(outputPath) / "runtime_graph_pages.csv"),
            "layout analyzer runtime-page report was not written");
    Require(std::filesystem::exists(
                std::filesystem::path(outputPath) / "cache_manifest.txt"),
            "layout analyzer cache manifest was not written");
    Require(chmv::streaming::analysis::DatasetLayoutAnalyzer::DefaultReportDirectory(graphPath) ==
                std::filesystem::path(graphPath).parent_path() / "layout-analysis",
            "layout analyzer default report directory mismatch");
    Require(chmv::streaming::analysis::DatasetLayoutAnalyzer::HasValidCachedReport(
                graphPath, rangePath, config, outputPath),
            "layout analyzer should reuse a matching cached report");
    Require(chmv::streaming::analysis::DatasetLayoutAnalyzer::ReadCachedSummary(outputPath).find(
                "CH_MapViewer dataset layout analysis") != std::string::npos,
            "layout analyzer cached summary could not be read");

    const auto indexPath = chmv::streaming::index::CHIndex::DefaultPath(graphPath);
    Require(indexPath == std::filesystem::path(base.string() + ".chidx"),
            "streaming index default path mismatch");
    chmv::streaming::index::CHIndex::Write(report, indexPath);
    Require(chmv::streaming::index::CHIndex::IsValid(
                indexPath, graphPath, rangePath, config),
            "streaming index should match source files");
    const auto index = chmv::streaming::index::CHIndex::Load(indexPath);
    Require(index.nodeCount == 4 && index.edgeCount == 4,
            "streaming index dataset counts mismatch");
    Require(index.nodeBlocks.size() == 2 && index.graphEdgeBlocks.size() == 2 &&
                index.rangeEdgeBlocks.size() == 2,
            "streaming index source block counts mismatch");
    Require(index.graphEdgeBlocks[0].byteOffset == report.graphSourceBlocks[0].byteOffset,
            "streaming index graph offset mismatch");
    Require(index.rangeEdgeBlocks[0].byteOffset == report.rangeSourceBlocks[0].byteOffset,
            "streaming index range offset mismatch");
    Require(index.graphPages.size() == report.runtimeGraphPages.size(),
            "streaming index graph page count mismatch");
    Require(index.GraphPageHasAliveEdges(0, 2),
            "streaming index page LOD mask lost alive edges");
    const auto page0Edges = chmv::streaming::index::CHIndex::ReadGraphPageEdgeIds(index, 0);
    Require(page0Edges.size() == index.graphPages[0].edgeCount,
            "streaming index page membership decode mismatch");

    auto changedConfig = config;
    ++changedConfig.lodBandWidth;
    Require(!chmv::streaming::analysis::DatasetLayoutAnalyzer::HasValidCachedReport(
                graphPath, rangePath, changedConfig, outputPath),
            "layout analyzer should invalidate cache when configuration changes");

    {
        std::ofstream ranges(rangePath, std::ios::app);
        ranges << "# cache invalidation\n";
    }
    Require(!chmv::streaming::analysis::DatasetLayoutAnalyzer::HasValidCachedReport(
                graphPath, rangePath, config, outputPath),
            "layout analyzer should invalidate cache when source files change");
    Require(!chmv::streaming::index::CHIndex::IsValid(
                indexPath, graphPath, rangePath, config),
            "streaming index should invalidate when source files change");

    std::filesystem::remove(graphPath);
    std::filesystem::remove(rangePath);
    std::filesystem::remove(indexPath);
    std::filesystem::remove_all(outputPath);
}




void TestHierarchicalSpatialOwnership() {
    const auto base = std::filesystem::temp_directory_path() / "chmv_spatial_owner_test";
    const auto graphPath = base.string() + ".sch";
    const auto rangePath = base.string() + ".sch.ranges";

    {
        std::ofstream graph(graphPath);
        graph << "3\n2\n";
        graph << "0 100 0.0 0.0 0 1\n";
        graph << "1 101 0.0 0.1 0 1\n";
        graph << "2 102 0.0 1.0 0 1\n";
        graph << "0 1 1.0 1 50 -1 -1\n";
        graph << "0 2 1.0 1 50 -1 -1\n";
    }
    {
        std::ofstream ranges(rangePath);
        ranges << "0 1 0\n";
        ranges << "1 1 0\n";
    }

    chmv::streaming::analysis::DatasetLayoutAnalysisConfig config;
    config.spatialGridSize = 8;
    config.lodBandWidth = 1;
    config.sourceBlockNodeCount = 2;
    config.sourceBlockEdgeCount = 2;
    config.virtualPageEdgeCount = 1;
    const auto report = chmv::streaming::analysis::DatasetLayoutAnalyzer::Analyze(
        graphPath, rangePath, config);

    Require(report.runtimeGraphPageEdgeIds.size() == 2,
            "hierarchical spatial ownership lost or duplicated an edge");
    std::vector<std::uint32_t> membership = report.runtimeGraphPageEdgeIds;
    std::sort(membership.begin(), membership.end());
    Require(membership == std::vector<std::uint32_t>({0, 1}),
            "hierarchical spatial ownership should store each drawable edge exactly once");

    bool crossingAtAncestor = false;
    bool localPushedDeeper = false;
    for (const auto& page : report.runtimeGraphPages) {
        const auto begin = report.runtimeGraphPageEdgeIds.begin() + page.edgeIdOffset;
        const auto end = begin + page.edgeCount;
        if (std::find(begin, end, 1u) != end && page.spatialLevel == 0) {
            crossingAtAncestor = true;
        }
        if (std::find(begin, end, 0u) != end && page.spatialLevel > 0) {
            localPushedDeeper = true;
        }
    }
    Require(crossingAtAncestor,
            "cross-tile edge should remain at its enclosing spatial ancestor");
    Require(localPushedDeeper,
            "local edge should descend into a finer spatial page when the parent is overloaded");

    std::filesystem::remove(graphPath);
    std::filesystem::remove(rangePath);
}

void TestDatasetPreprocessor() {
    const auto root = std::filesystem::temp_directory_path() / "chmv_preprocess_test_root";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "dataset");
    const auto graphPath = root / "dataset" / "sample.sch";
    const auto rangePath = root / "dataset" / "sample.sch.ranges";

    {
        std::ofstream graph(graphPath);
        graph << "5\n7\n";
        graph << "0 100 48.00 9.00 0 4\n";
        graph << "1 101 48.00 9.10 0 3\n";
        graph << "2 102 48.10 9.10 0 2\n";
        graph << "3 103 48.20 9.10 0 1\n";
        graph << "4 104 48.20 9.20 0 0\n";
        graph << "0 1 1.0 1 50 -1 -1\n";
        graph << "1 2 1.0 1 50 -1 -1\n";
        graph << "2 3 1.0 1 50 -1 -1\n";
        graph << "3 4 1.0 1 50 -1 -1\n";
        graph << "0 2 2.0 1 50 0 1\n";
        graph << "2 4 2.0 1 50 2 3\n";
        graph << "0 4 4.0 1 50 4 5\n";
    }
    {
        std::ofstream ranges(rangePath);
        ranges << "0 4 0\n";
        ranges << "1 4 0\n";
        ranges << "2 -1 -1\n";
        ranges << "3 -1 -1\n";
        ranges << "4 6 2\n";
        ranges << "5 6 2\n";
        ranges << "6 8 7\n";
    }

    const auto readAll = [](const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>());
    };
    const auto sourceGraphBytes = readAll(graphPath);
    const auto sourceRangeBytes = readAll(rangePath);
    const auto sourceGraph = chmv::data::CHLoader::Load(graphPath, rangePath);

    chmv::streaming::analysis::DatasetLayoutAnalysisConfig config;
    config.spatialGridSize = 16;
    config.lodBandWidth = 2;
    config.sourceBlockNodeCount = 2;
    config.sourceBlockEdgeCount = 2;
    config.virtualPageEdgeCount = 2;

    chmv::streaming::preprocess::PreprocessOptions preprocessOptions;
    preprocessOptions.memoryBudgetBytes = 4ull * 1024ull * 1024ull;
    preprocessOptions.writeDetailedAnalysis = false;
    const auto result = chmv::streaming::preprocess::DatasetPreprocessor::Run(
        graphPath, rangePath, config, true, {}, preprocessOptions);
    Require(!result.reused, "forced preprocess should rebuild the preprocessed dataset");
    Require(std::filesystem::is_regular_file(result.paths.graphPath),
            "preprocess graph was not written");
    Require(std::filesystem::is_regular_file(result.paths.rangesPath),
            "preprocess ranges were not written");
    Require(std::filesystem::is_regular_file(result.paths.indexPath),
            "preprocess index was not written");
    Require(result.paths.graphPath == graphPath && result.paths.rangesPath == rangePath,
            "preprocess should keep the active dataset filenames unchanged");
    Require(std::filesystem::is_regular_file(result.paths.originalGraphPath) &&
                std::filesystem::is_regular_file(result.paths.originalRangesPath),
            "preprocess did not preserve the original source pair under .org");
    Require(readAll(result.paths.originalGraphPath) == sourceGraphBytes &&
                readAll(result.paths.originalRangesPath) == sourceRangeBytes,
            "preprocess did not preserve the original source bytes");
    Require(chmv::streaming::preprocess::DatasetPreprocessor::HasValidPreprocess(
                graphPath, rangePath, config),
            "preprocess cache should validate against unchanged sources");
    const auto preprocessedIndex = chmv::streaming::index::CHIndex::Load(result.paths.indexPath);
    Require(preprocessedIndex.HasRootPayload(),
            "preprocess index did not persist the v6 binary root payload");
    Require(preprocessedIndex.rootPayloadRecordCount == sourceGraph.DrawableEdgeCount(),
            "preprocess root-payload count does not match drawable roots");
    Require(chmv::streaming::index::CHIndex::ReadEdgeSpatialBounds(
                preprocessedIndex, 0u, static_cast<std::uint32_t>(result.edgeCount)).size() ==
                result.edgeCount,
            "preprocess index did not persist one full-geometry spatial bound per edge");
    const auto firstRootPage = chmv::streaming::index::CHIndex::ReadGraphPageRootRecords(
        preprocessedIndex, 0u);
    Require(firstRootPage.size() == preprocessedIndex.graphPages.front().edgeCount,
            "preprocess root payload is not page-contiguous");

    const auto preprocessedGraph = chmv::data::CHLoader::Load(
        result.paths.graphPath, result.paths.rangesPath);
    Require(preprocessedGraph.NodeCount() == sourceGraph.NodeCount() &&
                preprocessedGraph.EdgeCount() == sourceGraph.EdgeCount(),
            "preprocess changed graph counts");
    Require(preprocessedGraph.ShortcutCount() == sourceGraph.ShortcutCount(),
            "preprocess changed shortcut count");

    auto nodeIds = [](const chmv::data::CHGraph& graph) {
        std::vector<std::uint64_t> ids;
        ids.reserve(graph.Nodes().size());
        for (const auto& node : graph.Nodes()) {
            ids.push_back(node.osmId);
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    };
    Require(nodeIds(sourceGraph) == nodeIds(preprocessedGraph),
            "preprocess changed the node set");

    using EdgeKey = std::tuple<std::uint64_t, std::uint64_t, std::uint32_t,
                               std::int32_t, std::int32_t, std::int32_t,
                               std::int32_t, bool>;
    const auto edgeKeys = [](const chmv::data::CHGraph& graph) {
        std::vector<EdgeKey> keys;
        keys.reserve(graph.Edges().size());
        for (std::size_t edgeId = 0; edgeId < graph.Edges().size(); ++edgeId) {
            const auto& edge = graph.Edges()[edgeId];
            const auto& range = graph.Ranges()[edgeId];
            keys.emplace_back(graph.Nodes()[edge.source].osmId,
                              graph.Nodes()[edge.target].osmId,
                              std::bit_cast<std::uint32_t>(edge.weight), edge.type,
                              edge.maxSpeed, range.birthLevel, range.deathLevel,
                              edge.IsShortcut());
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    };
    Require(edgeKeys(sourceGraph) == edgeKeys(preprocessedGraph),
            "preprocess changed edge/range semantics");
    for (const auto& edge : preprocessedGraph.Edges()) {
        if (edge.IsShortcut()) {
            Require(edge.childA < preprocessedGraph.EdgeCount() &&
                        edge.childB < preprocessedGraph.EdgeCount(),
                    "preprocess produced an invalid shortcut reference");
        }
    }

    const auto reportDirectory =
        chmv::streaming::analysis::DatasetLayoutAnalyzer::DefaultReportDirectory(graphPath);
    Require(std::filesystem::is_regular_file(reportDirectory / "preprocess_summary.txt"),
            "preprocess should always write the lightweight index-derived summary");

    chmv::streaming::preprocess::PreprocessOptions reuseOptions = preprocessOptions;
    reuseOptions.writeDetailedAnalysis = true;
    const auto reused = chmv::streaming::preprocess::DatasetPreprocessor::Run(
        graphPath, rangePath, config, false, {}, reuseOptions);
    Require(reused.reused, "matching preprocess should be reused");
    Require(std::filesystem::is_regular_file(reportDirectory / "summary.txt"),
            "reusing a small preprocess should restore the detailed layout reports");

    chmv::data::DatasetCatalog catalog(root);
    Require(catalog.Entries().size() == 1,
            "dataset catalog should hide internal preprocess copies");

    std::filesystem::remove_all(root);
}

void TestGraphPageStreamer() {
    const auto base = std::filesystem::temp_directory_path() / "chmv_streaming_test";
    const auto graphPath = base.string() + ".sch";
    const auto rangePath = base.string() + ".sch.ranges";
    const auto indexPath = base.string() + ".chidx";

    {
        std::ofstream graph(graphPath);
        graph << "4\n4\n";
        graph << "0 100 48.0 9.0 0 3\n";
        graph << "1 101 48.0 9.1 0 2\n";
        graph << "2 102 48.1 9.1 0 1\n";
        graph << "3 103 48.1 9.2 0 0\n";
        graph << "0 1 1.0 1 50 -1 -1\n";
        graph << "1 2 1.0 1 50 -1 -1\n";
        graph << "2 3 1.0 1 50 -1 -1\n";
        graph << "0 2 2.0 1 50 0 1\n";
    }
    {
        std::ofstream ranges(rangePath);
        ranges << "0 3 0\n";
        ranges << "1 3 0\n";
        ranges << "2 -1 -1\n";
        ranges << "3 3 2\n";
    }

    chmv::streaming::analysis::DatasetLayoutAnalysisConfig analysisConfig;
    analysisConfig.spatialCellSizeMeters = 4096.0;
    analysisConfig.lodBandWidth = 2;
    analysisConfig.sourceBlockNodeCount = 2;
    analysisConfig.sourceBlockEdgeCount = 2;
    analysisConfig.virtualPageEdgeCount = 2;
    chmv::streaming::preprocess::PreprocessOptions preprocessOptions;
    preprocessOptions.memoryBudgetBytes = 64ull * 1024ull * 1024ull;
    preprocessOptions.writeDetailedAnalysis = false;
    const auto preprocessResult = chmv::streaming::preprocess::DatasetPreprocessor::Run(
        graphPath, rangePath, analysisConfig, true, {}, preprocessOptions);
    auto index = chmv::streaming::index::CHIndex::Load(preprocessResult.paths.indexPath);
    Require(index.HasRootPayload(),
            "preprocess index did not persist the v6 binary root payload");
    Require(chmv::streaming::index::CHIndex::ReadEdgeSpatialBounds(
                index, 0u, static_cast<std::uint32_t>(index.edgeCount)).size() == index.edgeCount,
            "preprocess index did not persist full-geometry edge spatial bounds");
    Require(std::abs(index.spatialCellSizeMeters - analysisConfig.spatialCellSizeMeters) < 1e-9,
            "preprocess index did not persist the fixed world-space cell size");
    const auto streamedGraph = chmv::data::CHLoader::Load(graphPath, rangePath);

    chmv::streaming::runtime::GraphPageStreamerConfig streamingConfig;
    Require(streamingConfig.spatialPrefetchRadius == 3,
            "runtime streaming must default to a 7x7 fixed-grid spatial neighborhood");
    Require(streamingConfig.lodPrefetchNearSpatialRadius == 1 &&
                streamingConfig.lodPrefetchNearDistance == 2 &&
                streamingConfig.lodPrefetchGpuWarmDistance == 2,
            "runtime streaming must separate wide RAM LOD prefetch from a narrow GPU warm band");
    Require(!streamingConfig.prefetchLodChildren,
            "runtime streaming must not prefetch LOD children before GPU misses request them");
    streamingConfig.ramBudgetBytes = 16ull * 1024ull * 1024ull;
    streamingConfig.spatialPrefetchRadius = 1;
    {
        chmv::streaming::runtime::GraphPageStreamer streamer(
            std::move(index), graphPath, rangePath, streamingConfig);
        chmv::streaming::runtime::StreamingViewRequest view;
        view.lodLevel = 2;
        view.minLatitude = 47.9;
        view.minLongitude = 8.9;
        view.maxLatitude = 48.2;
        view.maxLongitude = 9.3;
        const auto token = streamer.RequestView(view);
        Require(token.requiredPageCount > 0, "streaming view did not request pages");
        const auto repeatedToken = streamer.RequestView(view);
        Require(repeatedToken.generation == token.generation,
                "identical streaming demand should not restart the request generation");
        Require(streamer.WaitForView(token.generation, std::chrono::seconds(2)),
                "streaming demand pages did not become resident");
        const auto stats = streamer.Snapshot();
        Require(stats.currentDemandReady, "streaming demand should be ready");
        Require(stats.requiredResidentCount == stats.requiredPageCount,
                "streaming required residency mismatch");
        Require(stats.totalCpuCacheBytes <= stats.ramBudgetBytes,
                "combined streaming caches exceeded the configured hard RAM budget");
        Require(stats.residentBytes <= stats.dataCacheBudgetBytes,
                "page/refinement cache exceeded its partition of the RAM budget");
        Require(stats.totalPageLoads > 0, "streaming cache did not load a page");
        const auto pageLoadsBeforeHit = stats.totalPageLoads;
        static_cast<void>(streamer.RequestView(view));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        Require(streamer.Snapshot().totalPageLoads == pageLoadsBeforeHit,
                "stable view unexpectedly re-read an already resident graph page");
        Require(!streamer.DesiredPageIds().empty(),
                "streaming runtime should expose the current 7x7 residency page set");
        const auto viewportWindow = streamer.CurrentViewportWindow();
        const auto residencyWindow = streamer.CurrentRefinementWindow();
        Require(viewportWindow.minX >= residencyWindow.minX &&
                    viewportWindow.minY >= residencyWindow.minY &&
                    viewportWindow.maxX <= residencyWindow.maxX &&
                    viewportWindow.maxY <= residencyWindow.maxY,
                "strict viewport window must be contained by the residency neighborhood");

        std::uint32_t shortcutId = chmv::data::InvalidEdgeId;
        for (std::uint32_t edgeId = 0; edgeId < streamedGraph.EdgeCount(); ++edgeId) {
            if (streamedGraph.Edges()[edgeId].IsShortcut()) {
                shortcutId = edgeId;
                break;
            }
        }
        Require(shortcutId != chmv::data::InvalidEdgeId,
                "preprocessed streaming test graph lost its shortcut");
        const auto requestedChild = streamedGraph.Edges()[shortcutId].childA;
        const auto requestedBlock = requestedChild / analysisConfig.sourceBlockEdgeCount;
        const chmv::streaming::runtime::RefinementEdgeRequest refinementRequest{
            shortcutId, requestedChild};
        const auto refinementToken = streamer.RequestRefinementEdges(
            std::span<const chmv::streaming::runtime::RefinementEdgeRequest>(
                &refinementRequest, 1));
        Require(refinementToken.uniqueBlockCount == 1,
                "refinement request should resolve one physical edge source block");
        const auto refinementDeadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!streamer.FindResidentRefinementBlock(requestedBlock) &&
               std::chrono::steady_clock::now() < refinementDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto block = streamer.FindResidentRefinementBlock(requestedBlock);
        Require(block && !block->edges.empty(),
                "requested refinement block did not become resident");
        Require(std::any_of(block->edges.begin(), block->edges.end(),
                            [requestedChild](const auto& edge) {
                                return edge.globalEdgeId == requestedChild;
                            }),
                "refinement block does not contain the requested child edge");

        // Direct GPU block faults are block-granular and must survive a camera generation
        // change. A small pan must not cancel hierarchy I/O that is still useful to the
        // persistent GPU cache.
        const std::uint32_t directBlock = 0;
        streamer.RequestBackingBlocks(std::span<const std::uint32_t>(&directBlock, 1));
        auto movedView = view;
        movedView.minLongitude += 0.01;
        movedView.maxLongitude += 0.01;
        static_cast<void>(streamer.RequestView(movedView));
        const auto backingDeadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!streamer.FindResidentRefinementBlock(directBlock) &&
               std::chrono::steady_clock::now() < backingDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Require(streamer.FindResidentRefinementBlock(directBlock) != nullptr,
                "persistent backing-block request was lost across a view generation");
        streamer.ReleaseBackingBlocks(std::span<const std::uint32_t>(&directBlock, 1));

        const auto loadsBeforeCacheHit = streamer.Snapshot().totalRefinementBlockLoads;
        streamer.RequestBackingBlocks(std::span<const std::uint32_t>(&directBlock, 1));
        const auto hitDeadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (!streamer.FindResidentRefinementBlock(directBlock) &&
               std::chrono::steady_clock::now() < hitDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto hitStats = streamer.Snapshot();
        Require(hitStats.totalRefinementBlockLoads == loadsBeforeCacheHit,
                "RAM backing-block cache hit unexpectedly re-read the source file");
        Require(hitStats.refinementBlockCacheHits > 0,
                "RAM backing-block cache did not report the cache hit");
        streamer.ReleaseBackingBlocks(std::span<const std::uint32_t>(&directBlock, 1));

        streamer.ReconfigureBudgets(64ull * 1024ull * 1024ull,
                                    8ull * 1024ull * 1024ull);
        const auto boundedStats = streamer.Snapshot();
        Require(boundedStats.totalCpuCacheBytes <= boundedStats.ramBudgetBytes,
                "runtime budget reconfiguration did not hard-bound total CPU cache memory");
    }

    const auto preprocessPaths = chmv::streaming::preprocess::DatasetPreprocessor::DefaultPaths(
        graphPath, rangePath);
    std::filesystem::remove(preprocessPaths.graphPath);
    std::filesystem::remove(preprocessPaths.rangesPath);
    std::filesystem::remove(preprocessPaths.indexPath);
    std::filesystem::remove(preprocessPaths.manifestPath);
    std::filesystem::remove(preprocessPaths.originalGraphPath);
    std::filesystem::remove(preprocessPaths.originalRangesPath);
}


void TestFixedGridBoundaryAndForwardShortcut() {
    const auto base = std::filesystem::temp_directory_path() / "chmv_fixed_grid_boundary_test";
    const auto graphPath = base.string() + ".sch";
    const auto rangePath = base.string() + ".sch.ranges";

    // 4096 * 4096 m is an exact power-of-two fixed-grid boundary in Web Mercator.
    // A tiny road crossing it must remain a small-scale page; the old aligned-quadtree
    // assignment would incorrectly promote this case to an enormous ancestor cell.
    constexpr double boundaryLongitude = -29.2877044222542;
    {
        std::ofstream graph(graphPath);
        graph << "3\n3\n";
        graph << "0 100 0.0 " << (boundaryLongitude - 0.001) << " 0 3\n";
        graph << "1 101 0.0 " << (boundaryLongitude + 0.001) << " 0 2\n";
        graph << "2 102 0.0 " << (boundaryLongitude + 0.002) << " 0 1\n";
        // Parent deliberately precedes its children to exercise the general DFS fallback.
        graph << "0 2 2.0 1 50 1 2\n";
        graph << "0 1 1.0 1 50 -1 -1\n";
        graph << "1 2 1.0 1 50 -1 -1\n";
    }
    {
        std::ofstream ranges(rangePath);
        ranges << "0 3 0\n";
        ranges << "1 -1 -1\n";
        ranges << "2 -1 -1\n";
    }

    chmv::streaming::analysis::DatasetLayoutAnalysisConfig config;
    config.spatialCellSizeMeters = 4096.0;
    config.sourceBlockNodeCount = 2;
    config.sourceBlockEdgeCount = 2;
    config.virtualPageEdgeCount = 2;
    chmv::streaming::preprocess::PreprocessOptions options;
    options.memoryBudgetBytes = 64ull * 1024ull * 1024ull;
    options.writeDetailedAnalysis = false;
    const auto result = chmv::streaming::preprocess::DatasetPreprocessor::Run(
        graphPath, rangePath, config, true, {}, options);
    auto index = chmv::streaming::index::CHIndex::Load(result.paths.indexPath);
    Require(index.graphPages.size() == 1,
            "fixed-grid boundary test should create exactly one drawable page");
    Require(index.graphPages.front().spatialLevel <= 1,
            "short edge crossing a power-of-two boundary was promoted to a coarse spatial scale");

    chmv::streaming::runtime::GraphPageStreamerConfig streamingConfig;
    streamingConfig.ramBudgetBytes = 32ull * 1024ull * 1024ull;
    streamingConfig.spatialPrefetchRadius = 0;
    {
        chmv::streaming::runtime::GraphPageStreamer streamer(
            std::move(index), graphPath, rangePath, streamingConfig);
        chmv::streaming::runtime::StreamingViewRequest view;
        view.lodLevel = 2;
        view.minLatitude = -0.001;
        view.maxLatitude = 0.001;
        view.minLongitude = boundaryLongitude - 0.0015;
        view.maxLongitude = boundaryLongitude - 0.0005;
        const auto token = streamer.RequestView(view);
        Require(token.requiredPageCount == 1,
                "fixed-grid sparse lookup missed a page crossing a cell boundary");
        Require(streamer.WaitForView(token.generation, std::chrono::seconds(2)),
                "fixed-grid boundary page did not become resident");
    }

    const auto paths = chmv::streaming::preprocess::DatasetPreprocessor::DefaultPaths(
        graphPath, rangePath);
    std::filesystem::remove(paths.graphPath);
    std::filesystem::remove(paths.rangesPath);
    std::filesystem::remove(paths.indexPath);
    std::filesystem::remove(paths.manifestPath);
    std::filesystem::remove(paths.originalGraphPath);
    std::filesystem::remove(paths.originalRangesPath);
}


void TestMixedOrderShortcutGeometryBounds() {
    const auto base = std::filesystem::temp_directory_path() / "chmv_mixed_order_bounds_test";
    const auto graphPath = base.string() + ".sch";
    const auto rangePath = base.string() + ".sch.ranges";

    // A valid shortcut DAG whose EdgeIDs alternate dependency direction. The root's own
    // endpoints are close together, while the child path detours far north. If preprocessing
    // ever treats direct endpoint bounds as final, the root would incorrectly stay in L0.
    {
        std::ofstream graph(graphPath);
        graph << "5\n7\n";
        graph << "0 100 48.0000 9.0000 0 5\n";
        graph << "1 101 48.0000 9.0010 0 4\n";
        graph << "2 102 48.1000 9.0000 0 3\n";
        graph << "3 103 48.2500 9.0000 0 2\n";
        graph << "4 104 48.1000 9.0010 0 1\n";
        // 0 -> (1,4), 4 -> (2,5), 2 -> (3,6). This requires alternating sweeps.
        graph << "0 1 4.0 1 50 1 4\n";   // edge 0, root shortcut
        graph << "0 2 1.0 1 50 -1 -1\n"; // edge 1
        graph << "2 4 2.0 1 50 3 6\n";   // edge 2
        graph << "2 3 1.0 1 50 -1 -1\n"; // edge 3
        graph << "2 1 3.0 1 50 2 5\n";   // edge 4
        graph << "4 1 1.0 1 50 -1 -1\n"; // edge 5
        graph << "3 4 1.0 1 50 -1 -1\n"; // edge 6
    }
    {
        std::ofstream ranges(rangePath);
        ranges << "0 8 0\n";
        for (std::uint32_t edge = 1; edge < 7; ++edge) {
            ranges << edge << " -1 -1\n";
        }
    }

    chmv::streaming::analysis::DatasetLayoutAnalysisConfig config;
    config.spatialCellSizeMeters = 4096.0;
    config.sourceBlockNodeCount = 2;
    config.sourceBlockEdgeCount = 2;
    config.virtualPageEdgeCount = 2;
    chmv::streaming::preprocess::PreprocessOptions options;
    options.memoryBudgetBytes = 64ull * 1024ull * 1024ull;
    options.writeDetailedAnalysis = false;
    const auto result = chmv::streaming::preprocess::DatasetPreprocessor::Run(
        graphPath, rangePath, config, true, {}, options);

    const auto graph = chmv::data::CHLoader::Load(result.paths.graphPath, result.paths.rangesPath);
    const auto index = chmv::streaming::index::CHIndex::Load(result.paths.indexPath);
    std::uint32_t rootEdge = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t edgeId = 0; edgeId < graph.EdgeCount(); ++edgeId) {
        if (graph.Ranges()[edgeId].birthLevel == 8) {
            rootEdge = edgeId;
            break;
        }
    }
    Require(rootEdge != std::numeric_limits<std::uint32_t>::max(),
            "mixed-order bounds test lost its drawable shortcut root");
    const auto packed = chmv::streaming::index::CHIndex::ReadEdgeSpatialBounds(
                            index, rootEdge, 1u)
                            .front();
    const auto minY = static_cast<std::uint32_t>((packed >> 16u) & 0xffffu);
    const auto maxY = static_cast<std::uint32_t>((packed >> 48u) & 0xffffu);
    Require(maxY > minY + 2u,
            "mixed-order shortcut resolver lost child geometry from the full root bounds");
    Require(!index.graphPages.empty() && index.graphPages.front().spatialLevel >= 2u,
            "mixed-order shortcut root was assigned from endpoint-only bounds");

    const auto paths = chmv::streaming::preprocess::DatasetPreprocessor::DefaultPaths(
        graphPath, rangePath);
    std::filesystem::remove(paths.graphPath);
    std::filesystem::remove(paths.rangesPath);
    std::filesystem::remove(paths.indexPath);
    std::filesystem::remove(paths.manifestPath);
    std::filesystem::remove(paths.originalGraphPath);
    std::filesystem::remove(paths.originalRangesPath);
}


void TestFixedGridPhysicalPagePacking() {
    const auto root = std::filesystem::temp_directory_path() / "chmv_fixed_grid_packing_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto graphPath = root / "packing.sch";
    const auto rangePath = root / "packing.sch.ranges";

    constexpr double pi = 3.14159265358979323846;
    constexpr double radius = 6378137.0;
    constexpr double halfExtent = pi * radius;
    constexpr double cellMeters = 4096.0;
    const auto longitudeForMeters = [](double xMeters) {
        return (xMeters - halfExtent) / radius * 180.0 / pi;
    };

    constexpr std::uint32_t edgeCount = 8;
    constexpr std::uint32_t nodeCount = edgeCount * 2u;
    {
        std::ofstream graph(graphPath);
        graph << nodeCount << '\n' << edgeCount << '\n';
        for (std::uint32_t edge = 0; edge < edgeCount; ++edge) {
            // Span several former 64x64 physical pack regions. Spatial lookup must stay exact,
            // but those artificial boundaries must no longer split a mostly-empty physical page.
            const auto cellX = 4780u + edge * 20u;
            const auto centerMeters = (static_cast<double>(cellX) + 0.5) * cellMeters;
            const auto lonA = longitudeForMeters(centerMeters - 100.0);
            const auto lonB = longitudeForMeters(centerMeters + 100.0);
            const auto a = edge * 2u;
            const auto b = a + 1u;
            graph << a << ' ' << (1000u + a) << " 0 " << lonA << " 0 1\n";
            graph << b << ' ' << (1000u + b) << " 0 " << lonB << " 0 1\n";
        }
        for (std::uint32_t edge = 0; edge < edgeCount; ++edge) {
            graph << edge * 2u << ' ' << edge * 2u + 1u
                  << " 1 1 50 -1 -1\n";
        }
    }
    {
        std::ofstream ranges(rangePath);
        for (std::uint32_t edge = 0; edge < edgeCount; ++edge) {
            ranges << edge << " 5 0\n";
        }
    }

    chmv::streaming::analysis::DatasetLayoutAnalysisConfig config;
    config.spatialCellSizeMeters = cellMeters;
    config.sourceBlockNodeCount = 16;
    config.sourceBlockEdgeCount = 16;
    config.virtualPageEdgeCount = 16;
    chmv::streaming::preprocess::PreprocessOptions options;
    options.memoryBudgetBytes = 64ull * 1024ull * 1024ull;
    options.writeDetailedAnalysis = false;
    const auto result = chmv::streaming::preprocess::DatasetPreprocessor::Run(
        graphPath, rangePath, config, true, {}, options);
    const auto index = chmv::streaming::index::CHIndex::Load(result.paths.indexPath);

    Require(index.graphPages.size() == 1,
            "one LOD group should pack densely across former fixed 64x64 region boundaries");
    Require(index.graphPages.front().edgeCount == edgeCount,
            "packed physical page lost drawable edges");
    Require(index.spatialPageRefs.size() == edgeCount,
            "packed physical page must retain exact cell-to-page lookup references");
    Require(std::all_of(index.spatialPageRefs.begin(), index.spatialPageRefs.end(),
                        [](const auto& ref) { return ref.pageId == 0; }),
            "spatial cells should all reference the shared dense packed page");

    std::filesystem::remove_all(root);
}

void TestCorrectnessValidator() {
    const std::vector<std::uint32_t> cpu{1, 2, 2, 4};
    const std::vector<std::uint32_t> gpuPass{4, 2, 1, 2};
    const auto pass = chmv::benchmark::CorrectnessValidator::Compare(cpu, gpuPass);
    Require(pass.Passed(), "validator should ignore output order");
    Require(pass.cpuDuplicateCount == 1 && pass.gpuDuplicateCount == 1,
            "duplicate counts mismatch");

    const std::vector<std::uint32_t> gpuFail{1, 2, 3, 4};
    const auto fail = chmv::benchmark::CorrectnessValidator::Compare(cpu, gpuFail);
    Require(!fail.Passed(), "validator should detect different edge multisets");
    Require(fail.missingEdgeCount == 1, "missing edge count mismatch");
    Require(fail.unexpectedEdgeCount == 1, "unexpected edge count mismatch");
}

} // namespace

int main() {
    try {
        TestLoaderAndReferenceUnfolding();
        TestCPURangeFilterAndPipeline();
        TestOrderedRangeRetrieval();
        TestAsyncDatasetLoader();
        TestDatasetLayoutAnalyzer();
        TestHierarchicalSpatialOwnership();
        TestDatasetPreprocessor();
        TestGraphPageStreamer();
        TestFixedGridBoundaryAndForwardShortcut();
        TestMixedOrderShortcutGeometryBounds();
        TestFixedGridPhysicalPagePacking();
        TestCorrectnessValidator();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
