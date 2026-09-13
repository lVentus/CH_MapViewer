#include "benchmark/CorrectnessValidator.h"
#include "data/AsyncDatasetLoader.h"
#include "data/ch/CHLoader.h"
#include "data/preprocessing/GeometryErrorPreprocessor.h"
#include "geometry/GeometryRefinement.h"
#include "pipeline/CPUReferencePipeline.h"
#include "reference/CPURangeFilter.h"
#include "reference/CPUReferenceUnfolder.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
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
    const auto base = std::filesystem::temp_directory_path() / "chmv_async_test";
    const auto graphPath = base.string() + ".sch";
    const auto rangePath = base.string() + ".sch.ranges";

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

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
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

    std::filesystem::remove(graphPath);
    std::filesystem::remove(rangePath);
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
        TestCorrectnessValidator();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
