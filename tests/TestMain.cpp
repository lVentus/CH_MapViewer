#include "benchmark/CorrectnessValidator.h"
#include "data/ch/CHLoader.h"
#include "pipeline/CPUReferencePipeline.h"
#include "reference/CPURangeFilter.h"
#include "reference/CPUReferenceUnfolder.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
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
        graph << "1 101 48.1 9.1 0 1\n";
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
    filter.Filter(graph, 1, alive);
    Require(alive.size() == 3, "unexpected CPU range-filter result");

    chmv::pipeline::CPUReferencePipeline pipeline;
    const auto result = pipeline.Process(graph, 1, true);
    Require(result.stats.aliveEdgeCount == 3, "unexpected CPU pipeline alive count");
    Require(result.edgeIds.size() == 4, "unexpected CPU pipeline unfolded count");
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
        TestCorrectnessValidator();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
