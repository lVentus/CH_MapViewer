#include "data/ch/CHLoader.h"
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

void TestLoaderAndReferenceUnfolding() {
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

    const auto graph = chmv::data::CHLoader::Load(graphPath, rangePath);
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

    std::filesystem::remove(graphPath);
    std::filesystem::remove(rangePath);
}

} // namespace

int main() {
    try {
        TestLoaderAndReferenceUnfolding();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
