#include "reference/CPURangeFilter.h"

#include "data/ch/CHGraph.h"

namespace chmv::reference {

void CPURangeFilter::SetGraph(const data::CHGraph& graph) {
    graph_ = &graph;
}

CPURangeFilterStats CPURangeFilter::Filter(const data::CHGraph& graph, std::int32_t level,
                                           std::vector<std::uint32_t>& output) {
    if (graph_ != &graph) {
        SetGraph(graph);
    }

    output.clear();
    const auto& index = graph.OrderedRanges();
    if (level < 0 || static_cast<std::size_t>(level) >= index.scanEndByLevel.size()) {
        return {};
    }

    const auto levelIndex = static_cast<std::size_t>(level);
    const auto scanEnd = index.scanEndByLevel[levelIndex];
    const auto expectedOutput = index.aliveCountByLevel[levelIndex];
    if (output.capacity() < expectedOutput) {
        output.reserve(expectedOutput);
    }

    for (std::size_t i = 0; i < scanEnd; ++i) {
        const auto& entry = index.entries[i];
        if (entry.deathLevel <= level) {
            output.push_back(entry.edgeId);
        }
    }

    return {
        scanEnd,
        output.size(),
    };
}

} // namespace chmv::reference
