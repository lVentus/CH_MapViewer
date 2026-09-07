#include "reference/CPURangeFilter.h"

#include "data/ch/CHGraph.h"

namespace chmv::reference {

void CPURangeFilter::Filter(const data::CHGraph& graph, std::int32_t level,
                            std::vector<std::uint32_t>& output) const {
    output.clear();
    if (output.capacity() < graph.DrawableEdgeCount()) {
        output.reserve(graph.DrawableEdgeCount());
    }

    for (std::uint32_t edgeId = 0; edgeId < graph.EdgeCount(); ++edgeId) {
        if (graph.Range(edgeId).IsAlive(level)) {
            output.push_back(edgeId);
        }
    }
}

} // namespace chmv::reference
