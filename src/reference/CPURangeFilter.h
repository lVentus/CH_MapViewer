#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chmv::data {
class CHGraph;
}

namespace chmv::reference {

struct CPURangeFilterStats {
    std::size_t scannedEdgeCount = 0;
    std::size_t outputEdgeCount = 0;
};

class CPURangeFilter {
public:
    void SetGraph(const data::CHGraph& graph);
    [[nodiscard]] CPURangeFilterStats Filter(const data::CHGraph& graph, std::int32_t level,
                                             std::vector<std::uint32_t>& output);

private:
    const data::CHGraph* graph_ = nullptr;
};

} // namespace chmv::reference
