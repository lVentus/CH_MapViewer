#pragma once

#include <cstdint>
#include <vector>

namespace chmv::data {
class CHGraph;
}

namespace chmv::reference {

class CPURangeFilter {
public:
    void Filter(const data::CHGraph& graph, std::int32_t level,
                std::vector<std::uint32_t>& output) const;
};

} // namespace chmv::reference
