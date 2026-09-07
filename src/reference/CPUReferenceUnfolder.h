#pragma once

#include "data/ch/CHGraph.h"

#include <cstdint>
#include <span>
#include <vector>

namespace chmv::reference {

class CPUReferenceUnfolder {
public:
    explicit CPUReferenceUnfolder(const data::CHGraph& graph) : graph_(graph) {}

    [[nodiscard]] std::vector<std::uint32_t> UnfoldFully(std::span<const std::uint32_t> roots) const;
    void UnfoldFully(std::span<const std::uint32_t> roots,
                     std::vector<std::uint32_t>& output) const;

private:
    const data::CHGraph& graph_;
};

} // namespace chmv::reference
