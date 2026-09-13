#pragma once

#include "data/ch/CHGraph.h"

#include <cstdint>
#include <filesystem>
#include <functional>

namespace chmv::data {

enum class CHLoadStage {
    Nodes,
    Edges,
    Ranges,
};

struct CHLoadProgress {
    CHLoadStage stage = CHLoadStage::Nodes;
    std::uint64_t current = 0;
    std::uint64_t total = 0;
    float overall = 0.0f;
};

class CHLoader {
public:
    using ProgressCallback = std::function<void(const CHLoadProgress&)>;

    [[nodiscard]] static CHGraph Load(const std::filesystem::path& graphPath,
                                      const std::filesystem::path& rangesPath,
                                      ProgressCallback progressCallback = {});
};

} // namespace chmv::data
