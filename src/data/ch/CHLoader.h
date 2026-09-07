#pragma once

#include "data/ch/CHGraph.h"

#include <filesystem>

namespace chmv::data {

class CHLoader {
public:
    [[nodiscard]] static CHGraph Load(const std::filesystem::path& graphPath,
                                      const std::filesystem::path& rangesPath);
};

} // namespace chmv::data
