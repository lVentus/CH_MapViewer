#pragma once

#include <functional>

namespace chmv::data {
class CHGraph;
}

namespace chmv::data::preprocessing {

class GeometryErrorPreprocessor {
public:
    using ProgressCallback = std::function<void(float)>;

    static void Run(CHGraph& graph, ProgressCallback progressCallback = {});
};

} // namespace chmv::data::preprocessing
