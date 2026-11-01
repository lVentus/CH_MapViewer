#pragma once

#include "geometry/GeometryRefinement.h"
#include "streaming/lod/LodPageFormat.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace chmv::gpu::lod {
class LodPagePool;
}

namespace chmv::gpu::unfolding {

// One expansion pass: turns the candidate list of a filter into the segments that are drawn
// (slot << 8 | record, appended to the segment buffer, counted in the draw command). Every
// strategy with the same threshold draws the same roads inside the view; they differ in how
// they get there.
struct UnfoldingInput {
    const lod::LodPagePool* pool = nullptr;
    std::uint32_t candidateList = 0;
    std::uint32_t candidateCapacity = 0;
    std::uint32_t level = 0;
    // Error code from ThresholdFor(Mode(), tolerance).
    std::uint32_t threshold = 0;
    // For traversals: view in fixed coordinates (minX, minY, maxX, maxY), the hierarchy height
    // and the uniform buffer of ErrorBoundMantissa() values.
    std::array<std::int32_t, 4> cullBox{};
    std::uint32_t hierarchyHeight = 0;
    std::uint32_t errorBounds = 0;
    std::uint32_t segmentBuffer = 0;
    std::uint32_t segmentCapacity = 0;
    // [vertexCount, instanceCount, firstVertex, baseInstance, segmentCount], count zero on entry.
    std::uint32_t drawCommand = 0;
    std::uint32_t statusBuffer = 0;
};

class IUnfoldingStrategy {
public:
    virtual ~IUnfoldingStrategy() = default;

    [[nodiscard]] virtual std::string_view Name() const = 0;
    // What is drawn: the roots as they are (None), every original edge (Full), or the coarsest
    // cut within the screen tolerance (Adaptive).
    [[nodiscard]] virtual geometry::RefinementMode Mode() const = 0;
    // Walks the hierarchy, so it needs the hierarchy extension of the page file.
    [[nodiscard]] virtual bool Traverses() const { return false; }
    [[nodiscard]] bool RefinesGeometry() const { return Mode() != geometry::RefinementMode::None; }
    [[nodiscard]] bool FullyUnfolds() const { return Mode() == geometry::RefinementMode::Full; }
    virtual void Execute(const UnfoldingInput& input) = 0;
};

// Error threshold of a mode: None keeps the largest code (only roots are drawn), Full keeps
// nothing but exact geometry (collinear shortcuts without error stay one segment), Adaptive
// uses the screen tolerance.
[[nodiscard]] inline std::uint32_t ThresholdFor(geometry::RefinementMode mode,
                                                std::uint32_t toleranceCode) {
    switch (mode) {
    case geometry::RefinementMode::None:
        return streaming::lod::kErrorCodeMax;
    case geometry::RefinementMode::Full:
        return 0u;
    case geometry::RefinementMode::Adaptive:
        break;
    }
    return toleranceCode;
}

} // namespace chmv::gpu::unfolding
