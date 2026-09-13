#include "pipeline/CPUReferencePipeline.h"

#include "data/ch/CHGraph.h"
#include "reference/CPUReferenceUnfolder.h"

#include <chrono>

namespace chmv::pipeline {
namespace {

using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

bool SameRefinement(const geometry::RefinementParameters& a,
                    const geometry::RefinementParameters& b) {
    if (a.mode != b.mode) {
        return false;
    }
    if (a.mode == geometry::RefinementMode::Adaptive) {
        return a.screenPixelScale == b.screenPixelScale &&
               a.maxScreenErrorPixels == b.maxScreenErrorPixels;
    }
    return true;
}

} // namespace

void CPUReferencePipeline::SetGraph(const data::CHGraph& graph) {
    graph_ = &graph;
    rangeFilter_.SetGraph(graph);
    aliveEdges_.clear();
    refinedEdges_.clear();
    refinementStack_.clear();
    cachedStats_ = {};
    cachedRefinement_ = {};
    cacheValid_ = false;
}

CPUProcessingResult CPUReferencePipeline::Process(
    const data::CHGraph& graph, std::int32_t lodLevel,
    const geometry::RefinementParameters& refinementParameters) {
    if (graph_ != &graph) {
        SetGraph(graph);
    }

    if (cacheValid_ && cachedLodLevel_ == lodLevel &&
        SameRefinement(cachedRefinement_, refinementParameters)) {
        auto stats = cachedStats_;
        stats.cacheHit = true;
        return {
            aliveEdges_,
            refinementParameters.Active() ? std::span<const std::uint32_t>(refinedEdges_)
                                          : std::span<const std::uint32_t>(aliveEdges_),
            stats,
        };
    }

    const auto filterStart = Clock::now();
    const auto filterStats = rangeFilter_.Filter(graph, lodLevel, aliveEdges_);
    const auto filterEnd = Clock::now();

    cachedStats_ = {};
    cachedStats_.rangeScannedEdgeCount = filterStats.scannedEdgeCount;
    cachedStats_.aliveEdgeCount = aliveEdges_.size();
    cachedStats_.rangeFilterMs = Milliseconds(filterEnd - filterStart);

    if (refinementParameters.Active()) {
        const auto refinementStart = Clock::now();
        reference::CPUReferenceUnfolder unfolder(graph);
        if (refinementParameters.mode == geometry::RefinementMode::Full) {
            unfolder.UnfoldFully(aliveEdges_, refinedEdges_, refinementStack_);
        } else {
            unfolder.UnfoldAdaptive(aliveEdges_, refinedEdges_, refinementStack_,
                                    refinementParameters.screenPixelScale,
                                    refinementParameters.maxScreenErrorPixels);
        }
        const auto refinementEnd = Clock::now();
        cachedStats_.outputEdgeCount = refinedEdges_.size();
        cachedStats_.geometryRefinementMs = Milliseconds(refinementEnd - refinementStart);
    } else {
        cachedStats_.outputEdgeCount = aliveEdges_.size();
    }

    cachedLodLevel_ = lodLevel;
    cachedRefinement_ = refinementParameters;
    cacheValid_ = true;

    return {
        aliveEdges_,
        refinementParameters.Active() ? std::span<const std::uint32_t>(refinedEdges_)
                                      : std::span<const std::uint32_t>(aliveEdges_),
        cachedStats_,
    };
}

} // namespace chmv::pipeline
