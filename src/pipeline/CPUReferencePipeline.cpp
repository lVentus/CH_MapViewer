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

} // namespace

CPUProcessingResult CPUReferencePipeline::Process(const data::CHGraph& graph,
                                                   std::int32_t lodLevel,
                                                   bool fullyUnfold) {
    const auto filterStart = Clock::now();
    rangeFilter_.Filter(graph, lodLevel, aliveEdges_);
    const auto filterEnd = Clock::now();

    CPUProcessingResult result;
    result.aliveEdgeIds = aliveEdges_;
    result.stats.aliveEdgeCount = aliveEdges_.size();
    result.stats.rangeFilterMs = Milliseconds(filterEnd - filterStart);

    if (!fullyUnfold) {
        result.edgeIds = aliveEdges_;
        result.stats.outputEdgeCount = aliveEdges_.size();
        return result;
    }

    const auto unfoldStart = Clock::now();
    reference::CPUReferenceUnfolder(graph).UnfoldFully(aliveEdges_, unfoldedEdges_);
    const auto unfoldEnd = Clock::now();

    result.edgeIds = unfoldedEdges_;
    result.stats.outputEdgeCount = unfoldedEdges_.size();
    result.stats.unfoldingMs = Milliseconds(unfoldEnd - unfoldStart);
    return result;
}

} // namespace chmv::pipeline
