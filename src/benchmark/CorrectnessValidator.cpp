#include "benchmark/CorrectnessValidator.h"

#include <algorithm>
#include <vector>

namespace chmv::benchmark {
namespace {

std::size_t DuplicateCount(std::span<const std::uint32_t> sortedEdges) {
    std::size_t duplicates = 0;
    for (std::size_t i = 1; i < sortedEdges.size(); ++i) {
        if (sortedEdges[i] == sortedEdges[i - 1]) {
            ++duplicates;
        }
    }
    return duplicates;
}

} // namespace

ValidationResult CorrectnessValidator::Compare(std::span<const std::uint32_t> cpuEdges,
                                                std::span<const std::uint32_t> gpuEdges) {
    std::vector<std::uint32_t> expected(cpuEdges.begin(), cpuEdges.end());
    std::vector<std::uint32_t> actual(gpuEdges.begin(), gpuEdges.end());
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());

    ValidationResult result;
    result.cpuEdgeCount = expected.size();
    result.gpuEdgeCount = actual.size();
    result.cpuDuplicateCount = DuplicateCount(expected);
    result.gpuDuplicateCount = DuplicateCount(actual);

    std::size_t cpuIndex = 0;
    std::size_t gpuIndex = 0;
    while (cpuIndex < expected.size() && gpuIndex < actual.size()) {
        if (expected[cpuIndex] == actual[gpuIndex]) {
            ++cpuIndex;
            ++gpuIndex;
        } else if (expected[cpuIndex] < actual[gpuIndex]) {
            ++result.missingEdgeCount;
            ++cpuIndex;
        } else {
            ++result.unexpectedEdgeCount;
            ++gpuIndex;
        }
    }

    result.missingEdgeCount += expected.size() - cpuIndex;
    result.unexpectedEdgeCount += actual.size() - gpuIndex;
    return result;
}

} // namespace chmv::benchmark
