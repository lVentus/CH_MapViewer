#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace chmv::benchmark {

struct ValidationResult {
    std::size_t cpuEdgeCount = 0;
    std::size_t gpuEdgeCount = 0;
    std::size_t missingEdgeCount = 0;
    std::size_t unexpectedEdgeCount = 0;
    std::size_t cpuDuplicateCount = 0;
    std::size_t gpuDuplicateCount = 0;

    [[nodiscard]] bool Passed() const {
        return missingEdgeCount == 0 && unexpectedEdgeCount == 0;
    }
};

struct PipelineValidationResult {
    ValidationResult rangeFilter;
    ValidationResult finalOutput;

    [[nodiscard]] bool Passed() const {
        return rangeFilter.Passed() && finalOutput.Passed();
    }
};

class CorrectnessValidator {
public:
    [[nodiscard]] static ValidationResult Compare(std::span<const std::uint32_t> cpuEdges,
                                                  std::span<const std::uint32_t> gpuEdges);
};

} // namespace chmv::benchmark
