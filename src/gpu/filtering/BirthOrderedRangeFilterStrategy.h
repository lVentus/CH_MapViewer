#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/GPUBuffer.h"
#include "gpu/filtering/IRangeFilterStrategy.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>


namespace chmv::data {
struct OrderedRangeIndex;
}

namespace chmv::gpu::filtering {

class BirthOrderedRangeFilterStrategy final : public IRangeFilterStrategy {
public:
    explicit BirthOrderedRangeFilterStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Birth-Ordered"; }
    [[nodiscard]] bool SupportsStreamingWorkingSet() const override { return true; }
    [[nodiscard]] PersistentStreamingRangeFilterKind PersistentStreamingKind() const override {
        return PersistentStreamingRangeFilterKind::BirthOrdered;
    }
    void SetGraph(const data::CHGraph& graph) override;
    void SetStreamingRanges(std::span<const data::EdgeRange> ranges) override;
    [[nodiscard]] RangeFilterExecutionStats Execute(const RangeFilterInput& input) override;
    [[nodiscard]] RangeFilterDiagnostics ReadBackDiagnostics(std::int32_t lodLevel) const override;

private:
    void UploadIndex(const data::OrderedRangeIndex& index);
    ComputeProgram prepareDispatchProgram_;
    ComputeProgram filterProgram_;
    GPUBuffer orderedRangeBuffer_;
    GPUBuffer scanEndBuffer_;
    GPUBuffer dispatchBuffer_;
    std::uint32_t levelCount_ = 0;
};

} // namespace chmv::gpu::filtering
