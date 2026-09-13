#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/filtering/IRangeFilterStrategy.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace chmv::gpu::filtering {

class FullScanRangeFilterStrategy final : public IRangeFilterStrategy {
public:
    explicit FullScanRangeFilterStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Full Scan"; }
    void SetGraph(const data::CHGraph& graph) override;
    [[nodiscard]] RangeFilterExecutionStats Execute(const RangeFilterInput& input) override;
    [[nodiscard]] RangeFilterDiagnostics ReadBackDiagnostics(std::int32_t lodLevel) const override;

private:
    ComputeProgram filterProgram_;
    std::uint32_t edgeCount_ = 0;
};

} // namespace chmv::gpu::filtering
