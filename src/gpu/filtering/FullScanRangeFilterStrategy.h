#pragma once

#include "gpu/ComputeProgram.h"
#include "gpu/filtering/IRangeFilterStrategy.h"

#include <cstdint>
#include <filesystem>

namespace chmv::gpu::filtering {

// Tests every record of every planned page: one workgroup per page, one invocation per record.
// The baseline; the work does not depend on the level at all.
class FullScanRangeFilterStrategy final : public IRangeFilterStrategy {
public:
    explicit FullScanRangeFilterStrategy(const std::filesystem::path& shaderDirectory);

    [[nodiscard]] std::string_view Name() const override { return "Full scan"; }
    void Execute(const RangeFilterInput& input) override;

    // Runs the scan kernel (lod_filter_scan.comp) over the pages of a GPU slot list.
    static void Scan(const ComputeProgram& program, const RangeFilterInput& input,
                     std::uint32_t slotList);

private:
    ComputeProgram scanProgram_;
};

} // namespace chmv::gpu::filtering
