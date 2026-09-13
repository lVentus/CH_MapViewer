#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace chmv::data {
class CHGraph;
}

namespace chmv::gpu::filtering {

struct RangeFilterInput {
    std::uint32_t graphEdgeBuffer = 0;
    std::uint32_t graphEdgeCount = 0;
    std::int32_t lodLevel = 0;
    std::uint32_t outputEdgeBuffer = 0;
    std::uint32_t outputDrawCommandBuffer = 0;
};

struct RangeFilterExecutionStats {
    std::optional<std::size_t> candidateEdgeCount;
};

struct RangeFilterDiagnostics {
    std::optional<std::size_t> candidateEdgeCount;
    std::optional<std::uint32_t> dispatchGroupCount;
    std::optional<std::uint32_t> localSizeX;
};

class IRangeFilterStrategy {
public:
    virtual ~IRangeFilterStrategy() = default;

    [[nodiscard]] virtual std::string_view Name() const = 0;
    virtual void SetGraph(const data::CHGraph& graph) = 0;
    virtual void Reset() {}
    [[nodiscard]] virtual RangeFilterExecutionStats Execute(const RangeFilterInput& input) = 0;
    [[nodiscard]] virtual RangeFilterDiagnostics ReadBackDiagnostics(std::int32_t) const {
        return {};
    }
};

} // namespace chmv::gpu::filtering
