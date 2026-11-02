#pragma once

#include "cpu/CpuPasses.h"
#include "renderer/ToleranceController.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace chmv::benchmark {
class LodPageZoomSweep;
}

namespace chmv::cpu {
struct ClassicStatus;
struct CpuBenchmarkResult;
} // namespace chmv::cpu

namespace chmv::core {
class Window;
}

namespace chmv::data {
class DatasetCatalog;
struct DatasetLoadSnapshot;
} // namespace chmv::data

namespace chmv::gpu::filtering {
class RangeFilterStrategyManager;
}

namespace chmv::gpu::lod {
struct LodPageBenchmarkResult;
struct LodPageRuntimeStats;
struct LodPageValidation;
} // namespace chmv::gpu::lod

namespace chmv::gpu::unfolding {
class UnfoldingStrategyManager;
}

namespace chmv::renderer {
class LODController;
class MapCamera2D;
} // namespace chmv::renderer

namespace chmv::ui {

struct DebugUIActions {
    std::optional<std::size_t> datasetRequest;
    bool validate = false;
    bool runBenchmark = false;
    bool startZoomSweep = false;
    bool cancelZoomSweep = false;
    std::optional<std::uint32_t> gpuBudgetMiB;
};

// What the panel shows and edits. Camera, LOD controller and stats are null without a dataset.
struct DebugUIState {
    data::DatasetCatalog& catalog;
    const data::DatasetLoadSnapshot& load;
    renderer::MapCamera2D* camera = nullptr;
    renderer::LODController* lod = nullptr;
    gpu::filtering::RangeFilterStrategyManager& filters;
    gpu::unfolding::UnfoldingStrategyManager& unfolds;
    float& pixelTolerance;
    // Manual or automatic tolerance, what the automatic one decided, and the tolerance of this
    // frame (doubled while moving, if asked for).
    renderer::ToleranceSettings& tolerance;
    const renderer::ToleranceStatus& toleranceStatus;
    float tolerancePixels = 1.0f;
    bool toleranceCoarsened = false;
    bool& preloadAll;
    const std::string& roadStyleStatus;
    bool roadStyleError = false;
    const gpu::lod::LodPageRuntimeStats* stats = nullptr;
    const gpu::lod::LodPageValidation* validation = nullptr;
    const gpu::lod::LodPageBenchmarkResult* benchmark = nullptr;
    const benchmark::LodPageZoomSweep* sweep = nullptr;
    // Who computes what is drawn, and the CPU side: method, threads, result (null when the CPU
    // does not run), classic method state, the GPU/CPU comparison of the current view (null
    // until there is one) and the CPU benchmark.
    cpu::ProcessingMode& mode;
    cpu::CpuMethod& cpuMethod;
    int& cpuThreads;
    const cpu::CpuResult* cpu = nullptr;
    // Why the CPU result is not drawn (empty when it is).
    const std::string& cpuDrawError;
    const cpu::ClassicStatus& classic;
    const cpu::CpuGpuComparison* comparison = nullptr;
    const cpu::CpuBenchmarkResult* cpuBenchmark = nullptr;
};

class DebugUI {
public:
    explicit DebugUI(const core::Window& window);
    ~DebugUI();

    DebugUI(const DebugUI&) = delete;
    DebugUI& operator=(const DebugUI&) = delete;

    void BeginFrame() const;
    [[nodiscard]] DebugUIActions Draw(const DebugUIState& state);
    void EndFrame() const;
    [[nodiscard]] bool WantsMouse() const;

private:
    void DrawDataset(const DebugUIState& state, DebugUIActions& actions);
    static void DrawView(const DebugUIState& state);
    static void DrawPipeline(const DebugUIState& state, DebugUIActions& actions);
    static void DrawProcessing(const DebugUIState& state);
    static void DrawTolerance(const DebugUIState& state);
    static void DrawCpuResults(const DebugUIState& state);
    // Labels the two halves of the window side by side.
    static void DrawSplitLabels(const DebugUIState& state);
    static void DrawPageStats(const gpu::lod::LodPageRuntimeStats& stats);
    void DrawMemory(const DebugUIState& state, DebugUIActions& actions);
    static void DrawZoomSweep(const DebugUIState& state, DebugUIActions& actions);

    std::size_t selectedDataset_ = 0;
    int gpuBudgetMiB_ = 0;
};

} // namespace chmv::ui
