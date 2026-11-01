#include "ui/DebugUI.h"

#include "benchmark/LodPageZoomSweep.h"
#include "core/Window.h"
#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "gpu/filtering/RangeFilterStrategyManager.h"
#include "gpu/lod/LodPageRuntime.h"
#include "gpu/unfolding/UnfoldingStrategyManager.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>

namespace chmv::ui {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;
const ImVec4 kWarning(1.0f, 0.75f, 0.3f, 1.0f);
const ImVec4 kFailure(1.0f, 0.35f, 0.35f, 1.0f);
const ImVec4 kSuccess(0.45f, 0.9f, 0.45f, 1.0f);

[[nodiscard]] double Megabytes(std::uint64_t bytes) {
    return static_cast<double>(bytes) / kMiB;
}

// ImGui expects UTF-8; path::string() would use the ANSI code page on Windows.
[[nodiscard]] std::string Utf8(const std::filesystem::path& path) {
    const auto text = path.u8string();
    return {text.begin(), text.end()};
}

[[nodiscard]] unsigned long long Count(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

const char* DatasetLoadPhaseName(data::DatasetLoadPhase phase) {
    switch (phase) {
    case data::DatasetLoadPhase::Idle:
        return "Idle";
    case data::DatasetLoadPhase::CheckingPageFile:
        return "Checking the page file";
    case data::DatasetLoadPhase::ReadingSource:
        return "Page file: reading .sch/.ranges";
    case data::DatasetLoadPhase::ComputingHierarchy:
        return "Page file: geometry errors and level contexts";
    case data::DatasetLoadPhase::LayingOutPages:
        return "Page file: sorting records";
    case data::DatasetLoadPhase::WritingPages:
        return "Page file: writing pages";
    case data::DatasetLoadPhase::VerifyingPages:
        return "Page file: checking against DFS";
    case data::DatasetLoadPhase::CompressingPages:
        return "Page file: compressing";
    case data::DatasetLoadPhase::LoadingIntoRam:
        return "Loading pages into RAM";
    case data::DatasetLoadPhase::Ready:
        return "Ready";
    case data::DatasetLoadPhase::Failed:
        return "Load failed";
    }
    return "Loading";
}

// Combo over the registered strategies of a manager.
template <typename Manager>
void StrategyCombo(const char* label, Manager& manager) {
    const auto strategies = manager.Strategies();
    if (strategies.empty()) {
        return;
    }
    const std::string current(manager.Current().Name());
    if (ImGui::BeginCombo(label, current.c_str())) {
        for (std::size_t i = 0; i < strategies.size(); ++i) {
            const std::string name(strategies[i]->Name());
            const bool selected = i == manager.CurrentIndex();
            if (ImGui::Selectable(name.c_str(), selected)) {
                manager.Select(i);
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void StatusLines(const std::array<std::uint32_t, gpu::lod::kStatusWords>& status) {
    using gpu::lod::LodPageStatus;
    const auto word = [&](LodPageStatus which) { return status[static_cast<std::size_t>(which)]; };
    if (word(LodPageStatus::CandidateOverflow) != 0u) {
        ImGui::TextColored(kFailure, "%u candidates dropped (list full)",
                           word(LodPageStatus::CandidateOverflow));
    }
    if (word(LodPageStatus::SegmentOverflow) != 0u) {
        ImGui::TextColored(kWarning, "%u segments dropped (buffer grows, pass repeats)",
                           word(LodPageStatus::SegmentOverflow));
    }
    if (word(LodPageStatus::FrontierOverflow) != 0u) {
        ImGui::TextColored(kWarning, "%u frontier entries dropped (buffer grows, pass repeats)",
                           word(LodPageStatus::FrontierOverflow));
    }
    if (word(LodPageStatus::StackOverflow) != 0u) {
        ImGui::TextColored(kFailure, "%u DFS stack overflows (hierarchy deeper than the stack)",
                           word(LodPageStatus::StackOverflow));
    }
    if (word(LodPageStatus::MissingPage) != 0u) {
        ImGui::TextDisabled("%u children skipped outside the loaded pages",
                            word(LodPageStatus::MissingPage));
    }
}

} // namespace

DebugUI::DebugUI(const core::Window& window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    const float contentScale = window.ContentScale();
    ImGui::GetStyle().ScaleAllSizes(contentScale);

    ImFontConfig fontConfig;
    fontConfig.SizePixels = 16.0f * contentScale;
    ImGui::GetIO().Fonts->AddFontDefault(&fontConfig);

    ImGui_ImplGlfw_InitForOpenGL(window.Handle(), true);
    ImGui_ImplOpenGL3_Init("#version 430 core");
}

DebugUI::~DebugUI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void DebugUI::BeginFrame() const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

DebugUIActions DebugUI::Draw(const DebugUIState& state) {
    DebugUIActions actions;
    ImGui::Begin("CH_MapViewer");
    DrawDataset(state, actions);
    ImGui::Separator();
    if (state.stats != nullptr) {
        DrawView(state);
        DrawPipeline(state, actions);
        DrawPageStats(*state.stats);
        DrawMemory(state, actions);
        DrawZoomSweep(state, actions);
    } else {
        ImGui::TextUnformatted("No dataset loaded.");
    }
    ImGui::End();
    return actions;
}

void DebugUI::DrawDataset(const DebugUIState& state, DebugUIActions& actions) {
    ImGui::TextUnformatted("Dataset");
    ImGui::TextDisabled("%s", Utf8(state.catalog.Directory()).c_str());

    const auto& datasets = state.catalog.Entries();
    if (selectedDataset_ >= datasets.size()) {
        selectedDataset_ = 0;
    }
    if (datasets.empty()) {
        ImGui::TextUnformatted("No datasets found.");
    } else {
        if (ImGui::BeginCombo("##dataset", datasets[selectedDataset_].name.c_str())) {
            for (std::size_t i = 0; i < datasets.size(); ++i) {
                const bool selected = i == selectedDataset_;
                if (ImGui::Selectable(datasets[i].name.c_str(), selected)) {
                    selectedDataset_ = i;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (state.load.Active()) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Load dataset")) {
            actions.datasetRequest = selectedDataset_;
        }
        if (state.load.Active()) {
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
    }
    if (ImGui::Button("Refresh datasets")) {
        state.catalog.Refresh();
    }
    ImGui::TextDisabled("Builds <name>.chlod when it is missing or outdated (complete CH + LOD "
                        "pages, checked against DFS, compressed); a .chlod without its sources "
                        "loads on its own. Pages are kept in RAM when they fit, otherwise read "
                        "from disk.");
    if (state.roadStyleError) {
        ImGui::TextColored(kFailure, "%s", state.roadStyleStatus.c_str());
    } else {
        ImGui::TextDisabled("%s", state.roadStyleStatus.c_str());
    }

    if (state.load.Active()) {
        ImGui::ProgressBar(state.load.progress, ImVec2(-1.0f, 0.0f),
                           DatasetLoadPhaseName(state.load.phase));
        ImGui::TextDisabled("%.1f%% | Loading runs on a background thread.",
                            static_cast<double>(state.load.progress) * 100.0);
    } else if (state.load.phase == data::DatasetLoadPhase::Failed) {
        ImGui::TextColored(kFailure, "Load failed: %s", state.load.error.c_str());
    }
}

void DebugUI::DrawView(const DebugUIState& state) {
    const auto& stats = *state.stats;
    ImGui::Text("Pages: %llu (%llu records)", Count(stats.pageCount), Count(stats.recordCount));
    auto& camera = *state.camera;
    auto& lod = *state.lod;
    ImGui::Text("Zoom %.2fx, vertical span %.2f km", static_cast<double>(camera.ZoomFactor()),
                camera.ViewHalfHeightMeters() * 2.0 / 1000.0);
    bool automatic = lod.Automatic();
    if (ImGui::Checkbox("Automatic LOD", &automatic)) {
        if (!automatic) {
            lod.SetManualLevel(lod.Level(camera.ZoomFactor()));
        }
        lod.SetAutomatic(automatic);
    }
    if (lod.Automatic()) {
        ImGui::SameLine();
        ImGui::Text("level %d of %d", lod.Level(camera.ZoomFactor()), lod.MaxLevel());
    } else {
        int manualLevel = lod.ManualLevel();
        if (ImGui::SliderInt("LOD level", &manualLevel, lod.MinLevel(), lod.MaxLevel())) {
            lod.SetManualLevel(manualLevel);
        }
    }
    if (ImGui::Button("Reset view")) {
        camera.Reset();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Mouse wheel: zoom | Left mouse: pan");
}

void DebugUI::DrawPipeline(const DebugUIState& state, DebugUIActions& actions) {
    if (!ImGui::CollapsingHeader("GPU pipeline", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    const auto& stats = *state.stats;
    ImGui::TextDisabled("Loading: virtual map. Pages follow from the camera alone and stream "
                        "into a GPU page pool.");
    ImGui::Checkbox("Preload all pages to the GPU if they fit", &state.preloadAll);
    if (state.preloadAll && !stats.preloadFits) {
        ImGui::TextColored(kWarning, "Not every page fits the GPU pool; streaming as needed.");
    }

    StrategyCombo("Filter", state.filters);
    StrategyCombo("Expansion", state.unfolds);
    const bool adaptive = state.unfolds.Current().Mode() == geometry::RefinementMode::Adaptive;
    if (!adaptive) {
        ImGui::BeginDisabled();
    }
    ImGui::SliderFloat("Pixel tolerance", &state.pixelTolerance, 0.5f, 16.0f, "%.2f px",
                       ImGuiSliderFlags_Logarithmic);
    if (!adaptive) {
        ImGui::EndDisabled();
    }
    ImGui::Text("Level %u, error code %u: %u candidates, %u segments drawn", stats.drawLevel,
                stats.drawThreshold, stats.candidates, stats.drawnSegments);
    if (stats.gpuFilterMs && stats.gpuUnfoldMs) {
        ImGui::Text("GPU filter %.3f ms, expansion %.3f ms, draw %.3f ms", *stats.gpuFilterMs,
                    *stats.gpuUnfoldMs, stats.gpuDrawMs.value_or(0.0));
    }
    StatusLines(stats.status);

    if (ImGui::Button("Validate against CPU")) {
        actions.validate = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Benchmark filter + expansion")) {
        actions.runBenchmark = true;
    }
    if (const auto* check = state.validation; check != nullptr && !check->error.empty()) {
        ImGui::TextColored(kFailure, "Could not check: %s", check->error.c_str());
    } else if (check != nullptr) {
        if (check->Passed()) {
            ImGui::TextColored(kSuccess, "Identical: %llu roads in view, %llu candidates",
                               Count(check->cpuSegmentsInView), Count(check->cpuCandidates));
        } else {
            ImGui::TextColored(kFailure,
                               "Mismatch: %llu roads missing, %llu extra, %llu candidates differ",
                               Count(check->missingOnGpu), Count(check->extraOnGpu),
                               Count(check->candidateMismatches));
        }
        ImGui::TextDisabled("Level %u, error code %u, %u pages; GPU drew %llu segments (%llu in "
                            "view); CPU check %.0f ms",
                            check->level, check->threshold, check->pages,
                            Count(check->gpuSegments), Count(check->gpuSegmentsInView),
                            check->cpuMilliseconds);
        StatusLines(check->status);
    }
    if (const auto* result = state.benchmark; result != nullptr && result->iterations != 0u) {
        ImGui::Text("%u passes over %u pages: filter %.3f ms (%.3f - %.3f), expansion %.3f ms "
                    "(%.3f - %.3f)",
                    result->iterations, result->pages, result->filterAverageMs,
                    result->filterMinMs, result->filterMaxMs, result->unfoldAverageMs,
                    result->unfoldMinMs, result->unfoldMaxMs);
        ImGui::TextDisabled("%u candidates, %u segments", result->candidates, result->segments);
    }
}

void DebugUI::DrawPageStats(const gpu::lod::LodPageRuntimeStats& stats) {
    if (!ImGui::CollapsingHeader("Pages", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    if (!stats.complete) {
        ImGui::TextColored(kWarning, "Loading %u of %u view pages; showing level %u meanwhile",
                           stats.missingRequiredPages, stats.requiredPages, stats.drawLevel);
    }
    ImGui::Text("View %u, with prefetch ring %u, drawn %u", stats.requiredPages,
                stats.prefetchPages, stats.drawPages);
    ImGui::Text("GPU pool: %u / %u pages (%.0f MiB%s), %llu evictions", stats.residentPages,
                stats.poolSlots, Megabytes(stats.poolBytes),
                stats.hierarchy ? ", with children" : "", Count(stats.evictions));
    if (stats.poolFull) {
        ImGui::TextColored(kFailure, "GPU pool full: this view needs more pages than fit");
    }
    ImGui::Text("Page file %.0f MiB on disk%s", Megabytes(stats.fileBytes),
                stats.compressedFile ? " (compressed)" : " (uncompressed)");
    if (stats.ramImage) {
        ImGui::Text("Pages served from RAM (%.0f MiB)", Megabytes(stats.pageSectionBytes));
    } else {
        ImGui::Text("Pages read from disk: %llu pages, %.1f MiB stored in %llu reads, %.1f ms "
                    "with decoding",
                    Count(stats.diskPagesRead), Megabytes(stats.diskBytesRead),
                    Count(stats.diskReadCalls), stats.diskReadMs);
        if (!stats.readerError.empty()) {
            ImGui::TextColored(kFailure, "Read error: %s", stats.readerError.c_str());
        }
    }
    if (stats.corruptPages != 0u) {
        ImGui::TextColored(kFailure,
                           "%llu corrupt pages skipped (delete the .chlod file to rebuild it)",
                           Count(stats.corruptPages));
    }
    ImGui::Text("Plan %.3f ms, upload %.3f ms (%.2f MiB, %u pages; budget %.1f MiB)",
                stats.planMs, stats.uploadMs, Megabytes(stats.uploadedBytes), stats.uploadedPages,
                Megabytes(stats.uploadBudgetBytes));
    ImGui::Text("CPU update %.3f ms", stats.updateMs);
    ImGui::Text("Until the view is complete: last %.1f ms, max %.1f ms (%llu loads)",
                stats.lastSettleMs, stats.maxSettleMs, Count(stats.settleCount));
    ImGui::Text("Frames with missing view pages: %llu", Count(stats.incompleteFrames));
}

void DebugUI::DrawMemory(const DebugUIState& state, DebugUIActions& actions) {
    if (!ImGui::CollapsingHeader("GPU memory")) {
        return;
    }
    ImGui::InputInt("Pool budget (MiB)", &gpuBudgetMiB_, 64, 1024);
    gpuBudgetMiB_ = std::max(gpuBudgetMiB_, 0);
    if (ImGui::Button("Apply budget")) {
        actions.gpuBudgetMiB = static_cast<std::uint32_t>(gpuBudgetMiB_);
    }
    ImGui::TextDisabled("0 = automatic: a third of free VRAM (an eighth of free RAM on integrated "
                        "GPUs), more when preloading everything. Currently %.0f MiB.",
                        Megabytes(state.stats->poolBytes));
}

void DebugUI::DrawZoomSweep(const DebugUIState& state, DebugUIActions& actions) {
    if (!ImGui::CollapsingHeader("Zoom sweep benchmark")) {
        return;
    }
    ImGui::TextDisabled("Overview -> 3 km in 12 s, hold 3 s, back out, hold 3 s. Writes a "
                        "per-frame CSV next to the dataset.");
    const auto* sweep = state.sweep;
    if (sweep != nullptr && sweep->Running()) {
        ImGui::Text("Running: %s", sweep->PhaseName());
        if (ImGui::Button("Cancel sweep")) {
            actions.cancelZoomSweep = true;
        }
    } else {
        if (ImGui::Button("Run zoom sweep")) {
            actions.startZoomSweep = true;
        }
        if (sweep != nullptr && !sweep->Summary().empty()) {
            ImGui::TextWrapped("%s", sweep->Summary().c_str());
        }
    }
}

void DebugUI::EndFrame() const {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool DebugUI::WantsMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

} // namespace chmv::ui
