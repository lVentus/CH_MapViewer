#include "ui/DebugUI.h"

#include "benchmark/CorrectnessValidator.h"
#include "core/Window.h"
#include "data/AsyncDatasetLoader.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHGraph.h"
#include "gpu/filtering/RangeFilterStrategyManager.h"
#include "gpu/unfolding/UnfoldingStrategyManager.h"
#include "pipeline/CPUReferencePipeline.h"
#include "pipeline/GPUDrivenPipeline.h"
#include "pipeline/ProcessingMode.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

namespace chmv::ui {
namespace {

const char* ProcessingModeName(pipeline::ProcessingMode mode) {
    switch (mode) {
    case pipeline::ProcessingMode::GPUDriven:
        return "GPU Driven";
    case pipeline::ProcessingMode::CPUReference:
        return "CPU Paper (Ordered)";
    case pipeline::ProcessingMode::Validation:
        return "Validation";
    }
    return "Unknown";
}

const char* DatasetLoadPhaseName(data::DatasetLoadPhase phase) {
    switch (phase) {
    case data::DatasetLoadPhase::Idle:
        return "Idle";
    case data::DatasetLoadPhase::ReadingNodes:
        return "Reading nodes";
    case data::DatasetLoadPhase::ReadingEdges:
        return "Reading edges";
    case data::DatasetLoadPhase::ReadingRanges:
        return "Reading ranges";
    case data::DatasetLoadPhase::PreparingRangeIndex:
        return "Preparing range index";
    case data::DatasetLoadPhase::PreparingGeometryErrors:
        return "Preparing geometry errors";
    case data::DatasetLoadPhase::Ready:
        return "Ready";
    case data::DatasetLoadPhase::Failed:
        return "Load failed";
    }
    return "Loading";
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

DebugUIActions DebugUI::Draw(
    const data::CHGraph* graph,
    gpu::filtering::RangeFilterStrategyManager& rangeFilterStrategyManager,
    gpu::unfolding::UnfoldingStrategyManager& strategyManager,
    renderer::MapCamera2D* camera,
    renderer::LODController* lodController,
    data::DatasetCatalog& datasetCatalog,
    const data::DatasetLoadSnapshot& datasetLoad,
    pipeline::ProcessingMode& processingMode,
    geometry::RefinementMode& cpuGeometryRefinement,
    float& maxScreenErrorPixels,
    const pipeline::CPUProcessingStats* cpuStats,
    const pipeline::GPUProcessingStats* gpuStats,
    const benchmark::PipelineValidationResult* validationResult,
    bool canValidate) {
    DebugUIActions actions;

    ImGui::Begin("CH_MapViewer");

    ImGui::TextUnformatted("Dataset");
    ImGui::TextDisabled("%s", datasetCatalog.Directory().string().c_str());

    const auto& datasets = datasetCatalog.Entries();
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

        if (datasetLoad.Active()) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Load dataset")) {
            actions.datasetRequest = selectedDataset_;
        }
        if (datasetLoad.Active()) {
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
    }

    if (ImGui::Button("Refresh datasets")) {
        datasetCatalog.Refresh();
    }

    if (datasetLoad.Active()) {
        const auto* phaseName = DatasetLoadPhaseName(datasetLoad.phase);
        ImGui::ProgressBar(datasetLoad.progress, ImVec2(-1.0f, 0.0f), phaseName);
        ImGui::TextDisabled("%.1f%% | Dataset loading runs on a background thread.",
                            datasetLoad.progress * 100.0f);
    } else if (datasetLoad.phase == data::DatasetLoadPhase::Failed) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Load failed: %s",
                           datasetLoad.error.c_str());
    }

    ImGui::Separator();

    if (graph) {
        ImGui::Text("Nodes: %zu", graph->NodeCount());
        ImGui::Text("Edges: %zu", graph->EdgeCount());
        ImGui::Text("Shortcuts: %zu", graph->ShortcutCount());
        ImGui::Text("Drawable ranges: %zu", graph->DrawableEdgeCount());
        if (camera && lodController) {
            ImGui::Text("Zoom: %.2fx", camera->ZoomFactor());
            ImGui::Text("LOD range: %d - %d", lodController->MinLevel(), lodController->MaxLevel());

            bool automaticLOD = lodController->Automatic();
            if (ImGui::Checkbox("Automatic LOD", &automaticLOD)) {
                if (!automaticLOD) {
                    lodController->SetManualLevel(lodController->Level(camera->ZoomFactor()));
                }
                lodController->SetAutomatic(automaticLOD);
            }

            if (lodController->Automatic()) {
                ImGui::Text("LOD level: %d", lodController->Level(camera->ZoomFactor()));
            } else {
                int manualLevel = lodController->ManualLevel();
                if (ImGui::SliderInt("LOD level", &manualLevel, lodController->MinLevel(),
                                     lodController->MaxLevel())) {
                    lodController->SetManualLevel(manualLevel);
                }
            }

            if (ImGui::Button("Reset view")) {
                camera->Reset();
            }
            ImGui::TextDisabled("Mouse wheel: zoom | Middle mouse: pan");
        }
    } else {
        ImGui::TextUnformatted("No CH dataset loaded.");
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Processing pipeline");

    if (ImGui::BeginCombo("##pipeline", ProcessingModeName(processingMode))) {
        constexpr pipeline::ProcessingMode modes[] = {
            pipeline::ProcessingMode::GPUDriven,
            pipeline::ProcessingMode::CPUReference,
            pipeline::ProcessingMode::Validation,
        };
        for (const auto mode : modes) {
            const bool selected = processingMode == mode;
            if (ImGui::Selectable(ProcessingModeName(mode), selected)) {
                processingMode = mode;
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    if (processingMode != pipeline::ProcessingMode::CPUReference) {
        ImGui::TextUnformatted("GPU filtering strategy");
        const auto rangeFilterStrategies = rangeFilterStrategyManager.Strategies();
        if (!rangeFilterStrategies.empty()) {
            const auto currentName = rangeFilterStrategyManager.Current().Name();
            if (ImGui::BeginCombo("##range-filter-strategy", currentName.data())) {
                for (std::size_t i = 0; i < rangeFilterStrategies.size(); ++i) {
                    const bool selected = i == rangeFilterStrategyManager.CurrentIndex();
                    if (ImGui::Selectable(rangeFilterStrategies[i]->Name().data(), selected)) {
                        rangeFilterStrategyManager.Select(i);
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }

        ImGui::TextUnformatted("GPU geometry refinement");
        const auto strategies = strategyManager.Strategies();
        if (!strategies.empty()) {
            const auto currentName = strategyManager.Current().Name();
            if (ImGui::BeginCombo("##strategy", currentName.data())) {
                for (std::size_t i = 0; i < strategies.size(); ++i) {
                    const bool selected = i == strategyManager.CurrentIndex();
                    if (ImGui::Selectable(strategies[i]->Name().data(), selected)) {
                        strategyManager.Select(i);
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }
    } else {
        ImGui::TextUnformatted("CPU geometry refinement");
        int refinement = static_cast<int>(cpuGeometryRefinement);
        constexpr const char* refinementNames[] = {
            "None (paper ranges only)",
            "Full unfold",
            "Adaptive (screen-space)",
        };
        if (ImGui::BeginCombo("##cpu-geometry-refinement", refinementNames[refinement])) {
            for (int i = 0; i < 3; ++i) {
                const bool selected = refinement == i;
                if (ImGui::Selectable(refinementNames[i], selected)) {
                    refinement = i;
                    cpuGeometryRefinement = static_cast<geometry::RefinementMode>(refinement);
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (cpuGeometryRefinement == geometry::RefinementMode::Full) {
            ImGui::TextDisabled(
                "Birth-ordered lifetime retrieval first, then full shortcut geometry unfolding.");
        } else if (cpuGeometryRefinement == geometry::RefinementMode::Adaptive) {
            ImGui::TextDisabled(
                "Birth-ordered lifetime retrieval first, then screen-space geometry refinement.");
        } else {
            ImGui::TextDisabled(
                "Paper runtime path: birth-ordered lifetime retrieval, no geometry refinement.");
        }
    }

    const bool adaptiveGPU =
        processingMode != pipeline::ProcessingMode::CPUReference &&
        strategyManager.Current().Mode() == geometry::RefinementMode::Adaptive;
    const bool adaptiveCPU =
        processingMode == pipeline::ProcessingMode::CPUReference &&
        cpuGeometryRefinement == geometry::RefinementMode::Adaptive;
    if (adaptiveGPU || adaptiveCPU) {
        ImGui::SliderFloat("Max screen error", &maxScreenErrorPixels, 1.0f, 512.0f, "%.1f px",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::TextDisabled(
            "1 px is the finest setting; larger values keep coarser shortcut geometry.");
    }

    if (gpuStats && processingMode != pipeline::ProcessingMode::CPUReference) {
        if (gpuStats->rangeCandidateEdgeCount.has_value()) {
            ImGui::Text("GPU range candidates: %zu", *gpuStats->rangeCandidateEdgeCount);
        } else {
            ImGui::TextUnformatted("GPU range candidates: GPU-managed");
        }
        if (gpuStats->rangeFilterMs) {
            ImGui::Text("GPU range filter: %.6f ms", *gpuStats->rangeFilterMs);
        }
        if (gpuStats->geometryRefinementMs) {
            ImGui::Text("GPU geometry refinement: %.6f ms", *gpuStats->geometryRefinementMs);
        } else {
            ImGui::TextDisabled("GPU geometry refinement: none (ranges only)");
        }
    }

    if (cpuStats && processingMode != pipeline::ProcessingMode::GPUDriven) {
        ImGui::Text("CPU scanned edges: %zu", cpuStats->rangeScannedEdgeCount);
        ImGui::Text("CPU alive edges: %zu", cpuStats->aliveEdgeCount);
        ImGui::Text("CPU output edges: %zu", cpuStats->outputEdgeCount);
        ImGui::Text("CPU range filter: %.6f ms", cpuStats->rangeFilterMs);
        if (cpuStats->geometryRefinementMs > 0.0) {
            ImGui::Text("CPU geometry refinement: %.6f ms", cpuStats->geometryRefinementMs);
        } else {
            ImGui::TextDisabled("CPU geometry refinement: none (ranges only)");
        }
        if (cpuStats->cacheHit) {
            ImGui::TextDisabled("CPU result reused for unchanged LOD.");
        }
    }

    if (processingMode == pipeline::ProcessingMode::Validation) {
        ImGui::Separator();
        ImGui::TextUnformatted("Validation view");

        int viewMode = splitScreenValidation_ ? 0 : 1;
        ImGui::RadioButton("Split screen", &viewMode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Overlay", &viewMode, 1);
        splitScreenValidation_ = viewMode == 0;

        if (splitScreenValidation_) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.20f, 1.0f), "CPU = left");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.20f, 0.75f, 1.0f, 1.0f), "GPU = right");
            ImGui::TextDisabled("Both views use the same camera center and zoom.");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.20f, 1.0f), "CPU");
            ImGui::SameLine();
            ImGui::TextUnformatted("+");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.20f, 0.75f, 1.0f, 1.0f), "GPU overlay");
        }

        if (strategyManager.Current().Mode() == geometry::RefinementMode::Full) {
            ImGui::TextDisabled(
                "CPU full geometry unfolding mirrors the selected GPU refinement for validation.");
        } else if (strategyManager.Current().Mode() == geometry::RefinementMode::Adaptive) {
            ImGui::TextDisabled(
                "CPU adaptive refinement uses the same precomputed error and pixel threshold.");
        } else {
            ImGui::TextDisabled(
                "CPU and GPU both render the topology-safe lifetime edge set directly.");
        }

        if (!canValidate) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Validate current result")) {
            actions.validateCurrentResult = true;
        }
        if (!canValidate) {
            ImGui::EndDisabled();
        }

        if (validationResult) {
            ImGui::Text("Exact validation: %s", validationResult->Passed() ? "PASS" : "FAIL");
            ImGui::Text("Range filter: %s", validationResult->rangeFilter.Passed() ? "PASS" : "FAIL");
            ImGui::Text("  CPU/GPU edges: %zu / %zu", validationResult->rangeFilter.cpuEdgeCount,
                        validationResult->rangeFilter.gpuEdgeCount);
            ImGui::Text("  Missing / unexpected: %zu / %zu",
                        validationResult->rangeFilter.missingEdgeCount,
                        validationResult->rangeFilter.unexpectedEdgeCount);
            ImGui::Text("Final output: %s",
                        validationResult->finalOutput.Passed() ? "PASS" : "FAIL");
            ImGui::Text("  CPU/GPU edges: %zu / %zu", validationResult->finalOutput.cpuEdgeCount,
                        validationResult->finalOutput.gpuEdgeCount);
            ImGui::Text("  Missing / unexpected: %zu / %zu",
                        validationResult->finalOutput.missingEdgeCount,
                        validationResult->finalOutput.unexpectedEdgeCount);
            ImGui::Text("  CPU/GPU duplicate occurrences: %zu / %zu",
                        validationResult->finalOutput.cpuDuplicateCount,
                        validationResult->finalOutput.gpuDuplicateCount);
        } else {
            ImGui::TextDisabled("Exact comparison runs only when requested and reads GPU output back to CPU.");
        }
    }

    ImGui::End();
    return actions;
}

void DebugUI::EndFrame() const {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool DebugUI::WantsMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

} // namespace chmv::ui
