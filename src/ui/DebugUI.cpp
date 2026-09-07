#include "ui/DebugUI.h"

#include "benchmark/CorrectnessValidator.h"
#include "core/Window.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHGraph.h"
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
        return "CPU Reference (Serial)";
    case pipeline::ProcessingMode::Validation:
        return "Validation";
    }
    return "Unknown";
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
    gpu::unfolding::UnfoldingStrategyManager& strategyManager,
    renderer::MapCamera2D* camera,
    renderer::LODController* lodController,
    data::DatasetCatalog& datasetCatalog,
    pipeline::ProcessingMode& processingMode,
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

        if (ImGui::Button("Load dataset")) {
            actions.datasetRequest = selectedDataset_;
        }
        ImGui::SameLine();
    }

    if (ImGui::Button("Refresh datasets")) {
        datasetCatalog.Refresh();
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
        ImGui::TextUnformatted("GPU unfolding strategy");
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
        ImGui::TextDisabled("Serial CPU reference uses full unfolding.");
    }

    if (gpuStats && processingMode != pipeline::ProcessingMode::CPUReference) {
        if (gpuStats->rangeFilterMs) {
            ImGui::Text("GPU range filter: %.3f ms", *gpuStats->rangeFilterMs);
        }
        if (gpuStats->unfoldingMs) {
            ImGui::Text("GPU unfolding: %.3f ms", *gpuStats->unfoldingMs);
        }
    }

    if (cpuStats && processingMode != pipeline::ProcessingMode::GPUDriven) {
        ImGui::Text("CPU alive edges: %zu", cpuStats->aliveEdgeCount);
        ImGui::Text("CPU output edges: %zu", cpuStats->outputEdgeCount);
        ImGui::Text("CPU range filter: %.3f ms", cpuStats->rangeFilterMs);
        ImGui::Text("CPU unfolding: %.3f ms", cpuStats->unfoldingMs);
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
