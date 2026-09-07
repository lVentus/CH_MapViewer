#include "ui/DebugUI.h"

#include "core/Window.h"
#include "data/DatasetCatalog.h"
#include "data/ch/CHGraph.h"
#include "gpu/unfolding/UnfoldingStrategyManager.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

namespace chmv::ui {

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

std::optional<std::size_t> DebugUI::Draw(
    const data::CHGraph* graph,
    gpu::unfolding::UnfoldingStrategyManager& strategyManager,
    renderer::MapCamera2D* camera,
    renderer::LODController* lodController,
    data::DatasetCatalog& datasetCatalog) {
    std::optional<std::size_t> datasetRequest;

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
            datasetRequest = selectedDataset_;
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
    ImGui::TextUnformatted("Unfolding strategy");

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

    if (strategyManager.Current().Name() == "Iterative DFS") {
        ImGui::TextDisabled("Fully unfolds selected shortcuts to original edges.");
    } else {
        ImGui::TextDisabled("Selected alive edges are rendered without unfolding.");
    }
    ImGui::End();

    return datasetRequest;
}

void DebugUI::EndFrame() const {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool DebugUI::WantsMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

} // namespace chmv::ui
