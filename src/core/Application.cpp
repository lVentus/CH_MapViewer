#include "core/Application.h"

#include "data/ch/CHLoader.h"
#include "gpu/unfolding/NoUnfoldingStrategy.h"
#include "gpu/unfolding/IterativeDFSUnfoldingStrategy.h"

#include <iostream>
#include <utility>

namespace chmv::core {

Application::Application(const std::filesystem::path& assetDirectory)
    : window_("CH_MapViewer"),
      context_(window_),
      roadRenderer_(assetDirectory / "shaders"),
      debugUI_(window_),
      datasetCatalog_(assetDirectory / "data") {
    strategyManager_.Register(std::make_unique<gpu::unfolding::NoUnfoldingStrategy>());
    strategyManager_.Register(
        std::make_unique<gpu::unfolding::IterativeDFSUnfoldingStrategy>(assetDirectory / "shaders"));
}

void Application::LoadDataset(const std::filesystem::path& graphPath,
                              const std::filesystem::path& rangesPath) {
    std::cout << "Loading " << graphPath << '\n';
    auto graph = data::CHLoader::Load(graphPath, rangesPath);
    std::cout << "Loaded " << graph.NodeCount() << " nodes and " << graph.EdgeCount()
              << " edges\n";
    graph_ = std::make_unique<data::CHGraph>(std::move(graph));
    roadRenderer_.SetGraph(*graph_);
    lodController_.SetLevelRange(roadRenderer_.MinLevel(), roadRenderer_.MaxLevel());
    lodController_.SetManualLevel(roadRenderer_.MaxLevel());
    camera_.Reset();
}

int Application::Run() {
    while (!window_.ShouldClose()) {
        window_.PollEvents();
        debugUI_.BeginFrame();
        UpdateCameraInput();

        int width = 0;
        int height = 0;
        window_.FramebufferSize(width, height);

        renderer_.BeginFrame(width, height);
        const auto lodLevel = lodController_.Level(camera_.ZoomFactor());
        roadRenderer_.Draw(camera_, lodLevel, width, height, strategyManager_.Current());
        const auto datasetRequest = debugUI_.Draw(
            graph_.get(), strategyManager_, graph_ ? &camera_ : nullptr,
            graph_ ? &lodController_ : nullptr, datasetCatalog_);
        debugUI_.EndFrame();

        if (datasetRequest) {
            const auto& dataset = datasetCatalog_.Entries()[*datasetRequest];
            LoadDataset(dataset.graphPath, dataset.rangesPath);
        }

        window_.SwapBuffers();
    }

    return 0;
}

void Application::UpdateCameraInput() {
    const auto scroll = window_.ConsumeScrollY();
    if (debugUI_.WantsMouse()) {
        wasPanning_ = false;
        return;
    }

    if (scroll != 0.0) {
        camera_.Zoom(static_cast<float>(scroll));
    }

    double cursorX = 0.0;
    double cursorY = 0.0;
    window_.CursorPosition(cursorX, cursorY);

    if (window_.MiddleMouseDown()) {
        if (wasPanning_) {
            int width = 0;
            int height = 0;
            window_.FramebufferSize(width, height);
            camera_.PanPixels(cursorX - lastCursorX_, cursorY - lastCursorY_, height);
        }
        wasPanning_ = true;
    } else {
        wasPanning_ = false;
    }

    lastCursorX_ = cursorX;
    lastCursorY_ = cursorY;
}

} // namespace chmv::core
