#pragma once

#include <cstddef>
#include <optional>

namespace chmv::core {
class Window;
}

namespace chmv::data {
class CHGraph;
class DatasetCatalog;
}

namespace chmv::gpu::unfolding {
class UnfoldingStrategyManager;
}

namespace chmv::renderer {
class LODController;
class MapCamera2D;
}

namespace chmv::ui {

class DebugUI {
public:
    explicit DebugUI(const core::Window& window);
    ~DebugUI();

    DebugUI(const DebugUI&) = delete;
    DebugUI& operator=(const DebugUI&) = delete;

    void BeginFrame() const;
    [[nodiscard]] std::optional<std::size_t> Draw(
        const data::CHGraph* graph,
        gpu::unfolding::UnfoldingStrategyManager& strategyManager,
        renderer::MapCamera2D* camera,
        renderer::LODController* lodController,
        data::DatasetCatalog& datasetCatalog);
    void EndFrame() const;
    [[nodiscard]] bool WantsMouse() const;

private:
    std::size_t selectedDataset_ = 0;
};

} // namespace chmv::ui
