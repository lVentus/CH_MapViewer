#pragma once

#include "gpu/lod/LodPagePool.h"
#include "gpu/lod/LodPageRenderer.h"
#include "renderer/RoadStyle.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/lod/LodPageReader.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace chmv::renderer {
class LODController;
class MapCamera2D;
} // namespace chmv::renderer

namespace chmv::gpu::lod {

struct LodPageRuntimeStats {
    // Dataset and memory.
    std::uint64_t pageCount = 0;
    std::uint64_t recordCount = 0;
    std::uint64_t pageSectionBytes = 0;
    bool ramImage = false;
    std::uint32_t poolSlots = 0;
    std::uint64_t poolBytes = 0;
    std::uint32_t residentPages = 0;

    // What the current frame asks for and what it draws.
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
    std::uint32_t drawLevel = 0;
    std::uint32_t drawThreshold = 0;
    std::uint32_t requiredPages = 0;
    std::uint32_t missingRequiredPages = 0;
    std::uint32_t prefetchPages = 0;
    std::uint32_t drawPages = 0;
    std::uint32_t drawnSegments = 0;
    bool complete = false;
    bool poolFull = false;
    bool cameraMoving = false;

    // Per-frame costs.
    double planMs = 0.0;
    double uploadMs = 0.0;
    std::uint64_t uploadedBytes = 0;
    std::uint32_t uploadedPages = 0;
    std::uint64_t uploadBudgetBytes = 0;
    double updateMs = 0.0;
    std::optional<double> gpuFilterMs;
    std::optional<double> gpuDrawMs;
    bool filtered = false;

    // Totals.
    std::uint64_t totalUploadedPages = 0;
    std::uint64_t totalUploadedBytes = 0;
    std::uint64_t evictions = 0;
    std::uint64_t filterDispatches = 0;
    std::uint64_t diskPagesRead = 0;
    std::uint64_t diskBytesRead = 0;
    std::uint64_t diskReadCalls = 0;
    double diskReadMs = 0.0;
    std::string readerError;
    std::uint64_t corruptPages = 0;

    // How long the view stayed incomplete before everything it needs was drawn.
    double lastSettleMs = 0.0;
    double maxSettleMs = 0.0;
    std::uint64_t settleCount = 0;
    std::uint64_t incompleteFrames = 0;
};

// CPU side of the LOD page runtime: plans pages from the camera alone (no GPU readback),
// streams missing pages into the GPU pool, and switches to a new level/tolerance only once
// every page it needs is resident, so a frame never shows a half-loaded level.
class LodPageRuntime {
public:
    // poolBudgetBytes = 0 sizes the GPU page pool from the memory the driver/OS report.
    LodPageRuntime(const std::filesystem::path& shaderDirectory,
                   chmv::streaming::lod::LodPageDataset dataset, std::uint64_t poolBudgetBytes = 0);

    void Update(const renderer::MapCamera2D& camera, const renderer::LODController& lod, int width,
                int height, float pixelTolerance);
    void Draw(const renderer::MapCamera2D& camera, int width, int height,
              const std::array<float, 4>& color, const renderer::RoadStyleConfig& styles);

    [[nodiscard]] const LodPageRuntimeStats& Stats() const { return stats_; }
    [[nodiscard]] const chmv::streaming::lod::LodPageFile& File() const { return *file_; }
    // Validation: (page, record) of every segment the GPU currently draws. Synchronous.
    [[nodiscard]] std::vector<std::pair<std::uint32_t, std::uint32_t>> ReadDrawnRecords() const;

private:
    struct DrawState {
        std::uint32_t level = 0;
        std::uint32_t threshold = 0;
        bool valid = false;
    };

    void UploadMissing(bool cameraMoving);
    [[nodiscard]] bool CameraMoved(const renderer::MapCamera2D& camera, int width, int height);

    std::shared_ptr<const chmv::streaming::lod::LodPageFile> file_;
    std::shared_ptr<const std::vector<std::uint32_t>> pageImage_;
    std::unique_ptr<chmv::streaming::lod::LodPageReader> reader_;
    LodPagePool pool_;
    LodPageRenderer renderer_;

    std::vector<std::uint32_t> required_;
    std::vector<std::uint32_t> prefetch_;
    std::vector<std::uint32_t> missing_;
    std::vector<std::uint32_t> demand_;
    std::vector<std::uint32_t> drawPages_;
    std::vector<std::uint32_t> drawSlots_;
    std::vector<std::uint64_t> filteredPages_;
    // Frame stamps per page: requested this frame, part of the view, held back after reading.
    std::vector<std::uint32_t> requestStamp_;
    std::vector<std::uint32_t> requiredStamp_;
    std::vector<std::uint32_t> heldStamp_;
    std::vector<std::uint8_t> corrupt_;
    std::vector<chmv::streaming::lod::LodPageData> loaded_;
    std::uint32_t frame_ = 0;
    // Last planned queries; an unchanged camera skips planning.
    chmv::streaming::lod::LodPageQuery plannedQuery_;
    chmv::streaming::lod::LodPageQuery plannedRing_;
    chmv::streaming::lod::LodPageQuery drawQuery_;
    bool planned_ = false;
    bool drawPagesFromTarget_ = true;
    DrawState committed_;
    DrawState filtered_;
    double uploadBytesPerMs_ = 0.0;
    std::optional<std::chrono::steady_clock::time_point> incompleteSince_;
    std::array<float, 3> lastCamera_{};
    std::array<int, 2> lastSize_{};
    LodPageRuntimeStats stats_;
};

} // namespace chmv::gpu::lod
