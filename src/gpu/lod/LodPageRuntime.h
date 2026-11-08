#pragma once

#include "gpu/lod/LodPageLists.h"
#include "gpu/lod/LodPagePool.h"
#include "gpu/lod/LodPageRenderer.h"
#include "renderer/RoadStyle.h"
#include "renderer/ToleranceController.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/lod/LodPageReader.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace chmv::gpu::filtering {
class IRangeFilterStrategy;
}

namespace chmv::gpu::unfolding {
class IUnfoldingStrategy;
}

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
    // The .chlod on disk.
    std::uint64_t fileBytes = 0;
    bool compressedFile = false;
    bool ramImage = false;
    std::uint32_t poolSlots = 0;
    std::uint64_t poolBytes = 0;
    std::uint32_t residentPages = 0;
    // The pool holds children blocks (a traversal is selected).
    bool hierarchy = false;
    // Every page is being loaded into the pool; it can only hold them all if it is that large.
    bool preloadAll = false;
    bool preloadFits = false;

    // What the current frame asks for and what it draws. level and threshold are the camera's,
    // or, when the view's pages do not fit in the GPU pool, those of the camera zoomOutOctaves
    // farther away (with a raised level where even that is not enough).
    std::uint32_t cameraLevel = 0;
    std::uint32_t cameraThreshold = 0;
    float zoomOutOctaves = 0.0f;
    bool fittedToPool = false;
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
    std::uint32_t drawLevel = 0;
    std::uint32_t drawThreshold = 0;
    std::uint32_t requiredPages = 0;
    std::uint32_t missingRequiredPages = 0;
    std::uint32_t prefetchPages = 0;
    std::uint32_t drawPages = 0;
    std::uint32_t candidates = 0;
    std::uint32_t drawnSegments = 0;
    std::array<std::uint32_t, kStatusWords> status{};
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
    std::optional<double> gpuUnfoldMs;
    std::optional<double> gpuDrawMs;
    std::optional<double> gpuCountMs;
    bool filtered = false;
    // For the automatic tolerance: the error code of one pixel now, and the segments each
    // tolerance step would draw in the drawn pages (when asked for and finished).
    std::uint32_t pixelCode = 0;
    std::optional<std::array<std::uint32_t, renderer::kToleranceSteps>> stepCounts;

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

// The current GPU result checked against the CPU: the candidates against a plain level test of
// the same pages, and the drawn roads inside the view against the dual-interval cut of the same
// level and threshold, evaluated record by record from the file (edges counted with
// multiplicity).
struct LodPageValidation {
    std::uint32_t level = 0;
    std::uint32_t threshold = 0;
    std::uint32_t pages = 0;
    std::uint64_t gpuCandidates = 0;
    std::uint64_t cpuCandidates = 0;
    std::uint64_t candidateMismatches = 0;
    std::uint64_t gpuSegments = 0;
    std::uint64_t gpuSegmentsInView = 0;
    std::uint64_t cpuSegmentsInView = 0;
    std::uint64_t missingOnGpu = 0;
    std::uint64_t extraOnGpu = 0;
    std::array<std::uint32_t, kStatusWords> status{};
    double cpuMilliseconds = 0.0;
    // Set when the file could not be read for the comparison (a damaged page, say).
    std::string error;

    // Children skipped for a page that is not resident are no failure: when the view is
    // complete, such children lie outside it (the comparison would show it otherwise).
    [[nodiscard]] bool Passed() const {
        bool clean = true;
        for (std::size_t word = 0; word < status.size(); ++word) {
            clean = clean && (status[word] == 0u ||
                              word == static_cast<std::size_t>(LodPageStatus::MissingPage));
        }
        return clean && error.empty() && candidateMismatches == 0u && missingOnGpu == 0u &&
               extraOnGpu == 0u;
    }
};

// CPU side of the LOD page runtime: plans pages from the camera alone (no GPU readback),
// streams missing pages into the GPU pool, and switches to a new level/tolerance only once
// every page it needs is resident, so a frame never shows a half-loaded level. Filtering and
// expansion are done by the strategies handed to Update.
class LodPageRuntime {
public:
    // poolBudgetBytes = 0 sizes the GPU page pool from the memory the driver/OS report.
    LodPageRuntime(const std::filesystem::path& shaderDirectory,
                   chmv::streaming::lod::LodPageDataset dataset, std::uint64_t poolBudgetBytes = 0);

    // countSteps also counts, with every pass of an adaptive expansion, the segments each
    // tolerance step would draw.
    void Update(const renderer::MapCamera2D& camera, const renderer::LODController& lod, int width,
                int height, float pixelTolerance, filtering::IRangeFilterStrategy& filter,
                unfolding::IUnfoldingStrategy& unfold, bool countSteps = false);
    void Draw(const renderer::MapCamera2D& camera, int width, int height,
              const renderer::RoadStyleConfig& styles);
    // Loads every page into the GPU pool when the pool can be made that large.
    void SetPreloadAll(bool preload);
    // GPU pool budget, 0 = automatic. Either change rebuilds the pool with the next frame.
    void SetPoolBudget(std::uint64_t bytes);
    // Drawn segments whose chord misses the view are dropped in the pass (it then reruns when
    // the view moves, as a traversal does).
    void SetCullSegments(bool cull) { cullSegments_ = cull; }

    [[nodiscard]] const LodPageRuntimeStats& Stats() const { return stats_; }
    [[nodiscard]] const chmv::streaming::lod::LodPageFile& File() const { return *file_; }
    // Query and view rectangle of what is drawn now.
    [[nodiscard]] const chmv::streaming::lod::LodPageQuery& DrawQuery() const {
        return drawQuery_;
    }
    // The CH edges drawn now whose chord touches the view, sorted (a synchronous readback).
    [[nodiscard]] std::vector<std::uint32_t> DrawnEdgesInView() const;
    // Status words of the last pass (a synchronous readback).
    [[nodiscard]] std::array<std::uint32_t, kStatusWords> CurrentStatus() const {
        return renderer_.ReadStatus();
    }
    // Synchronous checks of what the GPU currently draws.
    [[nodiscard]] LodPageValidation Validate() const;
    [[nodiscard]] LodPageBenchmarkResult
    Benchmark(filtering::IRangeFilterStrategy& filter, unfolding::IUnfoldingStrategy& unfold,
              std::uint32_t iterations, const renderer::MapCamera2D& camera, int width, int height,
              const renderer::RoadStyleConfig& styles);

private:
    struct DrawState {
        std::uint32_t level = 0;
        std::uint32_t threshold = 0;
        bool valid = false;
    };

    void SetUpPool(bool hierarchy);
    void QueuePreload();
    void UploadMissing(bool cameraMoving);
    [[nodiscard]] bool Upload(std::uint32_t page, std::span<const std::uint32_t> words,
                              std::span<const std::uint32_t> children);
    [[nodiscard]] bool CameraMoved(const renderer::MapCamera2D& camera, int width, int height);
    // The query whose view pages fit in the pool: query itself, the view planned `steps` steps
    // farther away (farther(steps)), or that with a raised level; `pages` gets its pages.
    [[nodiscard]] chmv::streaming::lod::LodPageQuery
    FitView(const chmv::streaming::lod::LodPageQuery& query,
            const std::function<chmv::streaming::lod::LodPageQuery(std::uint32_t)>& farther,
            std::vector<std::uint32_t>& pages, std::uint32_t& steps) const;
    [[nodiscard]] std::span<const chmv::streaming::lod::PageEntry> Directory() const;
    [[nodiscard]] std::span<const chmv::streaming::lod::PageGroup> Groups() const;
    [[nodiscard]] LodPagePass Pass() const;

    std::shared_ptr<const chmv::streaming::lod::LodPageFile> file_;
    std::shared_ptr<const std::vector<std::uint32_t>> pageImage_;
    std::shared_ptr<const std::vector<std::uint32_t>> childrenImage_;
    std::unique_ptr<chmv::streaming::lod::LodPageReader> reader_;
    LodPagePool pool_;
    LodPageRenderer renderer_;
    std::uint64_t poolBudgetBytes_ = 0;
    // Pool holds children blocks and pages are planned with the visit directory.
    bool hierarchy_ = false;
    bool poolReady_ = false;
    bool preloadAll_ = false;
    bool cullSegments_ = false;
    std::uint32_t preloadCursor_ = 0;

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
    // Last planned queries; an unchanged camera skips planning. drawQuery_ is the query the
    // drawn pages were planned with, drawView_ the visible part of it.
    chmv::streaming::lod::LodPageQuery plannedQuery_;
    chmv::streaming::lod::LodPageQuery plannedRing_;
    std::uint32_t fitSteps_ = 0;
    std::uint32_t fittedLevel_ = 0;
    std::uint32_t fittedThreshold_ = 0;
    chmv::streaming::lod::LodPageQuery drawQuery_;
    std::array<double, 4> drawView_{};
    bool planned_ = false;
    bool drawPagesFromTarget_ = true;
    DrawState committed_;
    // What the current GPU result was made from.
    std::optional<chmv::streaming::lod::LodPageQuery> filteredQuery_;
    const filtering::IRangeFilterStrategy* filteredBy_ = nullptr;
    const unfolding::IUnfoldingStrategy* unfoldedBy_ = nullptr;
    bool counted_ = false;
    bool culled_ = false;
    double uploadBytesPerMs_ = 0.0;
    std::optional<std::chrono::steady_clock::time_point> incompleteSince_;
    std::array<float, 3> lastCamera_{};
    std::array<int, 2> lastSize_{};
    LodPageRuntimeStats stats_;
};

} // namespace chmv::gpu::lod
