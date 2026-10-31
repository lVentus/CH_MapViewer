#pragma once

#include "data/ch/CHGraph.h"
#include "streaming/index/CHIndex.h"
#include "streaming/lod/LodPageFile.h"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace chmv::data {

struct StreamingDataset {
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;
    streaming::index::CHIndexData index;
    // When system RAM is large enough, the entire .chbin is read once by the background loader.
    // GraphPageStreamer then serves every root/backing miss from this immutable RAM image and
    // performs only RAM -> GPU residency streaming. Empty means SSD-backed Virtual Map.
    std::shared_ptr<const std::vector<std::byte>> runtimeResidentImage;
};

enum class DatasetLoadPhase {
    Idle,
    CheckingPreprocess,
    AnalyzingSourceLayout,
    LoadingSourceForPreprocess,
    ReorderingNodes,
    ResolvingGeometryBounds,
    ReorderingEdges,
    WritingPreprocessedGraph,
    WritingPreprocessedRanges,
    ValidatingPreprocessedLayout,
    WritingPreprocessedIndex,
    ReadingNodes,
    ReadingEdges,
    ReadingRanges,
    PreparingRangeIndex,
    PreparingGeometryErrors,
    PreparingRoadStyleMetadata,
    PreloadingRuntimeBinary,
    PreprocessedForStreaming,
    ReadingSourceForLodPages,
    ComputingLodPageHierarchy,
    LayingOutLodPages,
    WritingLodPages,
    VerifyingLodPages,
    PreloadingLodPages,
    LodPagesReady,
    Ready,
    Failed,
};

struct DatasetLoadSnapshot {
    DatasetLoadPhase phase = DatasetLoadPhase::Idle;
    float progress = 0.0f;
    std::string error;

    [[nodiscard]] bool Active() const {
        return phase != DatasetLoadPhase::Idle && phase != DatasetLoadPhase::Failed;
    }
};

class AsyncDatasetLoader {
public:
    AsyncDatasetLoader() = default;
    ~AsyncDatasetLoader();

    AsyncDatasetLoader(const AsyncDatasetLoader&) = delete;
    AsyncDatasetLoader& operator=(const AsyncDatasetLoader&) = delete;

    bool Start(const std::filesystem::path& graphPath, const std::filesystem::path& rangesPath);
    bool StartStreaming(const std::filesystem::path& graphPath,
                        const std::filesystem::path& rangesPath);
    // Builds <name>.chlod next to the source pair if it is missing or stale, then opens it.
    // With keepInRam the page section is read into RAM when it fits comfortably.
    bool StartLodPages(const std::filesystem::path& graphPath,
                       const std::filesystem::path& rangesPath, bool keepInRam);
    [[nodiscard]] DatasetLoadSnapshot Snapshot() const;
    [[nodiscard]] std::optional<CHGraph> TakeCompleted();
    [[nodiscard]] std::optional<StreamingDataset> TakeStreamingCompleted();
    [[nodiscard]] std::optional<streaming::lod::LodPageDataset> TakeLodPagesCompleted();

private:
    bool StartInternal(const std::filesystem::path& graphPath,
                       const std::filesystem::path& rangesPath,
                       bool streamingOnly);
    bool PrepareStart();
    void LoadLodPages(const std::filesystem::path& graphPath,
                      const std::filesystem::path& rangesPath, bool keepInRam,
                      const std::stop_token& stopToken);

    std::jthread worker_;
    std::atomic<DatasetLoadPhase> phase_{DatasetLoadPhase::Idle};
    std::atomic<float> progress_{0.0f};
    mutable std::mutex mutex_;
    std::optional<CHGraph> completedGraph_;
    std::optional<StreamingDataset> completedStreamingDataset_;
    std::optional<streaming::lod::LodPageDataset> completedLodPages_;
    std::string error_;
};

} // namespace chmv::data
