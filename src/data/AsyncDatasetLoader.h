#pragma once

#include "data/ch/CHGraph.h"
#include "streaming/index/CHIndex.h"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace chmv::data {

struct StreamingDataset {
    std::filesystem::path graphPath;
    std::filesystem::path rangesPath;
    streaming::index::CHIndexData index;
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
    PreprocessedForStreaming,
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
    [[nodiscard]] DatasetLoadSnapshot Snapshot() const;
    [[nodiscard]] std::optional<CHGraph> TakeCompleted();
    [[nodiscard]] std::optional<StreamingDataset> TakeStreamingCompleted();

private:
    bool StartInternal(const std::filesystem::path& graphPath,
                       const std::filesystem::path& rangesPath,
                       bool streamingOnly);

    std::jthread worker_;
    std::atomic<DatasetLoadPhase> phase_{DatasetLoadPhase::Idle};
    std::atomic<float> progress_{0.0f};
    mutable std::mutex mutex_;
    std::optional<CHGraph> completedGraph_;
    std::optional<StreamingDataset> completedStreamingDataset_;
    std::string error_;
};

} // namespace chmv::data
