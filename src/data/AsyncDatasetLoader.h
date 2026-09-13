#pragma once

#include "data/ch/CHGraph.h"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace chmv::data {

enum class DatasetLoadPhase {
    Idle,
    ReadingNodes,
    ReadingEdges,
    ReadingRanges,
    PreparingRangeIndex,
    PreparingGeometryErrors,
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
    [[nodiscard]] DatasetLoadSnapshot Snapshot() const;
    [[nodiscard]] std::optional<CHGraph> TakeCompleted();

private:
    std::jthread worker_;
    std::atomic<DatasetLoadPhase> phase_{DatasetLoadPhase::Idle};
    std::atomic<float> progress_{0.0f};
    mutable std::mutex mutex_;
    std::optional<CHGraph> completedGraph_;
    std::string error_;
};

} // namespace chmv::data
