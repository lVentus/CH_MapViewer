#pragma once

#include "streaming/lod/LodPageFile.h"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

namespace chmv::data {

enum class DatasetLoadPhase {
    Idle,
    CheckingPageFile,
    ReadingSource,
    ComputingHierarchy,
    LayingOutPages,
    WritingPages,
    VerifyingPages,
    CompressingPages,
    LoadingIntoRam,
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

// Prepares a dataset on a background thread: builds <name>.chlod next to the source pair when it
// is missing, older than the sources or of an older format, opens it, and reads its pages into
// RAM when they fit comfortably. A page file without its sources is used as it is.
class AsyncDatasetLoader {
public:
  // With a report root, a build writes its report (and every load a missing or outdated
  // layout report) into a folder named like the page file there (LodPageReport.h).
  explicit AsyncDatasetLoader(std::filesystem::path reportRoot = {})
      : reportRoot_(std::move(reportRoot)) {}
  ~AsyncDatasetLoader();

  AsyncDatasetLoader(const AsyncDatasetLoader&) = delete;
  AsyncDatasetLoader& operator=(const AsyncDatasetLoader&) = delete;

  bool Start(const std::filesystem::path& graphPath, const std::filesystem::path& rangesPath);
  bool StartPageFile(const std::filesystem::path& pageFilePath);
  [[nodiscard]] DatasetLoadSnapshot Snapshot() const;
  [[nodiscard]] std::optional<streaming::lod::LodPageDataset> TakeCompleted();

private:
    // Empty graphPath: the page file alone.
    bool Begin(std::filesystem::path pageFilePath, std::filesystem::path graphPath,
               std::filesystem::path rangesPath);
    void Load(const std::filesystem::path& pageFilePath, const std::filesystem::path& graphPath,
              const std::filesystem::path& rangesPath, const std::stop_token& stopToken);

    std::filesystem::path reportRoot_;
    std::jthread worker_;
    std::atomic<DatasetLoadPhase> phase_{DatasetLoadPhase::Idle};
    std::atomic<float> progress_{0.0f};
    mutable std::mutex mutex_;
    std::optional<streaming::lod::LodPageDataset> completed_;
    std::string error_;
};

} // namespace chmv::data
