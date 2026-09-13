#include "data/AsyncDatasetLoader.h"

#include "data/ch/CHLoader.h"
#include "data/preprocessing/GeometryErrorPreprocessor.h"

#include <exception>
#include <stdexcept>
#include <utility>

namespace chmv::data {
namespace {

struct LoadCancelled {};

DatasetLoadPhase ToDatasetPhase(CHLoadStage stage) {
    switch (stage) {
    case CHLoadStage::Nodes:
        return DatasetLoadPhase::ReadingNodes;
    case CHLoadStage::Edges:
        return DatasetLoadPhase::ReadingEdges;
    case CHLoadStage::Ranges:
        return DatasetLoadPhase::ReadingRanges;
    }
    return DatasetLoadPhase::ReadingNodes;
}

} // namespace

AsyncDatasetLoader::~AsyncDatasetLoader() {
    if (worker_.joinable()) {
        worker_.request_stop();
    }
}

bool AsyncDatasetLoader::Start(const std::filesystem::path& graphPath,
                               const std::filesystem::path& rangesPath) {
    const auto currentPhase = phase_.load();
    if (currentPhase != DatasetLoadPhase::Idle && currentPhase != DatasetLoadPhase::Failed) {
        return false;
    }

    if (worker_.joinable()) {
        worker_.join();
    }

    {
        std::lock_guard lock(mutex_);
        completedGraph_.reset();
        error_.clear();
    }

    progress_.store(0.0f);
    phase_.store(DatasetLoadPhase::ReadingNodes);

    worker_ = std::jthread([this, graphPath, rangesPath](std::stop_token stopToken) {
        try {
            auto graph = CHLoader::Load(
                graphPath, rangesPath, [this, stopToken](const CHLoadProgress& loadProgress) {
                    if (stopToken.stop_requested()) {
                        throw LoadCancelled{};
                    }
                    phase_.store(ToDatasetPhase(loadProgress.stage));
                    progress_.store(loadProgress.overall * 0.90f);
                });

            if (stopToken.stop_requested()) {
                throw LoadCancelled{};
            }

            phase_.store(DatasetLoadPhase::PreparingRangeIndex);
            progress_.store(0.90f);
            (void)graph.OrderedRanges();
            progress_.store(0.95f);

            phase_.store(DatasetLoadPhase::PreparingGeometryErrors);
            preprocessing::GeometryErrorPreprocessor::Run(
                graph, [this, stopToken](float progress) {
                    if (stopToken.stop_requested()) {
                        throw LoadCancelled{};
                    }
                    progress_.store(0.95f + progress * 0.045f);
                });
            progress_.store(0.995f);
            (void)graph.ShortcutCount();

            if (stopToken.stop_requested()) {
                throw LoadCancelled{};
            }

            {
                std::lock_guard lock(mutex_);
                completedGraph_ = std::move(graph);
            }
            progress_.store(1.0f);
            phase_.store(DatasetLoadPhase::Ready);
        } catch (const LoadCancelled&) {
            phase_.store(DatasetLoadPhase::Idle);
            progress_.store(0.0f);
        } catch (const std::exception& error) {
            {
                std::lock_guard lock(mutex_);
                error_ = error.what();
            }
            phase_.store(DatasetLoadPhase::Failed);
            progress_.store(0.0f);
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                error_ = "unknown error while loading dataset";
            }
            phase_.store(DatasetLoadPhase::Failed);
            progress_.store(0.0f);
        }
    });

    return true;
}

DatasetLoadSnapshot AsyncDatasetLoader::Snapshot() const {
    DatasetLoadSnapshot snapshot;
    snapshot.phase = phase_.load();
    snapshot.progress = progress_.load();
    if (snapshot.phase == DatasetLoadPhase::Failed) {
        std::lock_guard lock(mutex_);
        snapshot.error = error_;
    }
    return snapshot;
}

std::optional<CHGraph> AsyncDatasetLoader::TakeCompleted() {
    if (phase_.load() != DatasetLoadPhase::Ready) {
        return std::nullopt;
    }

    std::optional<CHGraph> graph;
    {
        std::lock_guard lock(mutex_);
        graph = std::move(completedGraph_);
        completedGraph_.reset();
    }
    phase_.store(DatasetLoadPhase::Idle);
    progress_.store(0.0f);
    return graph;
}

} // namespace chmv::data
