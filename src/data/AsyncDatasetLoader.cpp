#include "data/AsyncDatasetLoader.h"

#include "data/SystemMemory.h"
#include "streaming/lod/LodPageBuilder.h"
#include "streaming/lod/LodPageReport.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace chmv::data {
namespace {

namespace lod = streaming::lod;

struct LoadCancelled {};

DatasetLoadPhase ToDatasetPhase(lod::BuildStage stage) {
    using lod::BuildStage;
    switch (stage) {
    case BuildStage::ReadNodes:
    case BuildStage::ReadEdges:
    case BuildStage::ReadRanges:
        return DatasetLoadPhase::ReadingSource;
    case BuildStage::ResolveShortcuts:
    case BuildStage::ComputeErrors:
    case BuildStage::ComputeContexts:
        return DatasetLoadPhase::ComputingHierarchy;
    case BuildStage::LayoutPages:
        return DatasetLoadPhase::LayingOutPages;
    case BuildStage::WritePages:
    case BuildStage::WriteHierarchy:
        return DatasetLoadPhase::WritingPages;
    case BuildStage::Verify:
        return DatasetLoadPhase::VerifyingPages;
    case BuildStage::Compress:
        return DatasetLoadPhase::CompressingPages;
    }
    return DatasetLoadPhase::CheckingPageFile;
}

struct ProgressRange {
    float begin = 0.0f;
    float end = 0.0f;
};

ProgressRange BuildStageRange(lod::BuildStage stage) {
    using lod::BuildStage;
    switch (stage) {
    case BuildStage::ReadNodes:        return {0.00f, 0.02f};
    case BuildStage::ReadEdges:        return {0.02f, 0.19f};
    case BuildStage::ReadRanges:       return {0.19f, 0.23f};
    case BuildStage::ResolveShortcuts: return {0.23f, 0.32f};
    case BuildStage::ComputeErrors:    return {0.32f, 0.36f};
    case BuildStage::ComputeContexts:  return {0.36f, 0.40f};
    case BuildStage::LayoutPages:      return {0.40f, 0.48f};
    case BuildStage::WritePages:       return {0.48f, 0.63f};
    case BuildStage::WriteHierarchy:   return {0.63f, 0.66f};
    case BuildStage::Verify:           return {0.66f, 0.73f};
    case BuildStage::Compress:         return {0.73f, 0.97f};
    }
    return {0.0f, 1.0f};
}

float Fraction(std::uint64_t current, std::uint64_t total) {
    if (total == 0) {
        return 1.0f;
    }
    return std::clamp(static_cast<float>(current) / static_cast<float>(total), 0.0f, 1.0f);
}

// Keep a generous reserve for the OS and the rest of the viewer, and only then hold pages in
// memory.
bool FitsInRam(std::uint64_t bytes) {
    const auto memory = QuerySystemMemoryInfo();
    if (memory.availablePhysicalBytes == 0u) {
        return false;
    }
    constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
    const auto reserve = std::max<std::uint64_t>(6ull * kGiB, memory.totalPhysicalBytes / 5u);
    return bytes + kGiB + reserve < memory.availablePhysicalBytes;
}

} // namespace

AsyncDatasetLoader::~AsyncDatasetLoader() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
}

bool AsyncDatasetLoader::Start(const std::filesystem::path& graphPath,
                               const std::filesystem::path& rangesPath) {
    return Begin(lod::LodPageFile::DefaultPath(graphPath), graphPath, rangesPath);
}

bool AsyncDatasetLoader::StartPageFile(const std::filesystem::path& pageFilePath) {
    return Begin(pageFilePath, {}, {});
}

bool AsyncDatasetLoader::Begin(std::filesystem::path pageFilePath, std::filesystem::path graphPath,
                               std::filesystem::path rangesPath) {
    const auto currentPhase = phase_.load();
    if (currentPhase != DatasetLoadPhase::Idle && currentPhase != DatasetLoadPhase::Failed &&
        currentPhase != DatasetLoadPhase::Ready) {
        return false;
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    {
        std::lock_guard lock(mutex_);
        completed_.reset();
        error_.clear();
    }
    progress_.store(0.0f);
    phase_.store(DatasetLoadPhase::CheckingPageFile);

    worker_ = std::jthread([this, pageFilePath = std::move(pageFilePath),
                            graphPath = std::move(graphPath),
                            rangesPath = std::move(rangesPath)](std::stop_token stopToken) {
        const auto fail = [this](std::string message) {
            std::cerr << "Dataset load failed: " << message << '\n';
            {
                std::lock_guard lock(mutex_);
                error_ = std::move(message);
            }
            phase_.store(DatasetLoadPhase::Failed);
            progress_.store(0.0f);
        };
        try {
            Load(pageFilePath, graphPath, rangesPath, stopToken);
        } catch (const LoadCancelled&) {
            phase_.store(DatasetLoadPhase::Idle);
            progress_.store(0.0f);
        } catch (const std::exception& error) {
            fail(error.what());
        } catch (...) {
            fail("unknown error while loading the dataset");
        }
    });
    return true;
}

void AsyncDatasetLoader::Load(const std::filesystem::path& path,
                              const std::filesystem::path& graphPath,
                              const std::filesystem::path& rangesPath,
                              const std::stop_token& stopToken) {
    // Building reports every stage in its share of the bar; compressing an existing file alone
    // gets the whole bar.
    const auto buildProgress = [this, &stopToken](bool building) {
        return [this, &stopToken, building](const lod::BuildProgress& progress) {
            if (stopToken.stop_requested()) {
                throw LoadCancelled{};
            }
            phase_.store(ToDatasetPhase(progress.stage));
            const auto range =
                building ? BuildStageRange(progress.stage) : ProgressRange{0.0f, 0.97f};
            const auto fraction = Fraction(progress.current, progress.total);
            progress_.store(range.begin + (range.end - range.begin) * fraction);
        };
    };
    std::optional<lod::BuildResult> buildResult;
    const auto build = [&] {
        const auto& result = buildResult.emplace(
            lod::LodPageBuilder::Build(graphPath, rangesPath, path, buildProgress(true)));
        std::cout << "LOD page file ready after " << result.seconds << " s: " << result.recordCount
                  << " records in " << result.pageCount << " pages (" << result.extraContextRecords
                  << " extra contexts, " << result.coverGapEdges << " split covers); "
                  << result.verifiedCuts << " reference cuts match DFS exactly; "
                  << result.fileBytes / (1024u * 1024u) << " MiB compressed from "
                  << result.uncompressedBytes / (1024u * 1024u) << " MiB\n";
    };
    // A file of version 3 is the current file uncompressed: compressing it is enough. If that
    // fails (no disk space, say), it is used as it is.
    std::shared_ptr<const lod::LodPageFile> file;
    const auto open = [&] {
        file = lod::LodPageFile::Open(path);
        if (file->Compressed()) {
            return;
        }
        std::cout << "Compressing LOD page file " << path << " in background...\n";
        try {
            lod::LodPageBuilder::Compress(path, buildProgress(false));
        } catch (const std::exception& error) {
            std::cout << "Could not compress " << path << " (" << error.what()
                      << "); using it uncompressed\n";
        }
        file = lod::LodPageFile::Open(path);
    };

    bool built = false;
    if (graphPath.empty()) {
        std::cout << "Using LOD page file " << path << " (without its source files)\n";
    } else if (lod::LodPageFile::MatchesSource(path, graphPath, rangesPath)) {
        std::cout << "Using LOD page file " << path << '\n';
    } else {
        std::cout << "Building LOD page file " << path
                  << " (missing, older than its sources or of an older format) in background...\n";
        build();
        built = true;
    }
    try {
        open();
    } catch (const std::exception& error) {
        if (graphPath.empty()) {
            throw std::runtime_error(std::string(error.what()) +
                                     " (rebuilding it needs the .sch/.ranges it was made from)");
        }
        if (built) {
            throw;
        }
        // Damage past the header: rebuilt from the sources once.
        std::cout << "LOD page file " << path << " is damaged (" << error.what()
                  << "); rebuilding it in background...\n";
        build();
        open();
    }
    // Reports are for later analysis; failing to write them does not stop the load.
    if (!reportRoot_.empty()) {
        const auto reports = lod::ReportDirectory(reportRoot_, path);
        try {
            if (buildResult) {
                lod::WriteBuildReport(*buildResult, reports);
            }
            if (buildResult || lod::LayoutReportOutdated(*file, reports)) {
                lod::WriteLayoutReport(*file, reports);
                std::cout << "Reports written to " << reports << '\n';
            }
        } catch (const std::exception& error) {
            std::cout << "Could not write the reports to " << reports << ": " << error.what()
                      << '\n';
        }
    }
    const auto pageBytes = file->PageSectionBytes();
    const auto childBytes = file->Header().At(lod::Extension::PageChildren).bytes;
    const bool withChildren = FitsInRam(pageBytes + childBytes);
    lod::LodPageDataset dataset{file, nullptr, nullptr};
    if (withChildren || FitsInRam(pageBytes)) {
        phase_.store(DatasetLoadPhase::LoadingIntoRam);
        // Each part reports its own (stored) bytes; it counts by its uncompressed size.
        const auto total = pageBytes + (withChildren ? childBytes : 0u);
        const auto report = [this, &stopToken, total](std::uint64_t before, std::uint64_t bytes) {
            return [this, &stopToken, total, before, bytes](std::uint64_t current,
                                                            std::uint64_t partTotal) {
                if (stopToken.stop_requested()) {
                    throw LoadCancelled{};
                }
                const auto part = Fraction(current, partTotal) * static_cast<float>(bytes);
                progress_.store(0.97f + 0.03f * (static_cast<float>(before) + part) /
                                            static_cast<float>(std::max<std::uint64_t>(total, 1u)));
            };
        };
        dataset.pageImage = std::make_shared<const std::vector<std::uint32_t>>(
            file->ReadPageWords(report(0, pageBytes)));
        if (withChildren) {
            dataset.childrenImage = std::make_shared<const std::vector<std::uint32_t>>(
                file->ReadChildrenWords(report(pageBytes, childBytes)));
        }
    } else {
        std::cout << "LOD pages do not fit the RAM reserve; reading pages from disk.\n";
    }
    {
        std::lock_guard lock(mutex_);
        completed_ = std::move(dataset);
    }
    progress_.store(1.0f);
    phase_.store(DatasetLoadPhase::Ready);
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

std::optional<streaming::lod::LodPageDataset> AsyncDatasetLoader::TakeCompleted() {
    if (phase_.load() != DatasetLoadPhase::Ready) {
        return std::nullopt;
    }
    std::optional<streaming::lod::LodPageDataset> dataset;
    {
        std::lock_guard lock(mutex_);
        dataset = std::move(completed_);
        completed_.reset();
    }
    phase_.store(DatasetLoadPhase::Idle);
    progress_.store(0.0f);
    return dataset;
}

} // namespace chmv::data
