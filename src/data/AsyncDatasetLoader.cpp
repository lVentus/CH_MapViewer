#include "data/AsyncDatasetLoader.h"

#include "data/ch/CHLoader.h"
#include "data/preprocessing/GeometryErrorPreprocessor.h"
#include "streaming/analysis/DatasetLayoutAnalyzer.h"
#include "streaming/preprocess/DatasetPreprocessor.h"
#include "streaming/index/CHIndex.h"
#include "streaming/runtime/GraphPageStreamer.h"
#include "streaming/runtime/RuntimeData.h"
#include "streaming/analysis/TextSourceScanner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <new>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

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

DatasetLoadPhase ToDatasetPhase(streaming::preprocess::PreprocessStage stage) {
    using streaming::preprocess::PreprocessStage;
    switch (stage) {
    case PreprocessStage::AnalyzeSource:
        return DatasetLoadPhase::AnalyzingSourceLayout;
    case PreprocessStage::LoadSource:
        return DatasetLoadPhase::LoadingSourceForPreprocess;
    case PreprocessStage::ReorderNodes:
        return DatasetLoadPhase::ReorderingNodes;
    case PreprocessStage::ResolveGeometryBounds:
        return DatasetLoadPhase::ResolvingGeometryBounds;
    case PreprocessStage::ReorderEdges:
        return DatasetLoadPhase::ReorderingEdges;
    case PreprocessStage::WriteGraph:
        return DatasetLoadPhase::WritingPreprocessedGraph;
    case PreprocessStage::WriteRanges:
        return DatasetLoadPhase::WritingPreprocessedRanges;
    case PreprocessStage::ValidateOutput:
        return DatasetLoadPhase::ValidatingPreprocessedLayout;
    case PreprocessStage::WriteIndex:
        return DatasetLoadPhase::WritingPreprocessedIndex;
    }
    return DatasetLoadPhase::CheckingPreprocess;
}

struct PreprocessProgressRange {
    float begin = 0.0f;
    float end = 0.0f;
};

PreprocessProgressRange PreprocessStageRange(streaming::preprocess::PreprocessStage stage) {
    using streaming::preprocess::PreprocessStage;
    switch (stage) {
    case PreprocessStage::AnalyzeSource:          return {0.00f, 0.15f};
    case PreprocessStage::LoadSource:             return {0.15f, 0.35f};
    case PreprocessStage::ReorderNodes:           return {0.35f, 0.45f};
    case PreprocessStage::ResolveGeometryBounds:  return {0.45f, 0.60f};
    case PreprocessStage::ReorderEdges:           return {0.60f, 0.72f};
    case PreprocessStage::WriteGraph:             return {0.72f, 0.84f};
    case PreprocessStage::WriteRanges:            return {0.84f, 0.89f};
    case PreprocessStage::ValidateOutput:         return {0.89f, 0.92f};
    case PreprocessStage::WriteIndex:             return {0.92f, 0.95f};
    }
    return {0.0f, 0.0f};
}

float Fraction(std::uint64_t current, std::uint64_t total) {
    if (total == 0) {
        return 1.0f;
    }
    return std::clamp(static_cast<float>(current) / static_cast<float>(total), 0.0f, 1.0f);
}

bool CanWholeLoadGraph(std::uint64_t nodeCount, std::uint64_t edgeCount) {
    const auto memory = streaming::runtime::QuerySystemMemoryInfo();
    if (memory.availablePhysicalBytes == 0) {
        return true;
    }
    // Full-resident mode materializes CHGraph plus the temporary GPU upload arrays.
    // Refuse explicitly instead of silently switching to out-of-core: OFF must remain
    // a trustworthy no-streaming baseline.
    const auto estimatedBytes = nodeCount * 48ull + edgeCount * 80ull + (256ull << 20u);
    return estimatedBytes < memory.availablePhysicalBytes / 2u;
}

data::CHProjection RuntimeProjectionFromIndex(const streaming::index::CHIndexData& index) {
    constexpr double kPi = 3.14159265358979323846;
    data::CHProjection projection;
    projection.centerLatitude = (index.minLatitude + index.maxLatitude) * 0.5;
    projection.centerLongitude = (index.minLongitude + index.maxLongitude) * 0.5;
    projection.longitudeScale = std::cos(projection.centerLatitude * kPi / 180.0);
    const auto halfWidth = std::max(
        (index.maxLongitude - index.minLongitude) * 0.5 * projection.longitudeScale, 1e-12);
    const auto halfHeight =
        std::max((index.maxLatitude - index.minLatitude) * 0.5, 1e-12);
    projection.normalization = 0.95 / std::max(halfWidth, halfHeight);
    return projection;
}

template <typename ProgressFn>
data::CHGraph LoadWholeGraphFromRuntimeBinary(
    const std::filesystem::path& runtimePath,
    const streaming::index::CHIndexData& index,
    ProgressFn&& progress) {
    using streaming::runtime::RuntimeData;
    using streaming::runtime::RuntimeEdgeRecord;
    using streaming::runtime::RuntimeNodeRecord;

    const auto header = RuntimeData::ReadHeader(runtimePath);
    if (header.nodeCount != index.nodeCount || header.edgeCount != index.edgeCount) {
        throw std::runtime_error("runtime binary/index count mismatch in full-resident load");
    }
    if (header.nodeCount > std::numeric_limits<std::size_t>::max() ||
        header.edgeCount > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("runtime binary exceeds host addressable vector size");
    }

    const auto begin = std::chrono::steady_clock::now();
    std::ifstream input(runtimePath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open runtime binary for full-resident load: " +
                                 runtimePath.string());
    }

    data::CHGraph graph;
    auto& nodes = graph.Nodes();
    auto& edges = graph.Edges();
    auto& ranges = graph.Ranges();
    nodes.resize(static_cast<std::size_t>(header.nodeCount));
    edges.resize(static_cast<std::size_t>(header.edgeCount));
    ranges.resize(static_cast<std::size_t>(header.edgeCount));

    const auto projection = RuntimeProjectionFromIndex(index);
    constexpr std::size_t kNodeChunk = 1u << 20u;
    std::vector<RuntimeNodeRecord> nodeRecords(kNodeChunk);
    for (std::uint64_t first = 0; first < header.nodeCount; first += kNodeChunk) {
        const auto count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kNodeChunk, header.nodeCount - first));
        RuntimeData::ReadRecords(
            input, header.nodeSectionOffset + first * sizeof(RuntimeNodeRecord),
            std::span<RuntimeNodeRecord>(nodeRecords.data(), count));
        for (std::size_t i = 0; i < count; ++i) {
            const auto& src = nodeRecords[i];
            auto& dst = nodes[static_cast<std::size_t>(first) + i];
            dst.latitude = static_cast<double>(src.y) / projection.normalization +
                           projection.centerLatitude;
            const auto lonScale = projection.longitudeScale * projection.normalization;
            dst.longitude = std::abs(lonScale) > 1e-15
                                ? static_cast<double>(src.x) / lonScale +
                                      projection.centerLongitude
                                : projection.centerLongitude;
        }
        progress(DatasetLoadPhase::ReadingNodes, first + count, header.nodeCount);
    }

    constexpr std::size_t kEdgeChunk = 1u << 19u;
    std::vector<RuntimeEdgeRecord> edgeRecords(kEdgeChunk);
    for (std::uint64_t first = 0; first < header.edgeCount; first += kEdgeChunk) {
        const auto count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kEdgeChunk, header.edgeCount - first));
        RuntimeData::ReadRecords(
            input, header.edgeSectionOffset + first * sizeof(RuntimeEdgeRecord),
            std::span<RuntimeEdgeRecord>(edgeRecords.data(), count));
        for (std::size_t i = 0; i < count; ++i) {
            const auto& src = edgeRecords[i];
            const auto id = static_cast<std::size_t>(first) + i;
            edges[id] = {
                src.source, src.target, src.weight, static_cast<std::int32_t>(src.type),
                static_cast<std::int32_t>(src.maxSpeed), src.childA, src.childB,
                src.geometryError,
            };
            ranges[id] = {
                static_cast<std::int32_t>(src.birthLevel),
                static_cast<std::int32_t>(src.deathLevel),
            };
        }
        progress(DatasetLoadPhase::ReadingEdges, first + count, header.edgeCount);
    }

    // Preserve exactly the projection used to produce RuntimeNodeRecord. This avoids rebuilding
    // it from float-rounded coordinates when RoadRenderer uploads the whole graph.
    graph.SetProjection(projection);
    const auto elapsedMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - begin)
                               .count();
    const auto payloadMiB = static_cast<double>(
        header.nodeCount * sizeof(RuntimeNodeRecord) +
        header.edgeCount * sizeof(RuntimeEdgeRecord)) / (1024.0 * 1024.0);
    std::cout << "Full-resident binary load: " << payloadMiB << " MiB in "
              << elapsedMs << " ms ("
              << (elapsedMs > 0.0 ? payloadMiB / (elapsedMs / 1000.0) : 0.0)
              << " MiB/s, no text parsing)\n";
    return graph;
}

std::shared_ptr<const std::vector<std::byte>> TryPreloadRuntimeBinary(
    const std::filesystem::path& runtimePath,
    const streaming::index::CHIndexData& index,
    const std::function<void(std::uint64_t, std::uint64_t)>& progress) {
    if (!index.HasRuntimeBinary() || index.runtimeDataFileBytes == 0u) {
        return {};
    }
    const auto memory = streaming::runtime::QuerySystemMemoryInfo();
    if (memory.availablePhysicalBytes == 0u) {
        return {};
    }
    constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
    const auto reserve = std::max<std::uint64_t>(6ull * kGiB, memory.totalPhysicalBytes / 5u);
    // A full RAM image replaces runtime SSD reads; do not require room for the *maximum* normal
    // page-cache recommendation before allowing it. After the image is allocated, the streamer
    // sizes its decoded/page cache from the remaining physical memory. Requiring only a modest
    // working-cache floor here lets machines with enough RAM for the world graph actually select
    // RAM-backed OOC instead of needlessly falling back to SSD-backed streaming.
    constexpr std::uint64_t kMinimumWorkingCache = 1ull * kGiB;
    if (index.runtimeDataFileBytes + kMinimumWorkingCache + reserve >=
        memory.availablePhysicalBytes) {
        const auto needGiB = static_cast<double>(index.runtimeDataFileBytes +
                                                  kMinimumWorkingCache + reserve) /
                             static_cast<double>(kGiB);
        const auto availableGiB = static_cast<double>(memory.availablePhysicalBytes) /
                                  static_cast<double>(kGiB);
        std::cout << "RAM-backed Virtual Map not selected: needs about " << needGiB
                  << " GiB including safety/cache reserve, available " << availableGiB
                  << " GiB; using SSD-backed Virtual Map.\n";
        return {};
    }

    try {
        const auto begin = std::chrono::steady_clock::now();
        auto image = std::make_shared<std::vector<std::byte>>(
            static_cast<std::size_t>(index.runtimeDataFileBytes));
        std::ifstream input(runtimePath, std::ios::binary);
        if (!input) {
            return {};
        }
        constexpr std::size_t kChunkBytes = 64u << 20u;
        std::size_t offset = 0u;
        while (offset < image->size()) {
            const auto count = std::min(kChunkBytes, image->size() - offset);
            input.read(reinterpret_cast<char*>(image->data() + offset),
                       static_cast<std::streamsize>(count));
            if (!input) {
                return {};
            }
            offset += count;
            progress(offset, image->size());
        }
        const auto ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
        const auto mib = static_cast<double>(image->size()) / (1024.0 * 1024.0);
        std::cout << "RAM-backed Virtual Map: preloaded " << mib << " MiB .chbin in "
                  << ms << " ms ("
                  << (ms > 0.0 ? mib / (ms / 1000.0) : 0.0) << " MiB/s).\n";
        return image;
    } catch (const std::bad_alloc&) {
        std::cout << "RAM-backed preload allocation failed; using SSD-backed Virtual Map.\n";
        return {};
    }
}

using streaming::analysis::detail::TextSourceScanner;

bool RootRoadStyleMetadataReady(const streaming::index::CHIndexData& index) {
    // CHIDX v8 roots are emitted into .chbin with road-style metadata already encoded.
    if (index.HasRuntimeBinary() || !index.HasRootPayload() || index.rootPayloadRecordCount == 0u) {
        return true;
    }
    std::ifstream input(index.indexPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not inspect CH index road-style metadata");
    }
    const auto recordOffset = index.rootPayloadSectionOffset +
        (index.rootPayloadRecordCount - 1u) * sizeof(streaming::index::CHIndexRootRecord);
    if (recordOffset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("CH index road-style metadata offset exceeds stream range");
    }
    input.seekg(static_cast<std::streamoff>(recordOffset), std::ios::beg);
    streaming::index::CHIndexRootRecord record;
    input.read(reinterpret_cast<char*>(&record), sizeof(record));
    if (!input) {
        throw std::runtime_error("could not read CH index road-style metadata marker");
    }
    return streaming::index::HasRootRoadStyleMetadata(record.boundsLo, record.boundsHi);
}

template <typename ProgressFn>
void EnsureRootRoadStyleMetadata(const streaming::index::CHIndexData& index,
                                 const std::filesystem::path& graphPath,
                                 ProgressFn&& progress) {
    if (RootRoadStyleMetadataReady(index)) {
        return;
    }
    if (index.graphEdgeBlocks.empty() || !index.HasRootPayload()) {
        throw std::runtime_error("road-style metadata requires a current CHIDX RootPayload");
    }

    std::cout << "Preparing road-type metadata for existing CHIDX (one-time, no graph remap)...\n";
    TextSourceScanner graph(graphPath);
    graph.Seek(index.graphEdgeBlocks.front().byteOffset);

    std::fstream indexFile(index.indexPath,
                           std::ios::binary | std::ios::in | std::ios::out);
    if (!indexFile) {
        throw std::runtime_error("could not open CH index for road-style metadata update");
    }

    constexpr std::uint64_t kChunkRecords = 1u << 16u;
    std::vector<streaming::index::CHIndexRootRecord> records;
    records.reserve(static_cast<std::size_t>(kChunkRecords));
    std::uint64_t processed = 0;
    int lastPercent = -1;
    while (processed < index.rootPayloadRecordCount) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
            kChunkRecords, index.rootPayloadRecordCount - processed));
        records.resize(count);
        const auto byteOffset = index.rootPayloadSectionOffset +
            processed * sizeof(streaming::index::CHIndexRootRecord);
        if (byteOffset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            throw std::runtime_error("CH index road-style update offset exceeds stream range");
        }

        indexFile.clear();
        indexFile.seekg(static_cast<std::streamoff>(byteOffset), std::ios::beg);
        indexFile.read(reinterpret_cast<char*>(records.data()),
                       static_cast<std::streamsize>(count * sizeof(records.front())));
        if (!indexFile) {
            throw std::runtime_error("truncated CH index RootPayload during road-style update");
        }

        for (std::size_t i = 0; i < count; ++i) {
            const auto expectedEdge = processed + i;
            if (records[i].globalEdgeId != expectedEdge) {
                throw std::runtime_error(
                    "road-style metadata requires drawable RootPayload records to match reordered EdgeIDs");
            }
            graph.Read<std::uint32_t>();
            graph.Read<std::uint32_t>();
            graph.Read<float>();
            const auto roadType = graph.Read<std::int32_t>();
            graph.Read<std::int32_t>();
            graph.Read<std::int64_t>();
            graph.Read<std::int64_t>();
            streaming::index::EncodeRootRoadStyle(
                records[i].boundsLo, records[i].boundsHi,
                data::RoadStyleIndexFromType(roadType));
        }

        indexFile.clear();
        indexFile.seekp(static_cast<std::streamoff>(byteOffset), std::ios::beg);
        indexFile.write(reinterpret_cast<const char*>(records.data()),
                        static_cast<std::streamsize>(count * sizeof(records.front())));
        if (!indexFile) {
            throw std::runtime_error("failed to write CH index road-style metadata");
        }

        processed += count;
        progress(processed, index.rootPayloadRecordCount);
        const auto percent = static_cast<int>(processed * 100u / index.rootPayloadRecordCount);
        if (percent >= lastPercent + 5 || processed == index.rootPayloadRecordCount) {
            lastPercent = percent;
            std::cout << "Road-type metadata: " << percent << "% (" << processed << " / "
                      << index.rootPayloadRecordCount << " roots)\n";
        }
    }
    indexFile.flush();
    if (!indexFile) {
        throw std::runtime_error("failed to finalize CH index road-style metadata");
    }
    if (!RootRoadStyleMetadataReady(index)) {
        throw std::runtime_error("CH index road-style metadata validation failed");
    }
    std::cout << "Road-type metadata ready. Future loads skip this one-time pass.\n";
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
    return StartInternal(graphPath, rangesPath, false);
}

bool AsyncDatasetLoader::StartStreaming(const std::filesystem::path& graphPath,
                                        const std::filesystem::path& rangesPath) {
    return StartInternal(graphPath, rangesPath, true);
}

bool AsyncDatasetLoader::StartInternal(const std::filesystem::path& graphPath,
                                       const std::filesystem::path& rangesPath,
                                       bool streamingOnly) {
    const auto currentPhase = phase_.load();
    if (currentPhase != DatasetLoadPhase::Idle && currentPhase != DatasetLoadPhase::Failed &&
        currentPhase != DatasetLoadPhase::PreprocessedForStreaming) {
        return false;
    }

    if (worker_.joinable()) {
        worker_.join();
    }

    {
        std::lock_guard lock(mutex_);
        completedGraph_.reset();
        completedStreamingDataset_.reset();
        error_.clear();
    }

    progress_.store(0.0f);
    phase_.store(DatasetLoadPhase::CheckingPreprocess);

    worker_ = std::jthread([this, graphPath, rangesPath, streamingOnly](std::stop_token stopToken) {
        try {
            const streaming::analysis::DatasetLayoutAnalysisConfig analysisConfig;
            auto activeGraphPath = graphPath;
            auto activeRangesPath = rangesPath;
            bool preprocessedNow = false;

            if (!streaming::preprocess::DatasetPreprocessor::HasValidPreprocess(
                    graphPath, rangesPath, analysisConfig)) {
                std::cout << "No valid preprocess result found. Building fixed-grid out-of-core layout in background...\n";
                const auto preprocessResult = streaming::preprocess::DatasetPreprocessor::Run(
                    graphPath, rangesPath, analysisConfig, false,
                    [this, stopToken](const streaming::preprocess::PreprocessProgress& preprocessProgress) {
                        if (stopToken.stop_requested()) {
                            throw LoadCancelled{};
                        }

                        phase_.store(ToDatasetPhase(preprocessProgress.stage));
                        const auto range = PreprocessStageRange(preprocessProgress.stage);
                        progress_.store(range.begin +
                                        (range.end - range.begin) *
                                            Fraction(preprocessProgress.current, preprocessProgress.total));
                    });
                activeGraphPath = preprocessResult.paths.graphPath;
                activeRangesPath = preprocessResult.paths.rangesPath;
                preprocessedNow = !preprocessResult.reused;
                progress_.store(0.80f);
                std::cout << "Preprocess ready at " << preprocessResult.paths.directory << '\n';
            } else {
                const auto preprocessed = streaming::preprocess::DatasetPreprocessor::DefaultPaths(
                    graphPath, rangesPath);
                activeGraphPath = preprocessed.graphPath;
                activeRangesPath = preprocessed.rangesPath;
                std::cout << "Using cached preprocessed dataset from " << preprocessed.directory << '\n';
            }

            if (stopToken.stop_requested()) {
                throw LoadCancelled{};
            }

            const auto preprocessedPaths = streaming::preprocess::DatasetPreprocessor::DefaultPaths(
                graphPath, rangesPath);
            auto preprocessedIndex = streaming::index::CHIndex::Load(preprocessedPaths.indexPath);
            if (!RootRoadStyleMetadataReady(preprocessedIndex)) {
                phase_.store(DatasetLoadPhase::PreparingRoadStyleMetadata);
                EnsureRootRoadStyleMetadata(
                    preprocessedIndex, preprocessedPaths.graphPath,
                    [this, stopToken](std::uint64_t current, std::uint64_t total) {
                        if (stopToken.stop_requested()) {
                            throw LoadCancelled{};
                        }
                        progress_.store(Fraction(current, total));
                    });
            }
            if (streamingOnly) {
                std::shared_ptr<const std::vector<std::byte>> runtimeResidentImage;
                if (preprocessedIndex.HasRuntimeBinary()) {
                    phase_.store(DatasetLoadPhase::PreloadingRuntimeBinary);
                    runtimeResidentImage = TryPreloadRuntimeBinary(
                        preprocessedPaths.runtimeDataPath, preprocessedIndex,
                        [this, stopToken](std::uint64_t current, std::uint64_t total) {
                            if (stopToken.stop_requested()) {
                                throw LoadCancelled{};
                            }
                            progress_.store(0.95f + 0.05f * Fraction(current, total));
                        });
                }
                {
                    std::lock_guard lock(mutex_);
                    completedStreamingDataset_ = StreamingDataset{
                        preprocessedPaths.graphPath,
                        preprocessedPaths.rangesPath,
                        std::move(preprocessedIndex),
                        std::move(runtimeResidentImage),
                    };
                }
                std::cout << "Out-of-core ON: binary virtual-map runtime selected.\n";
                progress_.store(1.0f);
                phase_.store(DatasetLoadPhase::PreprocessedForStreaming);
                return;
            }

            // Out-of-core OFF is an explicit baseline: everything must become host-resident and
            // then GPU-resident. Never silently fall back to Virtual Map, otherwise OFF and ON are
            // not comparable benchmarks. v8+ loads the binary runtime directly (no ASCII parse).
            if (preprocessedIndex.HasRuntimeBinary()) {
                if (!CanWholeLoadGraph(preprocessedIndex.nodeCount, preprocessedIndex.edgeCount)) {
                    throw std::runtime_error(
                        "Out-of-core is OFF, but the full graph does not fit the safe host-RAM "
                        "budget. Enable out-of-core mode for this dataset.");
                }
                const auto runtimePath = preprocessedPaths.runtimeDataPath;
                auto graph = LoadWholeGraphFromRuntimeBinary(
                    runtimePath, preprocessedIndex,
                    [this, stopToken](DatasetLoadPhase stage, std::uint64_t current,
                                      std::uint64_t total) {
                        if (stopToken.stop_requested()) {
                            throw LoadCancelled{};
                        }
                        phase_.store(stage);
                        const auto base = stage == DatasetLoadPhase::ReadingNodes ? 0.0f : 0.35f;
                        const auto scale = stage == DatasetLoadPhase::ReadingNodes ? 0.35f : 0.55f;
                        progress_.store(base + scale * Fraction(current, total));
                    });
                if (stopToken.stop_requested()) {
                    throw LoadCancelled{};
                }
                phase_.store(DatasetLoadPhase::PreparingRangeIndex);
                progress_.store(0.92f);
                const auto rangeBegin = std::chrono::steady_clock::now();
                (void)graph.OrderedRanges();
                const auto rangeMs = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - rangeBegin)
                                         .count();
                std::cout << "Full-resident range index: " << rangeMs << " ms\n";
                progress_.store(0.99f);
                {
                    std::lock_guard lock(mutex_);
                    completedGraph_ = std::move(graph);
                }
                progress_.store(1.0f);
                phase_.store(DatasetLoadPhase::Ready);
                std::cout << "Out-of-core OFF: full binary graph is resident in RAM; GPU upload "
                             "will be performed as one no-streaming install.\n";
                return;
            }

            // Legacy pre-v8 fallback only. New preprocess results never take this text path.
            if (!CanWholeLoadGraph(preprocessedIndex.nodeCount, preprocessedIndex.edgeCount)) {
                throw std::runtime_error(
                    "Out-of-core is OFF, but the legacy graph does not fit the safe host-RAM "
                    "budget. Enable out-of-core mode.");
            }

            const float loadBase = preprocessedNow ? 0.80f : 0.0f;
            const float loadScale = preprocessedNow ? 0.15f : 0.90f;
            auto graph = CHLoader::Load(
                activeGraphPath, activeRangesPath,
                [this, stopToken, loadBase, loadScale](const CHLoadProgress& loadProgress) {
                    if (stopToken.stop_requested()) {
                        throw LoadCancelled{};
                    }
                    phase_.store(ToDatasetPhase(loadProgress.stage));
                    progress_.store(loadBase + loadProgress.overall * loadScale);
                });

            if (stopToken.stop_requested()) {
                throw LoadCancelled{};
            }

            phase_.store(DatasetLoadPhase::PreparingRangeIndex);
            progress_.store(preprocessedNow ? 0.95f : 0.90f);
            (void)graph.OrderedRanges();
            progress_.store(preprocessedNow ? 0.97f : 0.95f);

            phase_.store(DatasetLoadPhase::PreparingGeometryErrors);
            preprocessing::GeometryErrorPreprocessor::Run(
                graph, [this, stopToken, preprocessedNow](float progress) {
                    if (stopToken.stop_requested()) {
                        throw LoadCancelled{};
                    }
                    const float base = preprocessedNow ? 0.97f : 0.95f;
                    const float scale = preprocessedNow ? 0.025f : 0.045f;
                    progress_.store(base + progress * scale);
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
            std::cerr << "Dataset load failed: " << error.what() << '\n';
            {
                std::lock_guard lock(mutex_);
                error_ = error.what();
            }
            phase_.store(DatasetLoadPhase::Failed);
            progress_.store(0.0f);
        } catch (...) {
            std::cerr << "Dataset load failed: unknown error while loading dataset\n";
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

std::optional<StreamingDataset> AsyncDatasetLoader::TakeStreamingCompleted() {
    if (phase_.load() != DatasetLoadPhase::PreprocessedForStreaming) {
        return std::nullopt;
    }

    std::optional<StreamingDataset> dataset;
    {
        std::lock_guard lock(mutex_);
        dataset = std::move(completedStreamingDataset_);
        completedStreamingDataset_.reset();
    }
    if (!dataset) {
        return std::nullopt;
    }
    phase_.store(DatasetLoadPhase::Idle);
    progress_.store(0.0f);
    return dataset;
}

} // namespace chmv::data
