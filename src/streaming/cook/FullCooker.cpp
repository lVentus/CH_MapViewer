#include "streaming/cook/FullCooker.h"
#include "streaming/cook/ExternalFullCook.h"

#include "data/ch/CHLoader.h"
#include "data/ch/CHTypes.h"
#include "streaming/index/CHIndex.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace chmv::streaming::cook {
namespace {

constexpr std::uint32_t kManifestVersion = 4;
constexpr std::uint32_t kInvalidId = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kWriteBufferSize = 4u << 20u;

struct FileStamp {
    std::uint64_t size = 0;
    std::int64_t writeTime = 0;
};

struct CoordinateToken {
    double value = 0.0;
};

struct CompactFloatToken {
    float value = 0.0f;
};

FileStamp GetFileStamp(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        return {};
    }
    return {
        std::filesystem::file_size(path),
        static_cast<std::int64_t>(
            std::filesystem::last_write_time(path).time_since_epoch().count()),
    };
}

std::uint32_t Morton2D(std::uint32_t x, std::uint32_t y, std::uint32_t bits) {
    std::uint32_t result = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        result |= ((x >> bit) & 1u) << (2u * bit);
        result |= ((y >> bit) & 1u) << (2u * bit + 1u);
    }
    return result;
}

std::uint32_t SpatialMorton(double longitude, double latitude,
                            const analysis::DatasetLayoutAnalysisReport& report) {
    const auto normalize = [](double value, double minimum, double maximum) {
        if (!(maximum > minimum)) {
            return 0.0;
        }
        return std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
    };

    const auto gridSize = report.config.spatialGridSize;
    const auto toCell = [gridSize](double normalized) {
        const auto scaled = static_cast<std::uint64_t>(normalized * gridSize);
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(scaled, gridSize - 1u));
    };

    const auto cellX = toCell(normalize(longitude, report.minLongitude, report.maxLongitude));
    const auto cellY = toCell(normalize(latitude, report.minLatitude, report.maxLatitude));
    const auto gridBits = std::countr_zero(gridSize);
    return Morton2D(cellX, cellY, gridBits);
}

std::uint32_t PageMortonPrefix(const analysis::RuntimeGraphPageInfo& page,
                               std::uint32_t gridBits) {
    const auto prefix = Morton2D(page.tileX, page.tileY, page.spatialLevel);
    return prefix << (2u * (gridBits - page.spatialLevel));
}

struct CellBounds {
    std::uint32_t minX = 0;
    std::uint32_t minY = 0;
    std::uint32_t maxX = 0;
    std::uint32_t maxY = 0;
};

CellBounds UnpackCellBounds(std::uint64_t packed) {
    return {
        static_cast<std::uint32_t>(packed & 0xffffu),
        static_cast<std::uint32_t>((packed >> 16u) & 0xffffu),
        static_cast<std::uint32_t>((packed >> 32u) & 0xffffu),
        static_cast<std::uint32_t>((packed >> 48u) & 0xffffu),
    };
}

std::uint32_t BoundsMorton(std::uint64_t packed, std::uint32_t gridBits) {
    const auto bounds = UnpackCellBounds(packed);
    const auto x = (bounds.minX + bounds.maxX) / 2u;
    const auto y = (bounds.minY + bounds.maxY) / 2u;
    return Morton2D(x, y, gridBits);
}

void Report(const FullCooker::ProgressCallback& callback, FullCookStage stage,
            std::uint64_t current, std::uint64_t total) {
    if (callback) {
        callback({stage, current, total});
    }
}

class TextWriter {
public:
    explicit TextWriter(const std::filesystem::path& path)
        : buffer_(kWriteBufferSize) {
#ifdef _WIN32
        _wfopen_s(&file_, path.c_str(), L"wb");
#else
        file_ = std::fopen(path.c_str(), "wb");
#endif
        if (!file_) {
            throw std::runtime_error("could not create file: " + path.string());
        }
        std::setvbuf(file_, buffer_.data(), _IOFBF, buffer_.size());
    }

    ~TextWriter() {
        if (file_) {
            std::fclose(file_);
        }
    }

    TextWriter(const TextWriter&) = delete;
    TextWriter& operator=(const TextWriter&) = delete;

    template <typename... Values>
    void Line(const Values&... values) {
        line_.clear();
        bool first = true;
        (AppendField(values, first), ...);
        line_.push_back('\n');
        Write(line_);
    }

    void Raw(std::string_view text) {
        Write(text);
    }

    void Close() {
        if (!file_) {
            return;
        }
        if (std::fflush(file_) != 0 || std::fclose(file_) != 0) {
            file_ = nullptr;
            throw std::runtime_error("failed to finalize cooked text file");
        }
        file_ = nullptr;
    }

private:
    void BeginField(bool& first) {
        if (!first) {
            line_.push_back(' ');
        }
        first = false;
    }

    void AppendField(CoordinateToken token, bool& first) {
        BeginField(first);
        char local[96];
        const auto result = std::to_chars(local, local + sizeof(local), token.value,
                                          std::chars_format::fixed, 7);
        if (result.ec != std::errc{}) {
            throw std::runtime_error("failed to format cooked coordinate");
        }
        char* end = result.ptr;
        while (end > local && end[-1] == '0') {
            --end;
        }
        if (end > local && end[-1] == '.') {
            --end;
        }
        line_.append(local, end);
    }

    void AppendField(CompactFloatToken token, bool& first) {
        BeginField(first);
        char local[96];
        char* end = nullptr;
        std::errc error{};
        const auto rounded = std::round(token.value);
        if (std::isfinite(token.value) && token.value == rounded &&
            rounded >= static_cast<float>(std::numeric_limits<std::int32_t>::min()) &&
            rounded <= static_cast<float>(std::numeric_limits<std::int32_t>::max())) {
            const auto result = std::to_chars(local, local + sizeof(local),
                                              static_cast<std::int32_t>(rounded));
            end = result.ptr;
            error = result.ec;
        } else {
            const auto result = std::to_chars(
                local, local + sizeof(local), token.value, std::chars_format::general,
                std::numeric_limits<float>::max_digits10);
            end = result.ptr;
            error = result.ec;
        }
        if (error != std::errc{}) {
            throw std::runtime_error("failed to format cooked float");
        }
        line_.append(local, end);
    }

    template <typename T>
    void AppendField(T value, bool& first) {
        BeginField(first);
        char local[96];
        const auto result = std::to_chars(local, local + sizeof(local), value);
        if (result.ec != std::errc{}) {
            throw std::runtime_error("failed to format cooked text record");
        }
        line_.append(local, result.ptr);
    }

    void Write(std::string_view text) {
        if (std::fwrite(text.data(), 1, text.size(), file_) != text.size()) {
            throw std::runtime_error("failed to write cooked text file");
        }
    }

    std::FILE* file_ = nullptr;
    std::vector<char> buffer_;
    std::string line_;
};

std::string ReadPrefix(const std::filesystem::path& path, std::uint64_t byteCount) {
    if (byteCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("source prefix is too large");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not read source prefix: " + path.string());
    }
    std::string bytes(static_cast<std::size_t>(byteCount), '\0');
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("source prefix is truncated");
    }
    return bytes;
}

std::filesystem::path TemporaryPath(const std::filesystem::path& finalPath) {
    return std::filesystem::path(finalPath.string() + ".tmp");
}

void RemoveIfExists(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

void RequireCookDiskSpace(const std::filesystem::path& sourceGraphPath,
                          const std::filesystem::path& sourceRangesPath,
                          const std::filesystem::path& destinationDirectory) {
    std::error_code error;
    const auto graphBytes = std::filesystem::file_size(sourceGraphPath, error);
    if (error) {
        return;
    }
    const auto rangesBytes = std::filesystem::file_size(sourceRangesPath, error);
    if (error) {
        return;
    }
    const auto sourceBytes = graphBytes + rangesBytes;

    auto probe = destinationDirectory;
    while (!probe.empty() && !std::filesystem::exists(probe, error)) {
        probe = probe.parent_path();
    }
    if (probe.empty() || error) {
        return;
    }

    const auto space = std::filesystem::space(probe, error);
    if (error) {
        return;
    }

    constexpr std::uint64_t kMinimumMargin = 256ull << 20u;
    const auto percentageMargin = sourceBytes / 20u;
    const auto requiredBytes = sourceBytes + std::max(kMinimumMargin, percentageMargin);
    if (space.available < requiredBytes) {
        const auto mib = [](std::uint64_t bytes) { return bytes / (1024ull * 1024ull); };
        throw std::runtime_error(
            "not enough free disk space for FullCook: need about " +
            std::to_string(mib(requiredBytes)) + " MiB free, but only " +
            std::to_string(mib(space.available)) + " MiB is available");
    }
}

void CommitTemporaryFile(const std::filesystem::path& temporaryPath,
                         const std::filesystem::path& finalPath) {
    RemoveIfExists(finalPath);
    std::filesystem::rename(temporaryPath, finalPath);
}

void WriteManifest(const FullCookPaths& paths,
                   const analysis::DatasetLayoutAnalysisConfig& config,
                   std::uint64_t nodeCount, std::uint64_t edgeCount,
                   const std::filesystem::path& outputPath) {
    const auto sourceGraphStamp = GetFileStamp(paths.originalGraphPath);
    const auto sourceRangesStamp = GetFileStamp(paths.originalRangesPath);
    const auto cookedGraphStamp = GetFileStamp(paths.graphPath);
    const auto cookedRangesStamp = GetFileStamp(paths.rangesPath);
    std::ofstream output(outputPath, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not write full-cook manifest: " + outputPath.string());
    }
    output << "format_version=" << kManifestVersion << '\n';
    output << "source_graph=" << paths.originalGraphPath.filename().generic_string() << '|'
           << sourceGraphStamp.size << '|' << sourceGraphStamp.writeTime << '\n';
    output << "source_ranges=" << paths.originalRangesPath.filename().generic_string() << '|'
           << sourceRangesStamp.size << '|' << sourceRangesStamp.writeTime << '\n';
    output << "cooked_graph=" << paths.graphPath.filename().generic_string() << '|'
           << cookedGraphStamp.size << '|' << cookedGraphStamp.writeTime << '\n';
    output << "cooked_ranges=" << paths.rangesPath.filename().generic_string() << '|'
           << cookedRangesStamp.size << '|' << cookedRangesStamp.writeTime << '\n';
    output << "node_count=" << nodeCount << '\n';
    output << "edge_count=" << edgeCount << '\n';
    output << "spatial_grid_size=" << config.spatialGridSize << '\n';
    output << "lod_band_width=" << config.lodBandWidth << '\n';
    output << "source_block_node_count=" << config.sourceBlockNodeCount << '\n';
    output << "source_block_edge_count=" << config.sourceBlockEdgeCount << '\n';
    output << "virtual_page_edge_count=" << config.virtualPageEdgeCount << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize full-cook manifest");
    }
}

bool ParseStampedLine(const std::string& line, const std::string& key,
                      const std::filesystem::path& sourcePath) {
    const auto prefix = key + '=';
    if (!line.starts_with(prefix)) {
        return false;
    }
    const auto payload = line.substr(prefix.size());
    const auto first = payload.find('|');
    const auto second = first == std::string::npos ? std::string::npos : payload.find('|', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
        return false;
    }
    try {
        const auto size = static_cast<std::uint64_t>(
            std::stoull(payload.substr(first + 1, second - first - 1)));
        const auto time = static_cast<std::int64_t>(std::stoll(payload.substr(second + 1)));
        const auto stamp = GetFileStamp(sourcePath);
        return stamp.size == size && stamp.writeTime == time;
    } catch (...) {
        return false;
    }
}

bool ManifestMatches(const FullCookPaths& paths,
                     const analysis::DatasetLayoutAnalysisConfig& config) {
    if (!std::filesystem::is_regular_file(paths.manifestPath) ||
        !std::filesystem::is_regular_file(paths.graphPath) ||
        !std::filesystem::is_regular_file(paths.rangesPath) ||
        !std::filesystem::is_regular_file(paths.originalGraphPath) ||
        !std::filesystem::is_regular_file(paths.originalRangesPath) ||
        !std::filesystem::is_regular_file(paths.indexPath)) {
        return false;
    }

    std::ifstream input(paths.manifestPath);
    if (!input) {
        return false;
    }

    bool version = false;
    bool sourceGraph = false;
    bool sourceRanges = false;
    bool cookedGraph = false;
    bool cookedRanges = false;
    bool grid = false;
    bool band = false;
    bool nodeBlock = false;
    bool edgeBlock = false;
    bool pageEdges = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line == "format_version=" + std::to_string(kManifestVersion)) {
            version = true;
        } else if (line.starts_with("source_graph=")) {
            sourceGraph = ParseStampedLine(line, "source_graph", paths.originalGraphPath);
        } else if (line.starts_with("source_ranges=")) {
            sourceRanges = ParseStampedLine(line, "source_ranges", paths.originalRangesPath);
        } else if (line.starts_with("cooked_graph=")) {
            cookedGraph = ParseStampedLine(line, "cooked_graph", paths.graphPath);
        } else if (line.starts_with("cooked_ranges=")) {
            cookedRanges = ParseStampedLine(line, "cooked_ranges", paths.rangesPath);
        } else if (line == "spatial_grid_size=" + std::to_string(config.spatialGridSize)) {
            grid = true;
        } else if (line == "lod_band_width=" + std::to_string(config.lodBandWidth)) {
            band = true;
        } else if (line == "source_block_node_count=" + std::to_string(config.sourceBlockNodeCount)) {
            nodeBlock = true;
        } else if (line == "source_block_edge_count=" + std::to_string(config.sourceBlockEdgeCount)) {
            edgeBlock = true;
        } else if (line == "virtual_page_edge_count=" + std::to_string(config.virtualPageEdgeCount)) {
            pageEdges = true;
        }
    }

    return version && sourceGraph && sourceRanges && cookedGraph && cookedRanges &&
           grid && band && nodeBlock && edgeBlock && pageEdges &&
           index::CHIndex::IsValid(paths.indexPath, paths.graphPath, paths.rangesPath, config);
}

std::pair<std::filesystem::path, std::filesystem::path> AuthoritativeSourcePaths(
    const FullCookPaths& paths) {
    const bool graphBackup = std::filesystem::is_regular_file(paths.originalGraphPath);
    const bool rangesBackup = std::filesystem::is_regular_file(paths.originalRangesPath);
    if (graphBackup != rangesBackup) {
        throw std::runtime_error(
            "FullCook source backup is incomplete; expected both .sch.org and .sch.ranges.org");
    }
    if (graphBackup) {
        return {paths.originalGraphPath, paths.originalRangesPath};
    }
    return {paths.graphPath, paths.rangesPath};
}

void BackupOriginalSources(const FullCookPaths& paths) {
    if (std::filesystem::is_regular_file(paths.originalGraphPath) &&
        std::filesystem::is_regular_file(paths.originalRangesPath)) {
        return;
    }
    if (std::filesystem::exists(paths.originalGraphPath) ||
        std::filesystem::exists(paths.originalRangesPath)) {
        throw std::runtime_error("cannot create FullCook backup because an incomplete .org pair exists");
    }

    std::filesystem::rename(paths.graphPath, paths.originalGraphPath);
    try {
        std::filesystem::rename(paths.rangesPath, paths.originalRangesPath);
    } catch (...) {
        std::filesystem::rename(paths.originalGraphPath, paths.graphPath);
        throw;
    }
}

std::vector<std::uint32_t> BuildNodeOrder(
    const data::CHGraph& graph, const analysis::DatasetLayoutAnalysisReport& report,
    const FullCooker::ProgressCallback& progressCallback) {
    const auto count = graph.Nodes().size();
    std::vector<std::uint32_t> morton(count, 0);
    std::vector<std::uint32_t> newToOld(count, 0);

    Report(progressCallback, FullCookStage::ReorderNodes, 0, count);
    for (std::size_t oldId = 0; oldId < count; ++oldId) {
        const auto& node = graph.Nodes()[oldId];
        morton[oldId] = SpatialMorton(node.longitude, node.latitude, report);
        newToOld[oldId] = static_cast<std::uint32_t>(oldId);
    }

    std::sort(newToOld.begin(), newToOld.end(), [&](std::uint32_t left, std::uint32_t right) {
        if (morton[left] != morton[right]) {
            return morton[left] < morton[right];
        }
        const auto leftLevel = graph.Nodes()[left].level;
        const auto rightLevel = graph.Nodes()[right].level;
        if (leftLevel != rightLevel) {
            return leftLevel > rightLevel;
        }
        return left < right;
    });
    Report(progressCallback, FullCookStage::ReorderNodes, count, count);
    return newToOld;
}

std::vector<std::uint32_t> BuildEdgeOrder(
    const data::CHGraph& graph, const analysis::DatasetLayoutAnalysisReport& report,
    std::vector<std::uint32_t>& oldToNew,
    const FullCooker::ProgressCallback& progressCallback) {
    const auto edgeCount = graph.Edges().size();
    oldToNew.assign(edgeCount, kInvalidId);
    if (report.edgeSpatialBounds.size() != edgeCount) {
        throw std::runtime_error("layout report is missing full-geometry spatial bounds");
    }

    const auto gridBits = static_cast<std::uint32_t>(std::countr_zero(report.config.spatialGridSize));
    std::vector<std::uint32_t> pageOrder(report.runtimeGraphPages.size());
    std::iota(pageOrder.begin(), pageOrder.end(), 0u);
    std::sort(pageOrder.begin(), pageOrder.end(), [&](std::uint32_t leftId, std::uint32_t rightId) {
        const auto& left = report.runtimeGraphPages[leftId];
        const auto& right = report.runtimeGraphPages[rightId];
        if (left.levelMax != right.levelMax) {
            return left.levelMax > right.levelMax;
        }
        if (left.levelMin != right.levelMin) {
            return left.levelMin > right.levelMin;
        }
        const auto leftMorton = PageMortonPrefix(left, gridBits);
        const auto rightMorton = PageMortonPrefix(right, gridBits);
        if (leftMorton != rightMorton) {
            return leftMorton < rightMorton;
        }
        if (left.spatialLevel != right.spatialLevel) {
            return left.spatialLevel < right.spatialLevel;
        }
        return left.subPage < right.subPage;
    });

    std::vector<std::uint32_t> newToOld;
    newToOld.reserve(edgeCount);
    std::vector<std::uint32_t> pageEdges;
    Report(progressCallback, FullCookStage::ReorderEdges, 0, edgeCount);

    // Drawable roots are already partitioned into unique adaptive LOD/spatial pages.
    // Keep every page physically contiguous and preserve exact birth ordering inside it.
    for (const auto pageId : pageOrder) {
        const auto& page = report.runtimeGraphPages[pageId];
        const auto begin = report.runtimeGraphPageEdgeIds.begin() + page.edgeIdOffset;
        const auto end = begin + page.edgeCount;
        pageEdges.assign(begin, end);
        std::sort(pageEdges.begin(), pageEdges.end(), [&](std::uint32_t left, std::uint32_t right) {
            const auto& leftRange = graph.Ranges()[left];
            const auto& rightRange = graph.Ranges()[right];
            if (leftRange.birthLevel != rightRange.birthLevel) {
                return leftRange.birthLevel > rightRange.birthLevel;
            }
            if (leftRange.deathLevel != rightRange.deathLevel) {
                return leftRange.deathLevel > rightRange.deathLevel;
            }
            const auto leftMorton = BoundsMorton(report.edgeSpatialBounds[left], gridBits);
            const auto rightMorton = BoundsMorton(report.edgeSpatialBounds[right], gridBits);
            if (leftMorton != rightMorton) {
                return leftMorton < rightMorton;
            }
            return left < right;
        });

        for (const auto oldId : pageEdges) {
            if (oldId >= edgeCount || oldToNew[oldId] != kInvalidId) {
                throw std::runtime_error("invalid or duplicate runtime graph-page membership");
            }
            oldToNew[oldId] = static_cast<std::uint32_t>(newToOld.size());
            newToOld.push_back(oldId);
        }
        Report(progressCallback, FullCookStage::ReorderEdges, newToOld.size(), edgeCount);
    }

    const auto drawableCount = newToOld.size();
    if (drawableCount != report.drawableEdgeCount) {
        throw std::runtime_error("runtime graph pages do not cover every drawable edge exactly once");
    }

    // Refinement-only edges have no lifetime level. Give them the same unique
    // hierarchical spatial ownership rule: subdivide only overloaded spatial nodes,
    // keep boundary-crossing geometry at the smallest enclosing ancestor, and chunk
    // only when that owner's payload still exceeds the page target.
    std::vector<std::uint32_t> refinementEdges;
    refinementEdges.reserve(edgeCount - drawableCount);
    for (std::uint32_t oldId = 0; oldId < edgeCount; ++oldId) {
        if (oldToNew[oldId] == kInvalidId) {
            refinementEdges.push_back(oldId);
        }
    }

    const auto appendChunk = [&](std::vector<std::uint32_t>& ids) {
        if (ids.empty()) {
            return;
        }
        std::sort(ids.begin(), ids.end(), [&](std::uint32_t left, std::uint32_t right) {
            const auto leftMorton = BoundsMorton(report.edgeSpatialBounds[left], gridBits);
            const auto rightMorton = BoundsMorton(report.edgeSpatialBounds[right], gridBits);
            if (leftMorton != rightMorton) {
                return leftMorton < rightMorton;
            }
            return left < right;
        });
        for (const auto oldId : ids) {
            if (oldToNew[oldId] != kInvalidId) {
                throw std::runtime_error("refinement edge assigned more than once during FullCook");
            }
            oldToNew[oldId] = static_cast<std::uint32_t>(newToOld.size());
            newToOld.push_back(oldId);
        }
    };

    std::function<void(std::vector<std::uint32_t>&&, std::uint32_t)> partitionRefinement;
    partitionRefinement = [&](std::vector<std::uint32_t>&& ids, std::uint32_t spatialLevel) {
        if (ids.empty()) {
            return;
        }
        if (ids.size() <= report.config.virtualPageEdgeCount || spatialLevel >= gridBits) {
            for (std::size_t begin = 0; begin < ids.size();
                 begin += report.config.virtualPageEdgeCount) {
                const auto end = std::min<std::size_t>(
                    begin + report.config.virtualPageEdgeCount, ids.size());
                std::vector<std::uint32_t> chunk(
                    ids.begin() + static_cast<std::ptrdiff_t>(begin),
                    ids.begin() + static_cast<std::ptrdiff_t>(end));
                appendChunk(chunk);
            }
            return;
        }

        std::vector<std::uint32_t> stay;
        std::array<std::vector<std::uint32_t>, 4> children;
        stay.reserve(ids.size() / 8u + 1u);
        const auto childLevel = spatialLevel + 1u;
        const auto shift = gridBits - childLevel;
        for (const auto edgeId : ids) {
            const auto bounds = UnpackCellBounds(report.edgeSpatialBounds[edgeId]);
            const auto minTileX = bounds.minX >> shift;
            const auto maxTileX = bounds.maxX >> shift;
            const auto minTileY = bounds.minY >> shift;
            const auto maxTileY = bounds.maxY >> shift;
            if (minTileX == maxTileX && minTileY == maxTileY) {
                const auto childX = minTileX & 1u;
                const auto childY = minTileY & 1u;
                children[childX | (childY << 1u)].push_back(edgeId);
            } else {
                stay.push_back(edgeId);
            }
        }

        for (std::size_t begin = 0; begin < stay.size();
             begin += report.config.virtualPageEdgeCount) {
            const auto end = std::min<std::size_t>(
                begin + report.config.virtualPageEdgeCount, stay.size());
            std::vector<std::uint32_t> chunk(
                stay.begin() + static_cast<std::ptrdiff_t>(begin),
                stay.begin() + static_cast<std::ptrdiff_t>(end));
            appendChunk(chunk);
        }
        for (auto& child : children) {
            partitionRefinement(std::move(child), childLevel);
        }
    };

    partitionRefinement(std::move(refinementEdges), 0);
    if (newToOld.size() != edgeCount) {
        throw std::runtime_error("full-cook edge partition does not cover the full graph");
    }
    Report(progressCallback, FullCookStage::ReorderEdges, edgeCount, edgeCount);
    return newToOld;
}

void WriteCookedGraph(const data::CHGraph& graph,
                      const std::filesystem::path& sourceGraphPath,
                      std::uint64_t sourceNodeSectionOffset,
                      const std::vector<std::uint32_t>& newToOldNode,
                      const std::vector<std::uint32_t>& oldToNewNode,
                      const std::vector<std::uint32_t>& newToOldEdge,
                      const std::vector<std::uint32_t>& oldToNewEdge,
                      const std::filesystem::path& outputPath,
                      const FullCooker::ProgressCallback& progressCallback) {
    TextWriter writer(outputPath);
    writer.Raw(ReadPrefix(sourceGraphPath, sourceNodeSectionOffset));

    for (std::size_t newId = 0; newId < newToOldNode.size(); ++newId) {
        const auto& node = graph.Nodes()[newToOldNode[newId]];
        writer.Line(static_cast<std::uint32_t>(newId), node.osmId,
                    CoordinateToken{node.latitude}, CoordinateToken{node.longitude},
                    CompactFloatToken{node.elevation}, node.level);
        if ((newId + 1) % (1u << 18u) == 0 || newId + 1 == newToOldNode.size()) {
            Report(progressCallback, FullCookStage::WriteGraph, newId + 1,
                   newToOldNode.size() + newToOldEdge.size());
        }
    }

    for (std::size_t newId = 0; newId < newToOldEdge.size(); ++newId) {
        const auto& edge = graph.Edges()[newToOldEdge[newId]];
        if (edge.source >= oldToNewNode.size() || edge.target >= oldToNewNode.size()) {
            throw std::runtime_error("edge endpoint is out of range during full cook");
        }
        const auto childA = edge.childA == data::InvalidEdgeId
                                ? std::int64_t{-1}
                                : static_cast<std::int64_t>(oldToNewEdge[edge.childA]);
        const auto childB = edge.childB == data::InvalidEdgeId
                                ? std::int64_t{-1}
                                : static_cast<std::int64_t>(oldToNewEdge[edge.childB]);
        writer.Line(oldToNewNode[edge.source], oldToNewNode[edge.target],
                    CompactFloatToken{edge.weight}, edge.type, edge.maxSpeed, childA, childB);
        if ((newId + 1) % (1u << 18u) == 0 || newId + 1 == newToOldEdge.size()) {
            Report(progressCallback, FullCookStage::WriteGraph,
                   newToOldNode.size() + newId + 1,
                   newToOldNode.size() + newToOldEdge.size());
        }
    }
    writer.Close();
}

void WriteCookedRanges(const data::CHGraph& graph,
                       const std::vector<std::uint32_t>& newToOldEdge,
                       const std::filesystem::path& outputPath,
                       const FullCooker::ProgressCallback& progressCallback) {
    TextWriter writer(outputPath);
    for (std::size_t newId = 0; newId < newToOldEdge.size(); ++newId) {
        const auto& range = graph.Ranges()[newToOldEdge[newId]];
        writer.Line(static_cast<std::uint32_t>(newId), range.birthLevel, range.deathLevel);
        if ((newId + 1) % (1u << 18u) == 0 || newId + 1 == newToOldEdge.size()) {
            Report(progressCallback, FullCookStage::WriteRanges, newId + 1,
                   newToOldEdge.size());
        }
    }
    writer.Close();
}

std::uint64_t QueryAvailablePhysicalMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        return status.ullAvailPhys;
    }
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t valueKiB = 0;
    std::string unit;
    while (meminfo >> key >> valueKiB >> unit) {
        if (key == "MemAvailable:") {
            return valueKiB * 1024ull;
        }
    }
#endif
    return 0;
}

bool CanRunDetailedAnalyzer(std::uint64_t nodeCount, std::uint64_t edgeCount,
                            std::uint64_t drawableEdgeCount) {
    // Conservative approximation of the legacy analyzer's simultaneously resident
    // vectors. Keep at least 1 GiB of physical headroom beyond the estimate.
    const auto estimate = nodeCount * 24ull + edgeCount * 64ull +
                          drawableEdgeCount * 20ull + (256ull << 20u);
    const auto available = QueryAvailablePhysicalMemory();
    return available != 0 && available > estimate + (1ull << 30u);
}

std::uint64_t ResolveCookMemoryBudget(std::uint64_t requested) {
    if (requested != 0) {
        return std::max<std::uint64_t>(requested, 4ull << 20u);
    }
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        return std::clamp<std::uint64_t>(status.ullAvailPhys / 8u, 256ull << 20u, 1024ull << 20u);
    }
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t valueKiB = 0;
    std::string unit;
    while (meminfo >> key >> valueKiB >> unit) {
        if (key == "MemAvailable:") {
            return std::clamp<std::uint64_t>((valueKiB * 1024ull) / 8u,
                                             256ull << 20u, 1024ull << 20u);
        }
    }
#endif
    return 512ull << 20u;
}

bool ShouldUseExternalCook(const std::filesystem::path& graphPath,
                           const std::filesystem::path& rangesPath,
                           std::uint64_t memoryBudgetBytes,
                           bool forceExternal) {
    if (forceExternal) {
        return true;
    }
    std::error_code error;
    const auto graphBytes = std::filesystem::file_size(graphPath, error);
    if (error) {
        return true;
    }
    const auto rangeBytes = std::filesystem::file_size(rangesPath, error);
    if (error) {
        return true;
    }
    // CHGraph plus analysis/remap vectors are substantially larger than the text source.
    // Keep the in-memory path only when the source is comfortably below the explicit budget.
    return graphBytes + rangeBytes > memoryBudgetBytes / 2u;
}

} // namespace

FullCookPaths FullCooker::DefaultPaths(const std::filesystem::path& sourceGraphPath,
                                       const std::filesystem::path& sourceRangesPath) {
    FullCookPaths paths;
    paths.directory = sourceGraphPath.parent_path();
    paths.graphPath = sourceGraphPath;
    paths.rangesPath = sourceRangesPath;
    paths.indexPath = index::CHIndex::DefaultPath(paths.graphPath);
    const auto baseName = sourceGraphPath.extension() == ".sch"
                              ? sourceGraphPath.stem().string()
                              : sourceGraphPath.filename().string();
    paths.manifestPath = paths.directory / (baseName + ".fullcook_manifest");
    paths.originalGraphPath = std::filesystem::path(sourceGraphPath.string() + ".org");
    paths.originalRangesPath = std::filesystem::path(sourceRangesPath.string() + ".org");
    return paths;
}

bool FullCooker::HasValidCook(const std::filesystem::path& sourceGraphPath,
                              const std::filesystem::path& sourceRangesPath,
                              const analysis::DatasetLayoutAnalysisConfig& config) {
    try {
        return ManifestMatches(DefaultPaths(sourceGraphPath, sourceRangesPath), config);
    } catch (...) {
        return false;
    }
}

FullCookResult FullCooker::Cook(const std::filesystem::path& sourceGraphPath,
                                const std::filesystem::path& sourceRangesPath,
                                const analysis::DatasetLayoutAnalysisConfig& config,
                                bool force, ProgressCallback progressCallback,
                                const FullCookOptions& options) {
    const auto paths = DefaultPaths(sourceGraphPath, sourceRangesPath);
    if (!force && ManifestMatches(paths, config)) {
        const auto cookedIndex = index::CHIndex::Load(paths.indexPath);
        return {paths, cookedIndex.nodeCount, cookedIndex.edgeCount, true};
    }

    const auto [authoritativeGraphPath, authoritativeRangesPath] =
        AuthoritativeSourcePaths(paths);
    if (!std::filesystem::is_regular_file(authoritativeGraphPath) ||
        !std::filesystem::is_regular_file(authoritativeRangesPath)) {
        throw std::runtime_error("full cook requires an existing .sch/.ranges source pair");
    }

    RequireCookDiskSpace(authoritativeGraphPath, authoritativeRangesPath, paths.directory);
    const auto graphTemporary = TemporaryPath(paths.graphPath);
    const auto rangesTemporary = TemporaryPath(paths.rangesPath);
    const auto indexTemporary = TemporaryPath(paths.indexPath);
    const auto manifestTemporary = TemporaryPath(paths.manifestPath);
    RemoveIfExists(graphTemporary);
    RemoveIfExists(rangesTemporary);
    RemoveIfExists(indexTemporary);
    RemoveIfExists(manifestTemporary);

    const auto memoryBudgetBytes = ResolveCookMemoryBudget(options.memoryBudgetBytes);
    const auto useExternal = ShouldUseExternalCook(authoritativeGraphPath, authoritativeRangesPath,
                                                    memoryBudgetBytes, options.forceExternalMemory);

    try {
        if (useExternal) {
            const auto tempBase = options.tempDirectory.empty() ? paths.directory : options.tempDirectory;
            const auto tempRoot = tempBase / (paths.graphPath.filename().string() + ".fullcook_tmp");
            std::error_code tempError;
            std::filesystem::remove_all(tempRoot, tempError);
            const auto external = detail::RunExternalFullCook(
                authoritativeGraphPath, authoritativeRangesPath, graphTemporary, rangesTemporary,
                indexTemporary, config, memoryBudgetBytes, tempRoot, progressCallback);

            BackupOriginalSources(paths);
            CommitTemporaryFile(graphTemporary, paths.graphPath);
            CommitTemporaryFile(rangesTemporary, paths.rangesPath);
            CommitTemporaryFile(indexTemporary, paths.indexPath);
            if (!index::CHIndex::IsValid(paths.indexPath, paths.graphPath, paths.rangesPath, config)) {
                throw std::runtime_error("committed external FullCook index does not match the cooked dataset");
            }

            WriteManifest(paths, config, external.nodeCount, external.edgeCount, manifestTemporary);
            CommitTemporaryFile(manifestTemporary, paths.manifestPath);

            // Detailed Phase-0 CSVs are intentionally optional: on data sets that need
            // external cooking, forcing the legacy in-memory analyzer would defeat the
            // memory-bounded cook. Small data can still request the report after commit.
            if (options.writeDetailedAnalysis &&
                CanRunDetailedAnalyzer(external.nodeCount, external.edgeCount,
                                       external.drawableEdgeCount)) {
                auto cookedReport = analysis::DatasetLayoutAnalyzer::Analyze(
                    paths.graphPath, paths.rangesPath, config);
                analysis::DatasetLayoutAnalyzer::WriteReport(
                    cookedReport, analysis::DatasetLayoutAnalyzer::DefaultReportDirectory(paths.graphPath));
            }
            return {paths, external.nodeCount, external.edgeCount, false};
        }

        Report(progressCallback, FullCookStage::AnalyzeSource, 0, 1);
        auto sourceReport = analysis::DatasetLayoutAnalyzer::Analyze(
            authoritativeGraphPath, authoritativeRangesPath, config,
            [&](const analysis::DatasetLayoutAnalysisProgress& progress) {
                Report(progressCallback, FullCookStage::AnalyzeSource,
                       progress.current, progress.total);
            });
        Report(progressCallback, FullCookStage::AnalyzeSource, 1, 1);

        constexpr std::uint64_t kLoadProgressUnits = 1'000'000;
        Report(progressCallback, FullCookStage::LoadSource, 0, kLoadProgressUnits);
        auto graph = data::CHLoader::Load(
            authoritativeGraphPath, authoritativeRangesPath,
            [&](const data::CHLoadProgress& progress) {
                const auto current = static_cast<std::uint64_t>(
                    std::clamp(progress.overall, 0.0f, 1.0f) *
                    static_cast<float>(kLoadProgressUnits));
                Report(progressCallback, FullCookStage::LoadSource,
                       current, kLoadProgressUnits);
            });
        Report(progressCallback, FullCookStage::LoadSource,
               kLoadProgressUnits, kLoadProgressUnits);

        auto newToOldNode = BuildNodeOrder(graph, sourceReport, progressCallback);
        std::vector<std::uint32_t> oldToNewNode(newToOldNode.size(), kInvalidId);
        for (std::size_t newId = 0; newId < newToOldNode.size(); ++newId) {
            oldToNewNode[newToOldNode[newId]] = static_cast<std::uint32_t>(newId);
        }

        std::vector<std::uint32_t> oldToNewEdge;
        auto newToOldEdge = BuildEdgeOrder(graph, sourceReport, oldToNewEdge, progressCallback);

        Report(progressCallback, FullCookStage::WriteGraph, 0,
               newToOldNode.size() + newToOldEdge.size());
        const auto sourceNodeSectionOffset = sourceReport.nodeSourceBlocks.empty()
                                                 ? std::uint64_t{0}
                                                 : sourceReport.nodeSourceBlocks.front().byteOffset;
        WriteCookedGraph(graph, authoritativeGraphPath, sourceNodeSectionOffset, newToOldNode,
                         oldToNewNode, newToOldEdge, oldToNewEdge, graphTemporary,
                         progressCallback);
        Report(progressCallback, FullCookStage::WriteRanges, 0, newToOldEdge.size());
        WriteCookedRanges(graph, newToOldEdge, rangesTemporary, progressCallback);

        const auto nodeCount = static_cast<std::uint64_t>(graph.Nodes().size());
        const auto edgeCount = static_cast<std::uint64_t>(graph.Edges().size());
        graph = {};
        newToOldNode.clear();
        oldToNewNode.clear();
        newToOldEdge.clear();
        oldToNewEdge.clear();
        sourceReport = {};

        Report(progressCallback, FullCookStage::AnalyzeCooked, 0, 1);
        auto cookedReport = analysis::DatasetLayoutAnalyzer::Analyze(
            graphTemporary, rangesTemporary, config,
            [&](const analysis::DatasetLayoutAnalysisProgress& progress) {
                Report(progressCallback, FullCookStage::AnalyzeCooked,
                       progress.current, progress.total);
            });
        if (cookedReport.nodeCount != nodeCount || cookedReport.edgeCount != edgeCount ||
            cookedReport.invalidRangeCount != 0 || cookedReport.missingRangeRecordCount != 0 ||
            cookedReport.duplicateRangeRecordCount != 0) {
            throw std::runtime_error("full-cooked dataset failed structural validation");
        }

        Report(progressCallback, FullCookStage::WriteIndex, 0, 1);
        index::CHIndex::Write(cookedReport, indexTemporary);

        // Commit only after the complete cooked pair and its index have been validated.
        // The original source remains beside it under the .org suffix.
        BackupOriginalSources(paths);
        CommitTemporaryFile(graphTemporary, paths.graphPath);
        CommitTemporaryFile(rangesTemporary, paths.rangesPath);
        CommitTemporaryFile(indexTemporary, paths.indexPath);
        if (!index::CHIndex::IsValid(paths.indexPath, paths.graphPath, paths.rangesPath, config)) {
            throw std::runtime_error("committed FullCook index does not match the cooked dataset");
        }
        Report(progressCallback, FullCookStage::WriteIndex, 1, 1);

        WriteManifest(paths, config, nodeCount, edgeCount, manifestTemporary);
        CommitTemporaryFile(manifestTemporary, paths.manifestPath);

        cookedReport.graphPath = paths.graphPath;
        cookedReport.rangesPath = paths.rangesPath;
        analysis::DatasetLayoutAnalyzer::WriteReport(
            cookedReport, analysis::DatasetLayoutAnalyzer::DefaultReportDirectory(paths.graphPath));

        return {paths, nodeCount, edgeCount, false};
    } catch (...) {
        RemoveIfExists(graphTemporary);
        RemoveIfExists(rangesTemporary);
        RemoveIfExists(indexTemporary);
        RemoveIfExists(manifestTemporary);
        throw;
    }
}

} // namespace chmv::streaming::cook
