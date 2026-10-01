#include "streaming/cook/ExternalFullCook.h"

#include "data/ch/CHTypes.h"
#include "streaming/analysis/TextSourceScanner.h"
#include "streaming/index/CHIndex.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace chmv::streaming::cook::detail {
namespace {

using analysis::detail::TextSourceScanner;

constexpr std::uint32_t kInvalidId = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kProgressInterval = 1u << 18u;
constexpr std::size_t kWriteBufferSize = 4u << 20u;

struct DiskNode {
    std::uint64_t osmId = 0;
    double latitude = 0.0;
    double longitude = 0.0;
    float elevation = 0.0f;
    std::uint32_t level = 0;
};

struct DiskRange {
    std::int32_t birth = -1;
    std::int32_t death = -1;
};

struct DiskEdge {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    float weight = 0.0f;
    std::int32_t type = 0;
    std::int32_t maxSpeed = 0;
    std::uint32_t childA = data::InvalidEdgeId;
    std::uint32_t childB = data::InvalidEdgeId;
    std::int32_t birth = -1;
    std::int32_t death = -1;
    std::uint64_t bounds = 0;
};

struct NodeSortRecord {
    std::uint64_t morton = 0;
    std::uint32_t level = 0;
    std::uint32_t oldId = 0;
};

struct PartitionRecord {
    std::uint32_t oldId = 0;
    std::uint32_t reserved = 0;
    std::uint64_t bounds = 0;
};

struct RuntimeLodGroup {
    std::uint32_t groupIndex = 0;
    std::uint32_t levelMin = 0;
    std::uint32_t levelMax = 0;
    std::uint64_t edgeCount = 0;
};

struct BuiltPage {
    index::CHIndexGraphPage page;
    std::uint64_t firstNewEdgeId = 0;
};

struct CellBounds {
    std::uint32_t minX = 0;
    std::uint32_t minY = 0;
    std::uint32_t maxX = 0;
    std::uint32_t maxY = 0;
};

void Report(const FullCooker::ProgressCallback& callback, FullCookStage stage,
            std::uint64_t current, std::uint64_t total) {
    if (callback) {
        callback({stage, current, total});
    }
}

std::uint32_t DecodeChild(std::int64_t value) {
    if (value < 0) {
        return data::InvalidEdgeId;
    }
    if (value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("edge child id does not fit in uint32");
    }
    return static_cast<std::uint32_t>(value);
}

std::uint64_t PackCellBounds(const CellBounds& bounds) {
    return static_cast<std::uint64_t>(bounds.minX) |
           (static_cast<std::uint64_t>(bounds.minY) << 16u) |
           (static_cast<std::uint64_t>(bounds.maxX) << 32u) |
           (static_cast<std::uint64_t>(bounds.maxY) << 48u);
}

CellBounds UnpackCellBounds(std::uint64_t packed) {
    return {
        static_cast<std::uint32_t>(packed & 0xffffu),
        static_cast<std::uint32_t>((packed >> 16u) & 0xffffu),
        static_cast<std::uint32_t>((packed >> 32u) & 0xffffu),
        static_cast<std::uint32_t>((packed >> 48u) & 0xffffu),
    };
}

CellBounds UnionBounds(const CellBounds& a, const CellBounds& b) {
    return {
        std::min(a.minX, b.minX),
        std::min(a.minY, b.minY),
        std::max(a.maxX, b.maxX),
        std::max(a.maxY, b.maxY),
    };
}

std::pair<std::uint32_t, std::uint32_t> SpatialCell(
    double longitude, double latitude, double minLongitude, double maxLongitude,
    double minLatitude, double maxLatitude, std::uint32_t gridSize) {
    const auto normalize = [](double value, double minimum, double maximum) {
        if (!(maximum > minimum)) {
            return 0.0;
        }
        return std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
    };
    const auto toCell = [gridSize](double normalized) {
        const auto scaled = static_cast<std::uint64_t>(normalized * gridSize);
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(scaled, static_cast<std::uint64_t>(gridSize) - 1u));
    };
    return {toCell(normalize(longitude, minLongitude, maxLongitude)),
            toCell(normalize(latitude, minLatitude, maxLatitude))};
}

std::uint32_t Morton2D(std::uint32_t x, std::uint32_t y, std::uint32_t bits) {
    std::uint32_t result = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        result |= ((x >> bit) & 1u) << (2u * bit);
        result |= ((y >> bit) & 1u) << (2u * bit + 1u);
    }
    return result;
}

std::uint32_t BoundsMorton(std::uint64_t packed, std::uint32_t gridBits) {
    const auto bounds = UnpackCellBounds(packed);
    return Morton2D((bounds.minX + bounds.maxX) / 2u,
                    (bounds.minY + bounds.maxY) / 2u, gridBits);
}

std::uint32_t NodeMorton(const DiskNode& node, double minLongitude, double maxLongitude,
                         double minLatitude, double maxLatitude, std::uint32_t gridSize) {
    const auto [x, y] = SpatialCell(node.longitude, node.latitude, minLongitude, maxLongitude,
                                    minLatitude, maxLatitude, gridSize);
    return Morton2D(x, y, std::countr_zero(gridSize));
}

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

    void Raw(std::string_view text) { Write(text); }
    [[nodiscard]] std::uint64_t Tell() const { return bytesWritten_; }

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
    struct CoordinateToken { double value = 0.0; };
    struct CompactFloatToken { float value = 0.0f; };

public:
    static CoordinateToken Coordinate(double value) { return {value}; }
    static CompactFloatToken CompactFloat(float value) { return {value}; }

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
        bytesWritten_ += text.size();
    }

    std::FILE* file_ = nullptr;
    std::vector<char> buffer_;
    std::string line_;
    std::uint64_t bytesWritten_ = 0;
};

template <typename T>
class MappedArray {
public:
    MappedArray() = default;
    ~MappedArray() { Close(); }
    MappedArray(const MappedArray&) = delete;
    MappedArray& operator=(const MappedArray&) = delete;

    void Create(const std::filesystem::path& path, std::uint64_t count) {
        Close();
        path_ = path;
        count_ = count;
        if (count_ == 0) {
            return;
        }
        if (count_ > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
            throw std::runtime_error("mapped temporary array is too large");
        }
        const auto bytes = count_ * sizeof(T);
#ifdef _WIN32
        file_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("could not create mapped temporary file");
        }
        LARGE_INTEGER size{};
        size.QuadPart = static_cast<LONGLONG>(bytes);
        if (!SetFilePointerEx(file_, size, nullptr, FILE_BEGIN) || !SetEndOfFile(file_)) {
            Close();
            throw std::runtime_error("could not size mapped temporary file");
        }
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READWRITE,
                                      static_cast<DWORD>(bytes >> 32u),
                                      static_cast<DWORD>(bytes & 0xffffffffu), nullptr);
        if (!mapping_) {
            Close();
            throw std::runtime_error("could not create temporary file mapping");
        }
        data_ = static_cast<T*>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0,
                                              static_cast<SIZE_T>(bytes)));
        if (!data_) {
            Close();
            throw std::runtime_error("could not map temporary file");
        }
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd_ < 0 || ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
            Close();
            throw std::runtime_error("could not create mapped temporary file");
        }
        void* view = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd_, 0);
        if (view == MAP_FAILED) {
            data_ = nullptr;
            Close();
            throw std::runtime_error("could not map temporary file");
        }
        data_ = static_cast<T*>(view);
#endif
    }

    [[nodiscard]] T& operator[](std::uint64_t index) {
        if (index >= count_) {
            throw std::out_of_range("mapped temporary array index out of range");
        }
        return data_[index];
    }
    [[nodiscard]] const T& operator[](std::uint64_t index) const {
        if (index >= count_) {
            throw std::out_of_range("mapped temporary array index out of range");
        }
        return data_[index];
    }
    [[nodiscard]] std::uint64_t size() const { return count_; }

    void Flush() {
        if (!data_ || count_ == 0) {
            return;
        }
        const auto bytes = count_ * sizeof(T);
#ifdef _WIN32
        FlushViewOfFile(data_, static_cast<SIZE_T>(bytes));
#else
        msync(data_, static_cast<std::size_t>(bytes), MS_SYNC);
#endif
    }

private:
    void Close() {
        if (data_) {
#ifdef _WIN32
            UnmapViewOfFile(data_);
#else
            munmap(data_, static_cast<std::size_t>(count_ * sizeof(T)));
#endif
            data_ = nullptr;
        }
#ifdef _WIN32
        if (mapping_) {
            CloseHandle(mapping_);
            mapping_ = nullptr;
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        count_ = 0;
    }

    std::filesystem::path path_;
    std::uint64_t count_ = 0;
    T* data_ = nullptr;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

template <typename T>
void WriteBinaryRecord(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("failed to write FullCook temporary record");
    }
}

template <typename T>
bool ReadBinaryRecord(std::istream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (input.gcount() == 0) {
        return false;
    }
    if (input.gcount() != static_cast<std::streamsize>(sizeof(T))) {
        throw std::runtime_error("truncated FullCook temporary file");
    }
    return true;
}

std::uint64_t BinaryRecordCount(const std::filesystem::path& path, std::size_t recordSize) {
    const auto bytes = std::filesystem::file_size(path);
    if (recordSize == 0 || bytes % recordSize != 0) {
        throw std::runtime_error("invalid FullCook temporary file size");
    }
    return bytes / recordSize;
}

std::filesystem::path TempPath(const std::filesystem::path& directory,
                               std::string_view stem, std::uint64_t id = 0) {
    return directory / (std::string(stem) + "_" + std::to_string(id) + ".bin");
}

void RemoveFile(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

class TempDirectoryGuard {
public:
    explicit TempDirectoryGuard(std::filesystem::path path) : path_(std::move(path)) {}
    ~TempDirectoryGuard() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    TempDirectoryGuard(const TempDirectoryGuard&) = delete;
    TempDirectoryGuard& operator=(const TempDirectoryGuard&) = delete;
private:
    std::filesystem::path path_;
};

std::vector<RuntimeLodGroup> BuildRuntimeLodGroups(
    const std::vector<std::uint64_t>& birthCounts, std::uint32_t maxLevel,
    std::uint32_t targetEdgesPerPage) {
    std::vector<RuntimeLodGroup> groups;
    std::int64_t level = static_cast<std::int64_t>(maxLevel);
    while (level >= 0) {
        while (level >= 0 && birthCounts[static_cast<std::size_t>(level)] == 0) {
            --level;
        }
        if (level < 0) {
            break;
        }
        RuntimeLodGroup group;
        group.groupIndex = static_cast<std::uint32_t>(groups.size());
        group.levelMax = static_cast<std::uint32_t>(level);
        group.levelMin = group.levelMax;
        while (level >= 0) {
            const auto count = birthCounts[static_cast<std::size_t>(level)];
            if (group.edgeCount != 0 && count != 0 &&
                group.edgeCount + count > targetEdgesPerPage) {
                break;
            }
            group.levelMin = static_cast<std::uint32_t>(level);
            group.edgeCount += count;
            --level;
            if (group.edgeCount >= targetEdgesPerPage) {
                break;
            }
        }
        groups.push_back(group);
    }
    return groups;
}

bool NodeSortLess(const NodeSortRecord& left, const NodeSortRecord& right) {
    if (left.morton != right.morton) {
        return left.morton < right.morton;
    }
    if (left.level != right.level) {
        return left.level > right.level;
    }
    return left.oldId < right.oldId;
}

void MergeNodeRunBatch(const std::vector<std::filesystem::path>& inputs,
                       const std::filesystem::path& outputPath) {
    struct Item {
        NodeSortRecord record;
        std::size_t source = 0;
    };
    struct Greater {
        bool operator()(const Item& a, const Item& b) const {
            return NodeSortLess(b.record, a.record);
        }
    };

    std::vector<std::ifstream> streams;
    streams.reserve(inputs.size());
    for (const auto& path : inputs) {
        streams.emplace_back(path, std::ios::binary);
        if (!streams.back()) {
            throw std::runtime_error("could not open FullCook node-sort run");
        }
    }
    std::priority_queue<Item, std::vector<Item>, Greater> queue;
    for (std::size_t i = 0; i < streams.size(); ++i) {
        NodeSortRecord record;
        if (ReadBinaryRecord(streams[i], record)) {
            queue.push({record, i});
        }
    }

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not create merged FullCook node-sort run");
    }
    while (!queue.empty()) {
        const auto item = queue.top();
        queue.pop();
        WriteBinaryRecord(output, item.record);
        NodeSortRecord next;
        if (ReadBinaryRecord(streams[item.source], next)) {
            queue.push({next, item.source});
        }
    }
}

std::filesystem::path BuildNodeOrder(
    const MappedArray<DiskNode>& nodes, double minLongitude, double maxLongitude,
    double minLatitude, double maxLatitude, std::uint32_t gridSize,
    std::uint64_t memoryBudgetBytes, const std::filesystem::path& tempDirectory,
    MappedArray<std::uint32_t>& oldToNewNode,
    const FullCooker::ProgressCallback& progressCallback) {
    const auto chunkBudget = std::max<std::uint64_t>(sizeof(NodeSortRecord),
                                                     memoryBudgetBytes / 3u);
    const auto recordsPerRun = std::max<std::uint64_t>(
        1u, chunkBudget / static_cast<std::uint64_t>(sizeof(NodeSortRecord)));
    std::vector<std::filesystem::path> runs;
    std::vector<NodeSortRecord> chunk;
    chunk.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(recordsPerRun, nodes.size())));

    Report(progressCallback, FullCookStage::ReorderNodes, 0, nodes.size());
    std::uint64_t runId = 0;
    for (std::uint64_t oldId = 0; oldId < nodes.size(); ++oldId) {
        const auto& node = nodes[oldId];
        chunk.push_back({NodeMorton(node, minLongitude, maxLongitude, minLatitude, maxLatitude,
                                   gridSize),
                         node.level, static_cast<std::uint32_t>(oldId)});
        if (chunk.size() >= recordsPerRun || oldId + 1 == nodes.size()) {
            std::sort(chunk.begin(), chunk.end(), NodeSortLess);
            const auto runPath = TempPath(tempDirectory, "node_run", runId++);
            std::ofstream output(runPath, std::ios::binary | std::ios::trunc);
            for (const auto& record : chunk) {
                WriteBinaryRecord(output, record);
            }
            output.close();
            runs.push_back(runPath);
            chunk.clear();
        }
        if ((oldId + 1) % kProgressInterval == 0 || oldId + 1 == nodes.size()) {
            Report(progressCallback, FullCookStage::ReorderNodes, oldId + 1, nodes.size());
        }
    }

    constexpr std::size_t kMergeFanIn = 48;
    std::uint64_t mergeGeneration = 0;
    while (runs.size() > 1) {
        std::vector<std::filesystem::path> next;
        for (std::size_t begin = 0; begin < runs.size(); begin += kMergeFanIn) {
            const auto end = std::min(runs.size(), begin + kMergeFanIn);
            std::vector<std::filesystem::path> batch(runs.begin() + static_cast<std::ptrdiff_t>(begin),
                                                     runs.begin() + static_cast<std::ptrdiff_t>(end));
            const auto merged = TempPath(tempDirectory, "node_merge",
                                         mergeGeneration * 100000u + next.size());
            MergeNodeRunBatch(batch, merged);
            for (const auto& path : batch) {
                RemoveFile(path);
            }
            next.push_back(merged);
        }
        runs = std::move(next);
        ++mergeGeneration;
    }
    if (runs.empty()) {
        throw std::runtime_error("FullCook node sort produced no output");
    }

    const auto orderPath = TempPath(tempDirectory, "node_order");
    std::ifstream sorted(runs.front(), std::ios::binary);
    std::ofstream order(orderPath, std::ios::binary | std::ios::trunc);
    if (!sorted || !order) {
        throw std::runtime_error("could not finalize FullCook node order");
    }
    NodeSortRecord record;
    std::uint64_t newId = 0;
    while (ReadBinaryRecord(sorted, record)) {
        if (record.oldId >= nodes.size()) {
            throw std::runtime_error("node order contains invalid old id");
        }
        oldToNewNode[record.oldId] = static_cast<std::uint32_t>(newId);
        WriteBinaryRecord(order, record.oldId);
        ++newId;
    }
    if (newId != nodes.size()) {
        throw std::runtime_error("FullCook node order does not cover all nodes");
    }
    RemoveFile(runs.front());
    return orderPath;
}

void EncodeVarUInt32(std::uint32_t value, std::ostream& output,
                     std::uint64_t& byteCount) {
    while (value >= 0x80u) {
        const auto byte = static_cast<char>((value & 0x7fu) | 0x80u);
        output.write(&byte, 1);
        ++byteCount;
        value >>= 7u;
    }
    const auto byte = static_cast<char>(value);
    output.write(&byte, 1);
    ++byteCount;
    if (!output) {
        throw std::runtime_error("failed to write CH index membership stream");
    }
}

class EdgePartitionBuilder {
public:
    EdgePartitionBuilder(const MappedArray<DiskEdge>& edges,
                         const std::vector<RuntimeLodGroup>& groups,
                         std::uint32_t gridBits,
                         const analysis::DatasetLayoutAnalysisConfig& config,
                         const std::filesystem::path& tempDirectory,
                         const std::filesystem::path& edgeOrderPath,
                         const std::filesystem::path& membershipPath,
                         const FullCooker::ProgressCallback& progressCallback)
        : edges_(edges), groups_(groups), gridBits_(gridBits), config_(config),
          tempDirectory_(tempDirectory), edgeOrder_(edgeOrderPath, std::ios::binary | std::ios::trunc),
          membership_(membershipPath, std::ios::binary | std::ios::trunc),
          progressCallback_(progressCallback) {
        pagesByGroup_.resize(groups_.size());
        if (!edgeOrder_ || !membership_) {
            throw std::runtime_error("could not create FullCook edge-order temporary files");
        }
        lodMaskWordsPerPage_ = maxLevel_ / 64u + 1u;
    }

    void SetMaxLevel(std::uint32_t maxLevel) {
        maxLevel_ = maxLevel;
        lodMaskWordsPerPage_ = maxLevel / 64u + 1u;
    }

    void Build(const std::vector<std::filesystem::path>& groupFiles,
               const std::filesystem::path& refinementFile,
               std::uint64_t drawableCount) {
        Report(progressCallback_, FullCookStage::ReorderEdges, 0, edges_.size());
        for (std::size_t group = 0; group < groupFiles.size(); ++group) {
            PartitionDrawableFile(groupFiles[group], groups_[group], 0, 0, 0);
        }
        drawableCount_ = nextNewEdgeId_;
        if (drawableCount_ != drawableCount) {
            throw std::runtime_error("external FullCook did not cover all drawable edges exactly once");
        }
        PartitionRefinementFile(refinementFile, 0);
        if (nextNewEdgeId_ != edges_.size()) {
            throw std::runtime_error("external FullCook edge order does not cover the full graph");
        }
        edgeOrder_.close();
        membership_.close();
        Report(progressCallback_, FullCookStage::ReorderEdges, edges_.size(), edges_.size());
    }

    [[nodiscard]] const std::vector<BuiltPage>& Pages() const { return pages_; }
    [[nodiscard]] const std::vector<std::vector<std::uint32_t>>& PagesByGroup() const {
        return pagesByGroup_;
    }
    [[nodiscard]] const std::vector<std::uint32_t>& SourceRefs() const { return sourceRefs_; }
    [[nodiscard]] const std::vector<std::uint64_t>& AliveMasks() const { return aliveMasks_; }
    [[nodiscard]] std::uint64_t MembershipByteCount() const { return membershipByteCount_; }
    [[nodiscard]] std::uint64_t DrawableCount() const { return drawableCount_; }

private:
    std::filesystem::path NextPartitionPath(std::string_view stem) {
        return TempPath(tempDirectory_, stem, partitionFileId_++);
    }

    void AppendEdgeOrder(std::uint32_t oldId) {
        WriteBinaryRecord(edgeOrder_, oldId);
        ++nextNewEdgeId_;
        if (nextNewEdgeId_ % kProgressInterval == 0) {
            Report(progressCallback_, FullCookStage::ReorderEdges, nextNewEdgeId_, edges_.size());
        }
    }

    void AppendAliveMask(index::CHIndexGraphPage& page,
                         const std::vector<PartitionRecord>& records) {
        page.lodMaskOffset = static_cast<std::uint32_t>(aliveMasks_.size());
        aliveMasks_.resize(aliveMasks_.size() + lodMaskWordsPerPage_, 0);
        const auto base = static_cast<std::size_t>(page.lodMaskOffset);
        for (const auto& record : records) {
            const auto& edge = edges_[record.oldId];
            if (edge.birth < 0 || edge.death < 0) {
                continue;
            }
            const auto death = static_cast<std::uint32_t>(edge.death);
            const auto birth = static_cast<std::uint32_t>(edge.birth);
            const auto firstWord = death / 64u;
            const auto lastWord = birth / 64u;
            if (firstWord == lastWord) {
                const auto lowMask = ~std::uint64_t{0} << (death % 64u);
                const auto highBit = birth % 64u;
                const auto highMask = highBit == 63u
                                          ? ~std::uint64_t{0}
                                          : (std::uint64_t{1} << (highBit + 1u)) - 1u;
                aliveMasks_[base + firstWord] |= lowMask & highMask;
            } else {
                aliveMasks_[base + firstWord] |= ~std::uint64_t{0} << (death % 64u);
                for (auto word = firstWord + 1u; word < lastWord; ++word) {
                    aliveMasks_[base + word] = ~std::uint64_t{0};
                }
                const auto highBit = birth % 64u;
                aliveMasks_[base + lastWord] |=
                    highBit == 63u ? ~std::uint64_t{0}
                                   : (std::uint64_t{1} << (highBit + 1u)) - 1u;
            }
        }
    }

    void EmitDrawableChunk(std::vector<PartitionRecord>& records,
                           const RuntimeLodGroup& group, std::uint32_t spatialLevel,
                           std::uint32_t tileX, std::uint32_t tileY, std::uint32_t subPage) {
        if (records.empty()) {
            return;
        }
        std::sort(records.begin(), records.end(), [&](const auto& left, const auto& right) {
            const auto& leftEdge = edges_[left.oldId];
            const auto& rightEdge = edges_[right.oldId];
            if (leftEdge.birth != rightEdge.birth) {
                return leftEdge.birth > rightEdge.birth;
            }
            if (leftEdge.death != rightEdge.death) {
                return leftEdge.death > rightEdge.death;
            }
            const auto leftMorton = BoundsMorton(left.bounds, gridBits_);
            const auto rightMorton = BoundsMorton(right.bounds, gridBits_);
            return leftMorton != rightMorton ? leftMorton < rightMorton
                                              : left.oldId < right.oldId;
        });

        BuiltPage built;
        auto& page = built.page;
        page.pageId = static_cast<std::uint32_t>(pages_.size());
        page.bandIndex = group.groupIndex;
        page.levelMin = group.levelMin;
        page.levelMax = group.levelMax;
        page.spatialLevel = spatialLevel;
        page.tileX = tileX;
        page.tileY = tileY;
        page.subPage = subPage;
        page.edgeCount = static_cast<std::uint32_t>(records.size());
        built.firstNewEdgeId = nextNewEdgeId_;

        page.sourceBlockRefOffset = static_cast<std::uint32_t>(sourceRefs_.size());
        const auto firstBlock = static_cast<std::uint32_t>(
            built.firstNewEdgeId / config_.sourceBlockEdgeCount);
        const auto lastBlock = static_cast<std::uint32_t>(
            (built.firstNewEdgeId + records.size() - 1u) / config_.sourceBlockEdgeCount);
        for (auto block = firstBlock; block <= lastBlock; ++block) {
            sourceRefs_.push_back(block);
        }
        page.sourceBlockRefCount = lastBlock - firstBlock + 1u;
        AppendAliveMask(page, records);

        page.membershipByteOffset = membershipByteCount_;
        bool first = true;
        std::uint32_t previous = 0;
        for (const auto& record : records) {
            const auto newId = static_cast<std::uint32_t>(nextNewEdgeId_);
            EncodeVarUInt32(first ? newId : newId - previous, membership_, membershipByteCount_);
            previous = newId;
            first = false;
            AppendEdgeOrder(record.oldId);
        }
        page.membershipByteSize = static_cast<std::uint32_t>(
            membershipByteCount_ - page.membershipByteOffset);
        pagesByGroup_[group.groupIndex].push_back(page.pageId);
        pages_.push_back(built);
    }

    void EmitDrawableFile(const std::filesystem::path& path, const RuntimeLodGroup& group,
                          std::uint32_t spatialLevel, std::uint32_t tileX,
                          std::uint32_t tileY) {
        const auto count = BinaryRecordCount(path, sizeof(PartitionRecord));
        if (count == 0) {
            RemoveFile(path);
            return;
        }
        std::ifstream input(path, std::ios::binary);
        std::uint32_t subPage = 0;
        std::vector<PartitionRecord> chunk;
        chunk.reserve(config_.virtualPageEdgeCount);
        PartitionRecord record;
        while (ReadBinaryRecord(input, record)) {
            chunk.push_back(record);
            if (chunk.size() == config_.virtualPageEdgeCount) {
                EmitDrawableChunk(chunk, group, spatialLevel, tileX, tileY, subPage++);
                chunk.clear();
            }
        }
        EmitDrawableChunk(chunk, group, spatialLevel, tileX, tileY, subPage);
        input.close();
        RemoveFile(path);
    }

    void PartitionDrawableFile(const std::filesystem::path& path,
                               const RuntimeLodGroup& group, std::uint32_t spatialLevel,
                               std::uint32_t tileX, std::uint32_t tileY) {
        const auto count = BinaryRecordCount(path, sizeof(PartitionRecord));
        if (count == 0) {
            RemoveFile(path);
            return;
        }
        if (count <= config_.virtualPageEdgeCount || spatialLevel >= gridBits_) {
            EmitDrawableFile(path, group, spatialLevel, tileX, tileY);
            return;
        }

        const auto stayPath = NextPartitionPath("edge_stay");
        std::array<std::filesystem::path, 4> childPaths{};
        for (std::uint32_t child = 0; child < 4; ++child) {
            childPaths[child] = NextPartitionPath("edge_child");
        }
        std::ofstream stay(stayPath, std::ios::binary | std::ios::trunc);
        std::array<std::ofstream, 4> children;
        for (std::uint32_t child = 0; child < 4; ++child) {
            children[child].open(childPaths[child], std::ios::binary | std::ios::trunc);
        }

        std::ifstream input(path, std::ios::binary);
        const auto childLevel = spatialLevel + 1u;
        const auto shift = gridBits_ - childLevel;
        PartitionRecord record;
        while (ReadBinaryRecord(input, record)) {
            const auto bounds = UnpackCellBounds(record.bounds);
            const auto minTileX = bounds.minX >> shift;
            const auto maxTileX = bounds.maxX >> shift;
            const auto minTileY = bounds.minY >> shift;
            const auto maxTileY = bounds.maxY >> shift;
            if (minTileX == maxTileX && minTileY == maxTileY) {
                const auto childIndex = (minTileX & 1u) | ((minTileY & 1u) << 1u);
                WriteBinaryRecord(children[childIndex], record);
            } else {
                WriteBinaryRecord(stay, record);
            }
        }
        input.close();
        stay.close();
        for (auto& child : children) {
            child.close();
        }
        RemoveFile(path);

        EmitDrawableFile(stayPath, group, spatialLevel, tileX, tileY);
        for (std::uint32_t child = 0; child < 4; ++child) {
            const auto childX = child & 1u;
            const auto childY = (child >> 1u) & 1u;
            PartitionDrawableFile(childPaths[child], group, childLevel,
                                  tileX * 2u + childX, tileY * 2u + childY);
        }
    }

    void EmitRefinementFile(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            return;
        }
        std::ifstream input(path, std::ios::binary);
        std::vector<PartitionRecord> chunk;
        chunk.reserve(config_.virtualPageEdgeCount);
        PartitionRecord record;
        auto flush = [&]() {
            if (chunk.empty()) {
                return;
            }
            std::sort(chunk.begin(), chunk.end(), [&](const auto& left, const auto& right) {
                const auto leftMorton = BoundsMorton(left.bounds, gridBits_);
                const auto rightMorton = BoundsMorton(right.bounds, gridBits_);
                return leftMorton != rightMorton ? leftMorton < rightMorton
                                                  : left.oldId < right.oldId;
            });
            for (const auto& value : chunk) {
                AppendEdgeOrder(value.oldId);
            }
            chunk.clear();
        };
        while (ReadBinaryRecord(input, record)) {
            chunk.push_back(record);
            if (chunk.size() == config_.virtualPageEdgeCount) {
                flush();
            }
        }
        flush();
        input.close();
        RemoveFile(path);
    }

    void PartitionRefinementFile(const std::filesystem::path& path,
                                 std::uint32_t spatialLevel) {
        const auto count = BinaryRecordCount(path, sizeof(PartitionRecord));
        if (count == 0) {
            RemoveFile(path);
            return;
        }
        if (count <= config_.virtualPageEdgeCount || spatialLevel >= gridBits_) {
            EmitRefinementFile(path);
            return;
        }
        const auto stayPath = NextPartitionPath("refine_stay");
        std::array<std::filesystem::path, 4> childPaths{};
        for (std::uint32_t child = 0; child < 4; ++child) {
            childPaths[child] = NextPartitionPath("refine_child");
        }
        std::ofstream stay(stayPath, std::ios::binary | std::ios::trunc);
        std::array<std::ofstream, 4> children;
        for (std::uint32_t child = 0; child < 4; ++child) {
            children[child].open(childPaths[child], std::ios::binary | std::ios::trunc);
        }
        std::ifstream input(path, std::ios::binary);
        const auto childLevel = spatialLevel + 1u;
        const auto shift = gridBits_ - childLevel;
        PartitionRecord record;
        while (ReadBinaryRecord(input, record)) {
            const auto bounds = UnpackCellBounds(record.bounds);
            const auto minTileX = bounds.minX >> shift;
            const auto maxTileX = bounds.maxX >> shift;
            const auto minTileY = bounds.minY >> shift;
            const auto maxTileY = bounds.maxY >> shift;
            if (minTileX == maxTileX && minTileY == maxTileY) {
                const auto childIndex = (minTileX & 1u) | ((minTileY & 1u) << 1u);
                WriteBinaryRecord(children[childIndex], record);
            } else {
                WriteBinaryRecord(stay, record);
            }
        }
        input.close();
        stay.close();
        for (auto& child : children) {
            child.close();
        }
        RemoveFile(path);
        EmitRefinementFile(stayPath);
        for (auto& child : childPaths) {
            PartitionRefinementFile(child, childLevel);
        }
    }

    const MappedArray<DiskEdge>& edges_;
    const std::vector<RuntimeLodGroup>& groups_;
    std::uint32_t gridBits_ = 0;
    const analysis::DatasetLayoutAnalysisConfig& config_;
    std::filesystem::path tempDirectory_;
    std::ofstream edgeOrder_;
    std::ofstream membership_;
    const FullCooker::ProgressCallback& progressCallback_;
    std::uint32_t maxLevel_ = 0;
    std::uint32_t lodMaskWordsPerPage_ = 1;
    std::uint64_t nextNewEdgeId_ = 0;
    std::uint64_t drawableCount_ = 0;
    std::uint64_t membershipByteCount_ = 0;
    std::uint64_t partitionFileId_ = 0;
    std::vector<BuiltPage> pages_;
    std::vector<std::vector<std::uint32_t>> pagesByGroup_;
    std::vector<std::uint32_t> sourceRefs_;
    std::vector<std::uint64_t> aliveMasks_;
};

std::uint64_t PageKey(std::uint32_t level, std::uint32_t x, std::uint32_t y) {
    return static_cast<std::uint64_t>(level) |
           (static_cast<std::uint64_t>(x) << 8u) |
           (static_cast<std::uint64_t>(y) << 36u);
}

void BuildLodChildReferences(std::vector<BuiltPage>& pages,
                             const std::vector<std::vector<std::uint32_t>>& pagesByGroup,
                             std::vector<std::uint32_t>& output) {
    for (std::size_t group = 0; group + 1 < pagesByGroup.size(); ++group) {
        std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> descendantLookup;
        std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> exactLookup;
        for (const auto childId : pagesByGroup[group + 1]) {
            const auto& child = pages[childId].page;
            exactLookup[PageKey(child.spatialLevel, child.tileX, child.tileY)].push_back(childId);
            for (std::uint32_t level = 0; level <= child.spatialLevel; ++level) {
                const auto shift = child.spatialLevel - level;
                descendantLookup[PageKey(level, child.tileX >> shift, child.tileY >> shift)]
                    .push_back(childId);
            }
        }
        for (const auto parentId : pagesByGroup[group]) {
            auto& parent = pages[parentId].page;
            parent.lodChildPageRefOffset = static_cast<std::uint32_t>(output.size());
            std::vector<std::uint32_t> refs;
            if (const auto found = descendantLookup.find(
                    PageKey(parent.spatialLevel, parent.tileX, parent.tileY));
                found != descendantLookup.end()) {
                refs.insert(refs.end(), found->second.begin(), found->second.end());
            }
            for (std::uint32_t level = 0; level < parent.spatialLevel; ++level) {
                const auto shift = parent.spatialLevel - level;
                if (const auto found = exactLookup.find(
                        PageKey(level, parent.tileX >> shift, parent.tileY >> shift));
                    found != exactLookup.end()) {
                    refs.insert(refs.end(), found->second.begin(), found->second.end());
                }
            }
            std::sort(refs.begin(), refs.end());
            refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
            output.insert(output.end(), refs.begin(), refs.end());
            parent.lodChildPageRefCount = static_cast<std::uint32_t>(refs.size());
        }
    }
}

void ValidateConfig(const analysis::DatasetLayoutAnalysisConfig& config) {
    if (config.spatialGridSize == 0 ||
        (config.spatialGridSize & (config.spatialGridSize - 1u)) != 0 ||
        config.spatialGridSize > 1024u || config.sourceBlockNodeCount == 0 ||
        config.sourceBlockEdgeCount == 0 || config.virtualPageEdgeCount == 0) {
        throw std::runtime_error("invalid FullCook layout configuration");
    }
}

void ResolveGeometryBounds(MappedArray<DiskEdge>& edges,
                           const std::filesystem::path& statePath,
                           const FullCooker::ProgressCallback& progressCallback) {
    MappedArray<std::uint8_t> state;
    state.Create(statePath, edges.size());
    std::fill_n(&state[0], static_cast<std::size_t>(state.size()), std::uint8_t{0});

    struct Frame { std::uint32_t edgeId = 0; bool expanded = false; };
    std::vector<Frame> stack;
    std::uint64_t resolved = 0;
    for (std::uint32_t root = 0; root < edges.size(); ++root) {
        if (state[root] == 2) {
            continue;
        }
        stack.push_back({root, false});
        while (!stack.empty()) {
            auto& frame = stack.back();
            const auto edgeId = frame.edgeId;
            if (state[edgeId] == 2) {
                stack.pop_back();
                continue;
            }
            if (!frame.expanded) {
                if (state[edgeId] == 1) {
                    throw std::runtime_error("cycle detected in shortcut hierarchy during external FullCook");
                }
                state[edgeId] = 1;
                frame.expanded = true;
                const auto childA = edges[edgeId].childA;
                const auto childB = edges[edgeId].childB;
                for (const auto child : {childB, childA}) {
                    if (child == data::InvalidEdgeId) {
                        continue;
                    }
                    if (child >= edges.size()) {
                        throw std::runtime_error("shortcut child id out of range during external FullCook");
                    }
                    if (state[child] == 1) {
                        throw std::runtime_error("cycle detected in shortcut hierarchy during external FullCook");
                    }
                    if (state[child] == 0) {
                        stack.push_back({child, false});
                    }
                }
                continue;
            }
            auto bounds = UnpackCellBounds(edges[edgeId].bounds);
            for (const auto child : {edges[edgeId].childA, edges[edgeId].childB}) {
                if (child != data::InvalidEdgeId) {
                    bounds = UnionBounds(bounds, UnpackCellBounds(edges[child].bounds));
                }
            }
            edges[edgeId].bounds = PackCellBounds(bounds);
            state[edgeId] = 2;
            ++resolved;
            stack.pop_back();
            if (resolved % kProgressInterval == 0 || resolved == edges.size()) {
                Report(progressCallback, FullCookStage::ReorderEdges, resolved, edges.size());
            }
        }
    }
    state.Flush();
}


std::filesystem::path WriteOrderedEdgeSpatialBounds(
    const MappedArray<DiskEdge>& edges,
    const std::filesystem::path& edgeOrderPath,
    const std::filesystem::path& outputPath) {
    std::ifstream edgeOrder(edgeOrderPath, std::ios::binary);
    if (!edgeOrder) {
        throw std::runtime_error("could not read FullCook edge order for spatial bounds");
    }
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not create FullCook edge-spatial-bounds stream");
    }
    std::uint32_t oldEdgeId = 0;
    for (std::uint64_t newId = 0; newId < edges.size(); ++newId) {
        if (!ReadBinaryRecord(edgeOrder, oldEdgeId) || oldEdgeId >= edges.size()) {
            throw std::runtime_error("FullCook edge order is truncated while writing spatial bounds");
        }
        WriteBinaryRecord(output, edges[oldEdgeId].bounds);
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize FullCook edge-spatial-bounds stream");
    }
    return outputPath;
}

void WriteCookedFiles(const std::filesystem::path& sourceGraphPath,
                      std::uint64_t nodeSectionOffset,
                      const MappedArray<DiskNode>& nodes,
                      const MappedArray<DiskEdge>& edges,
                      const MappedArray<std::uint32_t>& oldToNewNode,
                      const MappedArray<std::uint32_t>& oldToNewEdge,
                      const std::filesystem::path& nodeOrderPath,
                      const std::filesystem::path& edgeOrderPath,
                      const std::filesystem::path& graphOutputPath,
                      const std::filesystem::path& rangesOutputPath,
                      const analysis::DatasetLayoutAnalysisConfig& config,
                      std::vector<index::CHIndexSourceBlock>& nodeBlocks,
                      std::vector<index::CHIndexSourceBlock>& edgeBlocks,
                      std::vector<index::CHIndexSourceBlock>& rangeBlocks,
                      const FullCooker::ProgressCallback& progressCallback) {
    TextWriter graphWriter(graphOutputPath);
    graphWriter.Raw(ReadPrefix(sourceGraphPath, nodeSectionOffset));
    std::ifstream nodeOrder(nodeOrderPath, std::ios::binary);
    if (!nodeOrder) {
        throw std::runtime_error("could not read FullCook node order");
    }
    Report(progressCallback, FullCookStage::WriteGraph, 0, nodes.size() + edges.size());
    std::uint32_t oldNodeId = 0;
    for (std::uint64_t newId = 0; newId < nodes.size(); ++newId) {
        if (!ReadBinaryRecord(nodeOrder, oldNodeId)) {
            throw std::runtime_error("FullCook node order is truncated");
        }
        if (newId % config.sourceBlockNodeCount == 0) {
            nodeBlocks.push_back({newId,
                                  static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                      config.sourceBlockNodeCount, nodes.size() - newId)),
                                  graphWriter.Tell()});
        }
        const auto& node = nodes[oldNodeId];
        graphWriter.Line(static_cast<std::uint32_t>(newId), node.osmId,
                         TextWriter::Coordinate(node.latitude),
                         TextWriter::Coordinate(node.longitude),
                         TextWriter::CompactFloat(node.elevation), node.level);
        if ((newId + 1) % kProgressInterval == 0 || newId + 1 == nodes.size()) {
            Report(progressCallback, FullCookStage::WriteGraph, newId + 1,
                   nodes.size() + edges.size());
        }
    }

    std::ifstream edgeOrder(edgeOrderPath, std::ios::binary);
    if (!edgeOrder) {
        throw std::runtime_error("could not read FullCook edge order");
    }
    std::uint32_t oldEdgeId = 0;
    for (std::uint64_t newId = 0; newId < edges.size(); ++newId) {
        if (!ReadBinaryRecord(edgeOrder, oldEdgeId)) {
            throw std::runtime_error("FullCook edge order is truncated");
        }
        if (newId % config.sourceBlockEdgeCount == 0) {
            edgeBlocks.push_back({newId,
                                  static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                      config.sourceBlockEdgeCount, edges.size() - newId)),
                                  graphWriter.Tell()});
        }
        const auto& edge = edges[oldEdgeId];
        const auto childA = edge.childA == data::InvalidEdgeId
                                ? std::int64_t{-1}
                                : static_cast<std::int64_t>(oldToNewEdge[edge.childA]);
        const auto childB = edge.childB == data::InvalidEdgeId
                                ? std::int64_t{-1}
                                : static_cast<std::int64_t>(oldToNewEdge[edge.childB]);
        graphWriter.Line(oldToNewNode[edge.source], oldToNewNode[edge.target],
                         TextWriter::CompactFloat(edge.weight), edge.type, edge.maxSpeed,
                         childA, childB);
        if ((newId + 1) % kProgressInterval == 0 || newId + 1 == edges.size()) {
            Report(progressCallback, FullCookStage::WriteGraph,
                   nodes.size() + newId + 1, nodes.size() + edges.size());
        }
    }
    graphWriter.Close();

    TextWriter rangeWriter(rangesOutputPath);
    edgeOrder.clear();
    edgeOrder.seekg(0, std::ios::beg);
    Report(progressCallback, FullCookStage::WriteRanges, 0, edges.size());
    for (std::uint64_t newId = 0; newId < edges.size(); ++newId) {
        if (!ReadBinaryRecord(edgeOrder, oldEdgeId)) {
            throw std::runtime_error("FullCook edge order is truncated while writing ranges");
        }
        if (newId % config.sourceBlockEdgeCount == 0) {
            rangeBlocks.push_back({newId,
                                   static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                       config.sourceBlockEdgeCount, edges.size() - newId)),
                                   rangeWriter.Tell()});
        }
        const auto& edge = edges[oldEdgeId];
        rangeWriter.Line(static_cast<std::uint32_t>(newId), edge.birth, edge.death);
        if ((newId + 1) % kProgressInterval == 0 || newId + 1 == edges.size()) {
            Report(progressCallback, FullCookStage::WriteRanges, newId + 1, edges.size());
        }
    }
    rangeWriter.Close();
}

void ValidateCookedStreaming(const std::filesystem::path& graphPath,
                             const std::filesystem::path& rangesPath,
                             std::uint64_t expectedNodes, std::uint64_t expectedEdges,
                             const FullCooker::ProgressCallback& progressCallback) {
    Report(progressCallback, FullCookStage::AnalyzeCooked, 0,
           expectedNodes + expectedEdges * 2u);
    TextSourceScanner graph(graphPath);
    const auto nodeCount = graph.Read<std::uint64_t>();
    const auto edgeCount = graph.Read<std::uint64_t>();
    if (nodeCount != expectedNodes || edgeCount != expectedEdges) {
        throw std::runtime_error("external FullCook changed graph counts");
    }
    for (std::uint64_t i = 0; i < nodeCount; ++i) {
        const auto id = graph.Read<std::uint32_t>();
        if (id != i) {
            throw std::runtime_error("external FullCook node ids are not sequential");
        }
        graph.Read<std::uint64_t>();
        graph.Read<double>();
        graph.Read<double>();
        graph.Read<float>();
        graph.Read<std::uint32_t>();
        if ((i + 1) % kProgressInterval == 0) {
            Report(progressCallback, FullCookStage::AnalyzeCooked, i + 1,
                   nodeCount + edgeCount * 2u);
        }
    }
    for (std::uint64_t edgeId = 0; edgeId < edgeCount; ++edgeId) {
        const auto source = graph.Read<std::uint32_t>();
        const auto target = graph.Read<std::uint32_t>();
        graph.Read<float>();
        graph.Read<std::int32_t>();
        graph.Read<std::int32_t>();
        const auto childA = graph.Read<std::int64_t>();
        const auto childB = graph.Read<std::int64_t>();
        if (source >= nodeCount || target >= nodeCount ||
            childA >= static_cast<std::int64_t>(edgeCount) ||
            childB >= static_cast<std::int64_t>(edgeCount)) {
            throw std::runtime_error("external FullCook produced an invalid graph reference");
        }
    }
    TextSourceScanner ranges(rangesPath);
    for (std::uint64_t record = 0; record < edgeCount; ++record) {
        const auto edgeId = ranges.Read<std::uint32_t>();
        const auto birth = ranges.Read<std::int32_t>();
        const auto death = ranges.Read<std::int32_t>();
        if (edgeId != record || !((birth == -1 && death == -1) ||
                                  (birth >= 0 && death >= 0 && death <= birth))) {
            throw std::runtime_error("external FullCook produced an invalid range record");
        }
        if ((record + 1) % kProgressInterval == 0 || record + 1 == edgeCount) {
            Report(progressCallback, FullCookStage::AnalyzeCooked,
                   nodeCount + edgeCount + record + 1,
                   nodeCount + edgeCount * 2u);
        }
    }
}

} // namespace

ExternalFullCookResult RunExternalFullCook(
    const std::filesystem::path& sourceGraphPath,
    const std::filesystem::path& sourceRangesPath,
    const std::filesystem::path& outputGraphPath,
    const std::filesystem::path& outputRangesPath,
    const std::filesystem::path& outputIndexPath,
    const analysis::DatasetLayoutAnalysisConfig& config,
    std::uint64_t memoryBudgetBytes,
    const std::filesystem::path& tempRoot,
    const FullCooker::ProgressCallback& progressCallback) {
    ValidateConfig(config);
    if (memoryBudgetBytes < 4u * 1024u * 1024u) {
        memoryBudgetBytes = 4u * 1024u * 1024u;
    }

    std::filesystem::create_directories(tempRoot);
    TempDirectoryGuard tempGuard(tempRoot);

    try {
        TextSourceScanner graph(sourceGraphPath);
        const auto nodeCount = graph.Read<std::uint64_t>();
        const auto edgeCount = graph.Read<std::uint64_t>();
        if (nodeCount == 0 || edgeCount == 0 ||
            nodeCount > std::numeric_limits<std::uint32_t>::max() ||
            edgeCount > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("external FullCook requires non-empty 32-bit-id graph data");
        }

        MappedArray<DiskNode> nodes;
        MappedArray<DiskRange> ranges;
        MappedArray<DiskEdge> edges;
        MappedArray<std::uint32_t> oldToNewNode;
        MappedArray<std::uint32_t> oldToNewEdge;
        nodes.Create(TempPath(tempRoot, "nodes"), nodeCount);
        ranges.Create(TempPath(tempRoot, "ranges"), edgeCount);
        edges.Create(TempPath(tempRoot, "edges"), edgeCount);
        oldToNewNode.Create(TempPath(tempRoot, "node_remap"), nodeCount);
        oldToNewEdge.Create(TempPath(tempRoot, "edge_remap"), edgeCount);

        Report(progressCallback, FullCookStage::AnalyzeSource, 0, nodeCount + edgeCount);
        double minLatitude = std::numeric_limits<double>::infinity();
        double maxLatitude = -std::numeric_limits<double>::infinity();
        double minLongitude = std::numeric_limits<double>::infinity();
        double maxLongitude = -std::numeric_limits<double>::infinity();
        std::uint64_t nodeSectionOffset = 0;
        for (std::uint64_t record = 0; record < nodeCount; ++record) {
            const auto [id, offset] = graph.ReadWithOffset<std::uint32_t>();
            if (id != record) {
                throw std::runtime_error(
                    "external FullCook currently requires source node records to be ID ordered");
            }
            if (record == 0) {
                nodeSectionOffset = offset;
            }
            DiskNode node;
            node.osmId = graph.Read<std::uint64_t>();
            node.latitude = graph.Read<double>();
            node.longitude = graph.Read<double>();
            node.elevation = graph.Read<float>();
            node.level = graph.Read<std::uint32_t>();
            nodes[id] = node;
            minLatitude = std::min(minLatitude, node.latitude);
            maxLatitude = std::max(maxLatitude, node.latitude);
            minLongitude = std::min(minLongitude, node.longitude);
            maxLongitude = std::max(maxLongitude, node.longitude);
            if ((record + 1) % kProgressInterval == 0 || record + 1 == nodeCount) {
                Report(progressCallback, FullCookStage::AnalyzeSource, record + 1,
                       nodeCount + edgeCount);
            }
        }

        TextSourceScanner rangeScanner(sourceRangesPath);
        std::vector<std::uint64_t> birthCounts(1, 0);
        std::uint32_t maxLevel = 0;
        std::uint64_t drawableCount = 0;
        for (std::uint64_t record = 0; record < edgeCount; ++record) {
            const auto edgeId = rangeScanner.Read<std::uint32_t>();
            const auto birth = rangeScanner.Read<std::int32_t>();
            const auto death = rangeScanner.Read<std::int32_t>();
            if (edgeId != record) {
                throw std::runtime_error(
                    "external FullCook currently requires range records to be EdgeID ordered");
            }
            if (!((birth == -1 && death == -1) ||
                  (birth >= 0 && death >= 0 && death <= birth))) {
                throw std::runtime_error("invalid lifetime range in FullCook source");
            }
            ranges[edgeId] = {birth, death};
            if (birth >= 0) {
                const auto level = static_cast<std::uint32_t>(birth);
                if (birthCounts.size() <= level) {
                    birthCounts.resize(static_cast<std::size_t>(level) + 1u, 0);
                }
                ++birthCounts[level];
                maxLevel = std::max(maxLevel, level);
                ++drawableCount;
            }
        }
        birthCounts.resize(static_cast<std::size_t>(maxLevel) + 1u, 0);

        const auto gridBits = static_cast<std::uint32_t>(std::countr_zero(config.spatialGridSize));
        Report(progressCallback, FullCookStage::LoadSource, 0, edgeCount);
        for (std::uint64_t edgeId = 0; edgeId < edgeCount; ++edgeId) {
            DiskEdge edge;
            edge.source = graph.Read<std::uint32_t>();
            edge.target = graph.Read<std::uint32_t>();
            edge.weight = graph.Read<float>();
            edge.type = graph.Read<std::int32_t>();
            edge.maxSpeed = graph.Read<std::int32_t>();
            edge.childA = DecodeChild(graph.Read<std::int64_t>());
            edge.childB = DecodeChild(graph.Read<std::int64_t>());
            if (edge.source >= nodeCount || edge.target >= nodeCount) {
                throw std::runtime_error("edge endpoint out of range during external FullCook");
            }
            edge.birth = ranges[edgeId].birth;
            edge.death = ranges[edgeId].death;
            const auto [sourceX, sourceY] = SpatialCell(
                nodes[edge.source].longitude, nodes[edge.source].latitude,
                minLongitude, maxLongitude, minLatitude, maxLatitude, config.spatialGridSize);
            const auto [targetX, targetY] = SpatialCell(
                nodes[edge.target].longitude, nodes[edge.target].latitude,
                minLongitude, maxLongitude, minLatitude, maxLatitude, config.spatialGridSize);
            edge.bounds = PackCellBounds({std::min(sourceX, targetX), std::min(sourceY, targetY),
                                          std::max(sourceX, targetX), std::max(sourceY, targetY)});
            edges[edgeId] = edge;
            if ((edgeId + 1) % kProgressInterval == 0 || edgeId + 1 == edgeCount) {
                Report(progressCallback, FullCookStage::LoadSource, edgeId + 1, edgeCount);
            }
        }
        nodes.Flush();
        ranges.Flush();
        edges.Flush();

        const auto nodeOrderPath = BuildNodeOrder(
            nodes, minLongitude, maxLongitude, minLatitude, maxLatitude,
            config.spatialGridSize, memoryBudgetBytes, tempRoot, oldToNewNode,
            progressCallback);
        oldToNewNode.Flush();

        ResolveGeometryBounds(edges, TempPath(tempRoot, "edge_state"), progressCallback);
        edges.Flush();

        const auto groups = BuildRuntimeLodGroups(birthCounts, maxLevel,
                                                  config.virtualPageEdgeCount);
        std::vector<std::uint32_t> levelToGroup(static_cast<std::size_t>(maxLevel) + 1u,
                                                kInvalidId);
        for (const auto& group : groups) {
            for (std::uint32_t level = group.levelMin; level <= group.levelMax; ++level) {
                levelToGroup[level] = group.groupIndex;
            }
        }

        std::vector<std::filesystem::path> groupFiles(groups.size());
        std::vector<std::ofstream> groupOutputs(groups.size());
        for (std::size_t group = 0; group < groups.size(); ++group) {
            groupFiles[group] = TempPath(tempRoot, "lod_group", group);
            groupOutputs[group].open(groupFiles[group], std::ios::binary | std::ios::trunc);
            if (!groupOutputs[group]) {
                throw std::runtime_error("could not create external FullCook LOD group file");
            }
        }
        const auto refinementPath = TempPath(tempRoot, "refinement");
        std::ofstream refinement(refinementPath, std::ios::binary | std::ios::trunc);
        for (std::uint32_t edgeId = 0; edgeId < edgeCount; ++edgeId) {
            const PartitionRecord record{edgeId, 0, edges[edgeId].bounds};
            if (edges[edgeId].birth >= 0) {
                const auto group = levelToGroup[static_cast<std::uint32_t>(edges[edgeId].birth)];
                if (group == kInvalidId || group >= groupOutputs.size()) {
                    throw std::runtime_error("drawable edge has no external FullCook LOD group");
                }
                WriteBinaryRecord(groupOutputs[group], record);
            } else {
                WriteBinaryRecord(refinement, record);
            }
        }
        for (auto& output : groupOutputs) {
            output.close();
        }
        refinement.close();

        const auto edgeOrderPath = TempPath(tempRoot, "edge_order");
        const auto membershipPath = TempPath(tempRoot, "membership");
        EdgePartitionBuilder partitionBuilder(edges, groups, gridBits, config, tempRoot,
                                              edgeOrderPath, membershipPath, progressCallback);
        partitionBuilder.SetMaxLevel(maxLevel);
        partitionBuilder.Build(groupFiles, refinementPath, drawableCount);

        std::fill_n(&oldToNewEdge[0], static_cast<std::size_t>(oldToNewEdge.size()), kInvalidId);
        std::ifstream edgeOrder(edgeOrderPath, std::ios::binary);
        std::uint32_t oldEdgeId = 0;
        std::uint64_t newEdgeId = 0;
        while (ReadBinaryRecord(edgeOrder, oldEdgeId)) {
            if (oldEdgeId >= edgeCount || oldToNewEdge[oldEdgeId] != kInvalidId) {
                throw std::runtime_error("external FullCook edge order contains duplicate/invalid id");
            }
            oldToNewEdge[oldEdgeId] = static_cast<std::uint32_t>(newEdgeId++);
        }
        if (newEdgeId != edgeCount) {
            throw std::runtime_error("external FullCook edge order is incomplete");
        }
        oldToNewEdge.Flush();

        const auto edgeSpatialBoundsPath = WriteOrderedEdgeSpatialBounds(
            edges, edgeOrderPath, TempPath(tempRoot, "edge_spatial_bounds"));

        auto builtPages = partitionBuilder.Pages();
        std::vector<std::uint32_t> childRefs;
        edgeOrder.clear();
        edgeOrder.seekg(0, std::ios::beg);
        MappedArray<std::uint32_t> orderedIds;
        orderedIds.Create(TempPath(tempRoot, "ordered_ids"), edgeCount);
        newEdgeId = 0;
        while (ReadBinaryRecord(edgeOrder, oldEdgeId)) {
            orderedIds[newEdgeId++] = oldEdgeId;
        }
        for (auto& built : builtPages) {
            auto& page = built.page;
            page.childSourceBlockRefOffset = static_cast<std::uint32_t>(childRefs.size());
            std::vector<std::uint32_t> refs;
            refs.reserve(static_cast<std::size_t>(page.edgeCount) * 2u);
            for (std::uint64_t i = 0; i < page.edgeCount; ++i) {
                const auto oldId = orderedIds[built.firstNewEdgeId + i];
                const auto& edge = edges[oldId];
                for (const auto child : {edge.childA, edge.childB}) {
                    if (child != data::InvalidEdgeId) {
                        refs.push_back(oldToNewEdge[child] / config.sourceBlockEdgeCount);
                    }
                }
            }
            std::sort(refs.begin(), refs.end());
            refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
            childRefs.insert(childRefs.end(), refs.begin(), refs.end());
            page.childSourceBlockRefCount = static_cast<std::uint32_t>(refs.size());
        }
        std::vector<std::uint32_t> lodChildRefs;
        BuildLodChildReferences(builtPages, partitionBuilder.PagesByGroup(), lodChildRefs);

        std::vector<index::CHIndexSourceBlock> nodeBlocks;
        std::vector<index::CHIndexSourceBlock> edgeBlocks;
        std::vector<index::CHIndexSourceBlock> rangeBlocks;
        WriteCookedFiles(sourceGraphPath, nodeSectionOffset, nodes, edges, oldToNewNode,
                         oldToNewEdge, nodeOrderPath, edgeOrderPath, outputGraphPath,
                         outputRangesPath, config, nodeBlocks, edgeBlocks, rangeBlocks,
                         progressCallback);

        ValidateCookedStreaming(outputGraphPath, outputRangesPath, nodeCount, edgeCount,
                                progressCallback);

        index::CHIndexBuildData indexData;
        indexData.graphPath = outputGraphPath;
        indexData.rangesPath = outputRangesPath;
        indexData.nodeCount = nodeCount;
        indexData.edgeCount = edgeCount;
        indexData.maxLevel = maxLevel;
        indexData.spatialGridSize = config.spatialGridSize;
        indexData.nodeBlockRecordCount = config.sourceBlockNodeCount;
        indexData.edgeBlockRecordCount = config.sourceBlockEdgeCount;
        indexData.lodBandWidth = config.lodBandWidth;
        indexData.lodMaskWordsPerPage = maxLevel / 64u + 1u;
        indexData.minLatitude = minLatitude;
        indexData.minLongitude = minLongitude;
        indexData.maxLatitude = maxLatitude;
        indexData.maxLongitude = maxLongitude;
        indexData.nodeBlocks = std::move(nodeBlocks);
        indexData.graphEdgeBlocks = std::move(edgeBlocks);
        indexData.rangeEdgeBlocks = std::move(rangeBlocks);
        indexData.graphPages.reserve(builtPages.size());
        for (const auto& page : builtPages) {
            indexData.graphPages.push_back(page.page);
        }
        indexData.pageSourceBlockRefs = partitionBuilder.SourceRefs();
        indexData.pageChildSourceBlockRefs = std::move(childRefs);
        indexData.pageLodChildRefs = std::move(lodChildRefs);
        indexData.pageAliveMasks = partitionBuilder.AliveMasks();
        indexData.membershipBytesPath = membershipPath;
        indexData.membershipByteCount = partitionBuilder.MembershipByteCount();
        indexData.edgeSpatialBoundsPath = edgeSpatialBoundsPath;
        indexData.edgeSpatialBoundsCount = edgeCount;

        Report(progressCallback, FullCookStage::WriteIndex, 0, 1);
        index::CHIndex::Write(indexData, outputIndexPath);
        Report(progressCallback, FullCookStage::WriteIndex, 1, 1);

        const auto graphPageCount = static_cast<std::uint32_t>(builtPages.size());
        return {nodeCount, edgeCount, drawableCount, graphPageCount};
    } catch (...) {
        throw;
    }
}

} // namespace chmv::streaming::cook::detail
