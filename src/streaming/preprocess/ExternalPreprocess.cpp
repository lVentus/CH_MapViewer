#include "streaming/preprocess/ExternalPreprocess.h"

#include "data/ch/CHProjection.h"
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

namespace chmv::streaming::preprocess::detail {
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
};

struct DiskHierarchyRecord {
    std::uint32_t childA = data::InvalidEdgeId;
    std::uint32_t childB = data::InvalidEdgeId;
};

static_assert(sizeof(DiskHierarchyRecord) == 8u);

struct NodeSortRecord {
    std::uint64_t morton = 0;
    std::uint32_t level = 0;
    std::uint32_t oldId = 0;
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

void Report(const DatasetPreprocessor::ProgressCallback& callback, PreprocessStage stage,
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

constexpr double kWebMercatorRadius = 6378137.0;
constexpr double kWebMercatorMaxLatitude = 85.0511287798066;
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kWorldHalfExtentMeters = kPi * kWebMercatorRadius;
constexpr double kWorldExtentMeters = 2.0 * kWorldHalfExtentMeters;

data::CHProjection RuntimeProjection(double minLatitude, double minLongitude,
                                     double maxLatitude, double maxLongitude) {
    data::CHProjection projection;
    projection.centerLatitude = (minLatitude + maxLatitude) * 0.5;
    projection.centerLongitude = (minLongitude + maxLongitude) * 0.5;
    projection.longitudeScale = std::cos(projection.centerLatitude * kPi / 180.0);
    const auto halfWidth = std::max(
        (maxLongitude - minLongitude) * 0.5 * projection.longitudeScale, 1e-12);
    const auto halfHeight = std::max((maxLatitude - minLatitude) * 0.5, 1e-12);
    projection.normalization = 0.95 / std::max(halfWidth, halfHeight);
    return projection;
}

data::ProjectedPoint WorldCellBoundaryPoint(std::uint32_t x, std::uint32_t y,
                                            double cellSizeMeters,
                                            const data::CHProjection& projection) {
    const auto xMeters = std::clamp(
        static_cast<double>(x) * cellSizeMeters - kWorldHalfExtentMeters,
        -kWorldHalfExtentMeters, kWorldHalfExtentMeters);
    const auto yMeters = std::clamp(
        static_cast<double>(y) * cellSizeMeters - kWorldHalfExtentMeters,
        -kWorldHalfExtentMeters, kWorldHalfExtentMeters);
    const auto longitude = xMeters / kWebMercatorRadius * 180.0 / kPi;
    const auto latitude =
        (2.0 * std::atan(std::exp(yMeters / kWebMercatorRadius)) - kPi * 0.5) *
        180.0 / kPi;
    return projection.Project(latitude, longitude);
}

double PointSegmentDistance(data::ProjectedPoint point, data::ProjectedPoint segmentStart,
                            data::ProjectedPoint segmentEnd) {
    const auto dx = segmentEnd.x - segmentStart.x;
    const auto dy = segmentEnd.y - segmentStart.y;
    const auto lengthSquared = dx * dx + dy * dy;
    if (lengthSquared <= 1e-30) {
        return std::hypot(point.x - segmentStart.x, point.y - segmentStart.y);
    }
    const auto px = point.x - segmentStart.x;
    const auto py = point.y - segmentStart.y;
    const auto t = std::clamp((px * dx + py * dy) / lengthSquared, 0.0, 1.0);
    const auto closestX = segmentStart.x + t * dx;
    const auto closestY = segmentStart.y + t * dy;
    return std::hypot(point.x - closestX, point.y - closestY);
}

float ConservativeGeometryError(std::uint64_t packedBounds,
                                data::ProjectedPoint source,
                                data::ProjectedPoint target,
                                double cellSizeMeters,
                                const data::CHProjection& projection) {
    const auto bounds = UnpackCellBounds(packedBounds);
    const std::uint32_t xs[] = {bounds.minX, bounds.maxX + 1u};
    const std::uint32_t ys[] = {bounds.minY, bounds.maxY + 1u};
    double error = 0.0;
    for (const auto x : xs) {
        for (const auto y : ys) {
            error = std::max(error, PointSegmentDistance(
                WorldCellBoundaryPoint(x, y, cellSizeMeters, projection), source, target));
        }
    }
    return static_cast<float>(error);
}

std::uint32_t WorldGridSize(double cellSizeMeters) {
    if (!(cellSizeMeters > 0.0) || !std::isfinite(cellSizeMeters)) {
        throw std::runtime_error("spatialCellSizeMeters must be finite and positive");
    }
    const auto count = static_cast<std::uint64_t>(std::ceil(kWorldExtentMeters / cellSizeMeters));
    if (count == 0 || count > 65535u) {
        throw std::runtime_error(
            "spatialCellSizeMeters is too small for 16-bit global fixed-grid coordinates");
    }
    return static_cast<std::uint32_t>(count);
}

std::pair<std::uint32_t, std::uint32_t> WorldCell(
    double longitude, double latitude, double cellSizeMeters) {
    const auto gridSize = WorldGridSize(cellSizeMeters);
    const auto lon = std::clamp(longitude, -180.0, 180.0);
    const auto lat = std::clamp(latitude, -kWebMercatorMaxLatitude, kWebMercatorMaxLatitude);
    const auto xMeters = kWebMercatorRadius * lon * kPi / 180.0 + kWorldHalfExtentMeters;
    const auto latRadians = lat * kPi / 180.0;
    const auto yMeters = kWebMercatorRadius *
                             std::log(std::tan(kPi * 0.25 + latRadians * 0.5)) +
                         kWorldHalfExtentMeters;
    const auto toCell = [gridSize, cellSizeMeters](double meters) {
        const auto value = meters <= 0.0 ? std::uint64_t{0}
                                         : static_cast<std::uint64_t>(meters / cellSizeMeters);
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(value, static_cast<std::uint64_t>(gridSize) - 1u));
    };
    return {toCell(xMeters), toCell(yMeters)};
}

std::uint32_t Morton2D(std::uint32_t x, std::uint32_t y, std::uint32_t bits) {
    std::uint32_t result = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        result |= ((x >> bit) & 1u) << (2u * bit);
        result |= ((y >> bit) & 1u) << (2u * bit + 1u);
    }
    return result;
}

std::uint32_t PackNodeCell(std::uint32_t x, std::uint32_t y) {
    if (x >= (1u << 16u) || y >= (1u << 16u)) {
        throw std::runtime_error("fixed-grid node cell exceeds packed 16-bit coordinates");
    }
    return x | (y << 16u);
}

std::pair<std::uint32_t, std::uint32_t> UnpackNodeCell(std::uint32_t packed) {
    return {packed & 0xffffu, packed >> 16u};
}

std::uint32_t NodeMorton(std::uint32_t packedCell) {
    const auto [x, y] = UnpackNodeCell(packedCell);
    return Morton2D(x, y, 16u);
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
            throw std::runtime_error("failed to finalize preprocessed text file");
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
            throw std::runtime_error("failed to format preprocessed coordinate");
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
            throw std::runtime_error("failed to format preprocessed float");
        }
        line_.append(local, end);
    }

    template <typename T>
    void AppendField(T value, bool& first) {
        BeginField(first);
        char local[96];
        const auto result = std::to_chars(local, local + sizeof(local), value);
        if (result.ec != std::errc{}) {
            throw std::runtime_error("failed to format preprocessed text record");
        }
        line_.append(local, result.ptr);
    }

    void Write(std::string_view text) {
        if (std::fwrite(text.data(), 1, text.size(), file_) != text.size()) {
            throw std::runtime_error("failed to write preprocessed text file");
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

    void OpenExisting(const std::filesystem::path& path, std::uint64_t count) {
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
                            OPEN_EXISTING, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("could not reopen mapped temporary file");
        }
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart < static_cast<LONGLONG>(bytes)) {
            Close();
            throw std::runtime_error("mapped temporary file is truncated");
        }
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READWRITE,
                                      static_cast<DWORD>(bytes >> 32u),
                                      static_cast<DWORD>(bytes & 0xffffffffu), nullptr);
        if (!mapping_) {
            Close();
            throw std::runtime_error("could not reopen temporary file mapping");
        }
        data_ = static_cast<T*>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0,
                                              static_cast<SIZE_T>(bytes)));
        if (!data_) {
            Close();
            throw std::runtime_error("could not remap temporary file");
        }
#else
        fd_ = ::open(path.c_str(), O_RDWR, 0600);
        if (fd_ < 0) {
            Close();
            throw std::runtime_error("could not reopen mapped temporary file");
        }
        struct stat statBuffer{};
        if (fstat(fd_, &statBuffer) != 0 ||
            static_cast<std::uint64_t>(statBuffer.st_size) < bytes) {
            Close();
            throw std::runtime_error("mapped temporary file is truncated");
        }
        void* view = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd_, 0);
        if (view == MAP_FAILED) {
            data_ = nullptr;
            Close();
            throw std::runtime_error("could not remap temporary file");
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

    // Temporary arrays only need process-local coherence. Avoid synchronous whole-file
    // flushes: on multi-gigabyte preprocess jobs they turn otherwise sequential work into
    // long writeback stalls. Releasing the mapping is sufficient before reopening the file.
    void Release() { Close(); }

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

class PackedBitset {
public:
    explicit PackedBitset(std::uint64_t bitCount = 0) { Reset(bitCount); }

    void Reset(std::uint64_t bitCount) {
        bitCount_ = bitCount;
        words_.assign(static_cast<std::size_t>((bitCount + 63u) / 64u), 0u);
    }

    [[nodiscard]] bool Test(std::uint64_t bit) const {
        return bit < bitCount_ &&
               (words_[static_cast<std::size_t>(bit >> 6u)] &
                (std::uint64_t{1} << (bit & 63u))) != 0;
    }

    bool Set(std::uint64_t bit) {
        auto& word = words_[static_cast<std::size_t>(bit >> 6u)];
        const auto mask = std::uint64_t{1} << (bit & 63u);
        const bool changed = (word & mask) == 0;
        word |= mask;
        return changed;
    }

    void Clear(std::uint64_t bit) {
        words_[static_cast<std::size_t>(bit >> 6u)] &=
            ~(std::uint64_t{1} << (bit & 63u));
    }

    [[nodiscard]] std::uint64_t ByteSize() const {
        return static_cast<std::uint64_t>(words_.size()) * sizeof(std::uint64_t);
    }

private:
    std::uint64_t bitCount_ = 0;
    std::vector<std::uint64_t> words_;
};

template <typename T>
void WriteBinaryRecord(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("failed to write preprocess temporary record");
    }
}

template <typename T>
bool ReadBinaryRecord(std::istream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (input.gcount() == 0) {
        return false;
    }
    if (input.gcount() != static_cast<std::streamsize>(sizeof(T))) {
        throw std::runtime_error("truncated preprocess temporary file");
    }
    return true;
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
            throw std::runtime_error("could not open preprocess node-sort run");
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
        throw std::runtime_error("could not create merged preprocess node-sort run");
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
    const MappedArray<DiskNode>& nodes, const MappedArray<std::uint32_t>& nodeCells,
    std::uint64_t memoryBudgetBytes, const std::filesystem::path& tempDirectory,
    MappedArray<std::uint32_t>& oldToNewNode,
    const DatasetPreprocessor::ProgressCallback& progressCallback) {
    const auto chunkBudget = std::max<std::uint64_t>(sizeof(NodeSortRecord),
                                                     memoryBudgetBytes / 3u);
    const auto recordsPerRun = std::max<std::uint64_t>(
        1u, chunkBudget / static_cast<std::uint64_t>(sizeof(NodeSortRecord)));
    std::vector<std::filesystem::path> runs;
    std::vector<NodeSortRecord> chunk;
    chunk.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(recordsPerRun, nodes.size())));

    Report(progressCallback, PreprocessStage::ReorderNodes, 0, nodes.size());
    std::uint64_t runId = 0;
    for (std::uint64_t oldId = 0; oldId < nodes.size(); ++oldId) {
        const auto& node = nodes[oldId];
        chunk.push_back({NodeMorton(nodeCells[oldId]),
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
            Report(progressCallback, PreprocessStage::ReorderNodes, oldId + 1, nodes.size());
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
        throw std::runtime_error("preprocess node sort produced no output");
    }

    const auto orderPath = TempPath(tempDirectory, "node_order");
    std::ifstream sorted(runs.front(), std::ios::binary);
    std::ofstream order(orderPath, std::ios::binary | std::ios::trunc);
    if (!sorted || !order) {
        throw std::runtime_error("could not finalize preprocess node order");
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
        throw std::runtime_error("preprocess node order does not cover all nodes");
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

struct DirectSpatialCell {
    std::uint32_t level = 0; // 0 = base cell, larger = coarser power-of-two cell.
    std::uint32_t x = 0;
    std::uint32_t y = 0;
};

DirectSpatialCell DirectCellForBounds(std::uint64_t packed) {
    const auto bounds = UnpackCellBounds(packed);

    // Pick the spatial scale from the geometry span, not from dyadic boundary alignment.
    // A short road that happens to cross a large power-of-two boundary must stay short-scale;
    // otherwise e.g. two adjacent 4 km cells could incorrectly become a world-scale page.
    const auto spanX = static_cast<std::uint64_t>(bounds.maxX) - bounds.minX + 1u;
    const auto spanY = static_cast<std::uint64_t>(bounds.maxY) - bounds.minY + 1u;
    const auto span = std::max(spanX, spanY);
    const auto level = span <= 1u
                           ? 0u
                           : static_cast<std::uint32_t>(std::bit_width(span - 1u));
    const auto centerX = (static_cast<std::uint64_t>(bounds.minX) + bounds.maxX) / 2u;
    const auto centerY = (static_cast<std::uint64_t>(bounds.minY) + bounds.maxY) / 2u;
    return {level, static_cast<std::uint32_t>(centerX >> level),
            static_cast<std::uint32_t>(centerY >> level)};
}

std::uint64_t DirectBucketKey(std::uint32_t group, const DirectSpatialCell& cell) {
    if (group >= (1u << 24u) || cell.level >= 256u || cell.x >= (1u << 16u) ||
        cell.y >= (1u << 16u)) {
        throw std::runtime_error("fixed-grid preprocess bucket key exceeds packed limits");
    }
    return static_cast<std::uint64_t>(cell.x) |
           (static_cast<std::uint64_t>(cell.y) << 16u) |
           (static_cast<std::uint64_t>(cell.level) << 32u) |
           (static_cast<std::uint64_t>(group) << 40u);
}

class DirectGridPageBuilder {
public:
    DirectGridPageBuilder(const MappedArray<DiskEdge>& edges,
                          const MappedArray<std::uint64_t>& geometryBounds,
                          const std::vector<RuntimeLodGroup>& groups,
                          const std::vector<std::uint32_t>& levelToGroup,
                          const analysis::DatasetLayoutAnalysisConfig& config,
                          const std::filesystem::path& edgeOrderPath,
                          const std::filesystem::path& membershipPath,
                          const DatasetPreprocessor::ProgressCallback& progressCallback)
        : edges_(edges), geometryBounds_(geometryBounds), groups_(groups), levelToGroup_(levelToGroup), config_(config),
          edgeOrderPath_(edgeOrderPath), membershipPath_(membershipPath),
          progressCallback_(progressCallback) {
        lodMaskWordsPerPage_ = 1u;
    }

    void SetMaxLevel(std::uint32_t maxLevel) {
        maxLevel_ = maxLevel;
        lodMaskWordsPerPage_ = maxLevel / 64u + 1u;
    }

    void Build(std::uint64_t drawableCount,
               MappedArray<std::uint32_t>& oldToNewEdge,
               const std::filesystem::path& edgeRemapPath,
               const std::filesystem::path& edgeSpatialBoundsPath) {
        drawableCount_ = drawableCount;
        CountBuckets();
        AllocateLayout();

        // Build both new-EdgeID products while the source edges/bounds are already being
        // scanned sequentially for the spatial scatter.  The old implementation waited until
        // ReorderEdges had reported 100%, then reread edge_order.bin and performed a second
        // whole-graph pass with random accesses into geometryBounds.  On continent-scale data
        // that looked like a long stall at exactly 72%.  Fusing the work here removes that pass
        // without changing a single EdgeID assignment.
        oldToNewEdge.Create(edgeRemapPath, edges_.size());
        edgeSpatialBounds_.Create(edgeSpatialBoundsPath, edges_.size());
        oldToNewEdge_ = &oldToNewEdge;
        ScatterEdges();
        oldToNewEdge_ = nullptr;
        edgeSpatialBounds_.Release();

        WriteMembership();
        edgeOrder_.Release();
        membership_.close();
        if (!membership_) {
            throw std::runtime_error("failed to finalize preprocess membership stream");
        }
        Report(progressCallback_, PreprocessStage::ReorderEdges, edges_.size(), edges_.size());
    }

    [[nodiscard]] const std::vector<BuiltPage>& Pages() const { return pages_; }
    [[nodiscard]] const std::vector<index::CHIndexSpatialPageRef>& SpatialRefs() const {
        return spatialRefs_;
    }
    [[nodiscard]] const std::vector<std::uint32_t>& SourceRefs() const { return sourceRefs_; }
    [[nodiscard]] const std::vector<std::uint64_t>& AliveMasks() const { return aliveMasks_; }
    [[nodiscard]] std::uint64_t MembershipByteCount() const { return membershipByteCount_; }
    [[nodiscard]] std::uint64_t DrawableCount() const { return drawableCount_; }

private:
    // Spatial cells stay fine-grained and fixed in world space, but they are NOT physical pages.
    // Physical pages are dense storage/I/O units. Within one LOD group all exact spatial cells,
    // including cells from different direct spatial scales, are ordered by a common base-grid
    // Morton key and then packed continuously into ~16K-edge pages. Exact cell -> page refs keep
    // runtime spatial lookup precise; no artificial pack-region boundary is allowed to create a
    // mostly-empty physical page.

    struct Bucket {
        std::uint32_t group = kInvalidId;
        DirectSpatialCell cell{};
        std::uint64_t count = 0;
        std::uint64_t start = 0;
        std::uint64_t cursor = 0;
        std::uint64_t packStart = 0;
        std::uint32_t firstPage = kInvalidId;
    };

    static std::uint64_t SpatialCellKey(const DirectSpatialCell& cell) {
        if (cell.level >= 256u || cell.x >= (1u << 16u) || cell.y >= (1u << 16u)) {
            throw std::runtime_error("fixed-grid preprocess spatial key exceeds packed limits");
        }
        return static_cast<std::uint64_t>(cell.x) |
               (static_cast<std::uint64_t>(cell.y) << 16u) |
               (static_cast<std::uint64_t>(cell.level) << 32u);
    }

    static std::uint32_t BaseCenter(std::uint32_t coordinate, std::uint32_t level) {
        const auto scale = std::uint64_t{1} << level;
        const auto center = static_cast<std::uint64_t>(coordinate) * scale + scale / 2u;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(center, 0xffffu));
    }

    static std::uint32_t BaseCenterX(const Bucket& bucket) {
        return BaseCenter(bucket.cell.x, bucket.cell.level);
    }

    static std::uint32_t BaseCenterY(const Bucket& bucket) {
        return BaseCenter(bucket.cell.y, bucket.cell.level);
    }

    static bool SamePhysicalRun(const Bucket& a, const Bucket& b) {
        return a.group == b.group;
    }

    std::uint64_t KeyForEdge(std::uint32_t edgeId) const {
        const auto& edge = edges_[edgeId];
        const auto cell = DirectCellForBounds(geometryBounds_[edgeId]);
        if (edge.birth >= 0) {
            const auto level = static_cast<std::uint32_t>(edge.birth);
            if (level >= levelToGroup_.size()) {
                throw std::runtime_error("drawable edge birth level exceeds preprocess LOD groups");
            }
            const auto group = levelToGroup_[level];
            if (group == kInvalidId || group >= groups_.size()) {
                throw std::runtime_error("drawable edge has no preprocess LOD group");
            }
            return DirectBucketKey(group, cell);
        }
        return DirectBucketKey(kRefinementGroup, cell);
    }

    [[nodiscard]] std::uint64_t LayoutWorkTotal() const {
        return edges_.size() * 2u + drawableCount_;
    }

    void CountBuckets() {
        Report(progressCallback_, PreprocessStage::ReorderEdges, 0, LayoutWorkTotal());
        const auto reserveHint = static_cast<std::size_t>(std::min<std::uint64_t>(
            edges_.size() / std::max<std::uint32_t>(config_.virtualPageEdgeCount, 1u) * 8u +
                4096u,
            4'000'000u));
        buckets_.reserve(reserveHint);
        for (std::uint64_t edgeIndex = 0; edgeIndex < edges_.size(); ++edgeIndex) {
            const auto edgeId = static_cast<std::uint32_t>(edgeIndex);
            const auto key = KeyForEdge(edgeId);
            auto [it, inserted] = buckets_.try_emplace(key);
            auto& bucket = it->second;
            if (inserted) {
                bucket.cell = DirectCellForBounds(geometryBounds_[edgeId]);
                bucket.group = edges_[edgeId].birth >= 0
                                   ? levelToGroup_[static_cast<std::uint32_t>(edges_[edgeId].birth)]
                                   : kRefinementGroup;
            }
            ++bucket.count;
            if ((edgeIndex + 1u) % kProgressInterval == 0u || edgeIndex + 1u == edges_.size()) {
                Report(progressCallback_, PreprocessStage::ReorderEdges,
                       edgeIndex + 1u, LayoutWorkTotal());
            }
        }
    }

    static bool BucketOrder(const Bucket* left, const Bucket* right) {
        const bool leftRefinement = left->group == kRefinementGroup;
        const bool rightRefinement = right->group == kRefinementGroup;
        if (leftRefinement != rightRefinement) {
            return !leftRefinement;
        }
        if (left->group != right->group) {
            return left->group < right->group;
        }
        const auto leftMorton = Morton2D(BaseCenterX(*left), BaseCenterY(*left), 16u);
        const auto rightMorton = Morton2D(BaseCenterX(*right), BaseCenterY(*right), 16u);
        if (leftMorton != rightMorton) {
            return leftMorton < rightMorton;
        }
        if (left->cell.level != right->cell.level) {
            return left->cell.level < right->cell.level;
        }
        if (left->cell.y != right->cell.y) {
            return left->cell.y < right->cell.y;
        }
        return left->cell.x < right->cell.x;
    }

    void EmitPhysicalRun(std::size_t begin, std::size_t end, std::uint64_t runStart,
                         std::uint64_t runEnd) {
        if (begin >= end) {
            return;
        }
        auto& firstBucket = *orderedBuckets_[begin];
        if (firstBucket.group == kRefinementGroup) {
            for (std::size_t i = begin; i < end; ++i) {
                orderedBuckets_[i]->packStart = runStart;
            }
            return;
        }

        const auto runCount = runEnd - runStart;
        const auto firstPage = static_cast<std::uint32_t>(pages_.size());
        const auto pageCount = static_cast<std::uint32_t>(
            (runCount + config_.virtualPageEdgeCount - 1u) / config_.virtualPageEdgeCount);
        const auto& group = groups_[firstBucket.group];

        for (std::uint32_t subPage = 0; subPage < pageCount; ++subPage) {
            const auto pageFirst = runStart +
                static_cast<std::uint64_t>(subPage) * config_.virtualPageEdgeCount;
            const auto remaining = runCount -
                static_cast<std::uint64_t>(subPage) * config_.virtualPageEdgeCount;
            const auto count = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(remaining, config_.virtualPageEdgeCount));

            BuiltPage built;
            auto& page = built.page;
            page.pageId = static_cast<std::uint32_t>(pages_.size());
            page.bandIndex = group.groupIndex;
            page.levelMin = group.levelMin;
            page.levelMax = group.levelMax;
            // A dense physical page may span multiple direct spatial scales. These fields are
            // diagnostics only; exact runtime ownership comes from spatialRefs_. They are filled
            // from the buckets that actually touch this page below.
            page.spatialLevel = 0;
            page.tileX = 0;
            page.tileY = 0;
            page.subPage = subPage;
            page.edgeCount = count;
            page.rootPayloadRecordOffset = pageFirst;
            built.firstNewEdgeId = pageFirst;

            page.sourceBlockRefOffset = static_cast<std::uint32_t>(sourceRefs_.size());
            const auto firstBlock = static_cast<std::uint32_t>(
                pageFirst / config_.sourceBlockEdgeCount);
            const auto lastBlock = static_cast<std::uint32_t>(
                (pageFirst + count - 1u) / config_.sourceBlockEdgeCount);
            for (auto block = firstBlock; block <= lastBlock; ++block) {
                sourceRefs_.push_back(block);
            }
            page.sourceBlockRefCount = lastBlock - firstBlock + 1u;
            page.lodMaskOffset = static_cast<std::uint32_t>(aliveMasks_.size());
            aliveMasks_.resize(aliveMasks_.size() + lodMaskWordsPerPage_, 0);
            pages_.push_back(built);
        }

        for (std::size_t i = begin; i < end; ++i) {
            auto& bucket = *orderedBuckets_[i];
            bucket.packStart = runStart;
            bucket.firstPage = firstPage;
            const auto localFirst = bucket.start - runStart;
            const auto localLast = localFirst + bucket.count - 1u;
            const auto firstTouched = static_cast<std::uint32_t>(
                localFirst / config_.virtualPageEdgeCount);
            const auto lastTouched = static_cast<std::uint32_t>(
                localLast / config_.virtualPageEdgeCount);
            const auto spatialKey = SpatialCellKey(bucket.cell);
            for (auto localPage = firstTouched; localPage <= lastTouched; ++localPage) {
                const auto pageId = firstPage + localPage;
                spatialRefs_.push_back({spatialKey, pageId});
                auto& page = pages_[pageId].page;
                if (page.tileX == 0u && page.tileY == 0u && page.spatialLevel == 0u) {
                    page.tileX = BaseCenterX(bucket);
                    page.tileY = BaseCenterY(bucket);
                }
                page.spatialLevel = std::max(page.spatialLevel, bucket.cell.level);
            }
        }
    }

    void AllocateLayout() {
        orderedBuckets_.reserve(buckets_.size());
        for (auto& [key, bucket] : buckets_) {
            static_cast<void>(key);
            orderedBuckets_.push_back(&bucket);
        }
        std::sort(orderedBuckets_.begin(), orderedBuckets_.end(), BucketOrder);

        edgeOrder_.Create(edgeOrderPath_, edges_.size());
        membership_.open(membershipPath_, std::ios::binary | std::ios::trunc);
        if (!membership_) {
            throw std::runtime_error("could not create preprocess membership stream");
        }

        std::uint64_t nextEdge = 0;
        std::uint64_t drawableAssigned = 0;
        std::size_t runBegin = 0;
        while (runBegin < orderedBuckets_.size()) {
            std::size_t runEnd = runBegin + 1u;
            while (runEnd < orderedBuckets_.size() &&
                   SamePhysicalRun(*orderedBuckets_[runBegin], *orderedBuckets_[runEnd])) {
                ++runEnd;
            }

            const auto runStart = nextEdge;
            for (std::size_t i = runBegin; i < runEnd; ++i) {
                auto& bucket = *orderedBuckets_[i];
                bucket.start = nextEdge;
                bucket.cursor = 0;
                nextEdge += bucket.count;
                if (bucket.group != kRefinementGroup) {
                    drawableAssigned += bucket.count;
                }
            }
            EmitPhysicalRun(runBegin, runEnd, runStart, nextEdge);
            runBegin = runEnd;
        }

        if (nextEdge != edges_.size()) {
            throw std::runtime_error("direct-grid preprocess edge layout does not cover all edges");
        }
        if (drawableAssigned != drawableCount_) {
            throw std::runtime_error("direct-grid preprocess drawable edge count mismatch");
        }

        std::sort(spatialRefs_.begin(), spatialRefs_.end(), [](const auto& a, const auto& b) {
            return a.key != b.key ? a.key < b.key : a.pageId < b.pageId;
        });
        spatialRefs_.erase(std::unique(spatialRefs_.begin(), spatialRefs_.end(),
                                      [](const auto& a, const auto& b) {
                                          return a.key == b.key && a.pageId == b.pageId;
                                      }),
                           spatialRefs_.end());
    }

    void MarkAlive(std::uint32_t pageId, const DiskEdge& edge) {
        if (edge.birth < 0 || edge.death < 0) {
            return;
        }
        auto& page = pages_[pageId].page;
        const auto base = static_cast<std::size_t>(page.lodMaskOffset);
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
            return;
        }
        aliveMasks_[base + firstWord] |= ~std::uint64_t{0} << (death % 64u);
        for (auto word = firstWord + 1u; word < lastWord; ++word) {
            aliveMasks_[base + word] = ~std::uint64_t{0};
        }
        const auto highBit = birth % 64u;
        aliveMasks_[base + lastWord] |=
            highBit == 63u ? ~std::uint64_t{0}
                           : (std::uint64_t{1} << (highBit + 1u)) - 1u;
    }

    void ScatterEdges() {
        for (std::uint64_t edgeIndex = 0; edgeIndex < edges_.size(); ++edgeIndex) {
            const auto edgeId = static_cast<std::uint32_t>(edgeIndex);
            const auto key = KeyForEdge(edgeId);
            auto found = buckets_.find(key);
            if (found == buckets_.end()) {
                throw std::runtime_error("preprocess direct-grid bucket disappeared during scatter");
            }
            auto& bucket = found->second;
            const auto local = bucket.cursor++;
            if (local >= bucket.count) {
                throw std::runtime_error("preprocess direct-grid bucket cursor overflow");
            }
            const auto newId = bucket.start + local;
            edgeOrder_[newId] = edgeId;
            if (!oldToNewEdge_ || oldToNewEdge_->size() != edges_.size() ||
                edgeSpatialBounds_.size() != edges_.size()) {
                throw std::runtime_error("preprocess fused edge remap outputs are not initialized");
            }
            // edgeId advances sequentially, so the old->new remap is a sequential write.  The
            // bounds write follows the exact same newId slot as edgeOrder_, so it adds no extra
            // graph traversal and preserves byte-for-byte bounds semantics.
            (*oldToNewEdge_)[edgeId] = static_cast<std::uint32_t>(newId);
            edgeSpatialBounds_[newId] = geometryBounds_[edgeId];
            if (bucket.group != kRefinementGroup) {
                const auto pageOffset = static_cast<std::uint32_t>(
                    (newId - bucket.packStart) / config_.virtualPageEdgeCount);
                MarkAlive(bucket.firstPage + pageOffset, edges_[edgeId]);
            }
            if ((edgeIndex + 1u) % kProgressInterval == 0u || edgeIndex + 1u == edges_.size()) {
                Report(progressCallback_, PreprocessStage::ReorderEdges,
                       edges_.size() + edgeIndex + 1u, LayoutWorkTotal());
            }
        }
        for (const auto* bucket : orderedBuckets_) {
            if (bucket->cursor != bucket->count) {
                throw std::runtime_error("preprocess direct-grid bucket was not filled exactly");
            }
        }
    }

    void WriteMembership() {
        std::uint64_t processed = 0;
        std::uint64_t nextReport = kProgressInterval;
        for (auto& built : pages_) {
            auto& page = built.page;
            page.membershipByteOffset = membershipByteCount_;
            bool first = true;
            std::uint32_t previous = 0;
            for (std::uint32_t i = 0; i < page.edgeCount; ++i) {
                const auto newId = static_cast<std::uint32_t>(built.firstNewEdgeId + i);
                EncodeVarUInt32(first ? newId : newId - previous,
                                membership_, membershipByteCount_);
                previous = newId;
                first = false;
                ++processed;
                if (processed >= nextReport || processed == drawableCount_) {
                    Report(progressCallback_, PreprocessStage::ReorderEdges,
                           edges_.size() * 2u + processed, LayoutWorkTotal());
                    nextReport = processed + kProgressInterval;
                }
            }
            page.membershipByteSize = static_cast<std::uint32_t>(
                membershipByteCount_ - page.membershipByteOffset);
        }
        if (processed != drawableCount_) {
            throw std::runtime_error("preprocess membership edge count mismatch");
        }
    }

    static constexpr std::uint32_t kRefinementGroup = (1u << 24u) - 1u;

    const MappedArray<DiskEdge>& edges_;
    const MappedArray<std::uint64_t>& geometryBounds_;
    const std::vector<RuntimeLodGroup>& groups_;
    const std::vector<std::uint32_t>& levelToGroup_;
    const analysis::DatasetLayoutAnalysisConfig& config_;
    std::filesystem::path edgeOrderPath_;
    std::filesystem::path membershipPath_;
    DatasetPreprocessor::ProgressCallback progressCallback_;
    std::uint32_t maxLevel_ = 0;
    std::uint32_t lodMaskWordsPerPage_ = 1;
    std::uint64_t drawableCount_ = 0;
    MappedArray<std::uint32_t> edgeOrder_;
    MappedArray<std::uint64_t> edgeSpatialBounds_;
    MappedArray<std::uint32_t>* oldToNewEdge_ = nullptr;
    std::ofstream membership_;
    std::unordered_map<std::uint64_t, Bucket> buckets_;
    std::vector<Bucket*> orderedBuckets_;
    std::vector<BuiltPage> pages_;
    std::vector<index::CHIndexSpatialPageRef> spatialRefs_;
    std::vector<std::uint32_t> sourceRefs_;
    std::vector<std::uint64_t> aliveMasks_;
    std::uint64_t membershipByteCount_ = 0;
};

void ValidateConfig(const analysis::DatasetLayoutAnalysisConfig& config) {
    static_cast<void>(WorldGridSize(config.spatialCellSizeMeters));
    if (config.sourceBlockNodeCount == 0 || config.sourceBlockEdgeCount == 0 ||
        config.virtualPageEdgeCount == 0 || config.lodBandWidth == 0) {
        throw std::runtime_error("invalid preprocess layout configuration");
    }
}

bool HierarchyChildrenResolved(const DiskHierarchyRecord& hierarchy,
                               const PackedBitset& resolved) {
    for (const auto child : {hierarchy.childA, hierarchy.childB}) {
        if (child != data::InvalidEdgeId && !resolved.Test(child)) {
            return false;
        }
    }
    return true;
}

void ResolveOneGeometryBound(std::uint32_t edgeId,
                             const DiskHierarchyRecord& hierarchy,
                             MappedArray<std::uint64_t>& bounds,
                             PackedBitset& resolved) {
    auto merged = UnpackCellBounds(bounds[edgeId]);
    for (const auto child : {hierarchy.childA, hierarchy.childB}) {
        if (child != data::InvalidEdgeId) {
            merged = UnionBounds(merged, UnpackCellBounds(bounds[child]));
        }
    }
    bounds[edgeId] = PackCellBounds(merged);
    resolved.Set(edgeId);
}

std::uint64_t SweepGeometryBounds(const std::filesystem::path& hierarchyPath,
                                  std::uint64_t edgeCount,
                                  MappedArray<std::uint64_t>& bounds,
                                  PackedBitset& resolved,
                                  bool ascending,
                                  std::size_t recordsPerChunk,
                                  std::uint64_t& resolvedCount,
                                  const DatasetPreprocessor::ProgressCallback& progressCallback) {
    std::ifstream input(hierarchyPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not read compact shortcut hierarchy during preprocess");
    }
    std::vector<DiskHierarchyRecord> chunk(
        std::max<std::size_t>(1u, recordsPerChunk));
    std::uint64_t newlyResolved = 0;

    const auto process = [&](std::uint64_t edgeId, const DiskHierarchyRecord& hierarchy) {
        if (resolved.Test(edgeId) || !HierarchyChildrenResolved(hierarchy, resolved)) {
            return;
        }
        ResolveOneGeometryBound(static_cast<std::uint32_t>(edgeId), hierarchy, bounds, resolved);
        ++newlyResolved;
        ++resolvedCount;
    };

    if (ascending) {
        for (std::uint64_t begin = 0; begin < edgeCount; begin += chunk.size()) {
            const auto count = static_cast<std::size_t>(
                std::min<std::uint64_t>(chunk.size(), edgeCount - begin));
            input.read(reinterpret_cast<char*>(chunk.data()),
                       static_cast<std::streamsize>(count * sizeof(DiskHierarchyRecord)));
            if (input.gcount() != static_cast<std::streamsize>(count * sizeof(DiskHierarchyRecord))) {
                throw std::runtime_error("compact shortcut hierarchy is truncated");
            }
            for (std::size_t i = 0; i < count; ++i) {
                process(begin + i, chunk[i]);
            }
            Report(progressCallback, PreprocessStage::ResolveGeometryBounds,
                   resolvedCount, edgeCount);
        }
    } else {
        std::uint64_t end = edgeCount;
        while (end != 0) {
            const auto count = static_cast<std::size_t>(
                std::min<std::uint64_t>(chunk.size(), end));
            const auto begin = end - count;
            input.clear();
            input.seekg(static_cast<std::streamoff>(begin * sizeof(DiskHierarchyRecord)),
                        std::ios::beg);
            if (!input) {
                throw std::runtime_error("could not seek compact shortcut hierarchy");
            }
            input.read(reinterpret_cast<char*>(chunk.data()),
                       static_cast<std::streamsize>(count * sizeof(DiskHierarchyRecord)));
            if (input.gcount() != static_cast<std::streamsize>(count * sizeof(DiskHierarchyRecord))) {
                throw std::runtime_error("compact shortcut hierarchy is truncated");
            }
            for (std::size_t i = count; i-- > 0;) {
                process(begin + i, chunk[i]);
            }
            end = begin;
            Report(progressCallback, PreprocessStage::ResolveGeometryBounds,
                   resolvedCount, edgeCount);
        }
    }
    return newlyResolved;
}

void ResolveResidualGeometryBounds(const std::filesystem::path& hierarchyPath,
                                   std::uint64_t edgeCount,
                                   MappedArray<std::uint64_t>& bounds,
                                   PackedBitset& resolved,
                                   std::uint64_t& resolvedCount,
                                   const DatasetPreprocessor::ProgressCallback& progressCallback) {
    MappedArray<DiskHierarchyRecord> hierarchy;
    hierarchy.OpenExisting(hierarchyPath, edgeCount);
    PackedBitset visiting(edgeCount);

    struct Frame {
        std::uint32_t edgeId = 0;
        std::uint8_t nextChild = 0;
    };
    std::vector<Frame> stack;
    stack.reserve(256);

    for (std::uint64_t rootIndex = 0; rootIndex < edgeCount; ++rootIndex) {
        if (resolved.Test(rootIndex)) {
            continue;
        }
        stack.push_back({static_cast<std::uint32_t>(rootIndex), 0});
        while (!stack.empty()) {
            auto& frame = stack.back();
            const auto edgeId = frame.edgeId;
            if (resolved.Test(edgeId)) {
                visiting.Clear(edgeId);
                stack.pop_back();
                continue;
            }
            if (!visiting.Test(edgeId)) {
                visiting.Set(edgeId);
            }

            const auto& record = hierarchy[edgeId];
            bool descended = false;
            while (frame.nextChild < 2u) {
                const auto child = frame.nextChild++ == 0u ? record.childA : record.childB;
                if (child == data::InvalidEdgeId || resolved.Test(child)) {
                    continue;
                }
                if (child >= edgeCount) {
                    throw std::runtime_error(
                        "shortcut child id out of range during compact geometry resolve");
                }
                if (visiting.Test(child)) {
                    throw std::runtime_error(
                        "cycle detected in shortcut hierarchy during external preprocess");
                }
                stack.push_back({child, 0});
                descended = true;
                break;
            }
            if (descended) {
                continue;
            }

            if (!HierarchyChildrenResolved(record, resolved)) {
                throw std::runtime_error(
                    "shortcut geometry dependency did not resolve during external preprocess");
            }
            ResolveOneGeometryBound(edgeId, record, bounds, resolved);
            visiting.Clear(edgeId);
            ++resolvedCount;
            stack.pop_back();
            if (resolvedCount % kProgressInterval == 0 || resolvedCount == edgeCount) {
                Report(progressCallback, PreprocessStage::ResolveGeometryBounds,
                       resolvedCount, edgeCount);
            }
        }
    }
    hierarchy.Release();
}

void ResolveGeometryBounds(const std::filesystem::path& hierarchyPath,
                           std::uint64_t edgeCount,
                           MappedArray<std::uint64_t>& bounds,
                           PackedBitset& resolved,
                           std::uint64_t initialResolvedCount,
                           std::uint64_t memoryBudgetBytes,
                           std::uint64_t backwardReferenceCount,
                           std::uint64_t forwardReferenceCount,
                           const DatasetPreprocessor::ProgressCallback& progressCallback) {
    std::uint64_t resolvedCount = initialResolvedCount;
    Report(progressCallback, PreprocessStage::ResolveGeometryBounds,
           resolvedCount, edgeCount);
    if (resolvedCount == edgeCount) {
        return;
    }

    const auto sweepBudget = std::clamp<std::uint64_t>(
        memoryBudgetBytes / 64u, 8ull << 20u, 128ull << 20u);
    const auto recordsPerChunk = static_cast<std::size_t>(std::max<std::uint64_t>(
        1u, sweepBudget / sizeof(DiskHierarchyRecord)));

    std::fprintf(stdout,
                 "Shortcut bounds: %llu/%llu edges resolved in source pass; "
                 "child refs backward=%llu forward=%llu. Using compact sequential resolver.\n"
                 "Shortcut bounds working set: %.1f MiB bounds + %.1f MiB resolved bits; "
                 "sweep buffer %.1f MiB (full edge table is unmapped).\n",
                 static_cast<unsigned long long>(resolvedCount),
                 static_cast<unsigned long long>(edgeCount),
                 static_cast<unsigned long long>(backwardReferenceCount),
                 static_cast<unsigned long long>(forwardReferenceCount),
                 static_cast<double>(edgeCount * sizeof(std::uint64_t)) / (1024.0 * 1024.0),
                 static_cast<double>(resolved.ByteSize()) / (1024.0 * 1024.0),
                 static_cast<double>(recordsPerChunk * sizeof(DiskHierarchyRecord)) /
                     (1024.0 * 1024.0));
    std::fflush(stdout);

    // The source pass already resolves the normal CH case (children before parent). Residual
    // dependencies are swept in the direction most likely to satisfy them first. Alternating
    // large sequential sweeps resolves mixed-ID DAGs without random access. A bounded number of
    // pairs prevents a pathological ordering from turning into O(depth * E) I/O; any exact
    // residual is finished by the compact 8-byte hierarchy DFS below.
    constexpr std::uint32_t kMaxSweepPairs = 4u;
    bool descendingFirst = forwardReferenceCount > backwardReferenceCount;
    for (std::uint32_t pair = 0; pair < kMaxSweepPairs && resolvedCount < edgeCount; ++pair) {
        std::uint64_t pairProgress = 0;
        if (descendingFirst) {
            pairProgress += SweepGeometryBounds(hierarchyPath, edgeCount, bounds, resolved,
                                                false, recordsPerChunk, resolvedCount,
                                                progressCallback);
            if (resolvedCount < edgeCount) {
                pairProgress += SweepGeometryBounds(hierarchyPath, edgeCount, bounds, resolved,
                                                    true, recordsPerChunk, resolvedCount,
                                                    progressCallback);
            }
        } else {
            pairProgress += SweepGeometryBounds(hierarchyPath, edgeCount, bounds, resolved,
                                                true, recordsPerChunk, resolvedCount,
                                                progressCallback);
            if (resolvedCount < edgeCount) {
                pairProgress += SweepGeometryBounds(hierarchyPath, edgeCount, bounds, resolved,
                                                    false, recordsPerChunk, resolvedCount,
                                                    progressCallback);
            }
        }
        std::fprintf(stdout,
                     "Shortcut bounds sweep pair %u: +%llu resolved, %llu remaining.\n",
                     pair + 1u,
                     static_cast<unsigned long long>(pairProgress),
                     static_cast<unsigned long long>(edgeCount - resolvedCount));
        std::fflush(stdout);
        if (pairProgress == 0) {
            break;
        }
    }

    if (resolvedCount != edgeCount) {
        std::fprintf(stdout,
                     "Shortcut bounds: %llu residual edges require exact compact DFS.\n",
                     static_cast<unsigned long long>(edgeCount - resolvedCount));
        std::fflush(stdout);
        ResolveResidualGeometryBounds(hierarchyPath, edgeCount, bounds, resolved,
                                      resolvedCount, progressCallback);
    }
    if (resolvedCount != edgeCount) {
        throw std::runtime_error("shortcut geometry resolver did not cover every edge");
    }
    Report(progressCallback, PreprocessStage::ResolveGeometryBounds, edgeCount, edgeCount);
}


void WritePreprocessedFiles(const std::filesystem::path& sourceGraphPath,
                      std::uint64_t nodeSectionOffset,
                      const MappedArray<DiskNode>& nodes,
                      const MappedArray<DiskEdge>& edges,
                      const MappedArray<std::uint32_t>& oldToNewNode,
                      const MappedArray<std::uint32_t>& oldToNewEdge,
                      const MappedArray<std::uint64_t>& edgeSpatialBounds,
                      std::uint64_t drawableCount,
                      double minLatitude, double minLongitude,
                      double maxLatitude, double maxLongitude,
                      const std::filesystem::path& nodeOrderPath,
                      const std::filesystem::path& edgeOrderPath,
                      const std::filesystem::path& rootPayloadPath,
                      const std::filesystem::path& graphOutputPath,
                      const std::filesystem::path& rangesOutputPath,
                      const analysis::DatasetLayoutAnalysisConfig& config,
                      std::vector<index::CHIndexSourceBlock>& nodeBlocks,
                      std::vector<index::CHIndexSourceBlock>& edgeBlocks,
                      std::vector<index::CHIndexSourceBlock>& rangeBlocks,
                      const DatasetPreprocessor::ProgressCallback& progressCallback) {
    TextWriter graphWriter(graphOutputPath);
    graphWriter.Raw(ReadPrefix(sourceGraphPath, nodeSectionOffset));
    std::ifstream nodeOrder(nodeOrderPath, std::ios::binary);
    if (!nodeOrder) {
        throw std::runtime_error("could not read preprocess node order");
    }
    Report(progressCallback, PreprocessStage::WriteGraph, 0, nodes.size() + edges.size());
    std::uint32_t oldNodeId = 0;
    for (std::uint64_t newId = 0; newId < nodes.size(); ++newId) {
        if (!ReadBinaryRecord(nodeOrder, oldNodeId)) {
            throw std::runtime_error("preprocess node order is truncated");
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
            Report(progressCallback, PreprocessStage::WriteGraph, newId + 1,
                   nodes.size() + edges.size());
        }
    }

    std::ifstream edgeOrder(edgeOrderPath, std::ios::binary);
    if (!edgeOrder) {
        throw std::runtime_error("could not read preprocess edge order");
    }
    std::ofstream rootPayload(rootPayloadPath, std::ios::binary | std::ios::trunc);
    if (!rootPayload) {
        throw std::runtime_error("could not create preprocess root payload");
    }
    const auto projection = RuntimeProjection(minLatitude, minLongitude, maxLatitude, maxLongitude);
    std::uint32_t oldEdgeId = 0;
    for (std::uint64_t newId = 0; newId < edges.size(); ++newId) {
        if (!ReadBinaryRecord(edgeOrder, oldEdgeId)) {
            throw std::runtime_error("preprocess edge order is truncated");
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

        if (newId < drawableCount) {
            if (edge.birth < 0 || edge.death < 0 || newId >= edgeSpatialBounds.size()) {
                throw std::runtime_error("drawable root payload is inconsistent with reordered edges");
            }
            const auto packedBounds = edgeSpatialBounds[newId];
            const auto source = projection.Project(nodes[edge.source].latitude,
                                                   nodes[edge.source].longitude);
            const auto target = projection.Project(nodes[edge.target].latitude,
                                                   nodes[edge.target].longitude);
            index::CHIndexRootRecord record;
            record.globalEdgeId = static_cast<std::uint32_t>(newId);
            record.birthLevel = edge.birth;
            record.deathLevel = edge.death;
            record.boundsLo = static_cast<std::uint32_t>(packedBounds & 0xffffffffull);
            record.boundsHi = static_cast<std::uint32_t>(packedBounds >> 32u);
            const auto maxPackedCoordinate = std::max({
                record.boundsLo & 0xffffu, (record.boundsLo >> 16u) & 0xffffu,
                record.boundsHi & 0xffffu, (record.boundsHi >> 16u) & 0xffffu});
            if (maxPackedCoordinate > index::kPackedSpatialCoordinateMask) {
                throw std::runtime_error(
                    "road-style root metadata requires a fixed grid below 16384 cells per axis");
            }
            index::EncodeRootRoadStyle(
                record.boundsLo, record.boundsHi, data::RoadStyleIndexFromType(edge.type));
            record.childA = edge.childA == data::InvalidEdgeId
                                ? data::InvalidEdgeId
                                : oldToNewEdge[edge.childA];
            record.childB = edge.childB == data::InvalidEdgeId
                                ? data::InvalidEdgeId
                                : oldToNewEdge[edge.childB];
            record.geometryError =
                (record.childA == data::InvalidEdgeId || record.childB == data::InvalidEdgeId)
                    ? 0.0f
                    : ConservativeGeometryError(packedBounds, source, target,
                                                config.spatialCellSizeMeters, projection);
            record.sourceX = static_cast<float>(source.x);
            record.sourceY = static_cast<float>(source.y);
            record.targetX = static_cast<float>(target.x);
            record.targetY = static_cast<float>(target.y);
            WriteBinaryRecord(rootPayload, record);
        }
        if ((newId + 1) % kProgressInterval == 0 || newId + 1 == edges.size()) {
            Report(progressCallback, PreprocessStage::WriteGraph,
                   nodes.size() + newId + 1, nodes.size() + edges.size());
        }
    }
    graphWriter.Close();
    rootPayload.close();
    if (!rootPayload) {
        throw std::runtime_error("failed to finalize preprocess root payload");
    }

    TextWriter rangeWriter(rangesOutputPath);
    edgeOrder.clear();
    edgeOrder.seekg(0, std::ios::beg);
    Report(progressCallback, PreprocessStage::WriteRanges, 0, edges.size());
    for (std::uint64_t newId = 0; newId < edges.size(); ++newId) {
        if (!ReadBinaryRecord(edgeOrder, oldEdgeId)) {
            throw std::runtime_error("preprocess edge order is truncated while writing ranges");
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
            Report(progressCallback, PreprocessStage::WriteRanges, newId + 1, edges.size());
        }
    }
    rangeWriter.Close();
}

void ValidatePreprocessedStreaming(const std::filesystem::path& graphPath,
                             const std::filesystem::path& rangesPath,
                             std::uint64_t expectedNodes, std::uint64_t expectedEdges,
                             const DatasetPreprocessor::ProgressCallback& progressCallback) {
    Report(progressCallback, PreprocessStage::ValidateOutput, 0,
           expectedNodes + expectedEdges * 2u);
    TextSourceScanner graph(graphPath);
    const auto nodeCount = graph.Read<std::uint64_t>();
    const auto edgeCount = graph.Read<std::uint64_t>();
    if (nodeCount != expectedNodes || edgeCount != expectedEdges) {
        throw std::runtime_error("external preprocess changed graph counts");
    }
    for (std::uint64_t i = 0; i < nodeCount; ++i) {
        const auto id = graph.Read<std::uint32_t>();
        if (id != i) {
            throw std::runtime_error("external preprocess node ids are not sequential");
        }
        graph.Read<std::uint64_t>();
        graph.Read<double>();
        graph.Read<double>();
        graph.Read<float>();
        graph.Read<std::uint32_t>();
        if ((i + 1) % kProgressInterval == 0) {
            Report(progressCallback, PreprocessStage::ValidateOutput, i + 1,
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
            throw std::runtime_error("external preprocess produced an invalid graph reference");
        }
    }
    TextSourceScanner ranges(rangesPath);
    for (std::uint64_t record = 0; record < edgeCount; ++record) {
        const auto edgeId = ranges.Read<std::uint32_t>();
        const auto birth = ranges.Read<std::int32_t>();
        const auto death = ranges.Read<std::int32_t>();
        if (edgeId != record || !((birth == -1 && death == -1) ||
                                  (birth >= 0 && death >= 0 && death <= birth))) {
            throw std::runtime_error("external preprocess produced an invalid range record");
        }
        if ((record + 1) % kProgressInterval == 0 || record + 1 == edgeCount) {
            Report(progressCallback, PreprocessStage::ValidateOutput,
                   nodeCount + edgeCount + record + 1,
                   nodeCount + edgeCount * 2u);
        }
    }
}

} // namespace

ExternalPreprocessResult RunExternalPreprocess(
    const std::filesystem::path& sourceGraphPath,
    const std::filesystem::path& sourceRangesPath,
    const std::filesystem::path& outputGraphPath,
    const std::filesystem::path& outputRangesPath,
    const std::filesystem::path& outputIndexPath,
    const analysis::DatasetLayoutAnalysisConfig& config,
    std::uint64_t memoryBudgetBytes,
    const std::filesystem::path& tempRoot,
    bool strictOutputValidation,
    const DatasetPreprocessor::ProgressCallback& progressCallback) {
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
            throw std::runtime_error("external preprocess requires non-empty 32-bit-id graph data");
        }

        const auto runtimeGridSize = WorldGridSize(config.spatialCellSizeMeters);

        const auto nodesPath = TempPath(tempRoot, "nodes");
        const auto nodeCellsPath = TempPath(tempRoot, "node_cells");
        const auto edgesPath = TempPath(tempRoot, "edges");
        const auto nodeRemapPath = TempPath(tempRoot, "node_remap");
        const auto edgeRemapPath = TempPath(tempRoot, "edge_remap");
        const auto hierarchyPath = TempPath(tempRoot, "hierarchy_children");
        const auto geometryBoundsPath = TempPath(tempRoot, "geometry_bounds");

        MappedArray<DiskNode> nodes;
        MappedArray<std::uint32_t> nodeCells;
        MappedArray<DiskEdge> edges;
        MappedArray<std::uint32_t> oldToNewNode;
        MappedArray<std::uint32_t> oldToNewEdge;
        MappedArray<DiskHierarchyRecord> hierarchy;
        MappedArray<std::uint64_t> geometryBounds;
        nodes.Create(nodesPath, nodeCount);
        nodeCells.Create(nodeCellsPath, nodeCount);
        edges.Create(edgesPath, edgeCount);
        oldToNewNode.Create(nodeRemapPath, nodeCount);
        hierarchy.Create(hierarchyPath, edgeCount);
        geometryBounds.Create(geometryBoundsPath, edgeCount);

        Report(progressCallback, PreprocessStage::AnalyzeSource, 0, nodeCount + edgeCount);
        double minLatitude = std::numeric_limits<double>::infinity();
        double maxLatitude = -std::numeric_limits<double>::infinity();
        double minLongitude = std::numeric_limits<double>::infinity();
        double maxLongitude = -std::numeric_limits<double>::infinity();
        std::uint64_t nodeSectionOffset = 0;
        for (std::uint64_t record = 0; record < nodeCount; ++record) {
            const auto [id, offset] = graph.ReadWithOffset<std::uint32_t>();
            if (id != record) {
                throw std::runtime_error(
                    "external preprocess currently requires source node records to be ID ordered");
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
            const auto [cellX, cellY] = WorldCell(
                node.longitude, node.latitude, config.spatialCellSizeMeters);
            nodeCells[id] = PackNodeCell(cellX, cellY);
            minLatitude = std::min(minLatitude, node.latitude);
            maxLatitude = std::max(maxLatitude, node.latitude);
            minLongitude = std::min(minLongitude, node.longitude);
            maxLongitude = std::max(maxLongitude, node.longitude);
            if ((record + 1) % kProgressInterval == 0 || record + 1 == nodeCount) {
                Report(progressCallback, PreprocessStage::AnalyzeSource, record + 1,
                       nodeCount + edgeCount);
            }
        }

        TextSourceScanner rangeScanner(sourceRangesPath);
        std::vector<std::uint64_t> birthCounts(1, 0);
        std::uint32_t maxLevel = 0;
        std::uint64_t drawableCount = 0;
        PackedBitset resolvedBounds(edgeCount);
        std::uint64_t resolvedBoundsCount = 0;
        std::uint64_t backwardReferenceCount = 0;
        std::uint64_t forwardReferenceCount = 0;
        // Parse the graph edge section and .ranges together. Every edge keeps its exact source
        // record for later remapping, while geometry dependency data is mirrored into compact
        // 8-byte child and 8-byte bounds arrays. This lets the expensive resolver unmap the
        // ~48-byte DiskEdge table entirely instead of random-walking it out of core.
        Report(progressCallback, PreprocessStage::LoadSource, 0, edgeCount);
        for (std::uint64_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
            const auto rangeEdgeId = rangeScanner.Read<std::uint32_t>();
            const auto birth = rangeScanner.Read<std::int32_t>();
            const auto death = rangeScanner.Read<std::int32_t>();
            if (rangeEdgeId != edgeIndex) {
                throw std::runtime_error(
                    "external preprocess currently requires range records to be EdgeID ordered");
            }
            if (!((birth == -1 && death == -1) ||
                  (birth >= 0 && death >= 0 && death <= birth))) {
                throw std::runtime_error("invalid lifetime range in preprocess source");
            }

            DiskEdge edge;
            edge.source = graph.Read<std::uint32_t>();
            edge.target = graph.Read<std::uint32_t>();
            edge.weight = graph.Read<float>();
            edge.type = graph.Read<std::int32_t>();
            edge.maxSpeed = graph.Read<std::int32_t>();
            edge.childA = DecodeChild(graph.Read<std::int64_t>());
            edge.childB = DecodeChild(graph.Read<std::int64_t>());
            if (edge.source >= nodeCount || edge.target >= nodeCount) {
                throw std::runtime_error("edge endpoint out of range during external preprocess");
            }
            edge.birth = birth;
            edge.death = death;

            if (birth >= 0) {
                const auto level = static_cast<std::uint32_t>(birth);
                if (birthCounts.size() <= level) {
                    birthCounts.resize(static_cast<std::size_t>(level) + 1u, 0);
                }
                ++birthCounts[level];
                maxLevel = std::max(maxLevel, level);
                ++drawableCount;
            }

            const auto [sourceX, sourceY] = UnpackNodeCell(nodeCells[edge.source]);
            const auto [targetX, targetY] = UnpackNodeCell(nodeCells[edge.target]);
            auto bounds = CellBounds{std::min(sourceX, targetX), std::min(sourceY, targetY),
                                     std::max(sourceX, targetX), std::max(sourceY, targetY)};
            bool ready = true;
            for (const auto child : {edge.childA, edge.childB}) {
                if (child == data::InvalidEdgeId) {
                    continue;
                }
                if (child >= edgeCount) {
                    throw std::runtime_error(
                        "shortcut child id out of range during external preprocess");
                }
                if (child < edgeIndex) {
                    ++backwardReferenceCount;
                } else {
                    ++forwardReferenceCount;
                }
                if (child >= edgeIndex || !resolvedBounds.Test(child)) {
                    ready = false;
                }
            }
            if (ready) {
                for (const auto child : {edge.childA, edge.childB}) {
                    if (child != data::InvalidEdgeId) {
                        bounds = UnionBounds(bounds, UnpackCellBounds(geometryBounds[child]));
                    }
                }
            }
            const auto packedBounds = PackCellBounds(bounds);
            geometryBounds[edgeIndex] = packedBounds;
            hierarchy[edgeIndex] = {edge.childA, edge.childB};
            edges[edgeIndex] = edge;
            if (ready) {
                resolvedBounds.Set(edgeIndex);
                ++resolvedBoundsCount;
            }
            if ((edgeIndex + 1) % kProgressInterval == 0 || edgeIndex + 1 == edgeCount) {
                Report(progressCallback, PreprocessStage::LoadSource, edgeIndex + 1, edgeCount);
            }
        }
        birthCounts.resize(static_cast<std::size_t>(maxLevel) + 1u, 0);

        // Nothing in node sorting needs the full edge table or compact child table resident.
        // Unmapping them here is critical on continent-scale data: Windows can immediately
        // reclaim those pages instead of carrying a 48-byte/edge working set into the resolver.
        hierarchy.Release();
        edges.Release();

        const auto nodeOrderPath = BuildNodeOrder(
            nodes, nodeCells, memoryBudgetBytes, tempRoot, oldToNewNode, progressCallback);

        // Node data/remap are needed again only while writing the final .sch. Release all three
        // mappings during geometry resolution so the compact 8-byte bounds table owns the RAM.
        nodes.Release();
        nodeCells.Release();
        oldToNewNode.Release();

        ResolveGeometryBounds(hierarchyPath, edgeCount, geometryBounds, resolvedBounds,
                              resolvedBoundsCount, memoryBudgetBytes,
                              backwardReferenceCount, forwardReferenceCount, progressCallback);

        // Reopen source edge metadata only after the hierarchy is fully resolved. Runtime page
        // construction reads exact geometry bounds from the compact array, not DiskEdge.bounds.
        edges.OpenExisting(edgesPath, edgeCount);

        const auto groups = BuildRuntimeLodGroups(birthCounts, maxLevel,
                                                  config.virtualPageEdgeCount);
        std::vector<std::uint32_t> levelToGroup(static_cast<std::size_t>(maxLevel) + 1u,
                                                kInvalidId);
        for (const auto& group : groups) {
            for (std::uint32_t level = group.levelMin; level <= group.levelMax; ++level) {
                levelToGroup[level] = group.groupIndex;
            }
        }

        const auto edgeOrderPath = TempPath(tempRoot, "edge_order");
        const auto membershipPath = TempPath(tempRoot, "membership");
        const auto edgeSpatialBoundsPath = TempPath(tempRoot, "edge_spatial_bounds");
        DirectGridPageBuilder pageBuilder(edges, geometryBounds, groups, levelToGroup, config,
                                          edgeOrderPath, membershipPath, progressCallback);
        pageBuilder.SetMaxLevel(maxLevel);
        pageBuilder.Build(drawableCount, oldToNewEdge, edgeRemapPath, edgeSpatialBoundsPath);

        // The persistent GPU runtime requests hierarchy backing blocks from actual shader
        // misses. Speculative per-page child-block and next-LOD page lists only add another
        // O(E) preprocessing pass and large temporary arrays, so fixed-grid preprocessing
        // deliberately leaves these optional legacy prefetch sections empty.
        auto builtPages = pageBuilder.Pages();

        // Geometry bounds have now been persisted in new-EdgeID order for .chidx. The original
        // hierarchy-order mapping can be released; reopen only the reordered compact bounds while
        // writing the binary root payload so continent-scale preprocessing never holds both.
        geometryBounds.Release();
        MappedArray<std::uint64_t> newEdgeSpatialBounds;
        newEdgeSpatialBounds.OpenExisting(edgeSpatialBoundsPath, edgeCount);
        nodes.OpenExisting(nodesPath, nodeCount);
        oldToNewNode.OpenExisting(nodeRemapPath, nodeCount);

        const auto rootPayloadPath = TempPath(tempRoot, "root_payload");
        std::vector<index::CHIndexSourceBlock> nodeBlocks;
        std::vector<index::CHIndexSourceBlock> edgeBlocks;
        std::vector<index::CHIndexSourceBlock> rangeBlocks;
        WritePreprocessedFiles(sourceGraphPath, nodeSectionOffset, nodes, edges, oldToNewNode,
                         oldToNewEdge, newEdgeSpatialBounds, drawableCount,
                         minLatitude, minLongitude, maxLatitude, maxLongitude,
                         nodeOrderPath, edgeOrderPath, rootPayloadPath, outputGraphPath,
                         outputRangesPath, config, nodeBlocks, edgeBlocks, rangeBlocks,
                         progressCallback);
        newEdgeSpatialBounds.Release();

        if (strictOutputValidation) {
            ValidatePreprocessedStreaming(outputGraphPath, outputRangesPath, nodeCount, edgeCount,
                                          progressCallback);
        } else {
            Report(progressCallback, PreprocessStage::ValidateOutput, 1, 1);
        }

        index::CHIndexBuildData indexData;
        indexData.graphPath = outputGraphPath;
        indexData.rangesPath = outputRangesPath;
        indexData.nodeCount = nodeCount;
        indexData.edgeCount = edgeCount;
        indexData.maxLevel = maxLevel;
        indexData.spatialGridSize = runtimeGridSize;
        indexData.spatialCellSizeMeters = config.spatialCellSizeMeters;
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
        indexData.spatialPageRefs = pageBuilder.SpatialRefs();
        indexData.pageSourceBlockRefs = pageBuilder.SourceRefs();
        indexData.pageAliveMasks = pageBuilder.AliveMasks();
        indexData.membershipBytesPath = membershipPath;
        indexData.membershipByteCount = pageBuilder.MembershipByteCount();
        indexData.edgeSpatialBoundsPath = edgeSpatialBoundsPath;
        indexData.edgeSpatialBoundsCount = edgeCount;
        indexData.rootPayloadPath = rootPayloadPath;
        indexData.rootPayloadRecordCount = drawableCount;

        Report(progressCallback, PreprocessStage::WriteIndex, 0, 1);
        index::CHIndex::Write(indexData, outputIndexPath);
        Report(progressCallback, PreprocessStage::WriteIndex, 1, 1);

        const auto graphPageCount = static_cast<std::uint32_t>(builtPages.size());
        return {nodeCount, edgeCount, drawableCount, graphPageCount};
    } catch (...) {
        throw;
    }
}

} // namespace chmv::streaming::preprocess::detail
