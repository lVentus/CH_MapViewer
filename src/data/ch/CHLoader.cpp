#include "data/ch/CHLoader.h"

#include <charconv>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace chmv::data {
namespace {

class TextScanner {
public:
    explicit TextScanner(const std::filesystem::path& path)
        : buffer_(BufferSize) {
#ifdef _WIN32
        _wfopen_s(&file_, path.c_str(), L"rb");
#else
        file_ = std::fopen(path.c_str(), "rb");
#endif
        if (!file_) {
            throw std::runtime_error("could not open file: " + path.string());
        }
    }

    ~TextScanner() {
        if (file_) {
            std::fclose(file_);
        }
    }

    TextScanner(const TextScanner&) = delete;
    TextScanner& operator=(const TextScanner&) = delete;

    template <typename T>
    T Read() {
        const auto token = NextToken();
        T value{};

        if constexpr (std::is_integral_v<T>) {
            const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
            if (ec != std::errc{} || ptr != token.data() + token.size()) {
                throw std::runtime_error("invalid integer token");
            }
        } else {
            const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
            if (ec != std::errc{} || ptr != token.data() + token.size()) {
                throw std::runtime_error("invalid floating-point token");
            }
        }

        return value;
    }

private:
    std::string NextToken() {
        std::string token;
        char ch = 0;

        while (ReadChar(ch)) {
            if (ch == '#') {
                SkipLine();
                continue;
            }
            if (!IsSpace(ch)) {
                token.push_back(ch);
                break;
            }
        }

        if (token.empty()) {
            throw std::runtime_error("unexpected end of file");
        }

        while (ReadChar(ch)) {
            if (IsSpace(ch)) {
                break;
            }
            token.push_back(ch);
        }

        return token;
    }

    bool ReadChar(char& ch) {
        if (position_ == size_) {
            size_ = std::fread(buffer_.data(), 1, buffer_.size(), file_);
            position_ = 0;
            if (size_ == 0) {
                return false;
            }
        }

        ch = buffer_[position_++];
        return true;
    }

    void SkipLine() {
        char ch = 0;
        while (ReadChar(ch) && ch != '\n') {
        }
    }

    static bool IsSpace(char ch) {
        return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
    }

    static constexpr std::size_t BufferSize = 1 << 20;
    std::FILE* file_ = nullptr;
    std::vector<char> buffer_;
    std::size_t position_ = 0;
    std::size_t size_ = 0;
};

std::uint32_t DecodeChild(std::int64_t value) {
    if (value < 0) {
        return InvalidEdgeId;
    }
    if (value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("edge child id does not fit in uint32");
    }
    return static_cast<std::uint32_t>(value);
}

constexpr std::uint64_t kProgressInterval = 1u << 14;

void ReportProgress(const CHLoader::ProgressCallback& callback, CHLoadStage stage,
                    std::uint64_t current, std::uint64_t total, std::uint64_t completedBefore,
                    std::uint64_t overallTotal) {
    if (!callback) {
        return;
    }

    const auto overallCurrent = completedBefore + current;
    callback({
        stage,
        current,
        total,
        overallTotal == 0
            ? 1.0f
            : static_cast<float>(static_cast<double>(overallCurrent) /
                                 static_cast<double>(overallTotal)),
    });
}

} // namespace

CHGraph CHLoader::Load(const std::filesystem::path& graphPath,
                       const std::filesystem::path& rangesPath,
                       ProgressCallback progressCallback) {
    TextScanner graphScanner(graphPath);

    const auto nodeCount = graphScanner.Read<std::uint64_t>();
    const auto edgeCount = graphScanner.Read<std::uint64_t>();

    if (nodeCount > std::numeric_limits<std::uint32_t>::max() ||
        edgeCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("graph is too large for the current 32-bit id representation");
    }

    const auto overallTotal = nodeCount + edgeCount + edgeCount;

    CHGraph graph;
    graph.Nodes().resize(static_cast<std::size_t>(nodeCount));
    graph.Edges().resize(static_cast<std::size_t>(edgeCount));

    ReportProgress(progressCallback, CHLoadStage::Nodes, 0, nodeCount, 0, overallTotal);
    for (std::uint64_t i = 0; i < nodeCount; ++i) {
        const auto id = graphScanner.Read<std::uint32_t>();
        if (id >= nodeCount) {
            throw std::runtime_error("node id out of range");
        }

        CHNode node;
        node.osmId = graphScanner.Read<std::uint64_t>();
        node.latitude = graphScanner.Read<double>();
        node.longitude = graphScanner.Read<double>();
        node.elevation = graphScanner.Read<float>();
        node.level = graphScanner.Read<std::uint32_t>();
        graph.Nodes()[id] = node;
        if ((i + 1) % kProgressInterval == 0 || i + 1 == nodeCount) {
            ReportProgress(progressCallback, CHLoadStage::Nodes, i + 1, nodeCount, 0, overallTotal);
        }
    }

    ReportProgress(progressCallback, CHLoadStage::Edges, 0, edgeCount, nodeCount, overallTotal);
    for (std::uint64_t edgeId = 0; edgeId < edgeCount; ++edgeId) {
        CHEdge edge;
        edge.source = graphScanner.Read<std::uint32_t>();
        edge.target = graphScanner.Read<std::uint32_t>();
        edge.weight = graphScanner.Read<float>();
        edge.type = graphScanner.Read<std::int32_t>();
        edge.maxSpeed = graphScanner.Read<std::int32_t>();
        edge.childA = DecodeChild(graphScanner.Read<std::int64_t>());
        edge.childB = DecodeChild(graphScanner.Read<std::int64_t>());

        if (edge.source >= nodeCount || edge.target >= nodeCount) {
            throw std::runtime_error("edge endpoint out of range");
        }
        graph.Edges()[static_cast<std::size_t>(edgeId)] = edge;
        if ((edgeId + 1) % kProgressInterval == 0 || edgeId + 1 == edgeCount) {
            ReportProgress(progressCallback, CHLoadStage::Edges, edgeId + 1, edgeCount, nodeCount,
                           overallTotal);
        }
    }

    TextScanner rangeScanner(rangesPath);
    graph.Ranges().assign(static_cast<std::size_t>(edgeCount), EdgeRange{});

    ReportProgress(progressCallback, CHLoadStage::Ranges, 0, edgeCount, nodeCount + edgeCount,
                   overallTotal);
    for (std::uint64_t i = 0; i < edgeCount; ++i) {
        const auto edgeId = rangeScanner.Read<std::uint32_t>();
        if (edgeId >= edgeCount) {
            throw std::runtime_error("range edge id out of range");
        }

        EdgeRange range;
        range.birthLevel = rangeScanner.Read<std::int32_t>();
        range.deathLevel = rangeScanner.Read<std::int32_t>();
        graph.Ranges()[edgeId] = range;
        if ((i + 1) % kProgressInterval == 0 || i + 1 == edgeCount) {
            ReportProgress(progressCallback, CHLoadStage::Ranges, i + 1, edgeCount,
                           nodeCount + edgeCount, overallTotal);
        }
    }

    return graph;
}

} // namespace chmv::data
