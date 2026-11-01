#include "streaming/lod/LodPageHierarchy.h"

#include "streaming/lod/LodPageIo.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace chmv::streaming::lod {
namespace {

constexpr std::uint32_t kNoEdge = 0xffffffffu;
constexpr std::size_t kChunkRecords = 8192;
constexpr std::uint64_t kProgressMask = (1u << 20u) - 1u;

struct SplitPair {
    std::uint32_t edge = 0;
    std::uint32_t node = 0;
};

// A shortcut whose first child alone does not tell which node it bypasses.
struct OpenSplit {
    std::uint32_t edge = 0;
    std::uint32_t childB = 0;
    std::uint32_t from = 0;
    std::uint32_t to = 0;
    std::uint32_t a0 = 0;
    std::uint32_t a1 = 0;
};

// Shortcut (u, w) with child A = (u, v): the bypassed node is A's endpoint that is neither u
// nor w. kNoEdge when that is not unique (degenerate shortcuts are resolved with both children).
[[nodiscard]] std::uint32_t OtherEndpoint(std::uint32_t a0, std::uint32_t a1, std::uint32_t from,
                                          std::uint32_t to) {
    std::uint32_t candidate = kNoEdge;
    for (const auto node : {a0, a1}) {
        if (node == from || node == to) {
            continue;
        }
        if (candidate != kNoEdge && candidate != node) {
            return kNoEdge;
        }
        candidate = node;
    }
    return candidate;
}

// Common endpoint of both children, preferring one that is not an endpoint of the shortcut.
[[nodiscard]] std::uint32_t SharedEndpoint(const OpenSplit& open, std::uint32_t b0,
                                           std::uint32_t b1) {
    std::uint32_t split = kNoEdge;
    for (const auto a : {open.a0, open.a1}) {
        for (const auto b : {b0, b1}) {
            if (a == b && (split == kNoEdge || (a != open.from && a != open.to))) {
                split = a;
            }
        }
    }
    if (split == kNoEdge) {
        throw std::runtime_error("children of shortcut " + std::to_string(open.edge) +
                                 " do not share a node");
    }
    return split;
}

[[nodiscard]] double DistanceToSegment(const LodPageNodes& nodes, std::uint32_t point,
                                       std::uint32_t from, std::uint32_t to) {
    const double px = nodes.x[point];
    const double py = nodes.y[point];
    const double ax = nodes.x[from];
    const double ay = nodes.y[from];
    const double dx = nodes.x[to] - ax;
    const double dy = nodes.y[to] - ay;
    const double lengthSquared = dx * dx + dy * dy;
    const double t = lengthSquared > 0.0
                         ? std::clamp(((px - ax) * dx + (py - ay) * dy) / lengthSquared, 0.0, 1.0)
                         : 0.0;
    return std::hypot(px - (ax + t * dx), py - (ay + t * dy)) / kFixedCoordinateScale;
}

// Merges the per-pass split files (each sorted by edge) back into edge order.
class SplitStreams {
public:
    void Add(const std::filesystem::path& path, std::uint64_t count) {
        if (count != 0u) {
            streams_.push_back({ValueReader<SplitPair>(path, 0, count), count, {}});
            Advance(streams_.back());
        }
    }
    void AddResolved(std::vector<SplitPair> pairs) { resolved_ = std::move(pairs); }

    [[nodiscard]] std::uint32_t Take(std::uint32_t edge) {
        for (auto& stream : streams_) {
            if (stream.remaining != 0u && stream.head.edge == edge) {
                const auto node = stream.head.node;
                --stream.remaining;
                Advance(stream);
                return node;
            }
        }
        if (next_ < resolved_.size() && resolved_[next_].edge == edge) {
            return resolved_[next_++].node;
        }
        throw std::logic_error("LOD page build lost the split node of shortcut " +
                               std::to_string(edge));
    }

private:
    struct Stream {
        ValueReader<SplitPair> reader;
        std::uint64_t remaining = 0;
        SplitPair head;
    };

    static void Advance(Stream& stream) {
        if (stream.remaining != 0u) {
            stream.head = stream.reader.Next();
        }
    }

    std::vector<Stream> streams_;
    std::vector<SplitPair> resolved_;
    std::size_t next_ = 0;
};

} // namespace

void StageProgress::Report(std::uint32_t part, std::uint32_t parts, std::uint64_t current,
                           std::uint64_t total) {
    if (!progress_) {
        return;
    }
    const auto inside =
        total == 0u ? 1.0 : static_cast<double>(current) / static_cast<double>(total);
    const auto unit = static_cast<std::uint64_t>(
        1000.0 * (static_cast<double>(part) + std::min(inside, 1.0)) / static_cast<double>(parts));
    if (unit != last_) {
        last_ = unit;
        progress_({stage_, unit, 1000u});
    }
}

LevelChunks::LevelChunks(std::filesystem::path path, std::uint32_t levelCount)
    : path_(std::move(path)),
      file_(path_, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc),
      pending_(levelCount), chunks_(levelCount) {
    if (!file_) {
        throw std::runtime_error("could not create scratch file " + path_.string());
    }
}

LevelChunks::~LevelChunks() {
    file_.close();
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
}

std::uint64_t LevelChunks::BufferBytes(std::uint32_t levelCount) {
    return std::uint64_t{levelCount} * kChunkRecords * sizeof(LevelRecord);
}

void LevelChunks::Append(std::uint32_t level, const LevelRecord& record) {
    auto& pending = pending_[level];
    if (pending.empty()) {
        pending.reserve(kChunkRecords);
    }
    pending.push_back(record);
    ++count_;
    if (pending.size() == kChunkRecords) {
        WriteChunk(level);
    }
}

void LevelChunks::WriteChunk(std::uint32_t level) {
    auto& pending = pending_[level];
    const auto bytes = pending.size() * sizeof(LevelRecord);
    file_.seekp(static_cast<std::streamoff>(end_));
    file_.write(reinterpret_cast<const char*>(pending.data()), static_cast<std::streamsize>(bytes));
    if (!file_) {
        throw std::runtime_error("could not write scratch file " + path_.string() +
                                 " (disk full?)");
    }
    chunks_[level].push_back({end_, static_cast<std::uint32_t>(pending.size())});
    end_ += bytes;
    pending.clear();
}

void LevelChunks::Finish() {
    for (std::uint32_t level = 0; level < pending_.size(); ++level) {
        if (!pending_[level].empty()) {
            WriteChunk(level);
        }
        std::vector<LevelRecord>().swap(pending_[level]);
    }
    file_.flush();
}

void LevelChunks::ForEach(bool ascending, bool writeBack,
                          const std::function<void(std::span<LevelRecord>)>& visit) {
    std::vector<LevelRecord> buffer;
    const auto levels = chunks_.size();
    for (std::size_t i = 0; i < levels; ++i) {
        const auto& list = chunks_[ascending ? i : levels - 1u - i];
        for (std::size_t c = 0; c < list.size(); ++c) {
            const auto& chunk = list[ascending ? c : list.size() - 1u - c];
            buffer.resize(chunk.count);
            const auto bytes = static_cast<std::streamsize>(chunk.count * sizeof(LevelRecord));
            file_.seekg(static_cast<std::streamoff>(chunk.offset));
            file_.read(reinterpret_cast<char*>(buffer.data()), bytes);
            if (!file_) {
                throw std::runtime_error("could not read scratch file " + path_.string());
            }
            // Downwards is the exact reverse of upwards, also inside a level, so whatever order
            // the upward pass accepted, the downward pass sees parents before children.
            if (!ascending) {
                std::reverse(buffer.begin(), buffer.end());
            }
            visit(buffer);
            if (writeBack) {
                if (!ascending) {
                    std::reverse(buffer.begin(), buffer.end());
                }
                file_.seekp(static_cast<std::streamoff>(chunk.offset));
                file_.write(reinterpret_cast<const char*>(buffer.data()), bytes);
                if (!file_) {
                    throw std::runtime_error("could not write scratch file " + path_.string());
                }
            }
        }
    }
    file_.flush();
}

EdgeMask BuildLevelChunks(const std::filesystem::path& file, const FileHeader& header,
                          const LodPageNodes& nodes, std::uint64_t memoryBudget,
                          const std::filesystem::path& scratchBase, LevelChunks& chunks,
                          const LodPageBuilder::ProgressCallback& progress) {
    const auto m = header.edgeCount;
    const auto& sourceSection = header.At(Section::EdgeSource);
    const auto& targetSection = header.At(Section::EdgeTarget);
    StageProgress report(progress, BuildStage::ResolveShortcuts);

    // Endpoints of one range of child ids are held in memory per pass: 8 bytes per edge.
    const auto reserved = nodes.Bytes() + m / 8u +
                          LevelChunks::BufferBytes(nodes.maxNodeLevel + 1u) + (256ull << 20u);
    const auto available = memoryBudget > reserved ? memoryBudget - reserved : 0u;
    const auto sliceEdges = std::max<std::uint64_t>(std::min<std::uint64_t>(available / 8u, m),
                                                    std::min<std::uint64_t>(m, 1u << 20u));
    const auto passes =
        static_cast<std::uint32_t>(std::max<std::uint64_t>(1u, (m + sliceEdges - 1u) / sliceEdges));

    std::vector<TemporaryFile> splitFiles;
    std::vector<std::uint64_t> splitCounts;
    std::vector<OpenSplit> open;
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
        const auto first = std::uint64_t{pass} * sliceEdges;
        const auto count = std::min(sliceEdges, m - first);
        const auto sliceSource =
            ReadValues<std::uint32_t>(file, sourceSection.offset + first * 4u, count);
        const auto sliceTarget =
            ReadValues<std::uint32_t>(file, targetSection.offset + first * 4u, count);

        auto splitPath = scratchBase;
        splitPath += ".split" + std::to_string(pass);
        splitFiles.emplace_back(std::move(splitPath));
        ScratchWriter<SplitPair> pairs(splitFiles.back().Path());
        ValueReader<std::uint32_t> sources(file, sourceSection);
        ValueReader<std::uint32_t> targets(file, targetSection);
        ValueReader<std::uint32_t> childrenA(file, header.At(Section::EdgeChildA));
        for (std::uint64_t e = 0; e < m; ++e) {
            const auto from = sources.Next();
            const auto to = targets.Next();
            const auto childA = childrenA.Next();
            if (childA != kNoEdge && childA >= first && childA < first + count) {
                const auto a0 = sliceSource[childA - first];
                const auto a1 = sliceTarget[childA - first];
                const auto split = OtherEndpoint(a0, a1, from, to);
                if (split != kNoEdge) {
                    pairs.Push({static_cast<std::uint32_t>(e), split});
                } else {
                    open.push_back({static_cast<std::uint32_t>(e), 0u, from, to, a0, a1});
                }
            }
            if ((e & kProgressMask) == 0u) {
                report.Report(pass, passes + 2u, e, m);
            }
        }
        pairs.Close();
        splitCounts.push_back(pairs.Count());
    }

    // The few shortcuts whose bypassed node needs both children.
    std::vector<SplitPair> resolved;
    if (!open.empty()) {
        std::sort(open.begin(), open.end(),
                  [](const OpenSplit& a, const OpenSplit& b) { return a.edge < b.edge; });
        ValueReader<std::uint32_t> childrenB(file, header.At(Section::EdgeChildB));
        std::size_t next = 0;
        for (std::uint64_t e = 0; e < m && next < open.size(); ++e) {
            const auto childB = childrenB.Next();
            if (open[next].edge == e) {
                open[next++].childB = childB;
            }
        }
        std::sort(open.begin(), open.end(),
                  [](const OpenSplit& a, const OpenSplit& b) { return a.childB < b.childB; });
        std::ifstream sourceInput(file, std::ios::binary);
        std::ifstream targetInput(file, std::ios::binary);
        const auto readAt = [](std::ifstream& input, std::uint64_t offset) {
            std::uint32_t value = 0;
            input.seekg(static_cast<std::streamoff>(offset));
            input.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!input) {
                throw std::runtime_error("read failed while building LOD pages");
            }
            return value;
        };
        resolved.reserve(open.size());
        for (const auto& item : open) {
            const auto b0 = readAt(sourceInput, sourceSection.offset + item.childB * 4ull);
            const auto b1 = readAt(targetInput, targetSection.offset + item.childB * 4ull);
            resolved.push_back({item.edge, SharedEndpoint(item, b0, b1)});
        }
        std::sort(resolved.begin(), resolved.end(),
                  [](const SplitPair& a, const SplitPair& b) { return a.edge < b.edge; });
    }

    // File every shortcut under the level of its bypassed node.
    SplitStreams splits;
    for (std::size_t i = 0; i < splitFiles.size(); ++i) {
        splits.Add(splitFiles[i].Path(), splitCounts[i]);
    }
    splits.AddResolved(std::move(resolved));
    EdgeMask shortcuts(m);
    {
        ValueReader<std::uint32_t> sources(file, sourceSection);
        ValueReader<std::uint32_t> targets(file, targetSection);
        ValueReader<std::uint32_t> childrenA(file, header.At(Section::EdgeChildA));
        ValueReader<std::uint32_t> childrenB(file, header.At(Section::EdgeChildB));
        ValueReader<std::int16_t> births(file, header.At(Section::EdgeBirth));
        ValueReader<std::int16_t> deaths(file, header.At(Section::EdgeDeath));
        for (std::uint64_t e = 0; e < m; ++e) {
            const auto from = sources.Next();
            const auto to = targets.Next();
            const auto childA = childrenA.Next();
            const auto childB = childrenB.Next();
            const auto birth = births.Next();
            const auto death = deaths.Next();
            if (childA == kNoEdge) {
                continue;
            }
            const auto edge = static_cast<std::uint32_t>(e);
            const auto split = splits.Take(edge);
            shortcuts.Set(edge);
            chunks.Append(nodes.level[split],
                          {edge, childA, childB,
                           static_cast<float>(DistanceToSegment(nodes, split, from, to)), birth,
                           death});
            if ((e & kProgressMask) == 0u) {
                report.Report(passes, passes + 2u, e, m);
            }
        }
    }
    chunks.Finish();
    report.Report(passes + 1u, passes + 2u, 1u, 1u);
    return shortcuts;
}

ErrorTable ComputeErrors(LevelChunks& chunks, const EdgeMask& shortcuts, std::uint64_t edgeCount,
                         const LodPageBuilder::ProgressCallback& progress) {
    ErrorTable table;
    table.error.assign(static_cast<std::size_t>(edgeCount), 0.0f);
    std::vector<std::uint8_t> height(static_cast<std::size_t>(edgeCount), 0u);
    StageProgress report(progress, BuildStage::ComputeErrors);
    std::uint64_t done = 0;
    chunks.ForEach(true, true, [&](std::span<LevelRecord> records) {
        for (auto& record : records) {
            for (const auto child : {record.childA, record.childB}) {
                if (shortcuts.Test(child) && height[child] == 0u) {
                    throw std::runtime_error(
                        "CH node levels are not monotone: shortcut " + std::to_string(record.edge) +
                        " has child " + std::to_string(child) + " on the same or a higher level");
                }
            }
            const double childError =
                std::max(table.error[record.childA], table.error[record.childB]);
            record.value = static_cast<float>(static_cast<double>(record.value) + childError);
            table.error[record.edge] = record.value;
            const auto level = 1u + std::max(height[record.childA], height[record.childB]);
            if (level > 250u) {
                throw std::runtime_error("shortcut hierarchy deeper than 250 levels");
            }
            height[record.edge] = static_cast<std::uint8_t>(level);
            table.maxHeight = std::max(table.maxHeight, level);
        }
        done += records.size();
        report.Report(0u, 1u, done, chunks.Count());
    });
    return table;
}

ContextTable ComputeContexts(LevelChunks& chunks, std::uint64_t edgeCount,
                             const LodPageBuilder::ProgressCallback& progress) {
    const auto m = static_cast<std::size_t>(edgeCount);
    ContextTable table;
    table.low.assign(m, 1);
    table.high.assign(m, 0);
    table.parentCode.assign(m, 0);

    const auto add = [&table](std::uint32_t edge, LevelContext context) {
        if (!table.HasPrimary(edge)) {
            table.low[edge] = context.low;
            table.high[edge] = context.high;
            table.parentCode[edge] = context.parentCode;
            return;
        }
        // Joining is exact only for adjoining levels from equally coded parents; anything else
        // is a separate DFS visit and gets its own record.
        const auto low = table.low[edge];
        const auto high = table.high[edge];
        if (table.parentCode[edge] == context.parentCode &&
            (context.low == high + 1 || context.high + 1 == low)) {
            table.low[edge] = std::min(low, context.low);
            table.high[edge] = std::max(high, context.high);
        } else {
            table.extra[edge].push_back(context);
        }
    };

    StageProgress report(progress, BuildStage::ComputeContexts);
    std::uint64_t done = 0;
    std::vector<LevelContext> cover;
    chunks.ForEach(false, false, [&](std::span<LevelRecord> records) {
        for (const auto& record : records) {
            const auto code = EncodeGeometryError(record.value);
            cover.clear();
            if (record.birth >= 0 && record.death >= 0 && record.death <= record.birth) {
                cover.push_back({record.death, record.birth, code});
            }
            if (table.HasPrimary(record.edge)) {
                cover.push_back({table.low[record.edge], table.high[record.edge], code});
            }
            if (const auto it = table.extra.find(record.edge); it != table.extra.end()) {
                for (const auto& context : it->second) {
                    cover.push_back({context.low, context.high, code});
                }
            }
            for (const auto& levels : cover) {
                add(record.childA, levels);
                add(record.childB, levels);
            }
        }
        done += records.size();
        report.Report(0u, 1u, done, chunks.Count());
    });
    return table;
}

} // namespace chmv::streaming::lod
