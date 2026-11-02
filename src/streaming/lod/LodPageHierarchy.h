#pragma once

// Out-of-core passes over the shortcut hierarchy. Every shortcut is stored once in a scratch
// file, grouped by the CH level of the node it bypasses. In a valid CH that level is strictly
// lower for both children than for the parent, so walking the levels upwards visits children
// before parents (geometry errors) and walking them downwards visits parents before children
// (level contexts). Only a few bytes per edge stay in memory.

#include "streaming/lod/LodPageBuilder.h"
#include "streaming/lod/LodPageFormat.h"
#include "streaming/lod/LodPageSource.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

namespace chmv::streaming::lod {

struct LevelRecord {
    std::uint32_t edge = 0;
    std::uint32_t childA = 0;
    std::uint32_t childB = 0;
    // Distance of the bypassed node to the straight segment; after ComputeErrors the exact
    // recursive geometry error.
    float value = 0.0f;
    std::int16_t birth = -1;
    std::int16_t death = -1;
};

static_assert(sizeof(LevelRecord) == 20);

class LevelChunks {
public:
    LevelChunks(std::filesystem::path path, std::uint32_t levelCount);
    ~LevelChunks();

    LevelChunks(const LevelChunks&) = delete;
    LevelChunks& operator=(const LevelChunks&) = delete;

    void Append(std::uint32_t level, const LevelRecord& record);
    void Finish();
    // Visits every shortcut level by level; the descending order is the exact reverse of the
    // ascending one. With writeBack, changes made by `visit` are kept.
    void ForEach(bool ascending, bool writeBack,
                 const std::function<void(std::span<LevelRecord>)>& visit);

    [[nodiscard]] std::uint64_t Count() const { return count_; }
    [[nodiscard]] static std::uint64_t BufferBytes(std::uint32_t levelCount);

private:
    struct Chunk {
        std::uint64_t offset = 0;
        std::uint32_t count = 0;
    };

    void WriteChunk(std::uint32_t level);

    std::filesystem::path path_;
    std::fstream file_;
    std::vector<std::vector<LevelRecord>> pending_;
    std::vector<std::vector<Chunk>> chunks_;
    std::uint64_t end_ = 0;
    std::uint64_t count_ = 0;
};

// Bit per edge: set for shortcuts.
class EdgeMask {
public:
    explicit EdgeMask(std::uint64_t edgeCount) : bits_((edgeCount + 63u) / 64u, 0u) {}
    void Set(std::uint32_t edge) { bits_[edge >> 6u] |= std::uint64_t{1} << (edge & 63u); }
    [[nodiscard]] bool Test(std::uint32_t edge) const {
        return (bits_[edge >> 6u] >> (edge & 63u)) & 1u;
    }

private:
    std::vector<std::uint64_t> bits_;
};

// Finds the node every shortcut bypasses (by joining it with its children's endpoints in as
// many passes over child id ranges as the memory budget needs) and files the shortcut under
// that node's CH level. Returns the shortcut mask.
[[nodiscard]] EdgeMask BuildLevelChunks(const std::filesystem::path& file, const FileHeader& header,
                                        const LodPageNodes& nodes, std::uint64_t memoryBudget,
                                        const std::filesystem::path& scratchBase,
                                        LevelChunks& chunks,
                                        const LodPageBuilder::ProgressCallback& progress);

struct ErrorTable {
    std::vector<float> error;
    std::uint32_t maxHeight = 0;
};

// Exact recursive geometry error, children first. Stores the error back into the chunks.
[[nodiscard]] ErrorTable ComputeErrors(LevelChunks& chunks, const EdgeMask& shortcuts,
                                       std::uint64_t edgeCount,
                                       const LodPageBuilder::ProgressCallback& progress);

// The road style every edge draws with (EdgeStyle in LodPageFormat.h): originals from their
// type, then shortcuts from their halves, children first.
[[nodiscard]] std::vector<std::uint8_t>
ComputeStyles(const std::filesystem::path& file, const FileHeader& header,
              const LodPageNodes& nodes, LevelChunks& chunks, const EdgeMask& shortcuts,
              const LodPageBuilder::ProgressCallback& progress);

struct LevelContext {
    std::int16_t low = 1;
    std::int16_t high = 0;
    std::uint16_t parentCode = 0;
};

// Levels at which an edge is reached below a parent, with that parent's error code. The first
// context of every edge lives in the arrays; further ones (a second parent, a gap, an overlap)
// are rare and kept in `extra`.
struct ContextTable {
    std::vector<std::int16_t> low;
    std::vector<std::int16_t> high;
    std::vector<std::uint16_t> parentCode;
    std::unordered_map<std::uint32_t, std::vector<LevelContext>> extra;

    [[nodiscard]] bool HasPrimary(std::uint32_t edge) const { return low[edge] <= high[edge]; }
};

// cover(e) = alive(e) + contexts(e), pushed from parents to children, parents first.
[[nodiscard]] ContextTable ComputeContexts(LevelChunks& chunks, std::uint64_t edgeCount,
                                           const LodPageBuilder::ProgressCallback& progress);

// Reports one build stage as a fraction in thousandths, at most once per thousandth.
class StageProgress {
public:
    StageProgress(const LodPageBuilder::ProgressCallback& progress, BuildStage stage)
        : progress_(progress), stage_(stage) {}

    // Part `part` of `parts` equal parts, `current` of `total` done inside that part.
    void Report(std::uint32_t part, std::uint32_t parts, std::uint64_t current,
                std::uint64_t total);

private:
    const LodPageBuilder::ProgressCallback& progress_;
    BuildStage stage_;
    std::uint64_t last_ = ~std::uint64_t{0};
};

} // namespace chmv::streaming::lod
