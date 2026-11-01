#include "streaming/lod/LodPageCompression.h"

#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPageIo.h"

#include <zstd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <new>
#include <stdexcept>
#include <string>

namespace chmv::streaming::lod {
namespace {

// Uncompressed bytes compressed at once (in parallel) before they are written.
constexpr std::uint64_t kRoundBytes = 64ull << 20u;

struct ThreadContexts {
    ThreadContexts() = default;
    ThreadContexts(const ThreadContexts&) = delete;
    ThreadContexts& operator=(const ThreadContexts&) = delete;
    ~ThreadContexts() {
        ZSTD_freeCCtx(compress);
        ZSTD_freeDCtx(decompress);
    }

    ZSTD_CCtx* compress = nullptr;
    ZSTD_DCtx* decompress = nullptr;
    std::vector<std::byte> shuffled;
    std::vector<std::byte> candidate;
};

[[nodiscard]] ThreadContexts& Contexts() {
    thread_local ThreadContexts contexts;
    return contexts;
}

void Shuffle(std::span<const std::byte> input, std::uint32_t width, std::vector<std::byte>& out) {
    out.resize(input.size());
    const auto count = input.size() / width;
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t k = 0; k < width; ++k) {
            out[k * count + i] = input[i * width + k];
        }
    }
    std::copy(input.begin() + static_cast<std::ptrdiff_t>(count * width), input.end(),
              out.begin() + static_cast<std::ptrdiff_t>(count * width));
}

void Unshuffle(std::span<const std::byte> input, std::uint32_t width, std::span<std::byte> out) {
    const auto count = input.size() / width;
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t k = 0; k < width; ++k) {
            out[i * width + k] = input[k * count + i];
        }
    }
    std::copy(input.begin() + static_cast<std::ptrdiff_t>(count * width), input.end(),
              out.begin() + static_cast<std::ptrdiff_t>(count * width));
}

// Element size a column is shuffled with, when that compresses better.
[[nodiscard]] std::uint32_t ElementBytes(std::size_t index) {
    switch (index) {
    case StoredIndex(Section::NodeOsmId):
        return 8u;
    case StoredIndex(Section::EdgeType):
    case StoredIndex(Section::EdgeMaxSpeed):
    case StoredIndex(Section::EdgeBirth):
    case StoredIndex(Section::EdgeDeath):
    case StoredIndex(Section::EdgeCoverLow):
    case StoredIndex(Section::EdgeCoverHigh):
        return 2u;
    default:
        return 4u;
    }
}

class Compressor {
public:
    using Progress = std::function<void(std::uint64_t, std::uint64_t)>;
    using Source = std::function<void(std::uint64_t, std::span<std::byte>)>;

    Compressor(const LodPageFile& raw, const std::filesystem::path& output,
               const Progress& progress)
        : raw_(raw), input_(raw.Path(), std::ios::binary), output_(output), progress_(progress) {
        if (!input_) {
            throw std::runtime_error("could not open " + raw.Path().string());
        }
        for (std::size_t index = 0; index < kStoredCount; ++index) {
            total_ += raw.Header().Layout(index).bytes;
        }
        total_ += raw.Header().pageCount * 8u;
    }

    void Run() {
        auto header = raw_.Header();
        header.version = kFormatVersion;
        header.compression = Compression::Zstd;
        std::vector<std::uint32_t> pageFrames;
        std::vector<std::uint32_t> childFrames;
        for (std::size_t index = 0; index < kStoredCount; ++index) {
            const auto layout = header.Layout(index);
            if (index == StoredIndex(Section::Pages)) {
                header.stored[index] = Frames(raw_.Pages(), 0u, pageFrames);
            } else if (index == StoredIndex(Extension::PageChildren)) {
                header.stored[index] = Frames(raw_.VisitPages(), 4u, childFrames);
            } else {
                const auto blockBytes = index == StoredIndex(Section::RecordEdge)
                                            ? kSmallBlockBytes
                                            : kColumnBlockBytes;
                header.stored[index] =
                    Blocks(layout.bytes, blockBytes, ElementBytes(index),
                           [&](std::uint64_t first, std::span<std::byte> destination) {
                               ReadRaw(layout.offset + first, destination);
                           });
            }
        }
        header.pageFrames = Column(pageFrames);
        header.childFrames = Column(childFrames);

        std::vector<std::byte> headerBytes(kHeaderBytes);
        std::memcpy(headerBytes.data(), &header, sizeof(FileHeader));
        output_.WriteAt(0, headerBytes.data(), headerBytes.size());
        output_.Close();
    }

private:
    void Report() {
        if (progress_) {
            progress_(done_, total_);
        }
    }

    void ReadRaw(std::uint64_t offset, std::span<std::byte> destination) {
        input_.seekg(static_cast<std::streamoff>(offset));
        input_.read(reinterpret_cast<char*>(destination.data()),
                    static_cast<std::streamsize>(destination.size()));
        if (!input_) {
            throw std::runtime_error("read failed while compressing " + raw_.Path().string());
        }
    }

    void Write(std::span<const std::byte> bytes) {
        output_.WriteAt(position_, bytes.data(), bytes.size());
        position_ += bytes.size();
    }

    // A BlockTable, the block entries, then one frame per block.
    SectionEntry Blocks(std::uint64_t bytes, std::uint32_t blockBytes, std::uint32_t elementBytes,
                        const Source& read) {
        const auto blocks = (bytes + blockBytes - 1u) / blockBytes;
        if (blocks > 0xffffffffull) {
            throw std::runtime_error("LOD page compression: section too large");
        }
        const auto start = position_;
        std::vector<StoredBlock> table(static_cast<std::size_t>(blocks));
        position_ += sizeof(BlockTable) + blocks * sizeof(StoredBlock);
        const auto perRound = std::max<std::uint64_t>(1u, kRoundBytes / blockBytes);
        std::vector<std::byte> input;
        std::vector<std::vector<std::byte>> frames;
        for (std::uint64_t first = 0; first < blocks; first += perRound) {
            const auto last = std::min(blocks, first + perRound);
            const auto begin = first * blockBytes;
            const auto end = std::min(bytes, last * blockBytes);
            input.resize(static_cast<std::size_t>(end - begin));
            read(begin, input);
            frames.resize(static_cast<std::size_t>(last - first));
            RunParallel(
                frames.size(),
                [&](std::size_t task, unsigned) {
                    const auto offset = task * blockBytes;
                    const auto slice = std::span<const std::byte>(input).subspan(
                        offset, std::min<std::size_t>(blockBytes, input.size() - offset));
                    frames[task].clear();
                    const auto shuffle =
                        AppendSmallerFrame(slice, kBlockLevel, elementBytes, frames[task]);
                    table[static_cast<std::size_t>(first) + task] = {
                        static_cast<std::uint32_t>(frames[task].size()), shuffle};
                },
                [&] { Report(); });
            for (const auto& frame : frames) {
                Write(frame);
            }
            done_ += end - begin;
            Report();
        }
        const BlockTable head{blockBytes, static_cast<std::uint32_t>(blocks)};
        output_.WriteAt(start, &head, sizeof(head));
        output_.WriteAt(start + sizeof(head), table.data(), table.size() * sizeof(StoredBlock));
        return {start, position_ - start};
    }

    SectionEntry Column(const std::vector<std::uint32_t>& values) {
        return Blocks(values.size() * 4u, kColumnBlockBytes, 4u,
                      [&](std::uint64_t first, std::span<std::byte> destination) {
                          std::memcpy(destination.data(),
                                      reinterpret_cast<const std::byte*>(values.data()) + first,
                                      destination.size());
                      });
    }

    // One frame per entry, in entry order; `sizes` gets the frame sizes. elementBytes != 0 also
    // tries the shuffled frame (4: words).
    SectionEntry Frames(std::span<const PageEntry> entries, std::uint32_t elementBytes,
                        std::vector<std::uint32_t>& sizes) {
        const auto start = position_;
        sizes.assign(entries.size(), 0u);
        std::vector<std::byte> input;
        std::vector<std::vector<std::byte>> chunks;
        for (std::size_t first = 0; first < entries.size();) {
            // Consecutive entries lie back to back in the uncompressed file.
            auto last = first;
            std::uint64_t bytes = 0;
            while (last < entries.size() && (last == first || bytes < kRoundBytes / 2u)) {
                if (entries[last].fileOffset != entries[first].fileOffset + bytes) {
                    throw std::logic_error("LOD page compression: pages are not back to back");
                }
                bytes += entries[last].wordCount * 4ull;
                ++last;
            }
            input.resize(static_cast<std::size_t>(bytes));
            ReadRaw(entries[first].fileOffset, input);
            const auto count = last - first;
            chunks.resize(std::min<std::size_t>(count, std::size_t{WorkerCount()} * 4u));
            RunParallel(
                chunks.size(),
                [&](std::size_t task, unsigned) {
                    auto& chunk = chunks[task];
                    chunk.clear();
                    const auto begin = first + count * task / chunks.size();
                    const auto end = first + count * (task + 1u) / chunks.size();
                    for (auto p = begin; p < end; ++p) {
                        const auto before = chunk.size();
                        const auto slice = std::span<const std::byte>(input).subspan(
                            static_cast<std::size_t>(entries[p].fileOffset -
                                                     entries[first].fileOffset),
                            entries[p].wordCount * 4u);
                        std::uint32_t shuffle = 0;
                        if (elementBytes != 0u) {
                            shuffle = AppendSmallerFrame(slice, kFrameLevel, elementBytes, chunk);
                        } else {
                            AppendFrame(slice, kFrameLevel, 0u, chunk);
                        }
                        sizes[p] = static_cast<std::uint32_t>(chunk.size() - before) |
                                   (shuffle != 0u ? kFrameShuffled : 0u);
                    }
                },
                [&] { Report(); });
            for (const auto& chunk : chunks) {
                Write(chunk);
            }
            done_ += bytes;
            Report();
            first = last;
        }
        return {start, position_ - start};
    }

    const LodPageFile& raw_;
    std::ifstream input_;
    LodPageOutput output_;
    const Progress& progress_;
    std::uint64_t position_ = kHeaderBytes;
    std::uint64_t done_ = 0;
    std::uint64_t total_ = 0;
};

} // namespace

void AppendFrame(std::span<const std::byte> input, int level, std::uint32_t shuffleBytes,
                 std::vector<std::byte>& out) {
    auto& contexts = Contexts();
    if (contexts.compress == nullptr) {
        contexts.compress = ZSTD_createCCtx();
        if (contexts.compress == nullptr) {
            throw std::bad_alloc();
        }
    }
    auto source = input;
    if (shuffleBytes > 1u) {
        Shuffle(input, shuffleBytes, contexts.shuffled);
        source = contexts.shuffled;
    }
    // With a checksum, a damaged frame fails to decode instead of giving other bytes.
    const auto check = [](std::size_t result) {
        if (ZSTD_isError(result) != 0u) {
            throw std::runtime_error(std::string("zstd compression failed: ") +
                                     ZSTD_getErrorName(result));
        }
        return result;
    };
    check(ZSTD_CCtx_setParameter(contexts.compress, ZSTD_c_compressionLevel, level));
    check(ZSTD_CCtx_setParameter(contexts.compress, ZSTD_c_checksumFlag, 1));
    const auto start = out.size();
    out.resize(start + ZSTD_compressBound(source.size()));
    const auto written = check(ZSTD_compress2(contexts.compress, out.data() + start,
                                              out.size() - start, source.data(), source.size()));
    out.resize(start + written);
}

std::uint32_t AppendSmallerFrame(std::span<const std::byte> input, int level,
                                 std::uint32_t elementBytes, std::vector<std::byte>& out) {
    const auto start = out.size();
    AppendFrame(input, level, 0u, out);
    if (elementBytes < 2u || input.size() < 2u * elementBytes) {
        return 0u;
    }
    auto& candidate = Contexts().candidate;
    candidate.clear();
    AppendFrame(input, level, elementBytes, candidate);
    if (candidate.size() >= out.size() - start) {
        return 0u;
    }
    out.resize(start);
    out.insert(out.end(), candidate.begin(), candidate.end());
    return elementBytes;
}

bool DecodeFrame(std::span<const std::byte> frame, std::uint32_t shuffleBytes,
                 std::span<std::byte> output) {
    auto& contexts = Contexts();
    if (contexts.decompress == nullptr) {
        contexts.decompress = ZSTD_createDCtx();
        if (contexts.decompress == nullptr) {
            throw std::bad_alloc();
        }
    }
    auto target = output;
    if (shuffleBytes > 1u) {
        contexts.shuffled.resize(output.size());
        target = contexts.shuffled;
    }
    const auto size = ZSTD_decompressDCtx(contexts.decompress, target.data(), target.size(),
                                          frame.data(), frame.size());
    if (ZSTD_isError(size) != 0u || size != target.size()) {
        return false;
    }
    if (shuffleBytes > 1u) {
        Unshuffle(target, shuffleBytes, output);
    }
    return true;
}

void CompressPageFile(const LodPageFile& raw, const std::filesystem::path& output,
                      const std::function<void(std::uint64_t, std::uint64_t)>& progress) {
    if (raw.Compressed()) {
        throw std::logic_error("LOD page compression: the file is already compressed");
    }
    Compressor(raw, output, progress).Run();
}

} // namespace chmv::streaming::lod
