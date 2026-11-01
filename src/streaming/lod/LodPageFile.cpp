#include "streaming/lod/LodPageFile.h"

#include "streaming/lod/LodPageCompression.h"
#include "streaming/lod/LodPageIo.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace chmv::streaming::lod {
namespace {

constexpr std::uint64_t kReadChunkBytes = 64ull << 20u;
// Stored bytes a whole-section read decodes at once, in parallel.
constexpr std::uint64_t kDecodeRoundBytes = 32ull << 20u;

[[noreturn]] void Fail(const std::filesystem::path& path, const std::string& message) {
    throw std::runtime_error(path.filename().string() + ": " + message);
}

[[nodiscard]] bool InsideFile(const SectionEntry& entry, std::uint64_t fileBytes) {
    return entry.offset >= kHeaderBytes && entry.offset <= fileBytes &&
           entry.bytes <= fileBytes - entry.offset;
}

[[nodiscard]] bool Framed(std::size_t index) {
    return index == StoredIndex(Section::Pages) || index == StoredIndex(Extension::PageChildren);
}

void ValidateHeader(const std::filesystem::path& path, const FileHeader& header,
                    std::uint64_t fileBytes) {
    if (header.magic != kFileMagic) {
        Fail(path, "not a LOD page file");
    }
    if (header.version != kFormatVersion &&
        (header.version != kUncompressedFormatVersion || header.compression != Compression::None)) {
        Fail(path, "unsupported LOD page format version " + std::to_string(header.version));
    }
    if (header.headerBytes != kHeaderBytes || header.pageSlotWords != kPageSlotWords) {
        Fail(path, "unexpected header or page slot size");
    }
    if (header.compression == Compression::None) {
        for (std::size_t index = 0; index < kStoredCount; ++index) {
            if (!InsideFile(header.Layout(index), fileBytes)) {
                Fail(path, "section outside the file (truncated?)");
            }
        }
    } else if (header.compression == Compression::Zstd) {
        for (const auto& stored : header.stored) {
            if (!InsideFile(stored, fileBytes)) {
                Fail(path, "compressed section outside the file (truncated?)");
            }
        }
        if (!InsideFile(header.pageFrames, fileBytes) ||
            !InsideFile(header.childFrames, fileBytes)) {
            Fail(path, "frame table outside the file (truncated?)");
        }
    } else {
        Fail(path, "unknown compression " +
                       std::to_string(static_cast<std::uint32_t>(header.compression)));
    }
    const auto expect = [&](Section section, std::uint64_t bytes, const char* name) {
        if (header.At(section).bytes != bytes) {
            Fail(path, std::string("unexpected size of section ") + name);
        }
    };
    if (header.nodeCount >= 0xffffffffull || header.edgeCount >= 0xffffffffull ||
        header.pageCount > 0xffffffffull || header.recordCount > 0xffffffffull) {
        Fail(path, "too many nodes, edges, pages or records");
    }
    const auto n = header.nodeCount;
    const auto m = header.edgeCount;
    expect(Section::NodeOsmId, n * 8u, "node OSM id");
    expect(Section::NodeLatitude, n * 4u, "node latitude");
    expect(Section::NodeLongitude, n * 4u, "node longitude");
    expect(Section::NodeElevation, n * 4u, "node elevation");
    expect(Section::NodeLevel, n * 4u, "node level");
    expect(Section::EdgeSource, m * 4u, "edge source");
    expect(Section::EdgeTarget, m * 4u, "edge target");
    expect(Section::EdgeWeight, m * 4u, "edge weight");
    expect(Section::EdgeType, m * 2u, "edge type");
    expect(Section::EdgeMaxSpeed, m * 2u, "edge max speed");
    expect(Section::EdgeChildA, m * 4u, "edge child A");
    expect(Section::EdgeChildB, m * 4u, "edge child B");
    expect(Section::EdgeBirth, m * 2u, "edge birth");
    expect(Section::EdgeDeath, m * 2u, "edge death");
    expect(Section::EdgeError, m * 4u, "edge error");
    expect(Section::EdgeCoverLow, m * 2u, "edge cover low");
    expect(Section::EdgeCoverHigh, m * 2u, "edge cover high");
    expect(Section::RecordEdge, header.recordCount * 4u, "record edge");
    expect(Section::PageDirectory, header.pageCount * sizeof(PageEntry), "page directory");
    expect(Section::PageGroups, header.groupCount * sizeof(PageGroup), "page groups");
    if (header.maxLevel < 0 || header.maxLevel > kMaxStoredLevel) {
        Fail(path, "level range outside the supported range");
    }
    if (header.At(Extension::VisitDirectory).bytes != header.pageCount * sizeof(PageEntry) ||
        header.At(Extension::VisitGroups).bytes != header.groupCount * sizeof(PageGroup) ||
        header.At(Extension::PageChildren).bytes < header.pageCount * kChildMaskWords * 4u) {
        Fail(path, "missing or unexpected hierarchy sections");
    }
}

void ValidatePages(const std::filesystem::path& path, const FileHeader& header,
                   std::span<const PageEntry> pages, std::span<const PageGroup> groups) {
    const auto& section = header.At(Section::Pages);
    for (const auto& page : pages) {
        const auto bytes = static_cast<std::uint64_t>(page.wordCount) * 4u;
        const auto relative = page.fileOffset - section.offset;
        if (page.recordCount == 0u || page.recordCount > kPageRecordCapacity ||
            page.vertexCount > kPageVertexCapacity ||
            page.wordCount != PageWordCount(page.recordCount, page.vertexCount) ||
            page.fileOffset < section.offset || relative > section.bytes ||
            bytes > section.bytes - relative || relative % 16u != 0u ||
            page.minLevel > page.maxLevel ||
            static_cast<std::uint64_t>(page.firstRecord) + page.recordCount > header.recordCount) {
            Fail(path, "corrupt page directory");
        }
    }
    std::uint64_t covered = 0;
    for (const auto& group : groups) {
        if (group.firstPage != covered || group.minLevel > group.maxLevel ||
            static_cast<std::uint64_t>(group.firstPage) + group.pageCount > pages.size()) {
            Fail(path, "corrupt page groups");
        }
        covered += group.pageCount;
    }
    if (covered != pages.size()) {
        Fail(path, "page groups do not cover the directory");
    }
}

// Visit entries may be empty (a page without representative records); their children blocks
// must lie inside the children section and match the page they belong to.
void ValidateVisits(const std::filesystem::path& path, const FileHeader& header,
                    std::span<const PageEntry> pages, std::span<const PageEntry> visits,
                    std::span<const PageGroup> groups) {
    const auto& section = header.At(Extension::PageChildren);
    for (std::size_t p = 0; p < visits.size(); ++p) {
        const auto& visit = visits[p];
        const auto bytes = static_cast<std::uint64_t>(visit.wordCount) * 4u;
        const auto relative = visit.fileOffset - section.offset;
        if (visit.firstRecord != pages[p].firstRecord ||
            visit.recordCount != pages[p].recordCount || visit.wordCount < kChildMaskWords ||
            visit.wordCount > kChildSlotWords || (visit.wordCount - kChildMaskWords) % 2u != 0u ||
            visit.fileOffset < section.offset || relative > section.bytes ||
            bytes > section.bytes - relative) {
            Fail(path, "corrupt visit directory");
        }
    }
    std::uint64_t covered = 0;
    for (const auto& group : groups) {
        if (group.firstPage != covered ||
            static_cast<std::uint64_t>(group.firstPage) + group.pageCount > visits.size()) {
            Fail(path, "corrupt visit groups");
        }
        covered += group.pageCount;
    }
    if (covered != visits.size()) {
        Fail(path, "visit groups do not cover the directory");
    }
}

// A compressed file stores pages (children blocks) as frames in page order, so the uncompressed
// section must be exactly their concatenation.
void RequireBackToBack(const std::filesystem::path& path, std::span<const PageEntry> entries,
                       const SectionEntry& section) {
    auto next = section.offset;
    for (const auto& entry : entries) {
        if (entry.fileOffset != next) {
            Fail(path, "pages of a compressed file are not back to back");
        }
        next += entry.wordCount * 4ull;
    }
    if (next != section.offset + section.bytes) {
        Fail(path, "pages of a compressed file do not fill their section");
    }
}

} // namespace

std::shared_ptr<const LodPageFile> LodPageFile::Open(const std::filesystem::path& path) {
    std::shared_ptr<LodPageFile> file(new LodPageFile());
    file->path_ = path;

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        Fail(path, "could not open");
    }
    const auto fileBytes = std::filesystem::file_size(path);
    if (fileBytes < kHeaderBytes) {
        Fail(path, "file is too small");
    }
    input.read(reinterpret_cast<char*>(&file->header_), sizeof(FileHeader));
    if (!input) {
        Fail(path, "could not read header");
    }
    ValidateHeader(path, file->header_, fileBytes);
    file->fileBytes_ = fileBytes;
    const auto& header = file->header_;

    if (file->Compressed()) {
        for (std::size_t index = 0; index < kStoredCount; ++index) {
            if (!Framed(index)) {
                file->blocks_[index] =
                    file->ReadBlockTable(header.stored[index], header.Layout(index).bytes);
            }
        }
        file->pageFrames_ =
            file->ReadFrames(file->ReadBlockTable(header.pageFrames, header.pageCount * 4u),
                             header.stored[StoredIndex(Section::Pages)]);
        file->childFrames_ =
            file->ReadFrames(file->ReadBlockTable(header.childFrames, header.pageCount * 4u),
                             header.stored[StoredIndex(Extension::PageChildren)]);
    }
    const auto read = [&](auto& values, std::uint64_t count, auto section) {
        values.resize(static_cast<std::size_t>(count));
        file->ReadRange(StoredIndex(section), 0u, values.data(), values.size() * sizeof(values[0]));
    };
    read(file->pages_, header.pageCount, Section::PageDirectory);
    read(file->groups_, header.groupCount, Section::PageGroups);
    ValidatePages(path, header, file->pages_, file->groups_);
    read(file->visitPages_, header.pageCount, Extension::VisitDirectory);
    read(file->visitGroups_, header.groupCount, Extension::VisitGroups);
    ValidateVisits(path, header, file->pages_, file->visitPages_, file->visitGroups_);
    if (file->Compressed()) {
        RequireBackToBack(path, file->pages_, header.At(Section::Pages));
        RequireBackToBack(path, file->visitPages_, header.At(Extension::PageChildren));
    }
    return file;
}

std::filesystem::path LodPageFile::DefaultPath(const std::filesystem::path& graphPath) {
    auto path = graphPath;
    if (path.extension() == ".sch") {
        path.replace_extension(".chlod");
    } else {
        path += ".chlod";
    }
    return path;
}

bool LodPageFile::MatchesSource(const std::filesystem::path& path,
                                const std::filesystem::path& graphPath,
                                const std::filesystem::path& rangesPath) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    FileHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(FileHeader));
    if (!input) {
        return false;
    }
    try {
        // A damaged or truncated header is simply rebuilt (damage elsewhere shows on Open).
        ValidateHeader(path, header, std::filesystem::file_size(path));
        const auto graph = QuerySourceIdentity(graphPath);
        const auto ranges = QuerySourceIdentity(rangesPath);
        return header.sourceGraphBytes == graph.bytes &&
               header.sourceGraphWriteTime == graph.writeTime &&
               header.sourceRangesBytes == ranges.bytes &&
               header.sourceRangesWriteTime == ranges.writeTime;
    } catch (const std::exception&) {
        return false;
    }
}

LodPageFile::Blocks LodPageFile::ReadBlockTable(const SectionEntry& stored,
                                                std::uint64_t bytes) const {
    BlockTable table;
    if (stored.bytes < sizeof(BlockTable)) {
        Fail(path_, "corrupt block table");
    }
    ReadBytes(stored.offset, &table, sizeof(table), {});
    const auto tableBytes =
        sizeof(BlockTable) + std::uint64_t{table.blockCount} * sizeof(StoredBlock);
    if (table.blockBytes == 0u ||
        (bytes + table.blockBytes - 1u) / table.blockBytes != table.blockCount ||
        tableBytes > stored.bytes) {
        Fail(path_, "corrupt block table");
    }
    std::vector<StoredBlock> entries(table.blockCount);
    ReadBytes(stored.offset + sizeof(BlockTable), entries.data(),
              entries.size() * sizeof(StoredBlock), {});
    Blocks blocks;
    blocks.blockBytes = table.blockBytes;
    blocks.bytes = bytes;
    blocks.offsets.reserve(entries.size() + 1u);
    blocks.shuffles.reserve(entries.size());
    const auto end = stored.offset + stored.bytes;
    auto offset = stored.offset + tableBytes;
    blocks.offsets.push_back(offset);
    for (const auto& entry : entries) {
        if (entry.storedBytes == 0u || entry.storedBytes > end - offset || entry.shuffle == 1u ||
            entry.shuffle > 256u) {
            Fail(path_, "corrupt block table");
        }
        offset += entry.storedBytes;
        blocks.offsets.push_back(offset);
        blocks.shuffles.push_back(entry.shuffle);
    }
    if (offset != end) {
        Fail(path_, "corrupt block table");
    }
    return blocks;
}

LodPageFile::Frames LodPageFile::ReadFrames(const Blocks& sizes, const SectionEntry& stored) const {
    std::vector<std::uint32_t> values(static_cast<std::size_t>(sizes.bytes / 4u));
    ReadBlocks(sizes, 0u, reinterpret_cast<std::byte*>(values.data()), values.size() * 4u, false);
    Frames frames;
    frames.offsets.reserve(values.size() + 1u);
    frames.shuffled.reserve(values.size());
    const auto end = stored.offset + stored.bytes;
    auto offset = stored.offset;
    frames.offsets.push_back(offset);
    for (const auto value : values) {
        const auto bytes = value & ~kFrameShuffled;
        if (bytes == 0u || bytes > end - offset) {
            Fail(path_, "corrupt frame table");
        }
        offset += bytes;
        frames.offsets.push_back(offset);
        frames.shuffled.push_back((value & kFrameShuffled) != 0u ? std::uint8_t{1}
                                                                 : std::uint8_t{0});
    }
    if (offset != end) {
        Fail(path_, "corrupt frame table");
    }
    return frames;
}

void LodPageFile::ReadBlocks(const Blocks& blocks, std::uint64_t first, std::byte* destination,
                             std::uint64_t bytes, bool cache) const {
    if (first > blocks.bytes || bytes > blocks.bytes - first) {
        Fail(path_, "read outside a compressed section");
    }
    std::vector<std::byte> stored;
    const auto decode = [&](std::uint64_t block, std::span<std::byte> out) {
        const auto offset = blocks.offsets[static_cast<std::size_t>(block)];
        stored.resize(static_cast<std::size_t>(
            blocks.offsets[static_cast<std::size_t>(block) + 1u] - offset));
        ReadBytes(offset, stored.data(), stored.size(), {});
        if (!DecodeFrame(stored, blocks.shuffles[static_cast<std::size_t>(block)], out)) {
            Fail(path_, "damaged compressed block");
        }
    };
    std::vector<std::byte> partial;
    while (bytes != 0u) {
        const auto block = first / blocks.blockBytes;
        const auto begin = block * blocks.blockBytes;
        const auto size = std::min<std::uint64_t>(blocks.blockBytes, blocks.bytes - begin);
        const auto inside = first - begin;
        const auto count = std::min(bytes, size - inside);
        if (count == size) {
            decode(block, {destination, static_cast<std::size_t>(size)});
        } else if (cache) {
            const std::lock_guard lock(cacheMutex_);
            if (cachedBlocks_ != &blocks || cachedBlock_ != block) {
                cachedBlocks_ = nullptr;
                cachedData_.resize(static_cast<std::size_t>(size));
                decode(block, cachedData_);
                cachedBlocks_ = &blocks;
                cachedBlock_ = block;
            }
            std::memcpy(destination, cachedData_.data() + inside, static_cast<std::size_t>(count));
        } else {
            partial.resize(static_cast<std::size_t>(size));
            decode(block, partial);
            std::memcpy(destination, partial.data() + inside, static_cast<std::size_t>(count));
        }
        destination += count;
        first += count;
        bytes -= count;
    }
}

void LodPageFile::ReadRange(std::size_t index, std::uint64_t first, void* destination,
                            std::uint64_t bytes) const {
    const auto& layout = header_.Layout(index);
    if (first > layout.bytes || bytes > layout.bytes - first) {
        Fail(path_, "read outside a section");
    }
    if (!Compressed()) {
        ReadBytes(layout.offset + first, destination, bytes, {});
    } else if (Framed(index)) {
        const auto words = ReadFramed(index != StoredIndex(Section::Pages), {});
        std::memcpy(destination, reinterpret_cast<const std::byte*>(words.data()) + first,
                    static_cast<std::size_t>(bytes));
    } else {
        ReadBlocks(blocks_[index], first, static_cast<std::byte*>(destination), bytes, true);
    }
}

std::uint64_t LodPageFile::HashSection(std::size_t index,
                                       const std::function<void(std::uint64_t)>& onRead) const {
    const auto& layout = header_.Layout(index);
    if (!Compressed()) {
        return HashFileRange(path_, layout.offset, layout.bytes, onRead);
    }
    StreamHash hash;
    if (Framed(index)) {
        DecodeFramed(index != StoredIndex(Section::Pages), true,
                     [&](std::span<const std::byte> decoded, std::uint64_t stored) {
                         hash.Update(decoded.data(), decoded.size());
                         if (onRead) {
                             onRead(stored);
                         }
                     });
        return hash.Finish();
    }
    const auto& blocks = blocks_[index];
    const auto count = blocks.shuffles.size();
    const auto perRound = std::max<std::size_t>(1u, kDecodeRoundBytes / blocks.blockBytes);
    std::vector<std::byte> stored;
    std::vector<std::byte> decoded;
    for (std::size_t first = 0; first < count; first += perRound) {
        const auto last = std::min(count, first + perRound);
        const auto storedBegin = blocks.offsets[first];
        stored.resize(static_cast<std::size_t>(blocks.offsets[last] - storedBegin));
        ReadBytes(storedBegin, stored.data(), stored.size(), {});
        const auto begin = std::uint64_t{first} * blocks.blockBytes;
        decoded.resize(static_cast<std::size_t>(
            std::min<std::uint64_t>(blocks.bytes, std::uint64_t{last} * blocks.blockBytes) -
            begin));
        RunParallel(last - first, [&](std::size_t task, unsigned) {
            const auto block = first + task;
            const auto offset = task * std::size_t{blocks.blockBytes};
            const auto frame = std::span<const std::byte>(stored).subspan(
                static_cast<std::size_t>(blocks.offsets[block] - storedBegin),
                static_cast<std::size_t>(blocks.offsets[block + 1u] - blocks.offsets[block]));
            const auto out = std::span<std::byte>(decoded).subspan(
                offset, std::min<std::size_t>(blocks.blockBytes, decoded.size() - offset));
            if (!DecodeFrame(frame, blocks.shuffles[block], out)) {
                Fail(path_, "damaged compressed block");
            }
        });
        hash.Update(decoded.data(), decoded.size());
        if (onRead) {
            onRead(stored.size());
        }
    }
    return hash.Finish();
}

void LodPageFile::DecodeFramed(
    bool children, bool strict,
    const std::function<void(std::span<const std::byte>, std::uint64_t)>& consume) const {
    const std::span<const PageEntry> entries = children ? visitPages_ : pages_;
    const auto& frames = children ? childFrames_ : pageFrames_;
    std::ifstream input(path_, std::ios::binary);
    if (!input) {
        Fail(path_, "could not open");
    }
    std::vector<std::byte> stored;
    std::vector<std::byte> decoded;
    for (std::size_t first = 0; first < entries.size();) {
        auto last = first + 1u;
        while (last < entries.size() &&
               frames.offsets[last + 1u] - frames.offsets[first] <= kDecodeRoundBytes) {
            ++last;
        }
        const auto storedBegin = frames.offsets[first];
        stored.resize(static_cast<std::size_t>(frames.offsets[last] - storedBegin));
        input.seekg(static_cast<std::streamoff>(storedBegin));
        input.read(reinterpret_cast<char*>(stored.data()),
                   static_cast<std::streamsize>(stored.size()));
        if (!input) {
            Fail(path_, "read failed (truncated file?)");
        }
        const auto decodedBegin = entries[first].fileOffset;
        decoded.assign(static_cast<std::size_t>(entries[last - 1u].fileOffset +
                                                entries[last - 1u].wordCount * 4ull - decodedBegin),
                       std::byte{0});
        const auto count = last - first;
        const auto tasks = std::min<std::size_t>(count, std::size_t{WorkerCount()} * 4u);
        RunParallel(tasks, [&](std::size_t task, unsigned) {
            for (auto p = first + count * task / tasks; p < first + count * (task + 1u) / tasks;
                 ++p) {
                const auto frame = std::span<const std::byte>(stored).subspan(
                    static_cast<std::size_t>(frames.offsets[p] - storedBegin),
                    static_cast<std::size_t>(frames.offsets[p + 1u] - frames.offsets[p]));
                const auto out = std::span<std::byte>(decoded).subspan(
                    static_cast<std::size_t>(entries[p].fileOffset - decodedBegin),
                    entries[p].wordCount * 4u);
                if (!DecodeFrame(frame, frames.shuffled[p] != 0u ? 4u : 0u, out)) {
                    if (strict) {
                        Fail(path_,
                             (children ? "damaged children block of page " : "damaged page ") +
                                 std::to_string(p));
                    }
                    std::fill(out.begin(), out.end(), std::byte{0});
                }
            }
        });
        consume(decoded, stored.size());
        first = last;
    }
}

std::vector<std::uint32_t> LodPageFile::ReadFramed(bool children,
                                                   const ProgressCallback& progress) const {
    const auto index =
        children ? StoredIndex(Extension::PageChildren) : StoredIndex(Section::Pages);
    const auto& layout = header_.Layout(index);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(layout.bytes / 4u));
    if (!Compressed()) {
        ReadBytes(layout.offset, words.data(), words.size() * 4u, progress);
        return words;
    }
    auto* out = reinterpret_cast<std::byte*>(words.data());
    const auto total = header_.stored[index].bytes;
    std::uint64_t done = 0;
    DecodeFramed(children, false, [&](std::span<const std::byte> decoded, std::uint64_t stored) {
        std::memcpy(out, decoded.data(), decoded.size());
        out += decoded.size();
        done += stored;
        if (progress) {
            progress(done, total);
        }
    });
    return words;
}

std::vector<std::uint32_t> LodPageFile::ReadPageWords(const ProgressCallback& progress) const {
    return ReadFramed(false, progress);
}

std::vector<std::uint32_t> LodPageFile::ReadChildrenWords(const ProgressCallback& progress) const {
    return ReadFramed(true, progress);
}

StoredRange LodPageFile::PageStorage(std::uint32_t page) const {
    if (!Compressed()) {
        return {pages_[page].fileOffset, pages_[page].wordCount * 4ull};
    }
    return {pageFrames_.offsets[page], pageFrames_.offsets[page + 1u] - pageFrames_.offsets[page]};
}

StoredRange LodPageFile::ChildrenStorage(std::uint32_t page) const {
    if (!Compressed()) {
        return {visitPages_[page].fileOffset, visitPages_[page].wordCount * 4ull};
    }
    return {childFrames_.offsets[page],
            childFrames_.offsets[page + 1u] - childFrames_.offsets[page]};
}

namespace {

[[nodiscard]] bool DecodeWords(std::span<const std::byte> stored, std::uint32_t wordCount,
                               bool compressed, std::uint32_t shuffle,
                               std::vector<std::uint32_t>& words) {
    words.resize(wordCount);
    const auto out = std::as_writable_bytes(std::span<std::uint32_t>(words));
    bool decoded = false;
    if (compressed) {
        decoded = DecodeFrame(stored, shuffle, out);
    } else if (stored.size() == out.size()) {
        std::memcpy(out.data(), stored.data(), out.size());
        decoded = true;
    }
    if (!decoded) {
        words.clear();
    }
    return decoded;
}

} // namespace

bool LodPageFile::DecodePage(std::uint32_t page, std::span<const std::byte> stored,
                             std::vector<std::uint32_t>& words) const {
    return DecodeWords(stored, pages_[page].wordCount, Compressed(),
                       Compressed() && pageFrames_.shuffled[page] != 0u ? 4u : 0u, words);
}

bool LodPageFile::DecodeChildren(std::uint32_t page, std::span<const std::byte> stored,
                                 std::vector<std::uint32_t>& words) const {
    return DecodeWords(stored, visitPages_[page].wordCount, Compressed(),
                       Compressed() && childFrames_.shuffled[page] != 0u ? 4u : 0u, words);
}

std::vector<std::uint32_t> LodPageFile::ReadPage(std::uint32_t page) const {
    const auto range = PageStorage(page);
    std::vector<std::byte> stored(static_cast<std::size_t>(range.bytes));
    ReadBytes(range.offset, stored.data(), stored.size(), {});
    std::vector<std::uint32_t> words;
    if (!DecodePage(page, stored, words)) {
        Fail(path_, "page " + std::to_string(page) + " is damaged");
    }
    return words;
}

std::vector<std::uint32_t> LodPageFile::ReadRecordEdges(std::uint64_t firstRecord,
                                                        std::uint32_t count) const {
    std::vector<std::uint32_t> edges(count);
    ReadRange(StoredIndex(Section::RecordEdge), firstRecord * 4u, edges.data(), edges.size() * 4u);
    return edges;
}

void LodPageFile::ReadBytes(std::uint64_t offset, void* destination, std::uint64_t bytes,
                            const ProgressCallback& progress) const {
    std::ifstream input(path_, std::ios::binary);
    if (!input) {
        Fail(path_, "could not open");
    }
    input.seekg(static_cast<std::streamoff>(offset));
    auto* out = static_cast<char*>(destination);
    std::uint64_t done = 0;
    while (done < bytes) {
        const auto count = std::min(kReadChunkBytes, bytes - done);
        input.read(out + done, static_cast<std::streamsize>(count));
        if (!input) {
            Fail(path_, "read failed (truncated file?)");
        }
        done += count;
        if (progress) {
            progress(done, bytes);
        }
    }
}

SourceIdentity QuerySourceIdentity(const std::filesystem::path& path) {
    SourceIdentity identity;
    identity.bytes = std::filesystem::file_size(path);
    identity.writeTime = static_cast<std::int64_t>(
        std::filesystem::last_write_time(path).time_since_epoch().count());
    return identity;
}

} // namespace chmv::streaming::lod
