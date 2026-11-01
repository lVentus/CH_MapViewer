#pragma once

#include "streaming/lod/LodPageFormat.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace chmv::streaming::lod {

// A byte range of the file.
struct StoredRange {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

// Read-only view of a .chlod file, compressed or not: header, page directory and page groups
// stay in memory, everything else is read (and decoded) on demand. Every read returns the
// uncompressed bytes.
class LodPageFile {
public:
    using ProgressCallback = std::function<void(std::uint64_t current, std::uint64_t total)>;

    [[nodiscard]] static std::shared_ptr<const LodPageFile> Open(const std::filesystem::path& path);
    [[nodiscard]] static std::filesystem::path DefaultPath(const std::filesystem::path& graphPath);
    // True if the file exists, can be read by this version (compressed or a version 3 file) and
    // was built from exactly these source files (size and modification time).
    [[nodiscard]] static bool MatchesSource(const std::filesystem::path& path,
                                            const std::filesystem::path& graphPath,
                                            const std::filesystem::path& rangesPath);

    [[nodiscard]] const std::filesystem::path& Path() const { return path_; }
    [[nodiscard]] const FileHeader& Header() const { return header_; }
    [[nodiscard]] bool Compressed() const { return header_.compression != Compression::None; }
    [[nodiscard]] std::uint64_t FileBytes() const { return fileBytes_; }
    [[nodiscard]] std::span<const PageEntry> Pages() const { return pages_; }
    [[nodiscard]] std::span<const PageGroup> Groups() const { return groups_; }
    // Directory and groups for traversals.
    [[nodiscard]] std::span<const PageEntry> VisitPages() const { return visitPages_; }
    [[nodiscard]] std::span<const PageGroup> VisitGroups() const { return visitGroups_; }
    // Word offset of a page's children block inside the children section.
    [[nodiscard]] std::uint64_t ChildrenWordOffset(std::uint32_t page) const {
        return (visitPages_[page].fileOffset - header_.At(Extension::PageChildren).offset) / 4u;
    }
    // Uncompressed size of the page section.
    [[nodiscard]] std::uint64_t PageSectionBytes() const {
        return header_.At(Section::Pages).bytes;
    }
    // Word offset of a page inside the page section (and inside a full RAM image of it).
    [[nodiscard]] std::uint64_t PageWordOffset(std::uint32_t page) const {
        return (pages_[page].fileOffset - header_.At(Section::Pages).offset) / 4u;
    }

    template <typename T> [[nodiscard]] std::vector<T> ReadSection(Section section) const {
        std::vector<T> values(static_cast<std::size_t>(header_.At(section).bytes / sizeof(T)));
        ReadRange(StoredIndex(section), 0u, values.data(), values.size() * sizeof(T));
        return values;
    }
    // Uncompressed bytes [first, first + bytes) of a section or extension (by StoredIndex).
    void ReadRange(std::size_t index, std::uint64_t first, void* destination,
                   std::uint64_t bytes) const;
    // Hash of a section or extension (by StoredIndex) as uncompressed, read back from disk;
    // `onRead` gets the stored bytes read, from the calling thread, and may throw to cancel.
    [[nodiscard]] std::uint64_t HashSection(std::size_t index,
                                            const std::function<void(std::uint64_t)>& onRead) const;

    // The whole page / children section. Pages that do not decode stay zero (and fail the
    // checks pages get before they are drawn).
    [[nodiscard]] std::vector<std::uint32_t> ReadPageWords(const ProgressCallback& progress) const;
    [[nodiscard]] std::vector<std::uint32_t>
    ReadChildrenWords(const ProgressCallback& progress) const;
    // Single reads, for checks: one page blob, and the CH edges of a run of records.
    [[nodiscard]] std::vector<std::uint32_t> ReadPage(std::uint32_t page) const;
    [[nodiscard]] std::vector<std::uint32_t> ReadRecordEdges(std::uint64_t firstRecord,
                                                             std::uint32_t count) const;

    // Where a page (its children block) lies in the file, and its words from those bytes;
    // false if they are damaged.
    [[nodiscard]] StoredRange PageStorage(std::uint32_t page) const;
    [[nodiscard]] StoredRange ChildrenStorage(std::uint32_t page) const;
    [[nodiscard]] bool DecodePage(std::uint32_t page, std::span<const std::byte> stored,
                                  std::vector<std::uint32_t>& words) const;
    [[nodiscard]] bool DecodeChildren(std::uint32_t page, std::span<const std::byte> stored,
                                      std::vector<std::uint32_t>& words) const;

private:
    // Blocks of a compressed section: file offsets (one more than blocks) and shuffles.
    struct Blocks {
        std::uint32_t blockBytes = 0;
        std::uint64_t bytes = 0;
        std::vector<std::uint64_t> offsets;
        std::vector<std::uint32_t> shuffles;
    };
    // Frames of the pages or children blocks: file offsets (one more than pages), shuffled.
    struct Frames {
        std::vector<std::uint64_t> offsets;
        std::vector<std::uint8_t> shuffled;
    };

    LodPageFile() = default;
    void ReadBytes(std::uint64_t offset, void* destination, std::uint64_t bytes,
                   const ProgressCallback& progress) const;
    [[nodiscard]] Blocks ReadBlockTable(const SectionEntry& stored, std::uint64_t bytes) const;
    [[nodiscard]] Frames ReadFrames(const Blocks& sizes, const SectionEntry& stored) const;
    // `cache` keeps the last partly read block (only for the tables in blocks_).
    void ReadBlocks(const Blocks& blocks, std::uint64_t first, std::byte* destination,
                    std::uint64_t bytes, bool cache) const;
    // Decodes the frames of every page (children block) a round at a time; `consume` gets each
    // round's uncompressed bytes (back to back) and stored size. A damaged frame throws when
    // `strict`, otherwise it stays zero.
    void DecodeFramed(
        bool children, bool strict,
        const std::function<void(std::span<const std::byte>, std::uint64_t)>& consume) const;
    [[nodiscard]] std::vector<std::uint32_t>
    ReadFramed(bool children, const ProgressCallback& progress) const;

    std::filesystem::path path_;
    std::uint64_t fileBytes_ = 0;
    FileHeader header_{};
    std::vector<PageEntry> pages_;
    std::vector<PageGroup> groups_;
    std::vector<PageEntry> visitPages_;
    std::vector<PageGroup> visitGroups_;
    std::array<Blocks, kStoredCount> blocks_{};
    Frames pageFrames_;
    Frames childFrames_;
    // The last block a partial read decoded (record edges are read a page at a time).
    mutable std::mutex cacheMutex_;
    mutable const Blocks* cachedBlocks_ = nullptr;
    mutable std::uint64_t cachedBlock_ = 0;
    mutable std::vector<std::byte> cachedData_;
};

// What the background loader hands to the runtime.
struct LodPageDataset {
    std::shared_ptr<const LodPageFile> file;
    // The whole page section when it fits in RAM; empty means pages are read from disk.
    std::shared_ptr<const std::vector<std::uint32_t>> pageImage;
    // The children section alongside it, when the file has one.
    std::shared_ptr<const std::vector<std::uint32_t>> childrenImage;
};

struct SourceIdentity {
    std::uint64_t bytes = 0;
    std::int64_t writeTime = 0;
};

[[nodiscard]] SourceIdentity QuerySourceIdentity(const std::filesystem::path& path);

} // namespace chmv::streaming::lod
