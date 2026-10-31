#pragma once

#include "streaming/lod/LodPageFormat.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace chmv::streaming::lod {

// Read-only view of a .chlod file: header, page directory and page groups stay in memory,
// everything else is read on demand.
class LodPageFile {
public:
    using ProgressCallback = std::function<void(std::uint64_t current, std::uint64_t total)>;

    [[nodiscard]] static std::shared_ptr<const LodPageFile> Open(const std::filesystem::path& path);
    [[nodiscard]] static std::filesystem::path DefaultPath(const std::filesystem::path& graphPath);
    // True if the file exists, has the current format version and was built from exactly these
    // source files (size and modification time).
    [[nodiscard]] static bool MatchesSource(const std::filesystem::path& path,
                                            const std::filesystem::path& graphPath,
                                            const std::filesystem::path& rangesPath);

    [[nodiscard]] const std::filesystem::path& Path() const { return path_; }
    [[nodiscard]] const FileHeader& Header() const { return header_; }
    [[nodiscard]] std::span<const PageEntry> Pages() const { return pages_; }
    [[nodiscard]] std::span<const PageGroup> Groups() const { return groups_; }
    [[nodiscard]] std::uint64_t PageSectionBytes() const {
        return header_.At(Section::Pages).bytes;
    }
    // Word offset of a page inside the page section (and inside a full RAM image of it).
    [[nodiscard]] std::uint64_t PageWordOffset(std::uint32_t page) const {
        return (pages_[page].fileOffset - header_.At(Section::Pages).offset) / 4u;
    }

    template <typename T> [[nodiscard]] std::vector<T> ReadSection(Section section) const {
        const auto& entry = header_.At(section);
        std::vector<T> values(static_cast<std::size_t>(entry.bytes / sizeof(T)));
        ReadBytes(entry.offset, values.data(), values.size() * sizeof(T), {});
        return values;
    }
    [[nodiscard]] std::vector<std::uint32_t> ReadPageWords(const ProgressCallback& progress) const;

private:
    LodPageFile() = default;
    void ReadBytes(std::uint64_t offset, void* destination, std::uint64_t bytes,
                   const ProgressCallback& progress) const;

    std::filesystem::path path_;
    FileHeader header_{};
    std::vector<PageEntry> pages_;
    std::vector<PageGroup> groups_;
};

// What the background loader hands to the runtime.
struct LodPageDataset {
    std::shared_ptr<const LodPageFile> file;
    // The whole page section when it fits in RAM; empty means pages are read from disk.
    std::shared_ptr<const std::vector<std::uint32_t>> pageImage;
};

struct SourceIdentity {
    std::uint64_t bytes = 0;
    std::int64_t writeTime = 0;
};

[[nodiscard]] SourceIdentity QuerySourceIdentity(const std::filesystem::path& path);

} // namespace chmv::streaming::lod
