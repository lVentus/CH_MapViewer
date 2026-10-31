#include "streaming/lod/LodPageFile.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace chmv::streaming::lod {
namespace {

constexpr std::uint64_t kReadChunkBytes = 64ull << 20u;

[[noreturn]] void Fail(const std::filesystem::path& path, const std::string& message) {
    throw std::runtime_error(path.filename().string() + ": " + message);
}

void ValidateHeader(const std::filesystem::path& path, const FileHeader& header,
                    std::uint64_t fileBytes) {
    if (header.magic != kFileMagic) {
        Fail(path, "not a LOD page file");
    }
    if (header.version != kFormatVersion) {
        Fail(path, "unsupported LOD page format version " + std::to_string(header.version));
    }
    if (header.headerBytes != kHeaderBytes || header.pageSlotWords != kPageSlotWords) {
        Fail(path, "unexpected header or page slot size");
    }
    for (const auto& section : header.sections) {
        if (section.offset < kHeaderBytes || section.offset > fileBytes ||
            section.bytes > fileBytes - section.offset) {
            Fail(path, "section outside the file (truncated?)");
        }
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

    file->pages_.resize(static_cast<std::size_t>(file->header_.pageCount));
    file->ReadBytes(file->header_.At(Section::PageDirectory).offset, file->pages_.data(),
                    file->pages_.size() * sizeof(PageEntry), {});
    file->groups_.resize(static_cast<std::size_t>(file->header_.groupCount));
    file->ReadBytes(file->header_.At(Section::PageGroups).offset, file->groups_.data(),
                    file->groups_.size() * sizeof(PageGroup), {});
    ValidatePages(path, file->header_, file->pages_, file->groups_);
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
        // A damaged or truncated file is simply rebuilt.
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

std::vector<std::uint32_t> LodPageFile::ReadPageWords(const ProgressCallback& progress) const {
    const auto& section = header_.At(Section::Pages);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(section.bytes / 4u));
    ReadBytes(section.offset, words.data(), words.size() * 4u, progress);
    return words;
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
