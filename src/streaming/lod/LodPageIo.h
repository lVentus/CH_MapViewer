#pragma once

// Small sequential I/O helpers for building LOD page files out of core: everything large is
// written and read in big chunks, never element by element on disk.

#include "streaming/lod/LodPageFormat.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace chmv::streaming::lod {

// 64-bit hash of a byte stream; the result does not depend on how the stream is chunked.
class StreamHash {
public:
    void Update(const void* data, std::uint64_t bytes);
    [[nodiscard]] std::uint64_t Finish() const;

private:
    void Mix(std::uint64_t word);

    std::uint64_t state_ = 0x9e3779b97f4a7c15ull;
    std::uint64_t length_ = 0;
    std::uint64_t carry_ = 0;
    unsigned carryBytes_ = 0;
};

// Random-access binary writer for the file under construction.
class LodPageOutput {
public:
    explicit LodPageOutput(const std::filesystem::path& path);

    void WriteAt(std::uint64_t offset, const void* data, std::uint64_t bytes);
    // Makes everything written so far visible to readers of the same file.
    void Flush();
    void Close();

private:
    std::filesystem::path path_;
    std::ofstream stream_;
};

// Appends values to one section, in order, hashing what it writes.
template <typename T> class ColumnWriter {
public:
    ColumnWriter(LodPageOutput& output, const SectionEntry& section, StreamHash& hash)
        : output_(&output), hash_(&hash), offset_(section.offset),
          end_(section.offset + section.bytes) {
        buffer_.reserve(kBufferElements);
    }

    void Push(const T& value) {
        buffer_.push_back(value);
        if (buffer_.size() == kBufferElements) {
            Flush();
        }
    }

    void Flush() {
        if (buffer_.empty()) {
            return;
        }
        const auto bytes = static_cast<std::uint64_t>(buffer_.size() * sizeof(T));
        if (offset_ + bytes > end_) {
            throw std::logic_error("LOD page column overflow");
        }
        output_->WriteAt(offset_, buffer_.data(), bytes);
        hash_->Update(buffer_.data(), bytes);
        offset_ += bytes;
        buffer_.clear();
    }

private:
    static constexpr std::size_t kBufferElements = 1u << 16u;
    LodPageOutput* output_;
    StreamHash* hash_;
    std::uint64_t offset_;
    std::uint64_t end_;
    std::vector<T> buffer_;
};

// Reads `count` values of type T starting at byte `offset` of a file, in large chunks.
template <typename T> class ValueReader {
public:
    ValueReader(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t count)
        : input_(path, std::ios::binary), remaining_(count) {
        if (!input_) {
            throw std::runtime_error("could not open " + path.string());
        }
        input_.seekg(static_cast<std::streamoff>(offset));
    }
    ValueReader(const std::filesystem::path& path, const SectionEntry& section)
        : ValueReader(path, section.offset, section.bytes / sizeof(T)) {}

    [[nodiscard]] T Next() {
        if (position_ == buffer_.size()) {
            Fill();
        }
        return buffer_[position_++];
    }

private:
    void Fill() {
        const auto count =
            static_cast<std::size_t>(std::min<std::uint64_t>(remaining_, kBufferElements));
        if (count == 0u) {
            throw std::logic_error("LOD page reader ran past its range");
        }
        buffer_.resize(count);
        input_.read(reinterpret_cast<char*>(buffer_.data()),
                    static_cast<std::streamsize>(count * sizeof(T)));
        if (!input_) {
            throw std::runtime_error("read failed while building LOD pages");
        }
        remaining_ -= count;
        position_ = 0;
    }

    static constexpr std::size_t kBufferElements = 1u << 18u;
    std::ifstream input_;
    std::uint64_t remaining_;
    std::vector<T> buffer_;
    std::size_t position_ = 0;
};

// Reads a range of values into memory at once.
template <typename T>
[[nodiscard]] std::vector<T> ReadValues(const std::filesystem::path& path, std::uint64_t offset,
                                        std::uint64_t count) {
    std::vector<T> values(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!input) {
        throw std::runtime_error("read failed while building LOD pages: " + path.string());
    }
    return values;
}

// Append-only scratch file of fixed-size values.
template <typename T> class ScratchWriter {
public:
    explicit ScratchWriter(const std::filesystem::path& path)
        : path_(path), stream_(path, std::ios::binary | std::ios::trunc) {
        if (!stream_) {
            throw std::runtime_error("could not create scratch file " + path.string());
        }
        buffer_.reserve(kBufferElements);
    }

    void Push(const T& value) {
        buffer_.push_back(value);
        if (buffer_.size() == kBufferElements) {
            Flush();
        }
    }

    void Close() {
        Flush();
        stream_.close();
        if (stream_.fail()) {
            throw std::runtime_error("could not write scratch file " + path_.string() +
                                     " (disk full?)");
        }
    }

    [[nodiscard]] std::uint64_t Count() const { return count_; }

private:
    void Flush() {
        stream_.write(reinterpret_cast<const char*>(buffer_.data()),
                      static_cast<std::streamsize>(buffer_.size() * sizeof(T)));
        count_ += buffer_.size();
        buffer_.clear();
        if (!stream_) {
            throw std::runtime_error("could not write scratch file " + path_.string() +
                                     " (disk full?)");
        }
    }

    static constexpr std::size_t kBufferElements = 1u << 16u;
    std::filesystem::path path_;
    std::ofstream stream_;
    std::vector<T> buffer_;
    std::uint64_t count_ = 0;
};

// Removes the file when it goes out of scope, unless it was moved into place.
class TemporaryFile {
public:
    explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {}
    ~TemporaryFile();
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    TemporaryFile(TemporaryFile&& other) noexcept : path_(std::move(other.path_)) {
        other.path_.clear();
    }
    TemporaryFile& operator=(TemporaryFile&&) = delete;

    [[nodiscard]] const std::filesystem::path& Path() const { return path_; }
    // rename replaces an existing file on every platform. On failure the old file survives and
    // this one is kept too (it is no longer removed), and the error names both.
    void MoveTo(const std::filesystem::path& destination);

private:
    std::filesystem::path path_;
};

// Hash of a byte range of a file, read back from disk. `read` gets the size of every chunk
// read and may throw to cancel.
[[nodiscard]] std::uint64_t HashFileRange(const std::filesystem::path& path, std::uint64_t offset,
                                          std::uint64_t bytes,
                                          const std::function<void(std::uint64_t)>& read = {});

} // namespace chmv::streaming::lod
