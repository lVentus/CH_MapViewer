#include "streaming/lod/LodPageIo.h"

#include <algorithm>
#include <system_error>

namespace chmv::streaming::lod {
namespace {

constexpr std::uint64_t kMultiplyA = 0x9e3779b97f4a7c15ull;
constexpr std::uint64_t kMultiplyB = 0xbf58476d1ce4e5b9ull;
constexpr std::uint64_t kHashChunkBytes = 16ull << 20u;

[[nodiscard]] std::uint64_t RotateLeft(std::uint64_t value, unsigned bits) {
    return (value << bits) | (value >> (64u - bits));
}

} // namespace

void StreamHash::Mix(std::uint64_t word) {
    state_ ^= word * kMultiplyA;
    state_ = RotateLeft(state_, 31u) * kMultiplyB;
}

void StreamHash::Update(const void* data, std::uint64_t bytes) {
    const auto* in = static_cast<const unsigned char*>(data);
    length_ += bytes;
    while (bytes != 0u && carryBytes_ != 0u) {
        carry_ |= static_cast<std::uint64_t>(*in++) << (8u * carryBytes_);
        --bytes;
        if (++carryBytes_ == 8u) {
            Mix(carry_);
            carry_ = 0;
            carryBytes_ = 0;
        }
    }
    while (bytes >= 8u) {
        std::uint64_t word = 0;
        std::memcpy(&word, in, 8u);
        Mix(word);
        in += 8;
        bytes -= 8u;
    }
    while (bytes != 0u) {
        carry_ |= static_cast<std::uint64_t>(*in++) << (8u * carryBytes_);
        ++carryBytes_;
        --bytes;
    }
}

std::uint64_t StreamHash::Finish() const {
    auto copy = *this;
    if (copy.carryBytes_ != 0u) {
        copy.Mix(copy.carry_);
    }
    auto h = copy.state_ ^ copy.length_;
    h = (h ^ (h >> 30u)) * kMultiplyB;
    h = (h ^ (h >> 27u)) * 0x94d049bb133111ebull;
    return h ^ (h >> 31u);
}

LodPageOutput::LodPageOutput(const std::filesystem::path& path)
    : path_(path), stream_(path, std::ios::binary | std::ios::out | std::ios::trunc) {
    if (!stream_) {
        throw std::runtime_error("could not create " + path.string());
    }
}

void LodPageOutput::WriteAt(std::uint64_t offset, const void* data, std::uint64_t bytes) {
    stream_.seekp(static_cast<std::streamoff>(offset));
    stream_.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream_) {
        throw std::runtime_error("write failed: " + path_.string() + " (disk full?)");
    }
}

void LodPageOutput::Flush() {
    stream_.flush();
    if (!stream_) {
        throw std::runtime_error("write failed: " + path_.string() + " (disk full?)");
    }
}

void LodPageOutput::Close() {
    stream_.close();
    if (stream_.fail()) {
        throw std::runtime_error("could not finish writing " + path_.string());
    }
}

TemporaryFile::~TemporaryFile() {
    if (!path_.empty()) {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
}

void TemporaryFile::MoveTo(const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::rename(path_, destination, error);
    if (error) {
        // The file is complete; keep it so the work is not lost.
        const auto kept = path_;
        path_.clear();
        throw std::filesystem::filesystem_error("could not replace the old file, the new one "
                                                "was kept under its temporary name",
                                                kept, destination, error);
    }
    path_.clear();
}

std::uint64_t HashFileRange(const std::filesystem::path& path, std::uint64_t offset,
                            std::uint64_t bytes, const std::function<void(std::uint64_t)>& read) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<char> buffer(static_cast<std::size_t>(std::min(bytes, kHashChunkBytes)));
    StreamHash hash;
    while (bytes != 0u) {
        const auto count = std::min<std::uint64_t>(bytes, buffer.size());
        input.read(buffer.data(), static_cast<std::streamsize>(count));
        if (!input) {
            throw std::runtime_error("read failed while checking " + path.string());
        }
        hash.Update(buffer.data(), count);
        bytes -= count;
        if (read) {
            read(count);
        }
    }
    return hash.Finish();
}

} // namespace chmv::streaming::lod
