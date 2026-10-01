#pragma once

#include <array>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace chmv::streaming::analysis::detail {

class TextSourceScanner {
public:
    explicit TextSourceScanner(const std::filesystem::path& path)
        : buffer_(BufferSize) {
#ifdef _WIN32
        _wfopen_s(&file_, path.c_str(), L"rb");
#else
        file_ = std::fopen(path.c_str(), "rb");
#endif
        if (!file_) {
            throw std::runtime_error("could not open file: " + path.string());
        }
        // stdio still owns the file handle; our larger scanner buffer minimizes calls into it.
        std::setvbuf(file_, nullptr, _IONBF, 0);
    }

    ~TextSourceScanner() {
        if (file_) {
            std::fclose(file_);
        }
    }

    TextSourceScanner(const TextSourceScanner&) = delete;
    TextSourceScanner& operator=(const TextSourceScanner&) = delete;

    void Seek(std::uint64_t byteOffset) {
#ifdef _WIN32
        if (byteOffset > static_cast<std::uint64_t>(std::numeric_limits<__int64>::max()) ||
            _fseeki64(file_, static_cast<__int64>(byteOffset), SEEK_SET) != 0) {
            throw std::runtime_error("could not seek text source");
        }
#else
        if (byteOffset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            fseeko(file_, static_cast<off_t>(byteOffset), SEEK_SET) != 0) {
            throw std::runtime_error("could not seek text source");
        }
#endif
        position_ = 0;
        size_ = 0;
        logicalOffset_ = byteOffset;
    }

    template <typename T>
    T Read() {
        return ReadWithOffset<T>().first;
    }

    template <typename T>
    std::pair<T, std::uint64_t> ReadWithOffset() {
        const auto token = NextToken();
        T value{};
        const auto* begin = token.text.data();
        const auto* end = begin + token.text.size();
        const auto [ptr, ec] = std::from_chars(begin, end, value);
        if (ec != std::errc{} || ptr != end) {
            throw std::runtime_error("invalid numeric token at byte offset " +
                                     std::to_string(token.byteOffset));
        }
        return {value, token.byteOffset};
    }

private:
    struct TokenView {
        std::string_view text;
        std::uint64_t byteOffset = 0;
    };

    TokenView NextToken() {
        std::size_t tokenSize = 0;
        std::uint64_t tokenOffset = 0;
        char ch = 0;
        std::uint64_t offset = 0;

        while (ReadChar(ch, offset)) {
            if (ch == '#') {
                SkipLine();
                continue;
            }
            if (!IsSpace(ch)) {
                tokenOffset = offset;
                tokenBuffer_[tokenSize++] = ch;
                break;
            }
        }
        if (tokenSize == 0) {
            throw std::runtime_error("unexpected end of file");
        }

        while (ReadChar(ch, offset)) {
            if (IsSpace(ch)) {
                break;
            }
            if (tokenSize == tokenBuffer_.size()) {
                throw std::runtime_error("numeric token is unexpectedly long at byte offset " +
                                         std::to_string(tokenOffset));
            }
            tokenBuffer_[tokenSize++] = ch;
        }
        return {std::string_view(tokenBuffer_.data(), tokenSize), tokenOffset};
    }

    bool ReadChar(char& ch, std::uint64_t& offset) {
        if (position_ == size_) {
            size_ = std::fread(buffer_.data(), 1, buffer_.size(), file_);
            position_ = 0;
            if (size_ == 0) {
                return false;
            }
        }
        offset = logicalOffset_;
        ch = buffer_[position_++];
        ++logicalOffset_;
        return true;
    }

    void SkipLine() {
        char ch = 0;
        std::uint64_t offset = 0;
        while (ReadChar(ch, offset) && ch != '\n') {
        }
    }

    static bool IsSpace(char ch) {
        return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
    }

    static constexpr std::size_t BufferSize = 8u << 20u;
    std::FILE* file_ = nullptr;
    std::vector<char> buffer_;
    std::array<char, 128> tokenBuffer_{};
    std::size_t position_ = 0;
    std::size_t size_ = 0;
    std::uint64_t logicalOffset_ = 0;
};

} // namespace chmv::streaming::analysis::detail
