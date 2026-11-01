#include "streaming/lod/LodPageReader.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <span>
#include <stdexcept>
#include <utility>

namespace chmv::streaming::lod {
namespace {

// Pages a thread claims at once; small enough that several threads share a view's demand.
constexpr std::uint64_t kBatchBytes = 512ull << 10u;
// Neighbouring pages closer than this are read in one call, gap included: one larger read
// costs less than two seeks, on SSDs as well as on spinning disks.
constexpr std::uint64_t kMaxGapBytes = 128ull << 10u;
constexpr std::uint64_t kMaxReadBytes = 4ull << 20u;
// Finished pages wait for upload up to this amount before the threads pause.
constexpr std::uint64_t kMaxFinishedBytes = 64ull << 20u;

[[nodiscard]] unsigned ReaderThreadCount() {
    return std::clamp(std::thread::hardware_concurrency() / 2u, 2u, 4u);
}

} // namespace

LodPageReader::LodPageReader(std::shared_ptr<const LodPageFile> file, bool withChildren)
    : file_(std::move(file)), withChildren_(withChildren) {
    const auto count = ReaderThreadCount();
    stats_.threads = count;
    threads_.reserve(count);
    for (unsigned i = 0; i < count; ++i) {
        threads_.emplace_back([this](std::stop_token stop) { Run(stop); });
    }
}

LodPageReader::~LodPageReader() {
    for (auto& thread : threads_) {
        thread.request_stop();
    }
    wake_.notify_all();
}

void LodPageReader::SetDemand(std::vector<std::uint32_t> pages) {
    {
        std::lock_guard lock(mutex_);
        demand_ = std::move(pages);
        cursor_ = 0;
        // Finished pages the new list no longer asks for would only evict useful ones.
        const std::unordered_set<std::uint32_t> wanted(demand_.begin(), demand_.end());
        std::erase_if(finished_, [&](const LodPageData& page) {
            if (wanted.contains(page.page)) {
                return false;
            }
            stats_.finishedBytes -= (page.words.size() + page.children.size()) * 4u;
            delivered_.erase(page.page);
            return true;
        });
    }
    wake_.notify_all();
}

void LodPageReader::Take(std::vector<LodPageData>& out, std::uint64_t maxBytes) {
    std::uint64_t taken = 0;
    {
        std::lock_guard lock(mutex_);
        while (!finished_.empty() && taken < maxBytes) {
            auto& page = finished_.front();
            const auto bytes = (page.words.size() + page.children.size()) * 4u;
            taken += bytes;
            stats_.finishedBytes -= bytes;
            delivered_.erase(page.page);
            out.push_back(std::move(page));
            finished_.pop_front();
        }
    }
    if (taken != 0u) {
        wake_.notify_all();
    }
}

LodPageReaderStats LodPageReader::Stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

void LodPageReader::Run(const std::stop_token& stop) {
    std::vector<std::uint32_t> batch;
    try {
        std::ifstream input(file_->Path(), std::ios::binary);
        if (!input) {
            throw std::runtime_error("could not open " + file_->Path().string());
        }
        const auto pages = file_->Pages();
        while (!stop.stop_requested()) {
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, stop, [&] {
                    return cursor_ < demand_.size() && stats_.finishedBytes < kMaxFinishedBytes;
                });
                if (stop.stop_requested()) {
                    return;
                }
                batch.clear();
                std::uint64_t bytes = 0;
                while (cursor_ < demand_.size() && bytes < kBatchBytes) {
                    const auto page = demand_[cursor_++];
                    if (page < pages.size() && !delivered_.contains(page) &&
                        inFlight_.insert(page).second) {
                        batch.push_back(page);
                        bytes += pages[page].wordCount * 4ull;
                    }
                }
            }
            if (!batch.empty()) {
                ReadBatch(input, batch);
            }
        }
    } catch (const std::exception& error) {
        // The other threads keep going; release this batch so they can pick it up.
        std::lock_guard lock(mutex_);
        stats_.error = error.what();
        for (const auto page : batch) {
            inFlight_.erase(page);
        }
        cursor_ = 0;
    }
    wake_.notify_all();
}

void LodPageReader::ReadBatch(std::ifstream& input, std::vector<std::uint32_t>& batch) {
    std::sort(batch.begin(), batch.end(), [&](std::uint32_t a, std::uint32_t b) {
        return file_->PageStorage(a).offset < file_->PageStorage(b).offset;
    });

    std::vector<LodPageData> done(batch.size());
    std::vector<std::byte> buffer;
    std::uint64_t bytesRead = 0;
    std::uint64_t calls = 0;
    const auto begin = std::chrono::steady_clock::now();
    // Both the pages and their children blocks lie in page order, so one sorted batch reads
    // either with few calls. Compressed pages are decoded here, off the render thread; one that
    // does not decode is handed over empty and rejected on upload.
    const auto readAll = [&](auto storage, auto decode) {
        std::size_t first = 0;
        while (first < batch.size()) {
            const auto offset = storage(batch[first]).offset;
            auto end = offset + storage(batch[first]).bytes;
            auto last = first + 1u;
            while (last < batch.size()) {
                const auto next = storage(batch[last]);
                const auto nextEnd = next.offset + next.bytes;
                if (next.offset > end + kMaxGapBytes || nextEnd - offset > kMaxReadBytes) {
                    break;
                }
                end = nextEnd;
                ++last;
            }
            buffer.resize(static_cast<std::size_t>(end - offset));
            input.seekg(static_cast<std::streamoff>(offset));
            input.read(reinterpret_cast<char*>(buffer.data()),
                       static_cast<std::streamsize>(end - offset));
            if (!input) {
                throw std::runtime_error("page read failed in " + file_->Path().string());
            }
            ++calls;
            bytesRead += end - offset;
            for (auto i = first; i < last; ++i) {
                const auto range = storage(batch[i]);
                decode(done[i], std::span<const std::byte>(buffer).subspan(
                                    static_cast<std::size_t>(range.offset - offset),
                                    static_cast<std::size_t>(range.bytes)));
            }
            first = last;
        }
    };
    for (std::size_t i = 0; i < batch.size(); ++i) {
        done[i].page = batch[i];
    }
    readAll([&](std::uint32_t page) { return file_->PageStorage(page); },
            [&](LodPageData& page, std::span<const std::byte> stored) {
                static_cast<void>(file_->DecodePage(page.page, stored, page.words));
            });
    if (withChildren_) {
        readAll([&](std::uint32_t page) { return file_->ChildrenStorage(page); },
                [&](LodPageData& page, std::span<const std::byte> stored) {
                    static_cast<void>(file_->DecodeChildren(page.page, stored, page.children));
                });
    }
    const auto milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();

    std::lock_guard lock(mutex_);
    for (auto& page : done) {
        inFlight_.erase(page.page);
        stats_.finishedBytes += (page.words.size() + page.children.size()) * 4u;
        delivered_.insert(page.page);
        finished_.push_back(std::move(page));
    }
    stats_.pagesRead += done.size();
    stats_.bytesRead += bytesRead;
    stats_.readCalls += calls;
    stats_.readMilliseconds += milliseconds;
}

} // namespace chmv::streaming::lod
