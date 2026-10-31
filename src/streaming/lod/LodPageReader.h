#pragma once

#include "streaming/lod/LodPageFile.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace chmv::streaming::lod {

struct LodPageData {
    std::uint32_t page = 0;
    std::vector<std::uint32_t> words;
};

struct LodPageReaderStats {
    std::uint64_t pagesRead = 0;
    std::uint64_t bytesRead = 0;
    std::uint64_t readCalls = 0;
    double readMilliseconds = 0.0;
    std::uint64_t finishedBytes = 0;
    std::uint32_t threads = 0;
    std::string error;
};

// Reads pages from disk on a few background threads. The render thread hands over the complete,
// prioritized list of pages it is still missing whenever that list changes; the newest list
// always replaces the old one, so a fast camera never builds up a backlog of stale reads.
// Several reads are in flight at once (SSDs need queue depth), and pages that lie close
// together in the file are fetched with one read.
class LodPageReader {
public:
    explicit LodPageReader(std::shared_ptr<const LodPageFile> file);
    ~LodPageReader();

    LodPageReader(const LodPageReader&) = delete;
    LodPageReader& operator=(const LodPageReader&) = delete;

    void SetDemand(std::vector<std::uint32_t> pages);
    // Moves finished pages into `out` until at least `maxBytes` have been taken or none are left.
    void Take(std::vector<LodPageData>& out, std::uint64_t maxBytes);
    [[nodiscard]] LodPageReaderStats Stats() const;

private:
    void Run(const std::stop_token& stop);
    void ReadBatch(std::ifstream& input, std::vector<std::uint32_t>& batch);

    std::shared_ptr<const LodPageFile> file_;
    mutable std::mutex mutex_;
    std::condition_variable_any wake_;
    std::vector<std::uint32_t> demand_;
    std::size_t cursor_ = 0;
    std::unordered_set<std::uint32_t> inFlight_;
    std::unordered_set<std::uint32_t> delivered_;
    std::deque<LodPageData> finished_;
    LodPageReaderStats stats_;
    std::vector<std::jthread> threads_;
};

} // namespace chmv::streaming::lod
