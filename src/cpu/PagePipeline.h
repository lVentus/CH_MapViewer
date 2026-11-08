#pragma once

#include "cpu/CpuPasses.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPagePlanner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace chmv::cpu {

class CpuWorkers;

// The pages a CPU pass reads: straight from the RAM image when the loader made one, otherwise
// read from the file and kept while they are used (the least recently used go first once the
// budget is reached). Also keeps the CH edges of their records and, for the birth-ordered
// filter, their records ordered by highest level.
class CpuPageStore {
public:
    CpuPageStore(std::shared_ptr<const streaming::lod::LodPageFile> file,
                 std::shared_ptr<const std::vector<std::uint32_t>> pageImage,
                 std::shared_ptr<const std::vector<std::uint32_t>> childrenImage);

    // Makes these pages readable (with their children blocks / record order when asked).
    // Pages that cannot be read or do not match their directory entry stay unavailable (and
    // children blocks likewise), as damaged pages do on the GPU.
    void Fetch(std::span<const std::uint32_t> pages, bool children, bool order);
    // Bytes the store would hold for a pass over these pages, and how much one pass may hold.
    [[nodiscard]] std::uint64_t PassBytes(std::span<const std::uint32_t> pages,
                                          bool children) const;
    [[nodiscard]] std::uint64_t PassLimit() const { return passLimit_; }

    [[nodiscard]] const streaming::lod::LodPageFile& File() const { return *file_; }
    [[nodiscard]] bool Has(std::uint32_t page) const { return slotOfPage_[page] != kNoSlot; }
    [[nodiscard]] std::span<const std::uint32_t> Words(std::uint32_t page) const;
    // Empty unless fetched with children.
    [[nodiscard]] std::span<const std::uint32_t> Children(std::uint32_t page) const;
    [[nodiscard]] std::span<const std::uint32_t> Edges(std::uint32_t page) const;
    // Record numbers, highest level first; empty unless fetched with order.
    [[nodiscard]] std::span<const std::uint8_t> Order(std::uint32_t page) const;

private:
    static constexpr std::uint32_t kNoSlot = 0xffffffffu;
    static constexpr std::uint8_t kBadWords = 1u;
    static constexpr std::uint8_t kBadChildren = 2u;

    struct Slot {
        std::uint32_t page = 0;
        std::span<const std::uint32_t> words;
        std::span<const std::uint32_t> children;
        std::vector<std::uint32_t> ownWords;
        std::vector<std::uint32_t> ownChildren;
        std::vector<std::uint32_t> edges;
        std::vector<std::uint8_t> order;
        std::uint64_t lastUse = 0;
    };

    void Evict();
    [[nodiscard]] std::uint64_t SlotBytes(const Slot& slot) const;

    std::shared_ptr<const streaming::lod::LodPageFile> file_;
    std::shared_ptr<const std::vector<std::uint32_t>> pageImage_;
    std::shared_ptr<const std::vector<std::uint32_t>> childrenImage_;
    std::vector<std::uint32_t> slotOfPage_;
    // kBadWords / kBadChildren per page once found damaged.
    std::vector<std::uint8_t> corrupt_;
    std::uint64_t passLimit_ = 0;
    std::vector<Slot> slots_;
    std::uint64_t bytes_ = 0;
    std::uint64_t use_ = 0;
};

// One CPU pass over the pages, mirroring the GPU kernels: plan the pages of the query (from
// the visit directory for traversals), find the candidates with the filter, expand them, and
// turn what is drawn into segments. With stepBase (the error code of one pixel) it also counts
// the segments each tolerance step would draw from the same pages. cullSegments drops drawn
// segments whose chord misses the query box.
[[nodiscard]] CpuResult RunPagePass(CpuPageStore& store, CpuWorkers& workers,
                                    const streaming::lod::LodPageQuery& query, CpuFilter filter,
                                    CpuExpansion expansion,
                                    std::optional<std::uint32_t> stepBase = std::nullopt,
                                    bool cullSegments = false);

} // namespace chmv::cpu
