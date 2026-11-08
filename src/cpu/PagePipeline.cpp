#include "cpu/PagePipeline.h"

#include "cpu/CpuWorkQueue.h"
#include "cpu/CpuWorkers.h"
#include "data/SystemMemory.h"
#include "streaming/lod/LodPagePlanner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace chmv::cpu {
namespace {

namespace lod = streaming::lod;

using Clock = std::chrono::steady_clock;
// Pages read from the file are kept up to this much (pages of the current pass always stay).
constexpr std::uint64_t kStoreBudgetBytes = 512ull << 20u;
constexpr std::size_t kPagesPerTask = 8;
constexpr std::size_t kItemsPerTask = 2048;
constexpr std::uint32_t kNoItem = 0xffffffffu;

[[nodiscard]] double Milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

[[nodiscard]] std::size_t TaskCount(std::size_t items, std::size_t perTask) {
    return (items + perTask - 1u) / perTask;
}

[[nodiscard]] bool InLevelRange(std::uint32_t packed, std::uint32_t level) {
    return level >= (packed & 0xffffu) && level <= (packed >> 16u);
}

[[nodiscard]] std::uint32_t HighestLevel(std::span<const std::uint32_t> words,
                                         std::uint32_t local) {
    const auto* record = words.data() + lod::kPageHeaderWords + local * lod::kRecordWords;
    return std::max(record[1] >> 16u, record[2] >> 16u);
}

// Concatenates what every worker found.
[[nodiscard]] std::vector<std::uint32_t> Gather(std::vector<std::vector<std::uint32_t>>& found) {
    std::size_t total = 0;
    for (const auto& part : found) {
        total += part.size();
    }
    std::vector<std::uint32_t> all;
    all.reserve(total);
    for (auto& part : found) {
        all.insert(all.end(), part.begin(), part.end());
        part.clear();
    }
    return all;
}

// The traversal helpers of lod_page_traversal.glsl, on the store instead of the pool.
class Walk {
  public:
    Walk(const CpuPageStore& store, const lod::LodPageQuery& query)
        : store_(store), threshold_(query.threshold), cull_(lod::TraversalCullBox(query)) {}

    [[nodiscard]] const std::uint32_t* Record(std::uint32_t item) const {
        return store_.Words(item >> 8u).data() + lod::kPageHeaderWords +
               (item & 0xffu) * lod::kRecordWords;
    }

    // The node's whole geometry lies within its error of its chord.
    [[nodiscard]] bool Visible(std::uint32_t item) const {
        return Touches(item, std::int64_t{lod::ErrorBound(Record(item)[3] & 0xffffu)});
    }

    // The chord itself touches the cull box (segment culling).
    [[nodiscard]] bool ChordInView(std::uint32_t item) const { return Touches(item, 0); }

    // Expanded unless the node is already accurate enough or is an original edge.
    [[nodiscard]] bool Expands(std::uint32_t item) const {
        if ((Record(item)[3] & 0xffffu) <= threshold_) {
            return false;
        }
        const auto block = store_.Children(item >> 8u);
        const auto local = item & 0xffu;
        return (block[local >> 5u] >> (local & 31u) & 1u) != 0u;
    }

    // Child 0 or 1 of an expanding node, or kNoItem if its page is not there (outside the view).
    [[nodiscard]] std::uint32_t Child(std::uint32_t item, std::uint32_t which,
                                      std::uint64_t& missing) const {
        const auto block = store_.Children(item >> 8u);
        const auto local = item & 0xffu;
        auto rank = static_cast<std::uint32_t>(
            std::popcount(block[local >> 5u] & ((1u << (local & 31u)) - 1u)));
        for (std::uint32_t word = 0; word < (local >> 5u); ++word) {
            rank += static_cast<std::uint32_t>(std::popcount(block[word]));
        }
        const auto child = block[lod::kChildMaskWords + rank * 2u + which];
        // A page the store holds without children (fetched for another pass) counts as missing
        // too; the planner never needs it for a visible node.
        if (child == lod::kNoChildRef || !store_.Has(child >> 8u) ||
            store_.Children(child >> 8u).empty()) {
            ++missing;
            return kNoItem;
        }
        return child;
    }

  private:
    // The chord's box, grown by `grow`, touches the cull box.
    [[nodiscard]] bool Touches(std::uint32_t item, std::int64_t grow) const {
        const auto words = store_.Words(item >> 8u);
        const auto* record = Record(item);
        const auto vertexBase = lod::kPageHeaderWords + (words[0] & 0xffffu) * lod::kRecordWords;
        const auto coordinate = [&](std::uint32_t vertex, std::uint32_t axis) {
            return std::int64_t{
                std::bit_cast<std::int32_t>(words[vertexBase + vertex * lod::kVertexWords + axis])};
        };
        const auto a = record[0] & 0xffffu;
        const auto b = record[0] >> 16u;
        return std::max(coordinate(a, 0), coordinate(b, 0)) >= cull_[0] - grow &&
               std::min(coordinate(a, 0), coordinate(b, 0)) <= cull_[2] + grow &&
               std::max(coordinate(a, 1), coordinate(b, 1)) >= cull_[1] - grow &&
               std::min(coordinate(a, 1), coordinate(b, 1)) <= cull_[3] + grow;
    }

    const CpuPageStore& store_;
    std::uint32_t threshold_;
    std::array<std::int32_t, 4> cull_;
};

} // namespace

CpuPageStore::CpuPageStore(std::shared_ptr<const lod::LodPageFile> file,
                           std::shared_ptr<const std::vector<std::uint32_t>> pageImage,
                           std::shared_ptr<const std::vector<std::uint32_t>> childrenImage)
    : file_(std::move(file)), pageImage_(std::move(pageImage)),
      childrenImage_(std::move(childrenImage)),
      slotOfPage_(static_cast<std::size_t>(file_->Header().pageCount), kNoSlot),
      corrupt_(static_cast<std::size_t>(file_->Header().pageCount), std::uint8_t{0}) {
    // One pass may hold a quarter of the RAM that is free now, at least 512 MiB.
    passLimit_ = std::max<std::uint64_t>(kStoreBudgetBytes,
                                         data::QuerySystemMemoryInfo().availablePhysicalBytes / 4u);
}

std::span<const std::uint32_t> CpuPageStore::Words(std::uint32_t page) const {
    return slots_[slotOfPage_[page]].words;
}

std::span<const std::uint32_t> CpuPageStore::Children(std::uint32_t page) const {
    return slots_[slotOfPage_[page]].children;
}

std::span<const std::uint32_t> CpuPageStore::Edges(std::uint32_t page) const {
    return slots_[slotOfPage_[page]].edges;
}

std::span<const std::uint8_t> CpuPageStore::Order(std::uint32_t page) const {
    return slots_[slotOfPage_[page]].order;
}

std::uint64_t CpuPageStore::SlotBytes(const Slot& slot) const {
    return (slot.ownWords.size() + slot.ownChildren.size() + slot.edges.size()) * 4u +
           slot.order.size();
}

std::uint64_t CpuPageStore::PassBytes(std::span<const std::uint32_t> pages, bool children) const {
    std::uint64_t bytes = 0;
    for (const auto page : pages) {
        if (!pageImage_) {
            bytes += file_->Pages()[page].wordCount * 4ull;
        }
        if (children && !childrenImage_) {
            bytes += file_->VisitPages()[page].wordCount * 4ull;
        }
        bytes += file_->Pages()[page].recordCount * 5ull;
    }
    return bytes;
}

void CpuPageStore::Fetch(std::span<const std::uint32_t> pages, bool children, bool order) {
    ++use_;
    const auto entries = file_->Pages();
    std::vector<std::uint32_t> readWords;
    std::vector<std::uint32_t> readChildren;
    for (const auto page : pages) {
        if ((corrupt_[page] & kBadWords) != 0u) {
            continue;
        }
        if (slotOfPage_[page] == kNoSlot) {
            auto& slot = slots_.emplace_back();
            slotOfPage_[page] = static_cast<std::uint32_t>(slots_.size() - 1u);
            slot.page = page;
            if (pageImage_) {
                slot.words = std::span<const std::uint32_t>(*pageImage_)
                                 .subspan(static_cast<std::size_t>(file_->PageWordOffset(page)),
                                          entries[page].wordCount);
                // The RAM image keeps pages that did not decode as zeros.
                if (!lod::PageMatchesEntry(entries[page], slot.words)) {
                    slot.words = {};
                    corrupt_[page] |= kBadWords;
                    continue;
                }
            } else {
                readWords.push_back(page);
            }
        }
        auto& slot = slots_[slotOfPage_[page]];
        slot.lastUse = use_;
        if (children && slot.children.empty() && (corrupt_[page] & kBadChildren) == 0u) {
            if (childrenImage_) {
                slot.children =
                    std::span<const std::uint32_t>(*childrenImage_)
                        .subspan(static_cast<std::size_t>(file_->ChildrenWordOffset(page)),
                                 file_->VisitPages()[page].wordCount);
                if (!lod::ChildrenMatchEntry(entries, file_->VisitPages()[page],
                                             entries[page].recordCount, slot.children)) {
                    slot.children = {};
                    corrupt_[page] |= kBadChildren;
                }
            } else {
                readChildren.push_back(page);
            }
        }
    }

    // Reads from the file in file order, decoding as the background reader does. A page (or
    // children block) that does not decode or does not match its directory entry is marked bad
    // and never read again, as on the GPU.
    const auto read = [&](std::vector<std::uint32_t>& list, bool readingChildren) {
        if (list.empty()) {
            return;
        }
        const auto storage = [&](std::uint32_t page) {
            return readingChildren ? file_->ChildrenStorage(page) : file_->PageStorage(page);
        };
        std::sort(list.begin(), list.end(), [&](std::uint32_t a, std::uint32_t b) {
            return storage(a).offset < storage(b).offset;
        });
        std::ifstream input(file_->Path(), std::ios::binary);
        if (!input) {
            throw std::runtime_error("could not open " + file_->Path().string());
        }
        std::vector<std::byte> stored;
        for (const auto page : list) {
            const auto range = storage(page);
            stored.resize(static_cast<std::size_t>(range.bytes));
            input.seekg(static_cast<std::streamoff>(range.offset));
            input.read(reinterpret_cast<char*>(stored.data()),
                       static_cast<std::streamsize>(stored.size()));
            auto& slot = slots_[slotOfPage_[page]];
            auto& words = readingChildren ? slot.ownChildren : slot.ownWords;
            bool good = false;
            if (readingChildren) {
                good = input && file_->DecodeChildren(page, stored, words) &&
                       lod::ChildrenMatchEntry(entries, file_->VisitPages()[page],
                                               entries[page].recordCount, words);
            } else {
                good = input && file_->DecodePage(page, stored, words) &&
                       lod::PageMatchesEntry(entries[page], words);
            }
            if (!good) {
                input.clear();
                words.clear();
                corrupt_[page] |= readingChildren ? kBadChildren : kBadWords;
                continue;
            }
            (readingChildren ? slot.children : slot.words) = words;
        }
    };
    read(readWords, false);
    read(readChildren, true);

    for (const auto page : pages) {
        const auto index = slotOfPage_[page];
        if (index == kNoSlot) {
            continue;
        }
        auto& slot = slots_[index];
        if (slot.edges.empty() && !slot.words.empty()) {
            try {
                slot.edges =
                    file_->ReadRecordEdges(entries[page].firstRecord, entries[page].recordCount);
            } catch (const std::exception&) {
                corrupt_[page] |= kBadWords;
                slot.words = {};
            }
        }
        if (order && slot.order.empty() && !slot.words.empty()) {
            // Highest level first; equal levels by record number, highest first (as on the GPU).
            const auto count = entries[page].recordCount;
            std::vector<std::uint32_t> keys(count);
            for (std::uint32_t local = 0; local < count; ++local) {
                keys[local] = ((HighestLevel(slot.words, local) + 1u) << 8u) | local;
            }
            std::sort(keys.begin(), keys.end(), std::greater<>());
            slot.order.resize(count);
            for (std::uint32_t k = 0; k < count; ++k) {
                slot.order[k] = static_cast<std::uint8_t>(keys[k] & 0xffu);
            }
        }
    }
    bytes_ = 0;
    bool unreadable = false;
    for (const auto& slot : slots_) {
        bytes_ += SlotBytes(slot);
        unreadable = unreadable || slot.words.empty();
    }
    if (unreadable || bytes_ > kStoreBudgetBytes) {
        Evict();
    }
}

// Drops unreadable pages always, and pages not used by this fetch while over budget, the least
// recently used first.
void CpuPageStore::Evict() {
    std::vector<std::uint32_t> order(slots_.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return slots_[a].lastUse < slots_[b].lastUse;
    });
    std::vector<bool> drop(slots_.size(), false);
    auto bytes = bytes_;
    for (const auto index : order) {
        const auto& slot = slots_[index];
        const bool unreadable = slot.words.empty();
        if (!unreadable && (slot.lastUse == use_ || bytes <= kStoreBudgetBytes)) {
            continue;
        }
        drop[index] = true;
        bytes -= SlotBytes(slot);
    }
    std::vector<Slot> kept;
    kept.reserve(slots_.size());
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        if (drop[i]) {
            slotOfPage_[slots_[i].page] = kNoSlot;
        } else {
            slotOfPage_[slots_[i].page] = static_cast<std::uint32_t>(kept.size());
            kept.push_back(std::move(slots_[i]));
        }
    }
    slots_ = std::move(kept);
    bytes_ = bytes;
}

namespace {

// Segments each tolerance step would draw from these pages, as lod_tolerance_counts.comp counts
// them: a record is drawn at threshold T when its error is within T and it is a root at the
// level, or lies below a parent there whose error is above T, so on a range of steps. With
// `cull`, records whose chord misses its view are left out.
[[nodiscard]] std::array<std::uint64_t, renderer::kToleranceSteps>
CountSteps(const CpuPageStore& store, CpuWorkers& workers, std::span<const std::uint32_t> pages,
           std::uint32_t level, std::uint32_t stepBase, const Walk* cull) {
    constexpr auto kSteps = static_cast<std::uint32_t>(renderer::kToleranceSteps);
    const auto firstStepReaching = [&](std::uint32_t code) {
        if (code <= stepBase) {
            return 0u;
        }
        const auto steps = (code - stepBase + renderer::kCodesPerToleranceStep - 1u) /
                           renderer::kCodesPerToleranceStep;
        return std::min(steps, kSteps);
    };
    std::vector<std::array<std::int64_t, kSteps + 1u>> changes(workers.Threads());
    workers.Run(TaskCount(pages.size(), kPagesPerTask), [&](std::size_t task, unsigned worker) {
        auto& out = changes[worker];
        const auto last = std::min(pages.size(), (task + 1u) * kPagesPerTask);
        for (auto index = task * kPagesPerTask; index < last; ++index) {
            const auto words = store.Words(pages[index]);
            const auto count = words[0] & 0xffffu;
            for (std::uint32_t local = 0; local < count; ++local) {
                const auto* record =
                    words.data() + lod::kPageHeaderWords + local * lod::kRecordWords;
                const bool root = InLevelRange(record[1], level);
                if ((!root && !InLevelRange(record[2], level)) ||
                    (cull != nullptr && !cull->ChordInView((pages[index] << 8u) | local))) {
                    continue;
                }
                const auto first = firstStepReaching(record[3] & 0xffffu);
                const auto end = root ? kSteps : firstStepReaching(record[3] >> 16u);
                if (first < end) {
                    ++out[first];
                    --out[end];
                }
            }
        }
    });
    std::array<std::uint64_t, renderer::kToleranceSteps> counts{};
    std::int64_t running = 0;
    for (std::uint32_t step = 0; step < kSteps; ++step) {
        for (const auto& out : changes) {
            running += out[step];
        }
        counts[step] = static_cast<std::uint64_t>(running);
    }
    return counts;
}

} // namespace

CpuResult RunPagePass(CpuPageStore& store, CpuWorkers& workers, const lod::LodPageQuery& query,
                      CpuFilter filter, CpuExpansion expansion,
                      std::optional<std::uint32_t> stepBase, bool cullSegments) {
    CpuResult result;
    result.threads = workers.Threads();
    const auto& file = store.File();
    const bool traverses = expansion == CpuExpansion::Dfs || expansion == CpuExpansion::Bfs ||
                           expansion == CpuExpansion::Queue;
    const auto directory = traverses ? file.VisitPages() : file.Pages();
    const auto groups = traverses ? file.VisitGroups() : file.Groups();
    const auto threads = workers.Threads();
    std::vector<std::vector<std::uint32_t>> found(threads);

    // Pages: the planner's choice, or for culling every page tested on its own (that is the
    // filter's own work, as on the GPU).
    auto begin = Clock::now();
    std::vector<std::uint32_t> pages;
    if (filter == CpuFilter::Culling) {
        const auto tasks = TaskCount(directory.size(), kItemsPerTask);
        workers.Run(tasks, [&](std::size_t task, unsigned worker) {
            const auto first = task * kItemsPerTask;
            const auto last = std::min(directory.size(), first + kItemsPerTask);
            for (auto page = first; page < last; ++page) {
                if (lod::MayDraw(directory[page], query)) {
                    found[worker].push_back(static_cast<std::uint32_t>(page));
                }
            }
        });
        pages = Gather(found);
        std::sort(pages.begin(), pages.end());
        result.filterMs += Milliseconds(begin);
    } else {
        lod::CollectLodPages(groups, directory, query, pages);
        result.prepareMs += Milliseconds(begin);
    }
    begin = Clock::now();
    if (const auto bytes = store.PassBytes(pages, traverses); bytes > store.PassLimit()) {
        result.unavailable = "this view needs " + std::to_string(bytes >> 20u) +
                             " MiB of pages, more than the CPU may hold (" +
                             std::to_string(store.PassLimit() >> 20u) + " MiB)";
        return result;
    }
    store.Fetch(pages, traverses, filter == CpuFilter::BirthOrdered);
    std::erase_if(pages, [&](std::uint32_t page) {
        return !store.Has(page) || (traverses && store.Children(page).empty());
    });
    result.pages = static_cast<std::uint32_t>(pages.size());
    result.prepareMs += Milliseconds(begin);

    // Candidates: records whose root or child interval holds the level.
    begin = Clock::now();
    const auto level = query.level;
    const auto candidateOf = [&](std::span<const std::uint32_t> words, std::uint32_t local) {
        const auto* record = words.data() + lod::kPageHeaderWords + local * lod::kRecordWords;
        return InLevelRange(record[1], level) || InLevelRange(record[2], level);
    };
    std::vector<std::uint32_t> candidates;
    if (filter == CpuFilter::TwoPass) {
        // Count per page (keeping which records matched), then every page writes its candidates
        // to its own place: no per-thread lists to merge, and the order is the page order.
        std::vector<std::array<std::uint32_t, lod::kPageRecordCapacity / 32u>> masks(pages.size());
        std::vector<std::uint32_t> offsets(pages.size() + 1u, 0u);
        workers.Run(TaskCount(pages.size(), kPagesPerTask), [&](std::size_t task, unsigned) {
            const auto last = std::min(pages.size(), (task + 1u) * kPagesPerTask);
            for (auto index = task * kPagesPerTask; index < last; ++index) {
                const auto words = store.Words(pages[index]);
                auto& mask = masks[index];
                mask.fill(0u);
                std::uint32_t count = 0;
                for (std::uint32_t local = 0; local < (words[0] & 0xffffu); ++local) {
                    if (candidateOf(words, local)) {
                        mask[local >> 5u] |= 1u << (local & 31u);
                        ++count;
                    }
                }
                offsets[index + 1u] = count;
            }
        });
        for (std::size_t index = 0; index < pages.size(); ++index) {
            offsets[index + 1u] += offsets[index];
        }
        candidates.resize(offsets.back());
        workers.Run(TaskCount(pages.size(), kPagesPerTask), [&](std::size_t task, unsigned) {
            const auto last = std::min(pages.size(), (task + 1u) * kPagesPerTask);
            for (auto index = task * kPagesPerTask; index < last; ++index) {
                auto at = offsets[index];
                for (std::uint32_t local = 0; local < lod::kPageRecordCapacity; ++local) {
                    if ((masks[index][local >> 5u] >> (local & 31u) & 1u) != 0u) {
                        candidates[at++] = (pages[index] << 8u) | local;
                    }
                }
            }
        });
    } else {
        workers.Run(TaskCount(pages.size(), kPagesPerTask), [&](std::size_t task, unsigned worker) {
            auto& out = found[worker];
            const auto last = std::min(pages.size(), (task + 1u) * kPagesPerTask);
            for (auto index = task * kPagesPerTask; index < last; ++index) {
                const auto page = pages[index];
                const auto words = store.Words(page);
                const auto count = words[0] & 0xffffu;
                const auto keep = [&](std::uint32_t local) {
                    if (candidateOf(words, local)) {
                        out.push_back((page << 8u) | local);
                    }
                };
                if (filter == CpuFilter::BirthOrdered) {
                    // Records of the level form a prefix of the order.
                    const auto order = store.Order(page);
                    std::uint32_t low = 0;
                    std::uint32_t high = count;
                    while (low < high) {
                        const auto middle = (low + high) / 2u;
                        if (HighestLevel(words, order[middle]) >= level) {
                            low = middle + 1u;
                        } else {
                            high = middle;
                        }
                    }
                    for (std::uint32_t k = 0; k < low; ++k) {
                        keep(order[k]);
                    }
                } else {
                    for (std::uint32_t local = 0; local < count; ++local) {
                        keep(local);
                    }
                }
            }
        });
        candidates = Gather(found);
    }
    result.candidates = candidates.size();
    result.filterMs += Milliseconds(begin);

    // Expansion, as the kernels do it.
    begin = Clock::now();
    const Walk walk(store, query);
    const auto threshold = query.threshold;
    std::vector<std::uint64_t> missing(threads, 0u);
    std::vector<std::uint32_t> drawn;
    // A segment to draw, unless culling drops it.
    const auto draw = [&](std::uint32_t item, unsigned worker) {
        if (!cullSegments || walk.ChordInView(item)) {
            found[worker].push_back(item);
        }
    };
    if (expansion == CpuExpansion::Roots || expansion == CpuExpansion::Intervals) {
        const bool childIntervals = expansion == CpuExpansion::Intervals;
        workers.Run(
            TaskCount(candidates.size(), kItemsPerTask), [&](std::size_t task, unsigned worker) {
                const auto last = std::min(candidates.size(), (task + 1u) * kItemsPerTask);
                for (auto i = task * kItemsPerTask; i < last; ++i) {
                    const auto* record = walk.Record(candidates[i]);
                    const auto errorCode = record[3] & 0xffffu;
                    const bool asChild = childIntervals && InLevelRange(record[2], level) &&
                                         threshold < (record[3] >> 16u);
                    if (errorCode <= threshold && (InLevelRange(record[1], level) || asChild)) {
                        draw(candidates[i], worker);
                    }
                }
            });
        drawn = Gather(found);
    } else if (expansion == CpuExpansion::Dfs) {
        std::vector<std::vector<std::uint32_t>> stacks(threads);
        workers.Run(TaskCount(candidates.size(), kItemsPerTask / 8u), [&](std::size_t task,
                                                                          unsigned worker) {
            auto& stack = stacks[worker];
            const auto perTask = kItemsPerTask / 8u;
            const auto last = std::min(candidates.size(), (task + 1u) * perTask);
            for (auto i = task * perTask; i < last; ++i) {
                if (!InLevelRange(walk.Record(candidates[i])[1], level)) {
                    continue;
                }
                stack.push_back(candidates[i]);
                while (!stack.empty()) {
                    const auto node = stack.back();
                    stack.pop_back();
                    if (!walk.Visible(node)) {
                        continue;
                    }
                    if (!walk.Expands(node)) {
                        draw(node, worker);
                        continue;
                    }
                    for (std::uint32_t which = 2u; which > 0u; --which) {
                        const auto child = walk.Child(node, which - 1u, missing[worker]);
                        if (child != kNoItem) {
                            stack.push_back(child);
                        }
                    }
                }
            }
        });
        drawn = Gather(found);
    } else if (expansion == CpuExpansion::Queue) {
        std::vector<std::uint32_t> roots;
        for (const auto item : candidates) {
            if (InLevelRange(walk.Record(item)[1], level)) {
                roots.push_back(item);
            }
        }
        // Room for every record of the pages once; a node visited more often than that (several
        // parents at one level) is walked by its worker.
        std::size_t records = 0;
        for (const auto page : pages) {
            records += store.Words(page)[0] & 0xffffu;
        }
        RunWorkQueue(workers, roots, records,
                     [&](std::uint32_t node, unsigned worker, std::array<std::uint32_t, 2>& next) {
                         if (!walk.Visible(node)) {
                             return 0;
                         }
                         if (!walk.Expands(node)) {
                             draw(node, worker);
                             return 0;
                         }
                         int count = 0;
                         for (std::uint32_t which = 0; which < 2u; ++which) {
                             const auto child = walk.Child(node, which, missing[worker]);
                             if (child != kNoItem) {
                                 next[static_cast<std::size_t>(count++)] = child;
                             }
                         }
                         return count;
                     });
        drawn = Gather(found);
    } else {
        std::vector<std::vector<std::uint32_t>> next(threads);
        std::vector<std::uint32_t> frontier;
        for (const auto item : candidates) {
            if (InLevelRange(walk.Record(item)[1], level)) {
                frontier.push_back(item);
            }
        }
        while (!frontier.empty()) {
            workers.Run(
                TaskCount(frontier.size(), kItemsPerTask), [&](std::size_t task, unsigned worker) {
                    const auto last = std::min(frontier.size(), (task + 1u) * kItemsPerTask);
                    for (auto i = task * kItemsPerTask; i < last; ++i) {
                        const auto node = frontier[i];
                        if (!walk.Visible(node)) {
                            continue;
                        }
                        if (!walk.Expands(node)) {
                            draw(node, worker);
                            continue;
                        }
                        for (std::uint32_t which = 0; which < 2u; ++which) {
                            const auto child = walk.Child(node, which, missing[worker]);
                            if (child != kNoItem) {
                                next[worker].push_back(child);
                            }
                        }
                    }
                });
            frontier = Gather(next);
        }
        drawn = Gather(found);
    }
    for (const auto count : missing) {
        result.missingPages += count;
    }
    result.expandMs = Milliseconds(begin);

    // Segments for drawing, and their CH edges for comparisons.
    begin = Clock::now();
    result.segments.resize(drawn.size());
    result.edges.resize(drawn.size());
    workers.Run(TaskCount(drawn.size(), kItemsPerTask), [&](std::size_t task, unsigned) {
        const auto last = std::min(drawn.size(), (task + 1u) * kItemsPerTask);
        for (auto i = task * kItemsPerTask; i < last; ++i) {
            const auto page = drawn[i] >> 8u;
            const auto local = drawn[i] & 0xffu;
            const auto words = store.Words(page);
            const auto recordCount = words[0] & 0xffffu;
            const auto vertexCount = words[0] >> 16u;
            const auto endpoints = words[lod::kPageHeaderWords + local * lod::kRecordWords];
            const auto vertexBase = lod::kPageHeaderWords + recordCount * lod::kRecordWords;
            const auto vertex = [&](std::uint32_t index, std::uint32_t axis) {
                return std::bit_cast<std::int32_t>(
                    words[vertexBase + index * lod::kVertexWords + axis]);
            };
            const auto typeWord =
                words[vertexBase + vertexCount * lod::kVertexWords + (local >> 2u)];
            auto& segment = result.segments[i];
            segment.ax = vertex(endpoints & 0xffffu, 0);
            segment.ay = vertex(endpoints & 0xffffu, 1);
            segment.bx = vertex(endpoints >> 16u, 0);
            segment.by = vertex(endpoints >> 16u, 1);
            segment.type = std::min((typeWord >> ((local & 3u) * 8u)) & 0xffu, 63u);
            result.edges[i] = store.Edges(page)[local];
        }
    });
    result.outputMs = Milliseconds(begin);
    if (stepBase) {
        begin = Clock::now();
        result.stepCounts =
            CountSteps(store, workers, pages, level, *stepBase, cullSegments ? &walk : nullptr);
        result.countMs = Milliseconds(begin);
    }
    return result;
}

} // namespace chmv::cpu
