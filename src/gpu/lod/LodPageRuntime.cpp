#include "gpu/lod/LodPageRuntime.h"

#include "cpu/CpuPasses.h"
#include "data/SystemMemory.h"
#include "gpu/GpuMemoryInfo.h"
#include "gpu/filtering/IRangeFilterStrategy.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <iterator>
#include <span>
#include <unordered_map>

namespace chmv::gpu::lod {
namespace {

namespace format = chmv::streaming::lod;

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
// Upload time slices: short while the camera moves so frames stay smooth, longer while it
// stands still so the view completes quickly. The byte budget follows measured throughput.
constexpr double kMovingUploadSliceMs = 4.0;
constexpr double kStillUploadSliceMs = 12.0;
// Pages for a ring of half a screen around the view are loaded after the view itself.
constexpr double kPrefetchMargin = 0.5;
// Pages queued per frame while preloading everything, after the view's own.
constexpr std::uint32_t kPreloadPagesPerFrame = 4096;
// A view too large for the GPU pool is drawn as from farther away, in quarter octaves, at most
// 16 octaves.
constexpr std::uint32_t kFitStepsPerOctave = 4;
constexpr std::uint32_t kMaxFitSteps = 64;

[[nodiscard]] double Milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

// GPU pool size: the requested budget, or else what the driver/OS report right now: a third of
// free VRAM on discrete GPUs, an eighth of free RAM on integrated ones (their "VRAM" is that
// RAM). To preload every page it may take up to three quarters of free VRAM (half of free RAM).
// Never more than the dataset needs.
// heldBytes: the pool being replaced, still counted as used by the driver/OS.
[[nodiscard]] std::uint32_t ChoosePoolSlots(std::uint64_t pageCount, std::uint64_t requested,
                                            std::uint32_t slotWords, bool preloadAll,
                                            std::uint64_t heldBytes) {
    const auto slotBytes = std::uint64_t{slotWords} * 4u;
    std::uint64_t budget = requested;
    if (budget == 0u) {
        const auto gpuMemory = QueryGpuMemoryInfo();
        const auto systemMemory = chmv::data::QuerySystemMemoryInfo();
        budget = 512u * kMiB;
        std::uint64_t most = budget;
        if (gpuMemory.architecture == GPUMemoryArchitecture::Unified &&
            systemMemory.availablePhysicalBytes != 0u) {
            const auto available = systemMemory.availablePhysicalBytes + heldBytes;
            budget = available / 8u;
            most = available / 2u;
        } else if (gpuMemory.reportedAvailableBytes != 0u) {
            const auto available = gpuMemory.reportedAvailableBytes + heldBytes;
            budget = available / 3u;
            most = available / 4u * 3u;
        }
        if (preloadAll && pageCount * slotBytes <= most) {
            budget = pageCount * slotBytes;
        }
        budget = std::max<std::uint64_t>(budget, 64u * kMiB);
    }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(pageCount, budget / slotBytes));
}

// The same test as for CPU results.
[[nodiscard]] bool ChordInView(std::span<const std::uint32_t> words, std::uint32_t record,
                               const std::array<double, 4>& view) {
    const auto recordCount = words[0] & 0xffffu;
    const auto endpoints = words[format::kPageHeaderWords + record * format::kRecordWords];
    const auto vertexBase = format::kPageHeaderWords + recordCount * format::kRecordWords;
    const auto coordinate = [&](std::uint32_t vertex, std::uint32_t axis) {
        return std::bit_cast<std::int32_t>(
            words[vertexBase + vertex * format::kVertexWords + axis]);
    };
    const auto a = endpoints & 0xffffu;
    const auto b = endpoints >> 16u;
    return chmv::cpu::ChordTouches(coordinate(a, 0), coordinate(a, 1), coordinate(b, 0),
                                   coordinate(b, 1), view);
}

} // namespace

LodPageRuntime::LodPageRuntime(const std::filesystem::path& shaderDirectory,
                               chmv::streaming::lod::LodPageDataset dataset,
                               std::uint64_t poolBudgetBytes)
    : file_(std::move(dataset.file)), pageImage_(std::move(dataset.pageImage)),
      childrenImage_(std::move(dataset.childrenImage)), renderer_(shaderDirectory),
      poolBudgetBytes_(poolBudgetBytes) {
    const auto& header = file_->Header();
    const auto pageCount = static_cast<std::size_t>(header.pageCount);
    requestStamp_.assign(pageCount, 0);
    requiredStamp_.assign(pageCount, 0);
    heldStamp_.assign(pageCount, 0);
    corrupt_.assign(pageCount, 0);
    stats_.pageCount = header.pageCount;
    stats_.recordCount = header.recordCount;
    stats_.pageSectionBytes = file_->PageSectionBytes();
    stats_.fileBytes = file_->FileBytes();
    stats_.compressedFile = file_->Compressed();
}

void LodPageRuntime::SetUpPool(bool hierarchy) {
    const auto& header = file_->Header();
    const auto pageCount = static_cast<std::uint32_t>(header.pageCount);
    const auto childWords = hierarchy ? format::kChildSlotWords : 0u;
    const auto slotWords = format::kPageSlotWords + childWords;
    const auto heldBytes = std::uint64_t{pool_.SlotCount()} * pool_.SlotWords() * 4u;
    pool_.Initialize(pageCount,
                     ChoosePoolSlots(pageCount, poolBudgetBytes_, slotWords, preloadAll_, heldBytes),
                     format::kPageSlotWords, childWords);
    hierarchy_ = hierarchy;
    poolReady_ = true;

    // Pages come from RAM when everything the mode needs is there, otherwise from disk.
    const bool fromRam = pageImage_ && (!hierarchy || childrenImage_);
    reader_.reset();
    if (!fromRam) {
        reader_ = std::make_unique<chmv::streaming::lod::LodPageReader>(file_, hierarchy);
    }
    std::fill(requestStamp_.begin(), requestStamp_.end(), 0u);
    std::fill(requiredStamp_.begin(), requiredStamp_.end(), 0u);
    std::fill(heldStamp_.begin(), heldStamp_.end(), 0u);
    // Pages are checked again on upload: a bad children block only matters with a hierarchy.
    std::fill(corrupt_.begin(), corrupt_.end(), std::uint8_t{0});
    stats_.corruptPages = 0;
    loaded_.clear();
    demand_.clear();
    drawPages_.clear();
    drawSlots_.clear();
    filteredPages_.clear();
    planned_ = false;
    drawPagesFromTarget_ = true;
    committed_ = {};
    filteredQuery_.reset();
    preloadCursor_ = 0;

    stats_.ramImage = fromRam;
    stats_.hierarchy = hierarchy;
    stats_.poolSlots = pool_.SlotCount();
    stats_.poolBytes = std::uint64_t{pool_.SlotCount()} * pool_.SlotWords() * 4u;
    stats_.preloadAll = preloadAll_;
    stats_.preloadFits = pool_.SlotCount() >= pageCount;
    std::cout << "LOD pages: " << header.pageCount << " pages, " << header.recordCount
              << " records; GPU pool " << pool_.SlotCount() << " slots ("
              << stats_.poolBytes / kMiB << " MiB" << (hierarchy ? ", with children" : "")
              << "); pages " << (fromRam ? "served from RAM" : "read from disk") << '\n';
}

void LodPageRuntime::SetPreloadAll(bool preload) {
    if (preload != preloadAll_) {
        preloadAll_ = preload;
        // The pool may have to grow (or may give memory back).
        poolReady_ = false;
    }
}

void LodPageRuntime::SetPoolBudget(std::uint64_t bytes) {
    if (bytes != poolBudgetBytes_) {
        poolBudgetBytes_ = bytes;
        poolReady_ = false;
    }
}

std::span<const chmv::streaming::lod::PageEntry> LodPageRuntime::Directory() const {
    return hierarchy_ ? file_->VisitPages() : file_->Pages();
}

std::span<const chmv::streaming::lod::PageGroup> LodPageRuntime::Groups() const {
    return hierarchy_ ? file_->VisitGroups() : file_->Groups();
}

LodPagePass LodPageRuntime::Pass() const {
    LodPagePass pass;
    pass.slots = drawSlots_;
    pass.directory = Directory();
    pass.query = drawQuery_;
    pass.hierarchyHeight = file_->Header().maxHierarchyHeight;
    pass.cullSegments = cullSegments_;
    return pass;
}

bool LodPageRuntime::CameraMoved(const renderer::MapCamera2D& camera, int width, int height) {
    const std::array<float, 3> current{camera.CenterX(), camera.CenterY(), camera.ZoomFactor()};
    const std::array<int, 2> size{width, height};
    const bool moved = current != lastCamera_ || size != lastSize_;
    lastCamera_ = current;
    lastSize_ = size;
    return moved;
}

void LodPageRuntime::Update(const renderer::MapCamera2D& camera, const renderer::LODController& lod,
                            int width, int height, float pixelTolerance,
                            filtering::IRangeFilterStrategy& filter,
                            unfolding::IUnfoldingStrategy& unfold, bool countSteps) {
    const auto begin = Clock::now();
    ++frame_;
    renderer_.Poll();
    if (width <= 0 || height <= 0) {
        return;
    }
    // A traversal needs the children blocks in the pool and pages planned by what it visits.
    const bool hierarchy = unfold.Traverses();
    if (!poolReady_ || hierarchy != hierarchy_) {
        SetUpPool(hierarchy);
    }
    pool_.BeginFrame();

    // Everything this frame needs follows from the camera: view rectangle, level, tolerance.
    const auto planned = chmv::streaming::lod::PlanView(camera, lod, width, height, pixelTolerance,
                                                        file_->Header().maxLevel);
    auto query = planned.query;
    query.threshold = unfolding::ThresholdFor(unfold.Mode(), planned.toleranceCode);
    const auto& view = planned.rectangle;
    const auto centerX = (view[0] + view[2]) / 2.0;
    const auto centerY = (view[1] + view[3]) / 2.0;
    const auto halfX = (view[2] - view[0]) / 2.0;
    const auto halfY = (view[3] - view[1]) / 2.0;

    auto ring = query;
    ring.minX = static_cast<float>(centerX - halfX * (1.0 + 2.0 * kPrefetchMargin));
    ring.minY = static_cast<float>(centerY - halfY * (1.0 + 2.0 * kPrefetchMargin));
    ring.maxX = static_cast<float>(centerX + halfX * (1.0 + 2.0 * kPrefetchMargin));
    ring.maxY = static_cast<float>(centerY + halfY * (1.0 + 2.0 * kPrefetchMargin));

    // A camera that has not moved asks for exactly the same pages as last frame. A view that
    // needs more pages than the GPU pool holds is drawn as from farther away (coarser level and
    // tolerance) until it fits.
    const auto planBegin = Clock::now();
    if (!planned_ || query != plannedQuery_ || ring != plannedRing_) {
        plannedQuery_ = query;
        plannedRing_ = ring;
        planned_ = true;
        const auto farther = [&](std::uint32_t steps) {
            const auto coarser = chmv::streaming::lod::PlanView(
                camera, lod, width, height, pixelTolerance, file_->Header().maxLevel,
                static_cast<double>(steps) / kFitStepsPerOctave);
            auto result = coarser.query;
            result.threshold = unfolding::ThresholdFor(unfold.Mode(), coarser.toleranceCode);
            return result;
        };
        const auto fitted = FitView(query, farther, required_, fitSteps_);
        fittedLevel_ = fitted.level;
        fittedThreshold_ = fitted.threshold;
        ring.level = fittedLevel_;
        ring.threshold = fittedThreshold_;
        prefetch_.clear();
        chmv::streaming::lod::CollectLodPages(Groups(), Directory(), ring, prefetch_);
    }
    stats_.planMs = Milliseconds(planBegin);
    stats_.cameraLevel = query.level;
    stats_.cameraThreshold = query.threshold;
    stats_.zoomOutOctaves = static_cast<float>(fitSteps_) / kFitStepsPerOctave;
    stats_.fittedToPool = fittedLevel_ != query.level || fittedThreshold_ != query.threshold;
    query.level = fittedLevel_;
    query.threshold = fittedThreshold_;

    // Pin what the target needs and, if the pool has room for both, what is on screen now;
    // a small pool gives up the old picture rather than never reaching the new one. Then queue
    // everything missing, the view itself first.
    for (const auto page : required_) {
        pool_.Touch(page);
        requiredStamp_[page] = frame_;
    }
    if ((required_.size() + drawPages_.size()) * 4u <= std::size_t{pool_.SlotCount()} * 3u) {
        for (const auto page : drawPages_) {
            pool_.Touch(page);
        }
    }
    // The ring is only worth loading when it cannot push the view itself out of the pool.
    const bool withRing = required_.size() + prefetch_.size() <= pool_.SlotCount() / 2u;
    missing_.clear();
    for (const auto& pages : {std::span<const std::uint32_t>(required_),
                              std::span<const std::uint32_t>(withRing ? prefetch_ : required_)}) {
        for (const auto page : pages) {
            if (!pool_.Resident(page) && requestStamp_[page] != frame_ && corrupt_[page] == 0u) {
                requestStamp_[page] = frame_;
                missing_.push_back(page);
            }
        }
    }
    QueuePreload();

    stats_.cameraMoving = CameraMoved(camera, width, height);
    UploadMissing(stats_.cameraMoving);

    // Corrupt pages never arrive; they must not keep the view incomplete forever.
    std::uint32_t missingRequired = 0;
    for (const auto page : required_) {
        missingRequired += pool_.Resident(page) || corrupt_[page] != 0u ? 0u : 1u;
    }
    const bool complete = missingRequired == 0u;
    // Only a full pool that keeps the view itself from loading is worth reporting.
    stats_.poolFull = stats_.poolFull && !complete;
    if (complete) {
        committed_ = {query.level, query.threshold, true};
        drawPages_ = required_;
        drawQuery_ = query;
        drawPagesFromTarget_ = true;
    } else if (committed_.valid) {
        // Keep showing the last complete level/tolerance (for the current view rectangle)
        // until the new one is fully resident.
        auto previous = query;
        previous.level = committed_.level;
        previous.threshold = committed_.threshold;
        if (previous != drawQuery_ || drawPagesFromTarget_) {
            drawPages_.clear();
            chmv::streaming::lod::CollectLodPages(Groups(), Directory(), previous, drawPages_);
            drawQuery_ = previous;
            drawPagesFromTarget_ = false;
        }
    } else {
        drawPages_ = required_;
        drawQuery_ = query;
        drawPagesFromTarget_ = true;
    }
    drawView_ = view;

    drawSlots_.clear();
    std::vector<std::uint64_t> filterKey;
    filterKey.reserve(drawPages_.size());
    for (const auto page : drawPages_) {
        const auto slot = pool_.SlotOf(page);
        if (slot != LodPagePool::kNoSlot) {
            drawSlots_.push_back(slot);
            filterKey.push_back((std::uint64_t{page} << 32u) | slot);
        }
    }
    // A traversal culls with the view, as does segment culling, so their result changes with
    // the view too; the interval rules alone only depend on level, threshold and pages.
    const auto sameQuery = [&](const chmv::streaming::lod::LodPageQuery& done) {
        return done.level == drawQuery_.level && done.threshold == drawQuery_.threshold &&
               ((!unfold.Traverses() && !cullSegments_) || done == drawQuery_);
    };
    const DrawState drawState{drawQuery_.level, drawQuery_.threshold, true};
    const bool retry = renderer_.TakeRetry();
    const bool count = countSteps && unfold.Mode() == geometry::RefinementMode::Adaptive;
    stats_.filtered = retry || !filteredQuery_ || !sameQuery(*filteredQuery_) ||
                      filterKey != filteredPages_ || filteredBy_ != &filter ||
                      unfoldedBy_ != &unfold || (count && !counted_) || culled_ != cullSegments_;
    if (stats_.filtered) {
        renderer_.Process(pool_, Pass(), filter, unfold,
                          count ? std::optional{planned.pixelCode} : std::nullopt);
        counted_ = count;
        culled_ = cullSegments_;
        filteredQuery_ = drawQuery_;
        filteredPages_ = std::move(filterKey);
        filteredBy_ = &filter;
        unfoldedBy_ = &unfold;
        ++stats_.filterDispatches;
    }

    const auto now = Clock::now();
    if (!complete) {
        if (!incompleteSince_) {
            incompleteSince_ = now;
        }
        ++stats_.incompleteFrames;
    } else if (incompleteSince_) {
        stats_.lastSettleMs =
            std::chrono::duration<double, std::milli>(now - *incompleteSince_).count();
        stats_.maxSettleMs = std::max(stats_.maxSettleMs, stats_.lastSettleMs);
        ++stats_.settleCount;
        incompleteSince_.reset();
    }

    stats_.level = query.level;
    stats_.threshold = query.threshold;
    stats_.drawLevel = drawState.level;
    stats_.drawThreshold = drawState.threshold;
    stats_.requiredPages = static_cast<std::uint32_t>(required_.size());
    stats_.missingRequiredPages = missingRequired;
    stats_.prefetchPages = static_cast<std::uint32_t>(prefetch_.size());
    stats_.drawPages = static_cast<std::uint32_t>(drawSlots_.size());
    stats_.candidates = renderer_.Candidates();
    stats_.drawnSegments = renderer_.DrawnSegments();
    stats_.status = renderer_.Status();
    stats_.complete = complete;
    stats_.residentPages = pool_.ResidentCount();
    stats_.evictions = pool_.Evictions();
    stats_.gpuFilterMs = renderer_.FilterMilliseconds();
    stats_.gpuUnfoldMs = renderer_.UnfoldMilliseconds();
    stats_.gpuDrawMs = renderer_.DrawMilliseconds();
    stats_.gpuCountMs = renderer_.CountMilliseconds();
    stats_.pixelCode = planned.pixelCode;
    stats_.stepCounts = renderer_.StepCounts();
    if (reader_) {
        const auto reader = reader_->Stats();
        stats_.diskPagesRead = reader.pagesRead;
        stats_.diskBytesRead = reader.bytesRead;
        stats_.diskReadCalls = reader.readCalls;
        stats_.diskReadMs = reader.readMilliseconds;
        stats_.readerError = reader.error;
    }
    stats_.updateMs = Milliseconds(begin);
}

// The pages of a view must fit in the pool (it then gives up the old picture rather than never
// showing the new one). Farther away, level and tolerance are coarser and fewer pages are
// needed. Where the farthest view still needs too many (the LOD curve stops at the overview
// level, or the level is set by hand), the level is raised beyond it, up to the coarsest, with
// the tolerance of the nearest view that reaches that level.
// Both searches double their step until the pages fit, then halve back to the nearest that
// fits; the coarsest is taken even if it does not fit.
chmv::streaming::lod::LodPageQuery LodPageRuntime::FitView(
    const chmv::streaming::lod::LodPageQuery& query,
    const std::function<chmv::streaming::lod::LodPageQuery(std::uint32_t)>& farther,
    std::vector<std::uint32_t>& pages, std::uint32_t& steps) const {
    const auto limit = std::size_t{pool_.SlotCount()};
    const auto fits = [&](const chmv::streaming::lod::LodPageQuery& candidate) {
        pages.clear();
        chmv::streaming::lod::CollectLodPages(Groups(), Directory(), candidate, pages);
        return pages.size() <= limit;
    };
    // The least value in (low, high] for which coarser(value) fits, high if none does.
    const auto search = [&](std::uint32_t low, std::uint32_t high, const auto& coarser) {
        auto last = high;
        for (std::uint32_t step = 1; low + step < high; step *= 2u) {
            if (fits(coarser(low + step))) {
                last = low + step;
                break;
            }
            low += step;
        }
        while (last - low > 1u) {
            const auto middle = low + (last - low) / 2u;
            if (fits(coarser(middle))) {
                last = middle;
            } else {
                low = middle;
            }
        }
        return last;
    };
    steps = 0;
    if (fits(query)) {
        return query;
    }
    steps = search(0u, kMaxFitSteps, farther);
    auto fitted = farther(steps);
    if (fits(fitted)) {
        return fitted;
    }
    // From the nearest step that already has the farthest level on: farther still would only
    // make the tolerance coarser.
    std::uint32_t low = 0;
    while (steps - low > 1u) {
        const auto middle = low + (steps - low) / 2u;
        (farther(middle).level == fitted.level ? steps : low) = middle;
    }
    fitted = farther(steps);
    const auto coarsest =
        static_cast<std::uint32_t>(std::max(file_->Header().maxLevel, std::int32_t{0}));
    if (fitted.level < coarsest) {
        const auto withLevel = [&](std::uint32_t level) {
            auto raised = fitted;
            raised.level = level;
            return raised;
        };
        fitted.level = search(fitted.level, coarsest, withLevel);
    }
    (void)fits(fitted);
    return fitted;
}

// With preloading, every page the view does not need yet is queued behind the view's own, a
// bounded window per frame, until all are resident. The window only moves on once its first
// pages are in, so pages being read or held stay wanted until they arrive. Only when the pool
// can hold every page; otherwise preloading would just evict pages the view needs.
void LodPageRuntime::QueuePreload() {
    const auto pageCount = pool_.PageCount();
    if (!preloadAll_ || pool_.SlotCount() < pageCount) {
        return;
    }
    while (preloadCursor_ < pageCount &&
           (pool_.Resident(preloadCursor_) || corrupt_[preloadCursor_] != 0u)) {
        ++preloadCursor_;
    }
    std::uint32_t queued = 0;
    for (auto page = preloadCursor_; page < pageCount && queued < kPreloadPagesPerFrame; ++page) {
        if (!pool_.Resident(page) && corrupt_[page] == 0u) {
            ++queued;
            if (requestStamp_[page] != frame_) {
                requestStamp_[page] = frame_;
                missing_.push_back(page);
            }
        }
    }
}

bool LodPageRuntime::Upload(std::uint32_t page, std::span<const std::uint32_t> words,
                            std::span<const std::uint32_t> children) {
    const auto& entry = file_->Pages()[page];
    // The shaders trust the page blob, so a page that does not match its directory entry is
    // never uploaded.
    if (!format::PageMatchesEntry(entry, words) ||
        (hierarchy_ && !format::ChildrenMatchEntry(file_->Pages(), file_->VisitPages()[page],
                                                   entry.recordCount, children))) {
        if (corrupt_[page] == 0u) {
            corrupt_[page] = 1u;
            ++stats_.corruptPages;
        }
        return true;
    }
    return pool_.Upload(page, words, hierarchy_ ? children : std::span<const std::uint32_t>{});
}

void LodPageRuntime::UploadMissing(bool cameraMoving) {
    const auto begin = Clock::now();
    const auto slice = cameraMoving ? kMovingUploadSliceMs : kStillUploadSliceMs;
    const auto budget =
        uploadBytesPerMs_ > 0.0
            ? std::clamp(static_cast<std::uint64_t>(uploadBytesPerMs_ * slice), kMiB, 256u * kMiB)
            : 16u * kMiB;

    std::uint64_t uploaded = 0;
    std::uint32_t pages = 0;
    stats_.poolFull = false;
    const auto upload = [&](std::uint32_t page, std::span<const std::uint32_t> words,
                            std::span<const std::uint32_t> children) {
        if (pool_.Resident(page)) {
            return true;
        }
        if (!Upload(page, words, children)) {
            stats_.poolFull = true;
            return false;
        }
        if (pool_.Resident(page)) {
            uploaded += words.size_bytes() + children.size_bytes();
            ++pages;
        }
        return true;
    };

    if (!reader_) {
        for (const auto page : missing_) {
            if (uploaded >= budget) {
                break;
            }
            const std::span<const std::uint32_t> words(
                pageImage_->data() + file_->PageWordOffset(page), file_->Pages()[page].wordCount);
            std::span<const std::uint32_t> children;
            if (hierarchy_) {
                children = std::span<const std::uint32_t>(
                    childrenImage_->data() + file_->ChildrenWordOffset(page),
                    file_->VisitPages()[page].wordCount);
            }
            if (!upload(page, words, children)) {
                break;
            }
        }
    } else {
        // Pages read earlier but not uploaded yet (budget, full pool) stay queued as long as
        // they are still wanted; the view's own pages go first.
        std::uint64_t held = 0;
        for (const auto& page : loaded_) {
            held += (page.words.size() + page.children.size()) * 4u;
        }
        if (held < budget) {
            reader_->Take(loaded_, budget - held);
        }
        std::erase_if(loaded_, [&](const chmv::streaming::lod::LodPageData& page) {
            return requestStamp_[page.page] != frame_ || pool_.Resident(page.page);
        });
        std::stable_partition(loaded_.begin(), loaded_.end(),
                              [&](const chmv::streaming::lod::LodPageData& page) {
                                  return requiredStamp_[page.page] == frame_;
                              });
        std::size_t done = 0;
        while (done < loaded_.size() && uploaded < budget &&
               upload(loaded_[done].page, loaded_[done].words, loaded_[done].children)) {
            ++done;
        }
        loaded_.erase(loaded_.begin(), loaded_.begin() + static_cast<std::ptrdiff_t>(done));
        for (const auto& page : loaded_) {
            heldStamp_[page.page] = frame_;
        }

        // Only now tell the reader what is still missing: pages that just arrived or are held
        // here must not be read a second time.
        std::erase_if(missing_, [&](std::uint32_t page) {
            return pool_.Resident(page) || heldStamp_[page] == frame_ || corrupt_[page] != 0u;
        });
        if (missing_ != demand_) {
            demand_ = missing_;
            reader_->SetDemand(demand_);
        }
    }

    const auto milliseconds = Milliseconds(begin);
    if (uploaded >= kMiB && milliseconds > 0.05) {
        const auto rate = static_cast<double>(uploaded) / milliseconds;
        uploadBytesPerMs_ = uploadBytesPerMs_ > 0.0 ? 0.8 * uploadBytesPerMs_ + 0.2 * rate : rate;
    }
    stats_.uploadMs = milliseconds;
    stats_.uploadedBytes = uploaded;
    stats_.uploadedPages = pages;
    stats_.uploadBudgetBytes = budget;
    stats_.totalUploadedBytes += uploaded;
    stats_.totalUploadedPages += pages;
}

void LodPageRuntime::Draw(const renderer::MapCamera2D& camera, int width, int height,
                          const renderer::RoadStyleConfig& styles) {
    if (poolReady_) {
        renderer_.Draw(pool_, camera, width, height, styles);
    }
}

LodPageBenchmarkResult LodPageRuntime::Benchmark(filtering::IRangeFilterStrategy& filter,
                                                 unfolding::IUnfoldingStrategy& unfold,
                                                 std::uint32_t iterations,
                                                 const renderer::MapCamera2D& camera, int width,
                                                 int height,
                                                 const renderer::RoadStyleConfig& styles) {
    if (!poolReady_) {
        return {};
    }
    return renderer_.Benchmark(pool_, Pass(), filter, unfold, iterations, camera, width, height,
                               styles);
}

std::vector<std::uint32_t> LodPageRuntime::DrawnEdgesInView() const {
    std::vector<std::uint32_t> edges;
    if (!poolReady_) {
        return edges;
    }
    struct PageData {
        std::vector<std::uint32_t> words;
        std::vector<std::uint32_t> edges;
    };
    std::unordered_map<std::uint32_t, PageData> cache;
    for (const auto item : renderer_.ReadSegments()) {
        const auto page = pool_.PageOfSlot(item >> 8u);
        auto [it, added] = cache.try_emplace(page);
        auto& data = it->second;
        if (added) {
            const auto& entry = file_->Pages()[page];
            if (pageImage_) {
                const auto* first = pageImage_->data() + file_->PageWordOffset(page);
                data.words.assign(first, first + entry.wordCount);
            } else {
                data.words = file_->ReadPage(page);
            }
            data.edges = file_->ReadRecordEdges(entry.firstRecord, entry.recordCount);
        }
        if (ChordInView(data.words, item & 0xffu, drawView_)) {
            edges.push_back(data.edges[item & 0xffu]);
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

LodPageValidation LodPageRuntime::Validate() const {
    const auto begin = Clock::now();
    LodPageValidation result;
    result.level = drawQuery_.level;
    result.threshold = drawQuery_.threshold;
    result.pages = static_cast<std::uint32_t>(drawSlots_.size());
    result.status = renderer_.ReadStatus();
    if (!poolReady_) {
        return result;
    }

    struct PageData {
        std::vector<std::uint32_t> words;
        std::vector<std::uint32_t> edges;
    };
    std::unordered_map<std::uint32_t, PageData> cache;
    const auto pageData = [&](std::uint32_t page) -> const PageData& {
        auto [it, added] = cache.try_emplace(page);
        if (added) {
            const auto& entry = file_->Pages()[page];
            if (pageImage_) {
                const auto* first = pageImage_->data() + file_->PageWordOffset(page);
                it->second.words.assign(first, first + entry.wordCount);
            } else {
                it->second.words = file_->ReadPage(page);
            }
            it->second.edges = file_->ReadRecordEdges(entry.firstRecord, entry.recordCount);
        }
        return it->second;
    };
    const auto record = [&](const PageData& data, std::uint32_t local) {
        return std::span<const std::uint32_t>(data.words).subspan(
            format::kPageHeaderWords + local * format::kRecordWords, format::kRecordWords);
    };

    // Candidates: the same set whatever the filter.
    std::vector<std::uint64_t> gpuCandidates;
    for (const auto item : renderer_.ReadCandidates()) {
        gpuCandidates.push_back((std::uint64_t{pool_.PageOfSlot(item >> 8u)} << 8u) |
                                (item & 0xffu));
    }
    std::vector<std::uint64_t> cpuCandidates;
    for (const auto slot : drawSlots_) {
        const auto page = pool_.PageOfSlot(slot);
        const auto& data = pageData(page);
        for (std::uint32_t local = 0; local < file_->Pages()[page].recordCount; ++local) {
            const auto words = record(data, local);
            const auto inRange = [&](std::uint32_t range) {
                return drawQuery_.level >= (range & 0xffffu) && drawQuery_.level <= (range >> 16u);
            };
            if (inRange(words[1]) || inRange(words[2])) {
                cpuCandidates.push_back((std::uint64_t{page} << 8u) | local);
            }
        }
    }
    std::sort(gpuCandidates.begin(), gpuCandidates.end());
    std::sort(cpuCandidates.begin(), cpuCandidates.end());
    std::vector<std::uint64_t> difference;
    std::set_symmetric_difference(gpuCandidates.begin(), gpuCandidates.end(),
                                  cpuCandidates.begin(), cpuCandidates.end(),
                                  std::back_inserter(difference));
    result.gpuCandidates = gpuCandidates.size();
    result.cpuCandidates = cpuCandidates.size();
    result.candidateMismatches = difference.size();

    // Drawn roads inside the view, as CH edges.
    std::vector<std::uint32_t> gpuEdges;
    const auto segments = renderer_.ReadSegments();
    result.gpuSegments = segments.size();
    for (const auto item : segments) {
        const auto page = pool_.PageOfSlot(item >> 8u);
        const auto& data = pageData(page);
        if (ChordInView(data.words, item & 0xffu, drawView_)) {
            gpuEdges.push_back(data.edges[item & 0xffu]);
        }
    }
    // The reference: the dual-interval rule on every page the main directory plans.
    std::vector<std::uint32_t> pages;
    chmv::streaming::lod::CollectLodPages(file_->Groups(), file_->Pages(), drawQuery_, pages);
    std::vector<std::uint32_t> cpuEdges;
    for (const auto page : pages) {
        const auto& data = pageData(page);
        for (std::uint32_t local = 0; local < file_->Pages()[page].recordCount; ++local) {
            const auto words = record(data, local);
            if (format::RecordDrawn(words[1], words[2], words[3], drawQuery_.level,
                                    drawQuery_.threshold) &&
                ChordInView(data.words, local, drawView_)) {
                cpuEdges.push_back(data.edges[local]);
            }
        }
    }
    std::sort(gpuEdges.begin(), gpuEdges.end());
    std::sort(cpuEdges.begin(), cpuEdges.end());
    std::vector<std::uint32_t> missing;
    std::vector<std::uint32_t> extra;
    std::set_difference(cpuEdges.begin(), cpuEdges.end(), gpuEdges.begin(), gpuEdges.end(),
                        std::back_inserter(missing));
    std::set_difference(gpuEdges.begin(), gpuEdges.end(), cpuEdges.begin(), cpuEdges.end(),
                        std::back_inserter(extra));
    result.gpuSegmentsInView = gpuEdges.size();
    result.cpuSegmentsInView = cpuEdges.size();
    result.missingOnGpu = missing.size();
    result.extraOnGpu = extra.size();
    result.cpuMilliseconds = Milliseconds(begin);
    return result;
}

} // namespace chmv::gpu::lod
