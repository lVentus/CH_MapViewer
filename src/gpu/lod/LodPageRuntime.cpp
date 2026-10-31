#include "gpu/lod/LodPageRuntime.h"

#include "gpu/streaming/PersistentStreamingPipeline.h"
#include "renderer/LODController.h"
#include "renderer/MapCamera2D.h"
#include "streaming/lod/LodPagePlanner.h"
#include "streaming/runtime/GraphPageStreamer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>

namespace chmv::gpu::lod {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
// Upload time slices: short while the camera moves so frames stay smooth, longer while it
// stands still so the view completes quickly. The byte budget follows measured throughput.
constexpr double kMovingUploadSliceMs = 4.0;
constexpr double kStillUploadSliceMs = 12.0;
// Pages for a ring of half a screen around the view are loaded after the view itself.
constexpr double kPrefetchMargin = 0.5;
// Roads just outside the screen still reach into it with half their stroke width
// (at most 5.6 px default width times the 4x width scale).
constexpr double kStrokeMarginPixels = 16.0;

// The shaders trust the page blob, so a page that does not match its directory entry is never
// uploaded.
[[nodiscard]] bool PageMatchesEntry(const chmv::streaming::lod::PageEntry& entry,
                                    std::span<const std::uint32_t> words) {
    if (words.size() != entry.wordCount ||
        words[0] != (static_cast<std::uint32_t>(entry.recordCount) |
                     (static_cast<std::uint32_t>(entry.vertexCount) << 16u))) {
        return false;
    }
    for (std::uint32_t i = 0; i < entry.recordCount; ++i) {
        const auto endpoints =
            words[chmv::streaming::lod::kPageHeaderWords + i * chmv::streaming::lod::kRecordWords];
        if ((endpoints & 0xffffu) >= entry.vertexCount || (endpoints >> 16u) >= entry.vertexCount) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] double Milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

// GPU pool size: the requested budget, or else what the driver/OS report right now: a third of
// free VRAM on discrete GPUs, an eighth of free RAM on integrated ones (their "VRAM" is that
// RAM). Never more than the dataset needs.
[[nodiscard]] std::uint32_t ChoosePoolSlots(std::uint64_t pageCount, std::uint64_t requested) {
    std::uint64_t budget = requested;
    if (budget == 0u) {
        const auto gpuMemory = streaming::QueryGpuMemoryInfo();
        const auto systemMemory = chmv::streaming::runtime::QuerySystemMemoryInfo();
        budget = 512u * kMiB;
        if (gpuMemory.architecture == streaming::GPUMemoryArchitecture::Unified &&
            systemMemory.availablePhysicalBytes != 0u) {
            budget = systemMemory.availablePhysicalBytes / 8u;
        } else if (gpuMemory.reportedAvailableBytes != 0u) {
            budget = gpuMemory.reportedAvailableBytes / 3u;
        }
        budget = std::max<std::uint64_t>(budget, 64u * kMiB);
    }
    const auto slotBytes = std::uint64_t{chmv::streaming::lod::kPageSlotWords} * 4u;
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(pageCount, budget / slotBytes));
}

} // namespace

LodPageRuntime::LodPageRuntime(const std::filesystem::path& shaderDirectory,
                               chmv::streaming::lod::LodPageDataset dataset,
                               std::uint64_t poolBudgetBytes)
    : file_(std::move(dataset.file)), pageImage_(std::move(dataset.pageImage)),
      renderer_(shaderDirectory) {
    const auto pageCount = static_cast<std::uint32_t>(file_->Pages().size());
    pool_.Initialize(pageCount, ChoosePoolSlots(pageCount, poolBudgetBytes),
                     chmv::streaming::lod::kPageSlotWords);
    const auto slots = pool_.SlotCount();
    requestStamp_.assign(pageCount, 0);
    requiredStamp_.assign(pageCount, 0);
    heldStamp_.assign(pageCount, 0);
    corrupt_.assign(pageCount, 0);
    if (!pageImage_) {
        reader_ = std::make_unique<chmv::streaming::lod::LodPageReader>(file_);
    }

    const auto& header = file_->Header();
    stats_.pageCount = header.pageCount;
    stats_.recordCount = header.recordCount;
    stats_.pageSectionBytes = file_->PageSectionBytes();
    stats_.ramImage = pageImage_ != nullptr;
    stats_.poolSlots = slots;
    stats_.poolBytes = std::uint64_t{slots} * chmv::streaming::lod::kPageSlotWords * 4u;
    std::cout << "LOD pages: " << header.pageCount << " pages, " << header.recordCount
              << " records; GPU pool " << slots << " slots (" << stats_.poolBytes / kMiB
              << " MiB); pages " << (pageImage_ ? "served from RAM" : "read from disk") << '\n';
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
                            int width, int height, float pixelTolerance) {
    const auto begin = Clock::now();
    ++frame_;
    pool_.BeginFrame();
    renderer_.Poll();
    if (width <= 0 || height <= 0) {
        return;
    }

    // Everything this frame needs follows from the camera: view rectangle, level, tolerance.
    const auto& header = file_->Header();
    const auto viewScale = std::max(static_cast<double>(camera.ViewScale()), 1e-12);
    const auto halfY = 1.0 / viewScale;
    const auto halfX = halfY * static_cast<double>(width) / static_cast<double>(height);
    const auto centerX = static_cast<double>(camera.CenterX());
    const auto centerY = static_cast<double>(camera.CenterY());
    const auto pixel = 2.0 / (viewScale * static_cast<double>(height));
    const auto margin = kStrokeMarginPixels * pixel;
    chmv::streaming::lod::LodPageQuery query;
    query.minX = static_cast<float>(centerX - halfX - margin);
    query.minY = static_cast<float>(centerY - halfY - margin);
    query.maxX = static_cast<float>(centerX + halfX + margin);
    query.maxY = static_cast<float>(centerY + halfY + margin);
    query.level = static_cast<std::uint32_t>(std::clamp<long>(
        std::lround(lod.ContinuousLevel(camera.ZoomFactor())), 0L, header.maxLevel));
    query.threshold = chmv::streaming::lod::EncodeErrorTolerance(
        static_cast<double>(std::max(pixelTolerance, 0.01f)) * pixel);

    auto ring = query;
    ring.minX = static_cast<float>(centerX - halfX * (1.0 + 2.0 * kPrefetchMargin));
    ring.minY = static_cast<float>(centerY - halfY * (1.0 + 2.0 * kPrefetchMargin));
    ring.maxX = static_cast<float>(centerX + halfX * (1.0 + 2.0 * kPrefetchMargin));
    ring.maxY = static_cast<float>(centerY + halfY * (1.0 + 2.0 * kPrefetchMargin));

    const auto planBegin = Clock::now();
    required_.clear();
    prefetch_.clear();
    chmv::streaming::lod::CollectLodPages(file_->Groups(), file_->Pages(), query, required_);
    chmv::streaming::lod::CollectLodPages(file_->Groups(), file_->Pages(), ring, prefetch_);
    stats_.planMs = Milliseconds(planBegin);

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

    stats_.cameraMoving = CameraMoved(camera, width, height);
    UploadMissing(stats_.cameraMoving);

    std::uint32_t missingRequired = 0;
    for (const auto page : required_) {
        missingRequired += pool_.Resident(page) ? 0u : 1u;
    }
    const bool complete = missingRequired == 0u;
    // Only a full pool that keeps the view itself from loading is worth reporting.
    stats_.poolFull = stats_.poolFull && !complete;
    if (complete) {
        committed_ = {query.level, query.threshold, true};
        drawPages_ = required_;
    } else if (committed_.valid) {
        // Keep showing the last complete level/tolerance (for the current view rectangle)
        // until the new one is fully resident.
        auto previous = query;
        previous.level = committed_.level;
        previous.threshold = committed_.threshold;
        drawPages_.clear();
        chmv::streaming::lod::CollectLodPages(file_->Groups(), file_->Pages(), previous,
                                              drawPages_);
    } else {
        drawPages_ = required_;
    }
    const auto drawState =
        committed_.valid ? committed_ : DrawState{query.level, query.threshold, true};

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
    stats_.filtered = !filtered_.valid || filtered_.level != drawState.level ||
                      filtered_.threshold != drawState.threshold || filterKey != filteredPages_;
    if (stats_.filtered) {
        renderer_.Filter(pool_, drawSlots_, drawState.level, drawState.threshold);
        filtered_ = drawState;
        filteredPages_ = std::move(filterKey);
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
    stats_.drawnSegments = renderer_.DrawnSegments();
    stats_.complete = complete;
    stats_.residentPages = pool_.ResidentCount();
    stats_.evictions = pool_.Evictions();
    stats_.gpuFilterMs = renderer_.FilterMilliseconds();
    stats_.gpuDrawMs = renderer_.DrawMilliseconds();
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
    const auto upload = [&](std::uint32_t page, std::span<const std::uint32_t> words) {
        if (pool_.Resident(page)) {
            return true;
        }
        if (!PageMatchesEntry(file_->Pages()[page], words)) {
            corrupt_[page] = 1u;
            ++stats_.corruptPages;
            return true;
        }
        if (!pool_.Upload(page, words)) {
            stats_.poolFull = true;
            return false;
        }
        uploaded += words.size_bytes();
        ++pages;
        return true;
    };

    if (pageImage_) {
        for (const auto page : missing_) {
            if (uploaded >= budget) {
                break;
            }
            const auto& entry = file_->Pages()[page];
            const std::span<const std::uint32_t> words(
                pageImage_->data() + file_->PageWordOffset(page), entry.wordCount);
            if (!upload(page, words)) {
                break;
            }
        }
    } else {
        // Pages read earlier but not uploaded yet (budget, full pool) stay queued as long as
        // they are still wanted; the view's own pages go first.
        std::uint64_t held = 0;
        for (const auto& page : loaded_) {
            held += page.words.size() * 4u;
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
               upload(loaded_[done].page, loaded_[done].words)) {
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
                          const std::array<float, 4>& color,
                          const renderer::RoadStyleConfig& styles) {
    renderer_.Draw(pool_, camera, width, height, color, styles);
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> LodPageRuntime::ReadDrawnRecords() const {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> records;
    for (const auto item : renderer_.ReadSegments()) {
        records.emplace_back(pool_.PageOfSlot(item >> 8u), item & 0xffu);
    }
    return records;
}

} // namespace chmv::gpu::lod
