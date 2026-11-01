#include "gpu/lod/LodPageRenderer.h"

#include "gpu/filtering/IRangeFilterStrategy.h"
#include "gpu/lod/LodPagePool.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"
#include "renderer/MapCamera2D.h"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace chmv::gpu::lod {
namespace {

using chmv::streaming::lod::kFixedCoordinateScale;
using chmv::streaming::lod::kPageRecordCapacity;

// The command buffer holds the indirect draw arguments followed by the segment count.
constexpr std::size_t kCommandWords = 5;
// Readback layout: command, candidate count, two spare words, status.
constexpr std::size_t kReadbackCandidateWord = 5;
constexpr std::size_t kReadbackStatusWord = 8;
constexpr std::size_t kReadbackWords = kReadbackStatusWord + kStatusWords;
// Segment room grows by doubling when a traversal draws more than the pages hold records.
constexpr std::uint32_t kMaxSegmentScale = 8;

[[nodiscard]] GLsync AsSync(void* fence) {
    return static_cast<GLsync>(fence);
}

[[nodiscard]] GLint FixedCameraCoordinate(float center) {
    constexpr double kLimit = 1073741823.0;
    const auto fixed = std::round(static_cast<double>(center) * kFixedCoordinateScale);
    return static_cast<GLint>(std::clamp(fixed, -kLimit, kLimit));
}

// The query box in fixed coordinates, rounded inward (and two units more), so a node that the
// traversal keeps always lies in a page the planner chose with the float box.
[[nodiscard]] std::array<std::int32_t, 4> CullBox(const chmv::streaming::lod::LodPageQuery& query) {
    const auto inward = [](float value, double direction) {
        constexpr double kLimit = 1073741824.0;
        const auto fixed = static_cast<double>(value) * kFixedCoordinateScale;
        const auto rounded = direction > 0.0 ? std::ceil(fixed) + 2.0 : std::floor(fixed) - 2.0;
        return static_cast<std::int32_t>(std::clamp(rounded, -kLimit, kLimit));
    };
    return {inward(query.minX, 1.0), inward(query.minY, 1.0), inward(query.maxX, -1.0),
            inward(query.maxY, -1.0)};
}

void CopyWords(const GPUBuffer& from, std::size_t fromWord, const GPUBuffer& to, std::size_t toWord,
               std::size_t words) {
    glBindBuffer(GL_COPY_READ_BUFFER, from.Id());
    glBindBuffer(GL_COPY_WRITE_BUFFER, to.Id());
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                        static_cast<GLintptr>(fromWord * 4u), static_cast<GLintptr>(toWord * 4u),
                        static_cast<GLsizeiptr>(words * 4u));
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

[[nodiscard]] std::vector<std::uint32_t> ReadWords(const GPUBuffer& buffer, std::size_t firstWord,
                                                   std::size_t words) {
    std::vector<std::uint32_t> values(words);
    if (words != 0u) {
        glBindBuffer(GL_COPY_READ_BUFFER, buffer.Id());
        glGetBufferSubData(GL_COPY_READ_BUFFER, static_cast<GLintptr>(firstWord * 4u),
                           static_cast<GLsizeiptr>(words * 4u), values.data());
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
    }
    return values;
}

} // namespace

LodPageRenderer::LodPageRenderer(const std::filesystem::path& shaderDirectory)
    : roadProgram_(shaderDirectory / "lod_page_road.vert", shaderDirectory / "lod_page_road.frag"),
      finishProgram_(shaderDirectory / "lod_draw_finish.comp") {
    glGenVertexArrays(1, &vertexArray_);
    const std::array<std::uint32_t, kCommandWords> empty{};
    commandBuffer_.Allocate(sizeof(empty), empty.data());
    const std::array<std::uint32_t, kStatusWords> clear{};
    statusBuffer_.Allocate(sizeof(clear), clear.data());
    for (auto& readback : readbacks_) {
        const std::array<std::uint32_t, kReadbackWords> zero{};
        readback.buffer.Allocate(sizeof(zero), zero.data(), GL_STREAM_READ);
    }
    std::array<std::uint32_t, chmv::streaming::lod::kErrorBoundMantissas> mantissas{};
    for (std::uint32_t step = 0; step < mantissas.size(); ++step) {
        mantissas[step] = chmv::streaming::lod::ErrorBoundMantissa(step);
    }
    errorBounds_.Allocate(sizeof(mantissas), mantissas.data(), GL_STATIC_DRAW);
    GLint64 maxBlockBytes = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockBytes);
    // 16 MiB is the least GL 4.3 guarantees.
    maxBlockBytes_ = std::max<std::int64_t>(maxBlockBytes, 16 << 20);
    EnsureCapacity(256, 256);
}

LodPageRenderer::~LodPageRenderer() {
    for (auto& readback : readbacks_) {
        if (readback.fence != nullptr) {
            glDeleteSync(AsSync(readback.fence));
        }
    }
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
    }
}

void LodPageRenderer::EnsureCapacity(std::size_t slotCount, std::size_t poolSlots) {
    // Lists are bound as one storage block each, like the pool.
    const auto blockWords = static_cast<std::size_t>(maxBlockBytes_ / 4) - kListHeaderWords;
    if (slotCount > slotCapacity_) {
        // Room for twice the pages, but never for more than the pool holds.
        slotCapacity_ = std::max(slotCount, std::min(slotCapacity_ * 2u, poolSlots));
        candidateCapacity_ = std::min(slotCapacity_ * kPageRecordCapacity, blockWords);
        slotList_.Allocate((kListHeaderWords + slotCapacity_) * 4u);
        candidateList_.Allocate((kListHeaderWords + candidateCapacity_) * 4u);
        allocatedScale_ = 0;
    }
    if (allocatedScale_ != segmentScale_) {
        segmentCapacity_ = std::min(slotCapacity_ * kPageRecordCapacity * segmentScale_, blockWords);
        segmentBuffer_.Allocate(segmentCapacity_ * 4u);
        allocatedScale_ = segmentScale_;
    }
}

LodPageRenderer::Capacities LodPageRenderer::StartPass(const LodPagePool& pool,
                                                       const LodPagePass& pass) {
    EnsureCapacity(pass.slots.size(), pool.SlotCount());
    Capacities capacities;
    capacities.candidates = static_cast<std::uint32_t>(candidateCapacity_);
    capacities.segments = static_cast<std::uint32_t>(segmentCapacity_);

    const auto grid = GroupGrid(pass.slots.size());
    slotListWords_.assign({static_cast<std::uint32_t>(pass.slots.size()), grid[0], grid[1],
                           grid[2]});
    slotListWords_.insert(slotListWords_.end(), pass.slots.begin(), pass.slots.end());
    slotList_.Update(0, slotListWords_.size() * 4u, slotListWords_.data());
    const std::array<std::uint32_t, kListHeaderWords> emptyList{};
    candidateList_.Update(0, sizeof(emptyList), emptyList.data());
    const std::array<std::uint32_t, kCommandWords> reset{0u, 1u, 0u, 0u, 0u};
    commandBuffer_.Update(0, sizeof(reset), reset.data());
    const std::array<std::uint32_t, kStatusWords> clear{};
    statusBuffer_.Update(0, sizeof(clear), clear.data());
    return capacities;
}

void LodPageRenderer::Filter(const LodPagePool& pool, const LodPagePass& pass,
                             Capacities capacities, filtering::IRangeFilterStrategy& filter) {
    filtering::RangeFilterInput input;
    input.pool = &pool;
    input.slots = pass.slots;
    input.slotList = slotList_.Id();
    input.directory = pass.directory;
    input.query = pass.query;
    input.candidateList = candidateList_.Id();
    input.candidateCapacity = capacities.candidates;
    input.statusBuffer = statusBuffer_.Id();
    filter.Execute(input);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
}

void LodPageRenderer::Unfold(const LodPagePool& pool, const LodPagePass& pass,
                             Capacities capacities, unfolding::IUnfoldingStrategy& unfold) {
    unfolding::UnfoldingInput input;
    input.pool = &pool;
    input.candidateList = candidateList_.Id();
    input.candidateCapacity = capacities.candidates;
    input.level = pass.query.level;
    input.threshold = pass.query.threshold;
    input.cullBox = CullBox(pass.query);
    input.hierarchyHeight = pass.hierarchyHeight;
    input.errorBounds = errorBounds_.Id();
    input.segmentBuffer = segmentBuffer_.Id();
    input.segmentCapacity = capacities.segments;
    input.drawCommand = commandBuffer_.Id();
    input.statusBuffer = statusBuffer_.Id();
    unfold.Execute(input);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    finishProgram_.Bind();
    statusBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 4);
    commandBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    glUniform1ui(glGetUniformLocation(finishProgram_.Id(), "uSegmentCapacity"),
                 capacities.segments);
    finishProgram_.Dispatch(1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);
}

void LodPageRenderer::Process(const LodPagePool& pool, const LodPagePass& pass,
                              filtering::IRangeFilterStrategy& filter,
                              unfolding::IUnfoldingStrategy& unfold) {
    const auto capacities = StartPass(pool, pass);
    if (pass.slots.empty()) {
        // Results of earlier passes still in flight would overwrite these.
        for (auto& readback : readbacks_) {
            if (readback.fence != nullptr) {
                glDeleteSync(AsSync(readback.fence));
                readback.fence = nullptr;
            }
        }
        drawnSegments_ = 0;
        candidates_ = 0;
        status_ = {};
        return;
    }
    filterTimer_.Begin();
    Filter(pool, pass, capacities, filter);
    filterTimer_.End();
    unfoldTimer_.Begin();
    Unfold(pool, pass, capacities, unfold);
    unfoldTimer_.End();

    auto& readback = readbacks_[nextReadback_];
    nextReadback_ = (nextReadback_ + 1u) % readbacks_.size();
    if (readback.fence != nullptr) {
        glDeleteSync(AsSync(readback.fence));
    }
    CopyWords(commandBuffer_, 0, readback.buffer, 0, kCommandWords);
    CopyWords(candidateList_, 0, readback.buffer, kReadbackCandidateWord, 1);
    CopyWords(statusBuffer_, 0, readback.buffer, kReadbackStatusWord, kStatusWords);
    readback.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    readback.segmentScale = segmentScale_;
}

LodPageBenchmarkResult LodPageRenderer::Benchmark(const LodPagePool& pool,
                                                  const LodPagePass& pass,
                                                  filtering::IRangeFilterStrategy& filter,
                                                  unfolding::IUnfoldingStrategy& unfold,
                                                  std::uint32_t iterations) {
    LodPageBenchmarkResult result;
    result.pages = static_cast<std::uint32_t>(pass.slots.size());
    if (pass.slots.empty() || iterations == 0u) {
        return result;
    }
    // Three timestamps per pass: start, after the filter, after the expansion.
    std::vector<GLuint> queries(std::size_t{iterations} * 3u);
    glGenQueries(static_cast<GLsizei>(queries.size()), queries.data());
    glFinish();
    for (std::uint32_t i = 0; i < iterations; ++i) {
        const auto capacities = StartPass(pool, pass);
        glQueryCounter(queries[i * 3u], GL_TIMESTAMP);
        Filter(pool, pass, capacities, filter);
        glQueryCounter(queries[i * 3u + 1u], GL_TIMESTAMP);
        Unfold(pool, pass, capacities, unfold);
        glQueryCounter(queries[i * 3u + 2u], GL_TIMESTAMP);
    }
    glFinish();

    std::vector<GLuint64> times(queries.size());
    for (std::size_t q = 0; q < queries.size(); ++q) {
        glGetQueryObjectui64v(queries[q], GL_QUERY_RESULT, &times[q]);
    }
    glDeleteQueries(static_cast<GLsizei>(queries.size()), queries.data());

    result.iterations = iterations;
    result.filterMinMs = result.unfoldMinMs = std::numeric_limits<double>::max();
    for (std::uint32_t i = 0; i < iterations; ++i) {
        const auto filterMs = static_cast<double>(times[i * 3u + 1u] - times[i * 3u]) * 1e-6;
        const auto unfoldMs = static_cast<double>(times[i * 3u + 2u] - times[i * 3u + 1u]) * 1e-6;
        result.filterMinMs = std::min(result.filterMinMs, filterMs);
        result.filterMaxMs = std::max(result.filterMaxMs, filterMs);
        result.filterAverageMs += filterMs / iterations;
        result.unfoldMinMs = std::min(result.unfoldMinMs, unfoldMs);
        result.unfoldMaxMs = std::max(result.unfoldMaxMs, unfoldMs);
        result.unfoldAverageMs += unfoldMs / iterations;
    }
    result.candidates = ReadWords(candidateList_, 0, 1)[0];
    result.segments = ReadWords(commandBuffer_, 0, kCommandWords)[4];
    drawnSegments_ = result.segments;
    candidates_ = result.candidates;
    status_ = ReadStatus();
    return result;
}

void LodPageRenderer::Poll() {
    // Oldest first, so the newest finished pass wins.
    for (std::size_t i = 0; i < readbacks_.size(); ++i) {
        auto& readback = readbacks_[(nextReadback_ + i) % readbacks_.size()];
        if (readback.fence == nullptr ||
            glClientWaitSync(AsSync(readback.fence), 0, 0) == GL_TIMEOUT_EXPIRED) {
            continue;
        }
        glDeleteSync(AsSync(readback.fence));
        readback.fence = nullptr;
        const auto words = ReadWords(readback.buffer, 0, kReadbackWords);
        drawnSegments_ = words[4];
        candidates_ = words[kReadbackCandidateWord];
        std::copy_n(words.begin() + kReadbackStatusWord, kStatusWords, status_.begin());

        const auto ranOutOfRoom =
            status_[static_cast<std::size_t>(LodPageStatus::SegmentOverflow)] != 0u ||
            status_[static_cast<std::size_t>(LodPageStatus::FrontierOverflow)] != 0u;
        if (ranOutOfRoom && readback.segmentScale == segmentScale_ &&
            segmentScale_ < kMaxSegmentScale) {
            segmentScale_ *= 2u;
            retry_ = true;
        }
    }
}

bool LodPageRenderer::TakeRetry() {
    return std::exchange(retry_, false);
}

std::vector<std::uint32_t> LodPageRenderer::ReadSegments() const {
    const auto command = ReadWords(commandBuffer_, 0, kCommandWords);
    return ReadWords(segmentBuffer_, 0, command[4]);
}

std::vector<std::uint32_t> LodPageRenderer::ReadCandidates() const {
    const auto count = ReadWords(candidateList_, 0, 1)[0];
    return ReadWords(candidateList_, kListHeaderWords, count);
}

std::array<std::uint32_t, kStatusWords> LodPageRenderer::ReadStatus() const {
    const auto words = ReadWords(statusBuffer_, 0, kStatusWords);
    std::array<std::uint32_t, kStatusWords> status{};
    std::copy(words.begin(), words.end(), status.begin());
    return status;
}

void LodPageRenderer::Draw(const LodPagePool& pool, const renderer::MapCamera2D& camera, int width,
                           int height, const std::array<float, 4>& color,
                           const renderer::RoadStyleConfig& styles) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const auto program = roadProgram_.Id();
    roadProgram_.Bind();
    pool.Buffer().BindBase(GL_SHADER_STORAGE_BUFFER, 0);
    segmentBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 3);
    glUniform1ui(glGetUniformLocation(program, "uSlotWords"), pool.SlotWords());
    glUniform2i(glGetUniformLocation(program, "uCameraFixed"),
                FixedCameraCoordinate(camera.CenterX()), FixedCameraCoordinate(camera.CenterY()));
    glUniform1f(glGetUniformLocation(program, "uFixedToClip"),
                static_cast<float>(static_cast<double>(camera.ViewScale()) /
                                   kFixedCoordinateScale));
    glUniform1f(glGetUniformLocation(program, "uAspectScale"),
                static_cast<float>(height) / static_cast<float>(width));
    glUniform2f(glGetUniformLocation(program, "uViewportSize"), static_cast<float>(width),
                static_cast<float>(height));
    glUniform4f(glGetUniformLocation(program, "uColor"), color[0], color[1], color[2], color[3]);

    const auto colorsLocation = glGetUniformLocation(program, "uRoadColors[0]");
    const auto widthsLocation = glGetUniformLocation(program, "uRoadWidths[0]");
    const bool styleValid = colorsLocation >= 0 && widthsLocation >= 0;
    glUniform1ui(glGetUniformLocation(program, "uRoadStyleDisabled"), styles.enabled ? 0u : 1u);
    glUniform1ui(glGetUniformLocation(program, "uRoadStyleConfigValid"), styleValid ? 1u : 0u);
    glUniform1f(glGetUniformLocation(program, "uRoadWidthScale"), styles.globalWidthScale);
    if (styleValid) {
        glUniform4fv(colorsLocation, static_cast<GLsizei>(styles.colors.size()),
                     styles.colors.front().data());
        glUniform1fv(widthsLocation, static_cast<GLsizei>(styles.widthsPixels.size()),
                     styles.widthsPixels.data());
    }

    drawTimer_.Begin();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, commandBuffer_.Id());
    glDrawArraysIndirect(GL_TRIANGLES, nullptr);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    drawTimer_.End();
}

} // namespace chmv::gpu::lod
