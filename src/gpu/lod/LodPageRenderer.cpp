#include "gpu/lod/LodPageRenderer.h"

#include "gpu/filtering/IRangeFilterStrategy.h"
#include "gpu/lod/LodPagePool.h"
#include "gpu/unfolding/IUnfoldingStrategy.h"
#include "renderer/MapCamera2D.h"

#include <glad/gl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace chmv::gpu::lod {
namespace {

using chmv::streaming::lod::kFixedCoordinateScale;
using chmv::streaming::lod::kPageRecordCapacity;

// The command buffer holds the indirect draw arguments followed by the segment count.
constexpr std::size_t kCommandWords = 5;
// Readback layout: command, candidate count, two spare words, status, segments per tolerance
// step.
constexpr std::size_t kReadbackCandidateWord = 5;
constexpr std::size_t kReadbackStatusWord = 8;
constexpr std::size_t kReadbackStepWord = kReadbackStatusWord + kStatusWords;
constexpr auto kSteps = static_cast<std::size_t>(chmv::renderer::kToleranceSteps);
constexpr std::size_t kReadbackWords = kReadbackStepWord + kSteps;
// Segment room grows by doubling when a traversal draws more than the pages hold records.
constexpr std::uint32_t kMaxSegmentScale = 8;
// Benchmark: unmeasured passes first (shaders, caches, buffer growth), and calls per batch of
// the wall-time measurement.
constexpr std::uint32_t kWarmUpPasses = 10;
constexpr std::uint32_t kWallBatch = 10;

[[nodiscard]] GLsync AsSync(void* fence) {
    return static_cast<GLsync>(fence);
}

[[nodiscard]] GLint FixedCameraCoordinate(float center) {
    constexpr double kLimit = 1073741823.0;
    const auto fixed = std::round(static_cast<double>(center) * kFixedCoordinateScale);
    return static_cast<GLint>(std::clamp(fixed, -kLimit, kLimit));
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
      finishProgram_(shaderDirectory / "lod_draw_finish.comp"),
      countProgram_(shaderDirectory / "lod_tolerance_counts.comp") {
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
    stepCountBuffer_.Allocate(kSteps * 4u);
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
    input.cullBox = chmv::streaming::lod::TraversalCullBox(pass.query);
    input.cullSegments = pass.cullSegments;
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

void LodPageRenderer::CountSteps(const LodPagePool& pool, const LodPagePass& pass,
                                 std::uint32_t stepBase) {
    const std::array<std::uint32_t, kSteps> zero{};
    stepCountBuffer_.Update(0, sizeof(zero), zero.data());
    countProgram_.Bind();
    const auto program = countProgram_.Id();
    pool.BindForKernel(program);
    slotList_.BindBase(GL_SHADER_STORAGE_BUFFER, 2);
    stepCountBuffer_.BindBase(GL_SHADER_STORAGE_BUFFER, 5);
    glUniform1ui(glGetUniformLocation(program, "uLevel"), pass.query.level);
    glUniform1ui(glGetUniformLocation(program, "uStepBase"), stepBase);
    const auto cullBox = chmv::streaming::lod::TraversalCullBox(pass.query);
    glUniform4i(glGetUniformLocation(program, "uCullBox"), cullBox[0], cullBox[1], cullBox[2],
                cullBox[3]);
    glUniform1ui(glGetUniformLocation(program, "uCullSegments"), pass.cullSegments ? 1u : 0u);
    ListDispatcher::Dispatch(slotList_.Id());
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

void LodPageRenderer::Process(const LodPagePool& pool, const LodPagePass& pass,
                              filtering::IRangeFilterStrategy& filter,
                              unfolding::IUnfoldingStrategy& unfold,
                              std::optional<std::uint32_t> stepBase) {
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
        stepCounts_.reset();
        return;
    }
    filterTimer_.Begin();
    Filter(pool, pass, capacities, filter);
    filterTimer_.End();
    unfoldTimer_.Begin();
    Unfold(pool, pass, capacities, unfold);
    unfoldTimer_.End();
    if (stepBase) {
        countTimer_.Begin();
        CountSteps(pool, pass, *stepBase);
        countTimer_.End();
    }

    auto& readback = readbacks_[nextReadback_];
    nextReadback_ = (nextReadback_ + 1u) % readbacks_.size();
    if (readback.fence != nullptr) {
        glDeleteSync(AsSync(readback.fence));
    }
    CopyWords(commandBuffer_, 0, readback.buffer, 0, kCommandWords);
    CopyWords(candidateList_, 0, readback.buffer, kReadbackCandidateWord, 1);
    CopyWords(statusBuffer_, 0, readback.buffer, kReadbackStatusWord, kStatusWords);
    if (stepBase) {
        CopyWords(stepCountBuffer_, 0, readback.buffer, kReadbackStepWord, kSteps);
    }
    readback.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    readback.segmentScale = segmentScale_;
    readback.counted = stepBase.has_value();
}

LodPageBenchmarkResult LodPageRenderer::Benchmark(const LodPagePool& pool, const LodPagePass& pass,
                                                  filtering::IRangeFilterStrategy& filter,
                                                  unfolding::IUnfoldingStrategy& unfold,
                                                  std::uint32_t iterations,
                                                  const renderer::MapCamera2D& camera, int width,
                                                  int height,
                                                  const renderer::RoadStyleConfig& styles) {
    LodPageBenchmarkResult result;
    result.pages = static_cast<std::uint32_t>(pass.slots.size());
    if (pass.slots.empty() || iterations == 0u) {
        return result;
    }
    using Clock = std::chrono::steady_clock;
    const auto runPass = [&] {
        const auto capacities = StartPass(pool, pass);
        Filter(pool, pass, capacities, filter);
        Unfold(pool, pass, capacities, unfold);
    };
    const auto draw = [&] { Draw(pool, camera, width, height, styles); };
    // Wall time per call, from batches of kWallBatch calls between two glFinish.
    const auto wallTimes = [&](const auto& work) {
        std::vector<double> perCall;
        for (std::uint32_t done = 0; done < iterations; done += kWallBatch) {
            const auto batch = std::min(kWallBatch, iterations - done);
            glFinish();
            const auto begin = Clock::now();
            for (std::uint32_t i = 0; i < batch; ++i) {
                work();
            }
            glFinish();
            perCall.push_back(
                std::chrono::duration<double, std::milli>(Clock::now() - begin).count() / batch);
        }
        return perCall;
    };
    for (std::uint32_t i = 0; i < kWarmUpPasses; ++i) {
        runPass();
    }

    // Timestamps: start, after the filter, after the expansion of every pass; around every draw.
    std::vector<GLuint> queries(std::size_t{iterations} * 5u);
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
    const auto drawQueries = std::size_t{iterations} * 3u;
    for (std::uint32_t i = 0; i < iterations; ++i) {
        glQueryCounter(queries[drawQueries + i * 2u], GL_TIMESTAMP);
        draw();
        glQueryCounter(queries[drawQueries + i * 2u + 1u], GL_TIMESTAMP);
    }
    glFinish();
    std::vector<GLuint64> times(queries.size());
    for (std::size_t q = 0; q < queries.size(); ++q) {
        glGetQueryObjectui64v(queries[q], GL_QUERY_RESULT, &times[q]);
    }
    glDeleteQueries(static_cast<GLsizei>(queries.size()), queries.data());
    const auto milliseconds = [&](std::size_t from, std::size_t to) {
        return static_cast<double>(times[to] - times[from]) * 1e-6;
    };
    std::vector<double> filterMs;
    std::vector<double> unfoldMs;
    std::vector<double> passMs;
    std::vector<double> drawMs;
    for (std::uint32_t i = 0; i < iterations; ++i) {
        filterMs.push_back(milliseconds(i * 3u, i * 3u + 1u));
        unfoldMs.push_back(milliseconds(i * 3u + 1u, i * 3u + 2u));
        passMs.push_back(milliseconds(i * 3u, i * 3u + 2u));
        drawMs.push_back(milliseconds(drawQueries + i * 2u, drawQueries + i * 2u + 1u));
    }
    result.iterations = iterations;
    result.filterGpu = benchmark::Summarize(std::move(filterMs));
    result.unfoldGpu = benchmark::Summarize(std::move(unfoldMs));
    result.passGpu = benchmark::Summarize(std::move(passMs));
    result.drawGpu = benchmark::Summarize(std::move(drawMs));
    result.passWall = benchmark::Summarize(wallTimes(runPass));
    result.drawWall = benchmark::Summarize(wallTimes(draw));

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
        if (readback.counted) {
            stepCounts_.emplace();
            std::copy_n(words.begin() + kReadbackStepWord, kSteps, stepCounts_->begin());
        } else {
            stepCounts_.reset();
        }

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
                           int height, const renderer::RoadStyleConfig& styles) {
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
    const auto colors = styles.TypeColors();
    const auto widths = styles.TypeWidths();
    glUniform4fv(glGetUniformLocation(program, "uRoadColors[0]"),
                 static_cast<GLsizei>(colors.size()), colors.front().data());
    glUniform1fv(glGetUniformLocation(program, "uRoadWidths[0]"),
                 static_cast<GLsizei>(widths.size()), widths.data());
    const auto depths = styles.TypeDepths();
    glUniform1fv(glGetUniformLocation(program, "uRoadDepths[0]"),
                 static_cast<GLsizei>(depths.size()), depths.data());

    drawTimer_.Begin();
    // Classes listed first on top: a road of an earlier class hides one of a later class, and
    // of two roads of one class the first drawn stays.
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, commandBuffer_.Id());
    glDrawArraysIndirect(GL_TRIANGLES, nullptr);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    drawTimer_.End();
}

} // namespace chmv::gpu::lod
