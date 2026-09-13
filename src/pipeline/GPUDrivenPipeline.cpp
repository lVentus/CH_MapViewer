#include "pipeline/GPUDrivenPipeline.h"

#include <glad/gl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace chmv::pipeline {
namespace {

struct DrawArraysIndirectCommand {
    std::uint32_t count;
    std::uint32_t instanceCount;
    std::uint32_t first;
    std::uint32_t baseInstance;
};

static_assert(sizeof(DrawArraysIndirectCommand) == 16);

void CheckBufferSize(std::size_t sizeBytes) {
    GLint64 maxBlockSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
    if (sizeBytes > static_cast<std::size_t>(maxBlockSize)) {
        throw std::runtime_error(
            "dataset exceeds the maximum SSBO size of this GPU; streaming is not implemented yet");
    }
}


} // namespace

void GPUDrivenPipeline::SetGraph(std::uint32_t edgeCount, std::uint32_t drawableEdgeCount) {
    if (edgeCount_ == edgeCount && visibleCapacity_ == drawableEdgeCount) {
        return;
    }

    edgeCount_ = edgeCount;
    visibleCapacity_ = drawableEdgeCount;
    rangeCandidateEdgeCount_.reset();

    const auto storageCount = std::max<std::uint32_t>(drawableEdgeCount, 1);
    const auto sizeBytes = static_cast<std::size_t>(storageCount) * sizeof(std::uint32_t);
    CheckBufferSize(sizeBytes);
    visibleEdgeBuffer_.Allocate(sizeBytes, nullptr, GL_DYNAMIC_DRAW);

    const DrawArraysIndirectCommand command{0, 1, 0, 0};
    indirectBuffer_.Allocate(sizeof(command), &command, GL_DYNAMIC_DRAW);
}

GPUProcessingResult GPUDrivenPipeline::Process(
    std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
    gpu::filtering::IRangeFilterStrategy& rangeFilterStrategy,
    gpu::unfolding::IUnfoldingStrategy& unfoldingStrategy,
    const geometry::RefinementParameters& refinementParameters) {
    if (edgeCount == 0) {
        return {};
    }

    if (edgeCount_ != edgeCount) {
        throw std::runtime_error("GPU pipeline graph state is out of date");
    }

    const std::uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indirectBuffer_.Id());
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                         GL_UNSIGNED_INT, &zero);

    const auto filterStats = rangeFilterStrategy.Execute({
        graphEdgeBuffer,
        edgeCount,
        lodLevel,
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
    });
    rangeCandidateEdgeCount_ = filterStats.candidateEdgeCount;

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    const bool geometryRefinementActive = unfoldingStrategy.RefinesGeometry();
    if (!geometryRefinementActive) {
        return {
            visibleEdgeBuffer_.Id(),
            indirectBuffer_.Id(),
            visibleEdgeBuffer_.Id(),
            indirectBuffer_.Id(),
        };
    }

    const gpu::unfolding::UnfoldingInput input{
        graphEdgeBuffer,
        edgeCount,
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
        edgeCount,
        refinementParameters.screenPixelScale,
        refinementParameters.maxScreenErrorPixels,
    };

    const auto refined = unfoldingStrategy.Execute(input);

    return {
        visibleEdgeBuffer_.Id(),
        indirectBuffer_.Id(),
        refined.edgeBuffer,
        refined.drawCommandBuffer,
    };
}

GPURangeBenchmarkResult GPUDrivenPipeline::RunRangeFilterBenchmark(
    std::uint32_t graphEdgeBuffer, std::uint32_t edgeCount, std::int32_t lodLevel,
    gpu::filtering::IRangeFilterStrategy& rangeFilterStrategy) {
    if (edgeCount == 0 || edgeCount_ != edgeCount) {
        throw std::runtime_error("GPU pipeline graph state is out of date");
    }

    constexpr std::size_t warmupIterations = 20;
    constexpr std::size_t measuredIterations = 200;
    constexpr std::size_t batchSize = 10;
    constexpr std::size_t sampleCount = measuredIterations / batchSize;

    const auto runOnce = [&]() {
        const std::uint32_t zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, indirectBuffer_.Id());
        glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(zero), GL_RED_INTEGER,
                             GL_UNSIGNED_INT, &zero);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_COMMAND_BARRIER_BIT);

        static_cast<void>(rangeFilterStrategy.Execute({
            graphEdgeBuffer,
            edgeCount,
            lodLevel,
            visibleEdgeBuffer_.Id(),
            indirectBuffer_.Id(),
        }));
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT |
                        GL_BUFFER_UPDATE_BARRIER_BIT);
    };

    glFinish();
    for (std::size_t i = 0; i < warmupIterations; ++i) {
        runOnce();
    }
    glFinish();

    std::array<double, sampleCount> samples{};
    double measuredTotalMs = 0.0;
    for (std::size_t sample = 0; sample < sampleCount; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < batchSize; ++i) {
            runOnce();
        }
        glFinish();
        const auto end = std::chrono::steady_clock::now();
        const auto batchMs = std::chrono::duration<double, std::milli>(end - start).count();
        measuredTotalMs += batchMs;
        samples[sample] = batchMs / static_cast<double>(batchSize);
    }

    const auto diagnostics = rangeFilterStrategy.ReadBackDiagnostics(lodLevel);

    DrawArraysIndirectCommand command{};
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer_.Id());
    glGetBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(command), &command);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    auto sortedSamples = samples;
    std::sort(sortedSamples.begin(), sortedSamples.end());

    const auto percentile = [&](double fraction) {
        const auto index = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(sortedSamples.size()))) - 1u;
        return sortedSamples[std::min(index, sortedSamples.size() - 1u)];
    };

    const auto mean = measuredTotalMs / static_cast<double>(measuredIterations);
    const auto middle = sortedSamples.size() / 2u;
    const auto median = (sortedSamples[middle - 1u] + sortedSamples[middle]) * 0.5;
    const auto squaredDifferenceSum = std::accumulate(
        samples.begin(), samples.end(), 0.0,
        [mean](double total, double value) {
            const auto difference = value - mean;
            return total + difference * difference;
        });

    return {
        std::string(rangeFilterStrategy.Name()),
        lodLevel,
        warmupIterations,
        measuredIterations,
        batchSize,
        diagnostics.candidateEdgeCount,
        command.count / 2u,
        measuredTotalMs,
        mean,
        median,
        sortedSamples.front(),
        sortedSamples.back(),
        percentile(0.95),
        percentile(0.99),
        std::sqrt(squaredDifferenceSum / static_cast<double>(samples.size())),
    };
}

void GPUDrivenPipeline::ReadBackEdgeIds(std::uint32_t edgeIdBuffer,
                                        std::uint32_t drawCommandBuffer,
                                        std::vector<std::uint32_t>& output) const {
    output.clear();
    if (edgeIdBuffer == 0 || drawCommandBuffer == 0) {
        return;
    }

    DrawArraysIndirectCommand command{};
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, drawCommandBuffer);
    glGetBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(command), &command);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    const auto outputEdgeCount = command.count / 2;
    output.resize(outputEdgeCount);
    if (outputEdgeCount == 0) {
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeIdBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       static_cast<GLsizeiptr>(outputEdgeCount * sizeof(std::uint32_t)),
                       output.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

GPUProcessingStats GPUDrivenPipeline::Stats() const {
    return {rangeCandidateEdgeCount_};
}

} // namespace chmv::pipeline
