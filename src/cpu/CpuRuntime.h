#pragma once

#include "benchmark/TimingStats.h"
#include "cpu/ClassicPipeline.h"
#include "cpu/CpuPasses.h"
#include "cpu/CpuWorkers.h"
#include "cpu/PagePipeline.h"
#include "streaming/lod/LodPageFile.h"
#include "streaming/lod/LodPagePlanner.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace chmv::cpu {

// Repeated passes of one CPU method, timed with the wall clock.
// Repeated CPU passes after an unmeasured one: preparing (pages), filter, expansion, output
// and the whole pass.
struct CpuBenchmarkResult {
    std::uint32_t iterations = 0;
    unsigned threads = 1;
    std::uint64_t candidates = 0;
    std::uint64_t segments = 0;
    benchmark::TimingStats prepare;
    benchmark::TimingStats filter;
    benchmark::TimingStats expand;
    benchmark::TimingStats output;
    benchmark::TimingStats total;
    std::string unavailable;
};

enum class ClassicState { NotLoaded, Loading, Ready, Failed };

struct ClassicStatus {
    ClassicState state = ClassicState::NotLoaded;
    std::string message;
    std::uint64_t bytes = 0;
    double loadSeconds = 0.0;
};

// The CPU side of the viewer: runs the selected CPU method for the current view whenever the
// view, the strategies, the method or the thread count change. The classic method needs the
// whole CH in RAM; it is loaded in the background the first time it is chosen.
class CpuRuntime {
public:
    CpuRuntime(std::shared_ptr<const streaming::lod::LodPageFile> file,
               std::shared_ptr<const std::vector<std::uint32_t>> pageImage,
               std::shared_ptr<const std::vector<std::uint32_t>> childrenImage);
    ~CpuRuntime();

    CpuRuntime(const CpuRuntime&) = delete;
    CpuRuntime& operator=(const CpuRuntime&) = delete;

    void SetThreads(unsigned threads);
    // Runs a pass unless nothing changed since the last one; true if the result is new. With
    // stepBase (the error code of one pixel) the page method also counts the segments each
    // tolerance step would draw.
    bool Update(const streaming::lod::LodPageQuery& query, CpuMethod method,
                const CpuStrategy& strategy,
                std::optional<std::uint32_t> stepBase = std::nullopt);
    [[nodiscard]] const CpuResult& Result() const { return result_; }
    // Query of the current result, and a number that grows with every new result.
    [[nodiscard]] const streaming::lod::LodPageQuery& ResultQuery() const { return key_.query; }
    [[nodiscard]] std::uint64_t ResultSerial() const { return serial_; }
    [[nodiscard]] ClassicStatus Classic() const;
    [[nodiscard]] CpuBenchmarkResult Benchmark(const streaming::lod::LodPageQuery& query,
                                               CpuMethod method, const CpuStrategy& strategy,
                                               std::uint32_t maxIterations, double maxSeconds);

private:
    struct Key {
        streaming::lod::LodPageQuery query;
        CpuMethod method = CpuMethod::Pages;
        std::optional<CpuFilter> filter;
        std::optional<CpuExpansion> expansion;
        bool cullSegments = false;
        unsigned threads = 0;
        ClassicState classic = ClassicState::NotLoaded;
        std::optional<std::uint32_t> stepBase;

        [[nodiscard]] bool operator==(const Key&) const = default;
    };

    [[nodiscard]] CpuResult Run(const streaming::lod::LodPageQuery& query, CpuMethod method,
                                const CpuStrategy& strategy,
                                std::optional<std::uint32_t> stepBase = std::nullopt);
    [[nodiscard]] CpuResult RunPass(const streaming::lod::LodPageQuery& query, CpuMethod method,
                                    const CpuStrategy& strategy,
                                    std::optional<std::uint32_t> stepBase);
    void StartClassicLoad();

    std::shared_ptr<const streaming::lod::LodPageFile> file_;
    CpuPageStore store_;
    std::unique_ptr<CpuWorkers> workers_;
    // A thread count the system refused to start.
    unsigned refusedThreads_ = 0;
    CpuResult result_;
    std::optional<Key> done_;
    Key key_;
    std::uint64_t serial_ = 0;

    mutable std::mutex classicMutex_;
    ClassicStatus classicStatus_;
    std::unique_ptr<ClassicGraph> classic_;
    std::jthread classicLoader_;
};

} // namespace chmv::cpu
