#include "cpu/CpuRuntime.h"

#include "data/SystemMemory.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <system_error>
#include <utility>

namespace chmv::cpu {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;

} // namespace

CpuRuntime::CpuRuntime(std::shared_ptr<const streaming::lod::LodPageFile> file,
                       std::shared_ptr<const std::vector<std::uint32_t>> pageImage,
                       std::shared_ptr<const std::vector<std::uint32_t>> childrenImage)
    : file_(file), store_(file, std::move(pageImage), std::move(childrenImage)),
      workers_(std::make_unique<CpuWorkers>(CpuWorkers::HardwareThreads())) {}

CpuRuntime::~CpuRuntime() {
    classicLoader_.request_stop();
}

void CpuRuntime::SetThreads(unsigned threads) {
    threads = std::max(threads, 1u);
    if (threads == workers_->Threads() || threads == refusedThreads_) {
        return;
    }
    try {
        workers_ = std::make_unique<CpuWorkers>(threads);
    } catch (const std::system_error&) {
        // The system would not start that many threads: keep the current ones (results show
        // how many ran) and do not try that count again every frame.
        refusedThreads_ = threads;
    }
}

ClassicStatus CpuRuntime::Classic() const {
    const std::lock_guard lock(classicMutex_);
    return classicStatus_;
}

void CpuRuntime::StartClassicLoad() {
    // It is asked for explicitly, so a smaller reserve than for the page RAM image will do.
    const auto needed = ClassicGraph::EstimateBytes(file_->Header());
    const auto memory = data::QuerySystemMemoryInfo();
    const auto reserve = std::max<std::uint64_t>(2048u * kMiB, memory.totalPhysicalBytes / 10u);
    {
        const std::lock_guard lock(classicMutex_);
        if (memory.availablePhysicalBytes != 0u &&
            needed + reserve > memory.availablePhysicalBytes) {
            classicStatus_.state = ClassicState::Failed;
            classicStatus_.message =
                "the whole CH needs about " + std::to_string(needed / kMiB) + " MiB of RAM, " +
                std::to_string(memory.availablePhysicalBytes / kMiB) + " MiB are free";
            return;
        }
        classicStatus_.state = ClassicState::Loading;
        classicStatus_.message =
            "loading the whole CH into RAM (about " + std::to_string(needed / kMiB) + " MiB)";
    }
    classicLoader_ = std::jthread([this](const std::stop_token& stop) {
        const auto begin = Clock::now();
        try {
            auto graph = ClassicGraph::Load(*file_, stop);
            const std::lock_guard lock(classicMutex_);
            classicStatus_.bytes = graph->Bytes();
            classicStatus_.loadSeconds =
                std::chrono::duration<double>(Clock::now() - begin).count();
            classicStatus_.message.clear();
            classic_ = std::move(graph);
            classicStatus_.state = ClassicState::Ready;
        } catch (const std::exception& error) {
            const std::lock_guard lock(classicMutex_);
            classicStatus_.state = ClassicState::Failed;
            classicStatus_.message = error.what();
        }
    });
}

bool CpuRuntime::Update(const streaming::lod::LodPageQuery& query, CpuMethod method,
                        const CpuStrategy& strategy, std::optional<std::uint32_t> stepBase) {
    if (method == CpuMethod::Classic && Classic().state == ClassicState::NotLoaded) {
        StartClassicLoad();
    }
    Key key;
    key.query = query;
    key.method = method;
    key.filter = strategy.filter;
    key.expansion = strategy.expansion;
    key.cullSegments = strategy.cullSegments;
    key.threads = workers_->Threads();
    key.classic = method == CpuMethod::Classic ? Classic().state : ClassicState::NotLoaded;
    key.stepBase = stepBase;
    if (done_ && *done_ == key) {
        return false;
    }
    result_ = Run(query, method, strategy, stepBase);
    done_ = key;
    key_ = key;
    ++serial_;
    return true;
}

CpuResult CpuRuntime::Run(const streaming::lod::LodPageQuery& query, CpuMethod method,
                          const CpuStrategy& strategy, std::optional<std::uint32_t> stepBase) {
    auto result = RunPass(query, method, strategy, stepBase);
    result.threads = workers_->Threads();
    result.level = query.level;
    result.threshold = query.threshold;
    return result;
}

CpuResult CpuRuntime::RunPass(const streaming::lod::LodPageQuery& query, CpuMethod method,
                              const CpuStrategy& strategy,
                              std::optional<std::uint32_t> stepBase) {
    CpuResult result;
    if (!strategy.filter || !strategy.expansion) {
        result.unavailable = "this filter or expansion has no CPU version yet";
        return result;
    }
    try {
        if (method == CpuMethod::Pages) {
            return RunPagePass(store_, *workers_, query, *strategy.filter, *strategy.expansion,
                               strategy.mode == geometry::RefinementMode::Adaptive ? stepBase
                                                                                   : std::nullopt,
                               strategy.cullSegments);
        }
        const ClassicGraph* graph = nullptr;
        {
            const std::lock_guard lock(classicMutex_);
            if (classicStatus_.state != ClassicState::Ready) {
                switch (classicStatus_.state) {
                case ClassicState::Loading:
                    result.unavailable = classicStatus_.message;
                    break;
                case ClassicState::Failed:
                    result.unavailable = "classic method unavailable: " + classicStatus_.message;
                    break;
                default:
                    result.unavailable = "the whole CH is not loaded yet";
                    break;
                }
                return result;
            }
            graph = classic_.get();
        }
        return graph->Run(*workers_, query, *strategy.filter, *strategy.expansion,
                          strategy.cullSegments);
    } catch (const std::exception& error) {
        result.unavailable = std::string("CPU pass failed: ") + error.what();
        return result;
    }
}

CpuBenchmarkResult CpuRuntime::Benchmark(const streaming::lod::LodPageQuery& query,
                                         CpuMethod method, const CpuStrategy& strategy,
                                         std::uint32_t maxIterations, double maxSeconds) {
    CpuBenchmarkResult benchmark;
    benchmark.threads = workers_->Threads();
    // The first pass fetches the pages and fills the caches.
    if (const auto first = Run(query, method, strategy); !first.unavailable.empty()) {
        benchmark.unavailable = first.unavailable;
        return benchmark;
    }
    const auto begin = Clock::now();
    std::vector<double> prepare;
    std::vector<double> filter;
    std::vector<double> expand;
    std::vector<double> output;
    std::vector<double> total;
    while (benchmark.iterations < maxIterations) {
        const auto run = Run(query, method, strategy);
        if (!run.unavailable.empty()) {
            benchmark.unavailable = run.unavailable;
            return benchmark;
        }
        prepare.push_back(run.prepareMs);
        filter.push_back(run.filterMs);
        expand.push_back(run.expandMs);
        output.push_back(run.outputMs);
        total.push_back(run.TotalMs());
        benchmark.candidates = run.candidates;
        benchmark.segments = run.segments.size();
        ++benchmark.iterations;
        if (benchmark.iterations >= 3u &&
            std::chrono::duration<double>(Clock::now() - begin).count() > maxSeconds) {
            break;
        }
    }
    benchmark.prepare = benchmark::Summarize(std::move(prepare));
    benchmark.filter = benchmark::Summarize(std::move(filter));
    benchmark.expand = benchmark::Summarize(std::move(expand));
    benchmark.output = benchmark::Summarize(std::move(output));
    benchmark.total = benchmark::Summarize(std::move(total));
    return benchmark;
}

} // namespace chmv::cpu
