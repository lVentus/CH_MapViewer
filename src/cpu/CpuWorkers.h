#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace chmv::cpu {

// A fixed set of threads for the CPU passes, started once and kept: a pass that runs every
// frame must not pay for creating threads. The calling thread works along, so one thread means
// the pass runs on the caller alone.
class CpuWorkers {
public:
    explicit CpuWorkers(unsigned threads);
    ~CpuWorkers();

    CpuWorkers(const CpuWorkers&) = delete;
    CpuWorkers& operator=(const CpuWorkers&) = delete;

    [[nodiscard]] unsigned Threads() const { return threads_; }
    // Runs work(task, worker) for every task in [0, tasks) and returns when all are done;
    // `worker` is below Threads(). The exception of the lowest failed task is rethrown.
    void Run(std::size_t tasks, const std::function<void(std::size_t, unsigned)>& work);

    // Threads the hardware offers (at least one).
    [[nodiscard]] static unsigned HardwareThreads();

private:
    void Work(unsigned worker);
    void Drain(unsigned worker);

    unsigned threads_ = 1;
    std::vector<std::jthread> pool_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(std::size_t, unsigned)>* work_ = nullptr;
    std::size_t tasks_ = 0;
    std::size_t next_ = 0;
    std::size_t finished_ = 0;
    std::size_t failedTask_ = 0;
    std::exception_ptr failure_;
    std::uint64_t generation_ = 0;
    bool stopping_ = false;
};

} // namespace chmv::cpu
