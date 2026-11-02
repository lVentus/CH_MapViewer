#include "cpu/CpuWorkers.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace chmv::cpu {

CpuWorkers::CpuWorkers(unsigned threads) : threads_(std::max(threads, 1u)) {
    pool_.reserve(threads_ - 1u);
    try {
        for (unsigned worker = 1; worker < threads_; ++worker) {
            pool_.emplace_back([this, worker] { Work(worker); });
        }
    } catch (...) {
        // The threads already started must end before the members they use go away.
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        pool_.clear();
        throw;
    }
}

CpuWorkers::~CpuWorkers() {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    pool_.clear();
}

unsigned CpuWorkers::HardwareThreads() {
    return std::max(std::thread::hardware_concurrency(), 1u);
}

void CpuWorkers::Run(std::size_t tasks, const std::function<void(std::size_t, unsigned)>& work) {
    if (tasks == 0u) {
        return;
    }
    {
        const std::lock_guard lock(mutex_);
        work_ = &work;
        tasks_ = tasks;
        next_ = 0;
        finished_ = 0;
        failedTask_ = std::numeric_limits<std::size_t>::max();
        failure_ = nullptr;
        ++generation_;
    }
    wake_.notify_all();
    Drain(0);
    std::exception_ptr failure;
    {
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return finished_ == tasks_; });
        work_ = nullptr;
        failure = std::exchange(failure_, nullptr);
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

void CpuWorkers::Drain(unsigned worker) {
    for (;;) {
        std::size_t task = 0;
        const std::function<void(std::size_t, unsigned)>* work = nullptr;
        {
            const std::lock_guard lock(mutex_);
            if (work_ == nullptr || next_ >= tasks_) {
                return;
            }
            task = next_++;
            work = work_;
        }
        try {
            (*work)(task, worker);
        } catch (...) {
            const std::lock_guard lock(mutex_);
            if (task < failedTask_) {
                failedTask_ = task;
                failure_ = std::current_exception();
            }
        }
        const std::lock_guard lock(mutex_);
        if (++finished_ == tasks_) {
            done_.notify_all();
        }
    }
}

void CpuWorkers::Work(unsigned worker) {
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] {
                return stopping_ || (generation_ != seen && work_ != nullptr && next_ < tasks_);
            });
            if (stopping_) {
                return;
            }
            seen = generation_;
        }
        Drain(worker);
    }
}

} // namespace chmv::cpu
