#pragma once

#include "cpu/CpuWorkers.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace chmv::cpu {

// The persistent work queue of lod_unfold_queue.comp on the CPU threads. Every worker takes
// tickets in order: a ticket below seeds.size() is a seed, the others are queue slots, and the
// holder of a slot polls it until some worker has put a node there or no work is left anywhere.
// A visited node hands back up to two children, which go to the end of the queue; children
// beyond its capacity are walked by the worker itself, so the result never depends on the
// capacity.
//
// visit(node, worker, children) returns how many of children[0], children[1] to visit next.
// If it throws, the other workers stop as well and the exception is passed on.
template <typename Visit>
void RunWorkQueue(CpuWorkers& workers, std::span<const std::uint32_t> seeds, std::size_t capacity,
                  const Visit& visit) {
    constexpr std::uint32_t kEmpty = 0xffffffffu;
    constexpr std::uint64_t kNoTicket = ~std::uint64_t{0};
    const auto slots = std::make_unique<std::atomic<std::uint32_t>[]>(capacity);
    for (std::size_t i = 0; i < capacity; ++i) {
        slots[i].store(kEmpty, std::memory_order_relaxed);
    }
    std::atomic<std::uint64_t> head{0};
    std::atomic<std::uint64_t> tail{0};
    // Nodes handed out but not yet visited; children are counted before their parent is done,
    // so it only reaches zero once all work is done.
    std::atomic<std::uint64_t> pending{seeds.size()};
    std::atomic<bool> failed{false};

    workers.Run(workers.Threads(), [&](std::size_t, unsigned worker) {
        try {
            std::vector<std::uint32_t> own;
            std::array<std::uint32_t, 2> children{};
            const auto expand = [&](std::uint32_t node) {
                const auto count = visit(node, worker, children);
                for (int i = 0; i < count; ++i) {
                    const auto at = tail.fetch_add(1u);
                    if (at < capacity) {
                        pending.fetch_add(1u);
                        slots[at].store(children[static_cast<std::size_t>(i)]);
                    } else {
                        own.push_back(children[static_cast<std::size_t>(i)]);
                    }
                }
            };
            auto ticket = kNoTicket;
            for (;;) {
                if (ticket == kNoTicket) {
                    ticket = head.fetch_add(1u);
                }
                auto node = kEmpty;
                if (ticket < seeds.size()) {
                    node = seeds[ticket];
                } else if (ticket - seeds.size() < capacity) {
                    node = slots[ticket - seeds.size()].exchange(kEmpty);
                }
                if (node != kEmpty) {
                    expand(node);
                    while (!own.empty()) {
                        const auto next = own.back();
                        own.pop_back();
                        expand(next);
                    }
                    pending.fetch_sub(1u);
                    ticket = kNoTicket;
                } else if (pending.load() == 0u || failed.load()) {
                    return;
                } else {
                    std::this_thread::yield();
                }
            }
        } catch (...) {
            // The node this worker held is never done, so the others would wait forever.
            failed.store(true);
            throw;
        }
    });
}

} // namespace chmv::cpu
