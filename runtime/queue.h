#pragma once

#include "inference.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <optional>
#include <vector>
#include <cstdint>

struct QueuedRequest {
    InferenceRequest                        request;
    std::chrono::steady_clock::time_point   enqueue_time;
    int                                     priority = 0;
};

class RequestQueue {
public:
    explicit RequestQueue(size_t max_size = 1000);

    bool push(const InferenceRequest& req, int priority = 0);

    std::optional<QueuedRequest> pop(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(100));

    std::optional<QueuedRequest> try_pop();

    std::vector<QueuedRequest> pop_batch(size_t max_n,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(50));

    size_t size()        const;
    bool   empty()       const;
    bool   is_shutdown() const { return shutdown_.load(); }
    void   shutdown();

    struct Stats {
        uint64_t total_enqueued  = 0;
        uint64_t total_processed = 0;
        uint64_t total_dropped   = 0;
        double   avg_wait_ms     = 0.0;
    };
    Stats get_stats() const;

private:
    size_t                    max_size_;
    std::queue<QueuedRequest> queue_;
    mutable std::mutex        mutex_;
    std::condition_variable   cv_;
    std::atomic<bool>         shutdown_{false};

    // Stats protected by mutex (double not atomic-safe on all platforms)
    uint64_t total_enqueued_  = 0;
    uint64_t total_processed_ = 0;
    uint64_t total_dropped_   = 0;
    double   avg_wait_ms_     = 0.0;
};