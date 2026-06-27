#pragma once

#include "queue.h"
#include "inference.h"
#include <vector>
#include <functional>
#include <thread>
#include <atomic>

enum class SchedulerPolicy {
    FCFS,              // First Come First Served
    PRIORITY,          // Higher priority first
    SHORTEST_FIRST,    // Shortest prompt first
};

// ── Result callback ─────────────────────────────────────────────────
using ResultCallback = std::function<void(InferenceResult)>;

// ── Scheduler: pulls from queue, dispatches to inference ───────────
class Scheduler {
public:
    Scheduler(RequestQueue&   queue,
              InferenceEngine& engine,
              SchedulerPolicy  policy     = SchedulerPolicy::FCFS,
              int              n_workers  = 1);
    ~Scheduler();

    void start();
    void stop();
    void set_result_callback(ResultCallback cb) { result_cb_ = cb; }

    struct Stats {
        uint64_t requests_served  = 0;
        double   avg_latency_ms   = 0.0;
        double   throughput_rps   = 0.0;   // requests per second
    };
    Stats get_stats() const;

private:
    RequestQueue&    queue_;
    InferenceEngine& engine_;
    SchedulerPolicy  policy_;
    int              n_workers_;
    ResultCallback   result_cb_;

    std::vector<std::thread>  workers_;
    std::atomic<bool>         running_{false};

    // Stats
    std::atomic<uint64_t> requests_served_{0};
    std::atomic<double>   total_latency_ms_{0.0};
    std::chrono::steady_clock::time_point start_time_;

    void worker_loop();
    QueuedRequest select_next(std::vector<QueuedRequest>& candidates);
};