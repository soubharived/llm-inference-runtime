#pragma once

#include "queue.h"
#include "inference.h"
#include "scheduler.h"
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <chrono>

// ── Dynamic Batcher ─────────────────────────────────────────────────
class DynamicBatcher {
public:
    struct Config {
        size_t max_batch_size = 8;
        int    max_wait_ms    = 50;
    };

    DynamicBatcher(RequestQueue&    queue,
                   InferenceEngine& engine,
                   Config           config);

    ~DynamicBatcher();

    void start();
    void stop();
    void set_result_callback(ResultCallback cb) { result_cb_ = cb; }

    struct Stats {
        uint64_t batches_processed = 0;
        double   avg_batch_size    = 0.0;
        double   throughput_rps    = 0.0;
        double   avg_latency_ms    = 0.0;
    };
    Stats get_stats() const;

private:
    RequestQueue&    queue_;
    InferenceEngine& engine_;
    Config           config_;
    ResultCallback   result_cb_;

    std::thread       worker_;
    std::atomic<bool> running_{false};

    std::atomic<uint64_t> batches_processed_{0};
    std::atomic<double>   total_batch_size_{0.0};
    std::atomic<double>   total_latency_ms_{0.0};
    std::chrono::steady_clock::time_point start_time_;

    void batcher_loop();
};