#include "batcher.h"
#include <iostream>
#include <chrono>

// ───────────────────────────────────────────────────────────────────
DynamicBatcher::DynamicBatcher(RequestQueue&    queue,
                                InferenceEngine& engine,
                                Config           config)
    : queue_(queue), engine_(engine), config_(config)
{}

DynamicBatcher::~DynamicBatcher() { stop(); }

// ───────────────────────────────────────────────────────────────────
void DynamicBatcher::start()
{
    running_    = true;
    start_time_ = std::chrono::steady_clock::now();
    worker_     = std::thread(&DynamicBatcher::batcher_loop, this);
    std::cout << "[Batcher] Started. max_batch=" << config_.max_batch_size
            << " max_wait=" << config_.max_wait_ms << "ms\n";
}

void DynamicBatcher::stop()
{
    running_ = false;
    queue_.shutdown();
    if (worker_.joinable()) worker_.join();
}

// ───────────────────────────────────────────────────────────────────
void DynamicBatcher::batcher_loop()
{
    while (running_) {
        // Collect a batch
        auto batch = queue_.pop_batch(
            config_.max_batch_size,
            std::chrono::milliseconds(config_.max_wait_ms));

        if (batch.empty()) continue;

        std::cout << "[Batcher] Processing batch of " << batch.size() << "\n";

        // Extract requests
        std::vector<InferenceRequest> reqs;
        reqs.reserve(batch.size());
        for (auto& qr : batch) reqs.push_back(qr.request);

        // Run batch inference
        auto t0      = std::chrono::high_resolution_clock::now();
        auto results = engine_.run_batch(reqs);
        auto t1      = std::chrono::high_resolution_clock::now();

        double batch_ms = std::chrono::duration<double, std::milli>(
            t1 - t0).count();

        // Update stats
        batches_processed_++;
        total_batch_size_.store(
            total_batch_size_.load() + (double)batch.size());
        total_latency_ms_.store(
            total_latency_ms_.load() + batch_ms);

        // Deliver results
        for (auto& r : results) {
            std::cout << "[Batcher] Request " << r.request_id
                      << " | latency=" << r.latency_ms << "ms"
                      << " | tokens=" << r.tokens_generated << "\n";
            if (result_cb_) result_cb_(r);
        }
    }
}

// ───────────────────────────────────────────────────────────────────
DynamicBatcher::Stats DynamicBatcher::get_stats() const
{
    Stats s;
    s.batches_processed = batches_processed_.load();
    s.avg_batch_size    = s.batches_processed > 0
        ? total_batch_size_.load() / s.batches_processed : 0.0;
    s.avg_latency_ms    = s.batches_processed > 0
        ? total_latency_ms_.load() / s.batches_processed : 0.0;

    auto now  = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    s.throughput_rps = elapsed > 0
        ? (total_batch_size_.load() / elapsed) : 0.0;

    return s;
}