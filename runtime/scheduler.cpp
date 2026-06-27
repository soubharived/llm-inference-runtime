#include "scheduler.h"
#include <iostream>
#include <algorithm>

// ───────────────────────────────────────────────────────────────────
Scheduler::Scheduler(RequestQueue&   queue,
                     InferenceEngine& engine,
                     SchedulerPolicy  policy,
                     int              n_workers)
    : queue_(queue)
    , engine_(engine)
    , policy_(policy)
    , n_workers_(n_workers)
{}

// ───────────────────────────────────────────────────────────────────
Scheduler::~Scheduler() { stop(); }

// ───────────────────────────────────────────────────────────────────
void Scheduler::start()
{
    running_     = true;
    start_time_  = std::chrono::steady_clock::now();

    for (int i = 0; i < n_workers_; ++i) {
        workers_.emplace_back(&Scheduler::worker_loop, this);
    }
    std::cout << "[Scheduler] Started with " << n_workers_
              << " worker(s), policy="
              << (policy_ == SchedulerPolicy::FCFS ? "FCFS" :
                  policy_ == SchedulerPolicy::PRIORITY ? "PRIORITY" :
                  "SHORTEST_FIRST") << "\n";
}

// ───────────────────────────────────────────────────────────────────
void Scheduler::stop()
{
    running_ = false;
    queue_.shutdown();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

// ───────────────────────────────────────────────────────────────────
void Scheduler::worker_loop()
{
    while (running_) {
        // Get a small batch of candidates for scheduling decision
        auto candidates = queue_.pop_batch(
            8, std::chrono::milliseconds(100));

        if (candidates.empty()) continue;

        // Pick the best request based on policy
        QueuedRequest chosen = select_next(candidates);

        // Put unchosen ones back (re-enqueue)
        for (auto& c : candidates) {
            if (c.request.id != chosen.request.id) {
                queue_.push(c.request, c.priority);
            }
        }

        // Run inference
        InferenceResult result = engine_.run(chosen.request);

        // Update stats
        requests_served_++;
        double prev = total_latency_ms_.load();
        total_latency_ms_.store(
            prev + (result.latency_ms - prev) / requests_served_.load());

        // Deliver result
        if (result_cb_) result_cb_(result);

        std::cout << "[Scheduler] Request " << result.request_id
                  << " done in " << result.latency_ms << "ms"
                  << " | tokens=" << result.tokens_generated << "\n";
    }
}

// ───────────────────────────────────────────────────────────────────
QueuedRequest Scheduler::select_next(std::vector<QueuedRequest>& candidates)
{
    if (candidates.size() == 1 || policy_ == SchedulerPolicy::FCFS) {
        // FCFS: just take first (already in arrival order)
        QueuedRequest chosen = candidates.front();
        candidates.erase(candidates.begin());
        return chosen;
    }

    if (policy_ == SchedulerPolicy::PRIORITY) {
        auto it = std::max_element(candidates.begin(), candidates.end(),
            [](const QueuedRequest& a, const QueuedRequest& b) {
                return a.priority < b.priority;
            });
        QueuedRequest chosen = *it;
        candidates.erase(it);
        return chosen;
    }

    if (policy_ == SchedulerPolicy::SHORTEST_FIRST) {
        auto it = std::min_element(candidates.begin(), candidates.end(),
            [](const QueuedRequest& a, const QueuedRequest& b) {
                return a.request.prompt.size() < b.request.prompt.size();
            });
        QueuedRequest chosen = *it;
        candidates.erase(it);
        return chosen;
    }

    // Default fallback
    QueuedRequest chosen = candidates.front();
    candidates.erase(candidates.begin());
    return chosen;
}

// ───────────────────────────────────────────────────────────────────
Scheduler::Stats Scheduler::get_stats() const
{
    Stats s;
    s.requests_served = requests_served_.load();
    s.avg_latency_ms  = total_latency_ms_.load();

    auto now     = std::chrono::steady_clock::now();
    double secs  = std::chrono::duration<double>(now - start_time_).count();
    s.throughput_rps = secs > 0 ? s.requests_served / secs : 0.0;

    return s;
}