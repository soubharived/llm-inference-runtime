#include "queue.h"

RequestQueue::RequestQueue(size_t max_size)
    : max_size_(max_size)
{}

bool RequestQueue::push(const InferenceRequest& req, int priority)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (shutdown_.load()) return false;

    if (queue_.size() >= max_size_) {
        total_dropped_++;
        return false;
    }

    QueuedRequest qr;
    qr.request      = req;
    qr.enqueue_time = std::chrono::steady_clock::now();
    qr.priority     = priority;

    queue_.push(std::move(qr));
    total_enqueued_++;
    cv_.notify_one();
    return true;
}

std::optional<QueuedRequest> RequestQueue::pop(
    std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    bool got = cv_.wait_for(lock, timeout, [this] {
        return !queue_.empty() || shutdown_.load();
    });

    if (!got || queue_.empty()) return std::nullopt;

    QueuedRequest qr = std::move(queue_.front());
    queue_.pop();
    total_processed_++;

    auto wait_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - qr.enqueue_time).count();
    avg_wait_ms_ = avg_wait_ms_ * 0.95 + wait_ms * 0.05;

    return qr;
}

std::optional<QueuedRequest> RequestQueue::try_pop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    QueuedRequest qr = std::move(queue_.front());
    queue_.pop();
    total_processed_++;
    return qr;
}

std::vector<QueuedRequest> RequestQueue::pop_batch(
    size_t max_n, std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, timeout, [this] {
        return !queue_.empty() || shutdown_.load();
    });

    std::vector<QueuedRequest> batch;
    batch.reserve(max_n);
    while (!queue_.empty() && batch.size() < max_n) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop();
        total_processed_++;
    }
    return batch;
}

size_t RequestQueue::size() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
}

bool RequestQueue::empty() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
}

void RequestQueue::shutdown()
{
    shutdown_.store(true);
    cv_.notify_all();
}

RequestQueue::Stats RequestQueue::get_stats() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    Stats s;
    s.total_enqueued  = total_enqueued_;
    s.total_processed = total_processed_;
    s.total_dropped   = total_dropped_;
    s.avg_wait_ms     = avg_wait_ms_;
    return s;
}