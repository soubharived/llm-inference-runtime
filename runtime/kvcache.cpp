#include "kvcache.h"
#include <functional>
#include <chrono>
#include <stdexcept>

// ───────────────────────────────────────────────────────────────────
KVCache::KVCache(size_t max_entries, size_t max_memory_mb)
    : max_entries_(max_entries), max_memory_mb_(max_memory_mb)
{}

// ───────────────────────────────────────────────────────────────────
std::string KVCache::make_key(const std::string& prompt, size_t prefix_len)
{
    std::string text = prefix_len > 0 && prefix_len < prompt.size()
        ? prompt.substr(0, prefix_len)
        : prompt;

    // FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return std::to_string(hash);
}

// ───────────────────────────────────────────────────────────────────
void KVCache::put(const std::string& key,
                  const std::vector<float>& kv_data,
                  size_t token_count)
{
    std::unique_lock<std::mutex> lock(mutex_);

    // If key already exists, update and move to front
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        lru_list_.erase(it->second.second);
        lru_list_.push_front(key);
        it->second.first.kv_data     = kv_data;
        it->second.first.token_count = token_count;
        it->second.second            = lru_list_.begin();
        return;
    }

    // Evict if over capacity
    while (cache_.size() >= max_entries_ ||
           current_memory_bytes() > max_memory_mb_ * 1024 * 1024) {
        evict(1);
    }

    // Insert new entry
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    KVEntry entry;
    entry.prefix_key  = key;
    entry.kv_data     = kv_data;
    entry.token_count = token_count;
    entry.last_access = ms;

    lru_list_.push_front(key);
    cache_[key] = {std::move(entry), lru_list_.begin()};
}

// ───────────────────────────────────────────────────────────────────
std::optional<KVEntry> KVCache::get(const std::string& key)
{
    std::unique_lock<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it == cache_.end()) {
        misses_++;
        return std::nullopt;
    }

    // Move to front (most recently used)
    lru_list_.erase(it->second.second);
    lru_list_.push_front(key);
    it->second.second = lru_list_.begin();

    hits_++;
    return it->second.first;
}

// ───────────────────────────────────────────────────────────────────
bool KVCache::contains(const std::string& key) const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return cache_.count(key) > 0;
}

// ───────────────────────────────────────────────────────────────────
void KVCache::evict(size_t n)
{
    // Called with lock already held
    for (size_t i = 0; i < n && !lru_list_.empty(); ++i) {
        const std::string& lru_key = lru_list_.back();
        cache_.erase(lru_key);
        lru_list_.pop_back();
        evictions_++;
    }
}

// ───────────────────────────────────────────────────────────────────
void KVCache::clear()
{
    std::unique_lock<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
}

// ───────────────────────────────────────────────────────────────────
size_t KVCache::current_memory_bytes() const
{
    // Called with lock held
    size_t total = 0;
    for (const auto& [k, v] : cache_) {
        total += v.first.kv_data.size() * sizeof(float);
        total += k.size();
    }
    return total;
}

// ───────────────────────────────────────────────────────────────────
KVCache::Stats KVCache::get_stats() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    Stats s;
    s.hits         = hits_.load();
    s.misses       = misses_.load();
    s.evictions    = evictions_.load();
    s.current_size = cache_.size();
    uint64_t total = s.hits + s.misses;
    s.hit_rate     = total > 0 ? (double)s.hits / total : 0.0;
    return s;
}