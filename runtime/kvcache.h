#pragma once

#include <unordered_map>
#include <list>
#include <vector>
#include <mutex>
#include <string>
#include <optional>
#include <atomic>
#include <cstdint>

// ── Cached KV entry ─────────────────────────────────────────────────
struct KVEntry {
    std::string              prefix_key;   // hash of prompt prefix
    std::vector<float>       kv_data;      // flattened K+V tensors
    size_t                   token_count;
    int64_t                  last_access;  // unix ms
};

// ── LRU KV Cache ────────────────────────────────────────────────────
class KVCache {
public:
    explicit KVCache(size_t max_entries = 128,
                     size_t max_memory_mb = 512);

    // Store KV data for a given prompt prefix
    void put(const std::string& key,
             const std::vector<float>& kv_data,
             size_t token_count);

    // Retrieve if exists
    std::optional<KVEntry> get(const std::string& key);

    // Check hit/miss
    bool contains(const std::string& key) const;

    // Evict least recently used entries
    void evict(size_t n = 1);

    // Clear all
    void clear();

    // Hash a prompt to a cache key
    static std::string make_key(const std::string& prompt,
                                 size_t prefix_len = 0);

    struct Stats {
        uint64_t hits         = 0;
        uint64_t misses       = 0;
        uint64_t evictions    = 0;
        size_t   current_size = 0;
        double   hit_rate     = 0.0;
    };
    Stats get_stats() const;

private:
    size_t max_entries_;
    size_t max_memory_mb_;

    // LRU: list front = most recent, back = least recent
    std::list<std::string>                           lru_list_;
    std::unordered_map<std::string,
        std::pair<KVEntry,
                  std::list<std::string>::iterator>> cache_;

    mutable std::mutex mutex_;

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};

    size_t current_memory_bytes() const;
};