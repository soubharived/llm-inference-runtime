#pragma once

#include <string>
#include <vector>
#include <functional>

// Forward declare to avoid exposing llama.h in headers
struct llama_model;
struct llama_context;

// ── Config ──────────────────────────────────────────────────────────
struct InferenceConfig {
    std::string model_path;
    int n_ctx         = 2048;
    int n_batch       = 512;
    int n_threads     = 4;
    int max_new_tokens = 256;
    float temperature = 0.7f;
    float min_p       = 0.05f;
};

// ── Request ─────────────────────────────────────────────────────────
struct InferenceRequest {
    int         id;
    std::string prompt;
    int         max_new_tokens = 256;
    float       temperature    = 0.7f;
};

// ── Result ──────────────────────────────────────────────────────────
struct InferenceResult {
    int         request_id       = 0;
    std::string output;
    int         tokens_generated = 0;
    double      latency_ms       = 0.0;
    bool        success          = true;
    std::string error_message;
};

using TokenCallback = std::function<void(const std::string&)>;

// ── Engine ──────────────────────────────────────────────────────────
class InferenceEngine {
public:
    explicit InferenceEngine(const InferenceConfig& config);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    InferenceResult run(const InferenceRequest& req,
                        TokenCallback on_token = nullptr);

    std::vector<InferenceResult> run_batch(
        const std::vector<InferenceRequest>& batch);

    bool is_ready() const { return ready_; }

private:
    InferenceConfig config_;
    llama_model*    model_ = nullptr;
    llama_context*  ctx_   = nullptr;
    bool            ready_ = false;

    void load_model();
};