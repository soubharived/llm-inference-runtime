#include "inference.h"
#include "llama.h"

#include <iostream>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <string>

// ───────────────────────────────────────────────────────────────────
InferenceEngine::InferenceEngine(const InferenceConfig& config)
    : config_(config)
{
    load_model();
}

// ───────────────────────────────────────────────────────────────────
InferenceEngine::~InferenceEngine()
{
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

// ───────────────────────────────────────────────────────────────────
void InferenceEngine::load_model()
{
    llama_backend_init();

    // ── Load model ──────────────────────────────────────────────────
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;   // CPU only

    // Correct API name (verified from llama.cpp source 2025)
    
    model_ = llama_model_load_from_file(config_.model_path.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error("[InferenceEngine] Failed to load model: "
                                  + config_.model_path);
    }

    // ── Create context ──────────────────────────────────────────────
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = config_.n_ctx;
    cparams.n_batch     = config_.n_batch;
    cparams.n_threads   = config_.n_threads;
    cparams.no_perf     = false;


    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("[InferenceEngine] Failed to create context");
    }

    ready_ = true;
    std::cout << "[InferenceEngine] Loaded: " << config_.model_path << "\n";
}

// ───────────────────────────────────────────────────────────────────
InferenceResult InferenceEngine::run(const InferenceRequest& req,
                                      TokenCallback on_token)
{
    InferenceResult result;
    result.request_id = req.id;

    if (!ready_) {
        result.success       = false;
        result.error_message = "Engine not ready";
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    try {
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        // Safe tokenization — allocate generously then resize
        std::vector<llama_token> tokens(req.prompt.size() + 64);
        int n_tokens = llama_tokenize(
            vocab,
            req.prompt.c_str(),
            (int)req.prompt.size(),
            tokens.data(),
            (int)tokens.size(),
            true,   // add BOS
            false
        );
        if (n_tokens < 0) {
            result.success       = false;
            result.error_message = "Tokenization failed";
            return result;
        }
        tokens.resize(n_tokens);

        // ── Clear KV cache and reset context ────────────────────────
        llama_memory_clear(llama_get_memory(ctx_), true);

        // ── Decode prompt ───────────────────────────────────────────
        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        if (llama_decode(ctx_, batch) != 0) {
            result.success       = false;
            result.error_message = "Prompt decode failed";
            return result;
        }

        // ── Setup sampler chain ─────────────────────────────────────
        llama_sampler* sampler = llama_sampler_chain_init(
            llama_sampler_chain_default_params()
        );
        llama_sampler_chain_add(sampler,
            llama_sampler_init_temp(req.temperature));
        llama_sampler_chain_add(sampler,
            llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler,
            llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // ── Generate tokens ─────────────────────────────────────────
        std::string output;
        int n_generated = 0;

        while (n_generated < req.max_new_tokens) {
            llama_token token_id = llama_sampler_sample(sampler, ctx_, -1);

            // End of generation
            if (llama_vocab_is_eog(vocab, token_id)) break;

            // Convert token to string piece
            char   buf[256];
            int    len = llama_token_to_piece(
                vocab, token_id, buf, sizeof(buf), 0, false);

            if (len > 0) {
                std::string piece(buf, len);
                output += piece;
                if (on_token) on_token(piece);
            }

            // Decode next token — pass current position
            llama_batch next = llama_batch_get_one(&token_id, 1);
            if (llama_decode(ctx_, next) != 0) break;

            ++n_generated;
        }

        llama_sampler_free(sampler);

        result.output           = output;
        result.tokens_generated = n_generated;
        result.success          = true;

    } catch (const std::exception& e) {
        result.success       = false;
        result.error_message = e.what();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    std::cout << "[Inference] req=" << req.id
              << " tokens=" << result.tokens_generated
              << " latency=" << result.latency_ms << "ms\n";

    return result;
}

// ───────────────────────────────────────────────────────────────────
std::vector<InferenceResult>
InferenceEngine::run_batch(const std::vector<InferenceRequest>& batch)
{
    std::vector<InferenceResult> results;
    results.reserve(batch.size());
    for (const auto& req : batch) {
        results.push_back(run(req));
    }
    return results;
}