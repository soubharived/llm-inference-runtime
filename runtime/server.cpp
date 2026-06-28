#include "inference.h"
#include "queue.h"
#include "scheduler.h"
#include "batcher.h"
#include "kvcache.h"

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#include <chrono>
#include <mutex>
#include <map>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/select.h>

// ── Global state ────────────────────────────────────────────────────
static std::atomic<bool>          g_running{true};
static RequestQueue*              g_queue    = nullptr;
static KVCache*                   g_kvcache  = nullptr;
static DynamicBatcher*            g_batcher  = nullptr;

static std::map<int, InferenceResult> g_results;
static std::mutex                     g_results_mutex;
static std::atomic<int>               g_next_id{1};

// ── Signal handler ──────────────────────────────────────────────────
void handle_signal(int) {
    g_running = false;
    if (g_queue) g_queue->shutdown();
}

// ── JSON string escaper ─────────────────────────────────────────────
// Prevents server crash on prompts with quotes, newlines, backslashes
std::string json_escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ── HTTP helpers ────────────────────────────────────────────────────
std::string http_200(const std::string& body,
                     const std::string& ct = "application/json")
{
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: " << ct << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n"
        << "\r\n" << body;
    return oss.str();
}

std::string http_err(int code, const std::string& msg)
{
    std::string body = "{\"error\":\"" + json_escape(msg) + "\"}";
    std::ostringstream oss;
    oss << "HTTP/1.1 " << code << " Error\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n"
        << "\r\n" << body;
    return oss.str();
}

// ── Minimal JSON string extractor ───────────────────────────────────
// Handles escaped characters inside JSON strings
std::string parse_json_string(const std::string& json, const std::string& key)
{
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    // Skip whitespace
    while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ')) ++pos;
    if (pos >= json.size() || json[pos] != '"') return "";

    ++pos; // skip opening quote
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            switch (json[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        ++pos;
    }
    return result;
}

// ── Handle one connection ───────────────────────────────────────────
void handle_connection(int client_fd)
{
    char buf[16384] = {};
    int n = 0, total = 0;
    // Read until we have full HTTP request including body
    while (total < (int)sizeof(buf) - 1) {
        n = recv(client_fd, buf + total, sizeof(buf) - 1 - total, 0);
        if (n <= 0) break;
        total += n;
        buf[total] = 0;
        // Check if we have complete request (headers + body)
        std::string so_far(buf, total);
        auto hend = so_far.find("\r\n\r\n");
        if (hend == std::string::npos) continue;
        // Check content-length
        auto cl_pos = so_far.find("content-length: ");
        if (cl_pos == std::string::npos)
            cl_pos = so_far.find("Content-Length: ");
        if (cl_pos != std::string::npos) {
            int cl = std::stoi(so_far.substr(cl_pos + 16));
            int body_received = total - (hend + 4);
            if (body_received >= cl) break;
        } else {
            break;
        }
    }
    if (total <= 0) { close(client_fd); return; }

    std::string request(buf, total);
    std::istringstream ss(request);
    std::string method, path;
    ss >> method >> path;

    std::string response;

    if (method == "POST" && path == "/generate") {
        auto pos  = request.find("\r\n\r\n");
        std::string body = (pos != std::string::npos)
            ? request.substr(pos + 4) : "";

        std::string prompt = parse_json_string(body, "prompt");
        if (prompt.empty()) {
            response = http_err(400, "missing prompt field");
        } else {
            InferenceRequest req;
            req.id             = g_next_id++;
            req.prompt         = prompt;
            req.max_new_tokens = 256;
            req.temperature    = 0.7f;

            if (g_queue->push(req)) {
                response = http_200(
                    "{\"request_id\":" + std::to_string(req.id) + "}");
            } else {
                response = http_err(503, "queue full");
            }
        }
    }
    else if (method == "GET" && path.rfind("/result/", 0) == 0) {
        try {
            int id = std::stoi(path.substr(8));
            std::unique_lock<std::mutex> lock(g_results_mutex);
            auto it = g_results.find(id);
            if (it == g_results.end()) {
                response = http_200("{\"status\":\"pending\"}");
            } else {
                auto& r = it->second;
                std::ostringstream j;
                j << "{\"status\":\"done\","
                  << "\"output\":\"" << json_escape(r.output) << "\","
                  << "\"tokens\":" << r.tokens_generated << ","
                  << "\"latency_ms\":" << r.latency_ms
                  << "}";
                response = http_200(j.str());
            }
        } catch (...) {
            response = http_err(400, "invalid request id");
        }
    }
    else if (method == "GET" && path == "/stats") {
        auto qs  = g_queue->get_stats();
        auto bs  = g_batcher->get_stats();
        auto kvs = g_kvcache->get_stats();

        std::ostringstream j;
        j << "{"
          << "\"queue\":{"
          <<   "\"size\":"      << g_queue->size()      << ","
          <<   "\"enqueued\":"  << qs.total_enqueued    << ","
          <<   "\"processed\":" << qs.total_processed   << ","
          <<   "\"dropped\":"   << qs.total_dropped     << "},"
          << "\"batcher\":{"
          <<   "\"batches\":"        << bs.batches_processed << ","
          <<   "\"avg_batch_size\":" << bs.avg_batch_size    << ","
          <<   "\"throughput_rps\":" << bs.throughput_rps    << ","
          <<   "\"avg_latency_ms\":" << bs.avg_latency_ms    << "},"
          << "\"kvcache\":{"
          <<   "\"hits\":"     << kvs.hits         << ","
          <<   "\"misses\":"   << kvs.misses        << ","
          <<   "\"hit_rate\":" << kvs.hit_rate      << ","
          <<   "\"size\":"     << kvs.current_size  << "}"
          << "}";
        response = http_200(j.str());
    }
    else if (method == "GET" && path == "/health") {
        response = http_200("{\"status\":\"ok\"}");
    }
    else if (method == "POST" && path == "/reset") {
        std::unique_lock<std::mutex> lock(g_results_mutex);
        g_results.clear();
        g_kvcache->clear();
        response = http_200("{\"status\":\"reset done\"}");
    }
    else {
        response = http_err(404, "not found");
    }

    send(client_fd, response.c_str(), response.size(), 0);
    close(client_fd);
}

// ── Main ────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    std::string model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    int port      = 8080;
    int n_threads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--model"   && i+1 < argc) model_path = argv[++i];
        if (a == "--port"    && i+1 < argc) port       = std::stoi(argv[++i]);
        if (a == "--threads" && i+1 < argc) n_threads  = std::stoi(argv[++i]);
    }

    std::signal(SIGINT,  handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::cout << "╔══════════════════════════════════════╗\n"
              << "║     LLM Inference Runtime v1.0       ║\n"
              << "╚══════════════════════════════════════╝\n"
              << "Model:   " << model_path << "\n"
              << "Port:    " << port       << "\n"
              << "Threads: " << n_threads  << "\n\n";

    // ── Init ────────────────────────────────────────────────────────
    InferenceConfig cfg;
    cfg.model_path = model_path;
    cfg.n_threads  = n_threads;
    cfg.n_ctx      = 2048;

    InferenceEngine engine(cfg);

    RequestQueue queue(500);
    g_queue = &queue;

    KVCache kvcache(128, 256);
    g_kvcache = &kvcache;

    DynamicBatcher::Config bcfg;
    bcfg.max_batch_size = 4;
    bcfg.max_wait_ms    = 50;

    DynamicBatcher batcher(queue, engine, bcfg);
    g_batcher = &batcher;

    batcher.set_result_callback([](InferenceResult r) {
        std::unique_lock<std::mutex> lock(g_results_mutex);
        g_results[r.request_id] = std::move(r);
    });

    batcher.start();

    // ── HTTP listener ───────────────────────────────────────────────
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[Server] bind() failed on port " << port << "\n";
        return 1;
    }
    listen(server_fd, 128);

    std::cout << "[Server] Listening on :" << port << "\n"
              << "  POST /generate  {\"prompt\":\"...\"}\n"
              << "  GET  /result/<id>\n"
              << "  GET  /stats\n"
              << "  GET  /health\n\n"
              << "  POST /reset\n\n";

    while (g_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(server_fd, &fds);
        timeval tv{1, 0};
        if (select(server_fd + 1, &fds, nullptr, nullptr, &tv) <= 0)
            continue;

        sockaddr_in client_addr{};
        socklen_t   client_len = sizeof(client_addr);
        int client_fd = accept(server_fd,
                               (sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;

        std::thread(handle_connection, client_fd).detach();
    }

    close(server_fd);
    batcher.stop();
    std::cout << "[Server] Shutdown complete.\n";
    return 0;
}