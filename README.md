# LLM Inference Runtime

A production-grade LLM inference serving system built from scratch in **C++17**, implementing core concepts from modern inference engines like vLLM, TensorRT-LLM, and Triton Inference Server.

![Benchmark Results](results/graphs/benchmark_20260627_172417.png)

---

## Overview

Most LLM projects wrap a Hugging Face pipeline and call it done. This project builds the **infrastructure layer** that sits between users and the model — the same layer that powers ChatGPT, Gemini, and other production AI systems at scale.

```
Multiple Users
      ↓
FastAPI Gateway (Python)
      ↓
C++ Inference Runtime
      ↓
┌─────────────────────────────────┐
│  Thread-Safe Request Queue      │
│  Multi-Policy Scheduler         │
│  Dynamic Batching Engine        │
│  LRU KV Cache Manager           │
│  HTTP Server                    │
└─────────────────────────────────┘
      ↓
llama.cpp → Qwen2.5-1.5B-Instruct
```

---

## Features

- **C++17 Runtime Core** — HTTP server, request queue, scheduler, batcher, KV cache all in C++17
- **Thread-Safe Request Queue** — Producer-consumer pattern with mutex and condition variables
- **Dynamic Batching** — Groups concurrent requests to maximize throughput
- **LRU KV Cache** — Memory-efficient caching with configurable eviction policy
- **Multi-Policy Scheduler** — FCFS, Priority, and Shortest-Job-First scheduling
- **FastAPI Gateway** — Clean REST API layer on top of C++ runtime
- **Live Dashboard** — Real-time monitoring of queue, throughput, latency, cache hit rate
- **Benchmark Suite** — Automated load testing with PNG graph generation

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Client Layer                      │
│         curl / Python client / Browser               │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP :8000
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Gateway                     │
│              api/main.py (:8000)                     │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP :8080
┌──────────────────────▼──────────────────────────────┐
│               C++ Inference Runtime                  │
│                                                      │
│  ┌─────────────┐    ┌─────────────┐                 │
│  │ HTTP Server │    │  KV Cache   │                 │
│  │ server.cpp  │    │ kvcache.cpp │                 │
│  └──────┬──────┘    └─────────────┘                 │
│         │                                            │
│  ┌──────▼──────┐    ┌─────────────┐                 │
│  │   Request   │    │  Scheduler  │                 │
│  │    Queue    │───▶│scheduler.cpp│                 │
│  │  queue.cpp  │    └──────┬──────┘                 │
│  └─────────────┘           │                        │
│                    ┌───────▼──────┐                 │
│                    │   Dynamic    │                 │
│                    │   Batcher    │                 │
│                    │ batcher.cpp  │                 │
│                    └───────┬──────┘                 │
│                    ┌───────▼──────┐                 │
│                    │  Inference   │                 │
│                    │   Engine     │                 │
│                    │inference.cpp │                 │
│                    └───────┬──────┘                 │
└────────────────────────────┼────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   llama.cpp     │
                    │ Qwen2.5-1.5B   │
                    └─────────────────┘
```

---

## OS Concepts Implemented

| Concept | Implementation |
|---------|---------------|
| Threads | Scheduler workers, batcher thread, per-connection threads |
| Mutex | Thread-safe queue and KV cache access |
| Condition Variables | Queue blocking and wakeup |
| Producer-Consumer | Request queue + batcher pattern |
| Scheduling Algorithms | FCFS, Priority, Shortest-Job-First |
| Memory Management | LRU eviction in KV cache |
| Sockets | Raw HTTP server using POSIX sockets |
| Process Management | 4 independent processes running together |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core Runtime | C++17 |
| Model Backend | llama.cpp |
| Model | Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) |
| API Layer | Python + FastAPI |
| Monitoring | Streamlit + Plotly |
| Build System | CMake |
| Benchmarking | Python + aiohttp + Matplotlib |

---

## Project Structure

```
llm-runtime/
│
├── runtime/              # C++17 core
│   ├── server.cpp        # HTTP server (POSIX sockets)
│   ├── queue.cpp/h       # Thread-safe request queue
│   ├── scheduler.cpp/h   # Multi-policy scheduler
│   ├── batcher.cpp/h     # Dynamic batching engine
│   ├── kvcache.cpp/h     # LRU KV cache manager
│   └── inference.cpp/h   # llama.cpp wrapper
│
├── api/
│   ├── main.py           # FastAPI gateway
│   └── client.py         # Test client
│
├── benchmark/
│   ├── load_test.py      # Concurrent load testing
│   └── metrics.py        # Live metrics collection
│
├── dashboard/
│   └── app.py            # Streamlit monitoring UI
│
├── results/
│   ├── graphs/           # Auto-generated PNG graphs
│   └── benchmarks/       # Raw JSON results
│
├── CMakeLists.txt
├── requirements.txt
└── setup.sh              # One-command setup
```

---

## Setup

### Prerequisites
- Ubuntu 22.04 (or WSL2)
- GCC 11+
- CMake 3.16+
- Python 3.10+
- 8GB RAM minimum

### One Command Setup

```bash
git clone https://github.com/soubharived/llm-inference-runtime.git
cd llm-inference-runtime
chmod +x setup.sh
./setup.sh
```

This automatically:
- Installs system dependencies
- Clones and builds llama.cpp
- Sets up Python virtual environment
- Downloads Qwen2.5-1.5B-Instruct model
- Builds the C++ runtime

---

## Running

Open 4 terminals:

```bash
# Terminal 1 — C++ Runtime
source venv/bin/activate
./build/llm_runtime

# Terminal 2 — FastAPI Gateway
source venv/bin/activate
python api/main.py

# Terminal 3 — Live Dashboard
source venv/bin/activate
streamlit run dashboard/app.py

# Terminal 4 — Benchmark
source venv/bin/activate
python benchmark/load_test.py --users 1 2 3
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Submit inference request |
| `/result/<id>` | GET | Poll for result |
| `/stats` | GET | Runtime statistics |
| `/health` | GET | Health check |
| `/reset` | POST | Reset runtime state |

### Example

```bash
# Submit request
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'

# Get result
curl http://localhost:8080/result/1
```

---

## Benchmark Results (CPU Baseline)

Tested on: AMD Ryzen 3, 8GB RAM, CPU-only inference

| Users | RPS | P50 Latency | P95 Latency | Tokens/sec |
|-------|-----|-------------|-------------|------------|
| 1 | 0.007 | 136756ms | 136756ms | 0.18 |
| 2 | 0.041 | 42220ms | 47437ms | 6.27 |
| 3 | 0.043 | 69542ms | 69606ms | 5.27 |

**Key finding:** Dynamic batching improved throughput by **5.7x** (0.007 → 0.041 req/s) when serving 2 concurrent users vs sequential processing.

---

## Dashboard

Live monitoring dashboard available at `http://localhost:8501`

Shows real-time:
- Request queue length
- Throughput (req/s)
- Average latency (ms)
- KV Cache hit rate
- Total requests processed

---

## Roadmap

- [x] CPU baseline implementation
- [x] Thread-safe request queue
- [x] Dynamic batching
- [x] LRU KV cache
- [x] Multi-policy scheduler
- [x] Live monitoring dashboard
- [x] Benchmark suite with graphs
- [ ] GPU support (CUDA) — in progress
- [ ] Adaptive dynamic batching
- [ ] Token streaming
- [ ] Baseline vs batching comparison
- [ ] 10-20 concurrent user benchmarks on GPU

---

## Comparison With Production Systems

| Feature | This Project | vLLM | TensorRT-LLM |
|---------|-------------|------|--------------|
| Request Queue | ✅ | ✅ | ✅ |
| Dynamic Batching | ✅ | ✅ | ✅ |
| KV Cache | ✅ | ✅ (Paged) | ✅ |
| Scheduler | ✅ | ✅ | ✅ |
| GPU Support | 🔄 Soon | ✅ | ✅ |
| Streaming | 🔄 Soon | ✅ | ✅ |

---

## Author

**Soubhari Ved**
M.Tech — Artificial Intelligence
IIT Patna

---

## License

MIT License
