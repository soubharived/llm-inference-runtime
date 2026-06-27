"""
Load test: simulate N concurrent users and collect metrics.
Saves results as JSON + generates PNG graphs automatically.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

CPP_RUNTIME_URL = "http://localhost:8080"
API_URL         = "http://localhost:8000"
RESULTS_DIR     = Path("results")
GRAPHS_DIR      = RESULTS_DIR / "graphs"
RESULTS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(exist_ok=True)

SAMPLE_PROMPTS = [
    "What is machine learning in one sentence?",
    "What is a neural network in one sentence?",
    "What is gradient descent in one sentence?",
    "What is overfitting in one sentence?",
    "What is a transformer model in one sentence?",
    "What is backpropagation in one sentence?",
    "What is supervised learning in one sentence?",
    "What is unsupervised learning in one sentence?",
    "What is reinforcement learning in one sentence?",
    "What is deep learning in one sentence?",
]

# ── Reset runtime between runs ───────────────────────────────────────
async def reset_runtime(session: aiohttp.ClientSession):
    try:
        await session.post(
            f"{CPP_RUNTIME_URL}/reset",
            timeout=aiohttp.ClientTimeout(total=5)
        )
        print("[Reset] Runtime state cleared")
        await asyncio.sleep(1)
    except Exception as e:
        print(f"[Reset] Warning: {e}")

# ── Single user simulation ───────────────────────────────────────────
async def single_user(session: aiohttp.ClientSession,
                      user_id: int,
                      prompt: str,
                      results: list):
    t0 = time.time()
    try:
        # Step 1: Submit request directly to C++ runtime
        async with session.post(
            f"{CPP_RUNTIME_URL}/generate",
            json={"prompt": prompt},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            data     = await resp.json()
            req_id   = data.get("request_id")

        if not req_id:
            raise Exception("No request_id returned")

        # Step 2: Poll for result with correct timeout
        max_wait = 300  # 5 minutes max
        poll_interval = 1.0
        elapsed = 0

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            async with session.get(
                f"{CPP_RUNTIME_URL}/result/{req_id}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                result_data = await r.json()

            if result_data.get("status") == "done":
                latency = (time.time() - t0) * 1000
                results.append({
                    "user_id":          user_id,
                    "request_id":       req_id,
                    "success":          True,
                    "latency_ms":       latency,
                    "tokens_generated": result_data.get("tokens", 0),
                    "runtime_latency":  result_data.get("latency_ms", 0),
                    "prompt_len":       len(prompt),
                })
                print(f"  [User {user_id}] done | "
                      f"latency={latency:.0f}ms | "
                      f"tokens={result_data.get('tokens', 0)}")
                return

        # Timeout
        results.append({
            "user_id": user_id,
            "success": False,
            "latency_ms": (time.time() - t0) * 1000,
            "error": "timeout after 300s",
        })
        print(f"  [User {user_id}] TIMEOUT")

    except Exception as e:
        results.append({
            "user_id":    user_id,
            "success":    False,
            "latency_ms": (time.time() - t0) * 1000,
            "error":      str(e),
        })
        print(f"  [User {user_id}] ERROR: {e}")

# ── Run one benchmark scenario ───────────────────────────────────────
async def run_benchmark(n_users: int) -> dict:
    print(f"\n{'='*50}")
    print(f"  Benchmark: {n_users} concurrent user(s)")
    print(f"{'='*50}")

    results   = []
    connector = aiohttp.TCPConnector(limit=n_users + 10)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Reset runtime before each run
        await reset_runtime(session)

        tasks = []
        for i in range(n_users):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            tasks.append(single_user(session, i+1, prompt, results))

        t_start = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - t_start

    successful = [r for r in results if r["success"]]
    failed     = [r for r in results if not r["success"]]
    latencies  = [r["latency_ms"] for r in successful]
    tokens     = [r.get("tokens_generated", 0) for r in successful]

    summary = {
        "n_users":         n_users,
        "total_time_s":    round(total_time, 2),
        "success_count":   len(successful),
        "fail_count":      len(failed),
        "throughput_rps":  round(len(successful) / total_time, 4) if total_time > 0 else 0,
        "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "latency_p50_ms":  round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "latency_p95_ms":  round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "latency_p99_ms":  round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        "latency_min_ms":  round(min(latencies), 1) if latencies else 0,
        "latency_max_ms":  round(max(latencies), 1) if latencies else 0,
        "avg_tokens":      round(statistics.mean(tokens), 1) if tokens else 0,
        "total_tokens":    sum(tokens),
        "tokens_per_sec":  round(sum(tokens) / total_time, 2) if total_time > 0 else 0,
        "raw_results":     results,
    }

    print(f"\n  Success      : {summary['success_count']}/{n_users}")
    print(f"  Total time   : {summary['total_time_s']}s")
    print(f"  Throughput   : {summary['throughput_rps']} req/s")
    print(f"  Latency mean : {summary['latency_mean_ms']} ms")
    print(f"  Latency p95  : {summary['latency_p95_ms']} ms")
    print(f"  Tokens/sec   : {summary['tokens_per_sec']}")

    return summary

# ── Generate graphs ──────────────────────────────────────────────────
def generate_graphs(all_summaries: list, timestamp: str):
    user_counts = [s["n_users"]         for s in all_summaries]
    throughputs = [s["throughput_rps"]  for s in all_summaries]
    lat_mean    = [s["latency_mean_ms"] for s in all_summaries]
    lat_p95     = [s["latency_p95_ms"]  for s in all_summaries]
    lat_p99     = [s["latency_p99_ms"]  for s in all_summaries]
    tokens_sec  = [s["tokens_per_sec"]  for s in all_summaries]
    success     = [s["success_count"]   for s in all_summaries]
    failed      = [s["fail_count"]      for s in all_summaries]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("LLM Inference Runtime — Benchmark Results",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Throughput ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(user_counts, throughputs, "o-",
             color="#2196F3", linewidth=2, markersize=8)
    ax1.fill_between(user_counts, throughputs, alpha=0.15, color="#2196F3")
    ax1.set_title("Throughput vs Concurrent Users")
    ax1.set_xlabel("Concurrent Users")
    ax1.set_ylabel("Requests / Second")
    ax1.grid(True, alpha=0.3)

    # ── 2. Latency percentiles ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(user_counts, lat_mean, "s-", label="Mean",
             color="#4CAF50", linewidth=2)
    ax2.plot(user_counts, lat_p95,  "^-", label="P95",
             color="#FF9800", linewidth=2)
    ax2.plot(user_counts, lat_p99,  "D-", label="P99",
             color="#F44336", linewidth=2)
    ax2.set_title("Latency Percentiles vs Users")
    ax2.set_xlabel("Concurrent Users")
    ax2.set_ylabel("Latency (ms)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3. Tokens/sec ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(user_counts, tokens_sec,
                   color="#9C27B0", alpha=0.8, edgecolor="white")
    ax3.set_title("Token Generation Throughput")
    ax3.set_xlabel("Concurrent Users")
    ax3.set_ylabel("Tokens / Second")
    for bar, val in zip(bars, tokens_sec):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    # ── 4. Latency distribution (last run) ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    last = all_summaries[-1]
    lats = [r["latency_ms"] for r in last["raw_results"] if r["success"]]
    if lats:
        ax4.hist(lats, bins=max(5, len(lats)),
                 color="#00BCD4", edgecolor="white", alpha=0.8)
        if len(lats) > 1:
            ax4.axvline(np.percentile(lats, 50),
                        color="green",  linestyle="--", label="P50")
            ax4.axvline(np.percentile(lats, 95),
                        color="orange", linestyle="--", label="P95")
            ax4.legend(fontsize=8)
    ax4.set_title(f"Latency Distribution ({last['n_users']} users)")
    ax4.set_xlabel("Latency (ms)")
    ax4.set_ylabel("Count")
    ax4.grid(True, alpha=0.3)

    # ── 5. Success vs Failed ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(user_counts))
    w = 0.35
    ax5.bar(x - w/2, success, w, label="Success",
            color="#4CAF50", alpha=0.8)
    ax5.bar(x + w/2, failed,  w, label="Failed",
            color="#F44336", alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(user_counts)
    ax5.set_title("Success vs Failed Requests")
    ax5.set_xlabel("Concurrent Users")
    ax5.set_ylabel("Count")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    # ── 6. Summary table ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = [["Users", "RPS", "P50(ms)", "P95(ms)", "Tok/s"]]
    for s in all_summaries:
        table_data.append([
            str(s["n_users"]),
            str(s["throughput_rps"]),
            str(s["latency_p50_ms"]),
            str(s["latency_p95_ms"]),
            str(s["tokens_per_sec"]),
        ])
    tbl = ax6.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ax6.set_title("Summary Table", pad=20)

    out = GRAPHS_DIR / f"benchmark_{timestamp}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n[Graphs] Saved → {out}")
    return str(out)

# ── Main ─────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", nargs="+", type=int,
                        default=[1, 2, 3],
                        help="List of concurrent user counts")
    args = parser.parse_args()

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries = []

    for n in args.users:
        summary = await run_benchmark(n)
        all_summaries.append(summary)
        print("\n[Cooldown] Waiting 3 seconds before next run...")
        await asyncio.sleep(3)

    # Save JSON
    json_out = RESULTS_DIR / "benchmarks" / f"results_{timestamp}.json"
    json_out.parent.mkdir(exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[Results] Saved → {json_out}")

    # Generate graphs
    generate_graphs(all_summaries, timestamp)

if __name__ == "__main__":
    asyncio.run(main())