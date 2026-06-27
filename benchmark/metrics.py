"""
Fetch live stats from the runtime and log them over time.
Generates time-series graphs of queue, cache, and throughput.
"""

import requests
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import argparse

API_URL    = "http://localhost:8000"
GRAPHS_DIR = Path("results/graphs")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

def collect_metrics(duration_s: int = 60, interval_s: float = 1.0) -> list:
    records = []
    print(f"Collecting metrics for {duration_s}s (every {interval_s}s)...")
    t_end = time.time() + duration_s

    while time.time() < t_end:
        try:
            r = requests.get(f"{API_URL}/stats", timeout=2)
            data = r.json()
            data["timestamp"] = time.time()
            records.append(data)
        except Exception:
            pass
        time.sleep(interval_s)

    return records

def plot_metrics(records: list, timestamp: str):
    if not records:
        print("No records to plot.")
        return

    times        = [r["timestamp"] - records[0]["timestamp"] for r in records]
    queue_sizes  = [r["queue"]["size"]              for r in records]
    processed    = [r["queue"]["processed"]         for r in records]
    throughputs  = [r["batcher"]["throughput_rps"]  for r in records]
    latencies    = [r["batcher"]["avg_latency_ms"]  for r in records]
    hit_rates    = [r["kvcache"]["hit_rate"] * 100  for r in records]
    cache_sizes  = [r["kvcache"]["size"]            for r in records]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("LLM Runtime — Live Metrics", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    def plot_line(ax, x, y, title, ylabel, color):
        ax.plot(x, y, color=color, linewidth=1.5)
        ax.fill_between(x, y, alpha=0.1, color=color)
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plot_line(fig.add_subplot(gs[0,0]), times, queue_sizes,
              "Queue Length over Time",   "Requests in Queue", "#2196F3")
    plot_line(fig.add_subplot(gs[0,1]), times, throughputs,
              "Throughput over Time",     "Requests / Second", "#4CAF50")
    plot_line(fig.add_subplot(gs[0,2]), times, latencies,
              "Avg Latency over Time",    "Latency (ms)",      "#FF9800")
    plot_line(fig.add_subplot(gs[1,0]), times, hit_rates,
              "KV Cache Hit Rate",        "Hit Rate (%)",      "#9C27B0")
    plot_line(fig.add_subplot(gs[1,1]), times, cache_sizes,
              "KV Cache Size",            "Cached Entries",    "#00BCD4")
    plot_line(fig.add_subplot(gs[1,2]), times, processed,
              "Total Processed",          "Requests",          "#F44336")

    out = GRAPHS_DIR / f"metrics_{timestamp}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Metrics] Graph saved → {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    records = collect_metrics(args.duration, args.interval)
    plot_metrics(records, ts)