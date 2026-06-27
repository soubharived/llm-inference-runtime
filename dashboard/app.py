"""
Streamlit live monitoring dashboard.
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="LLM Inference Runtime",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 LLM Inference Runtime — Dashboard")

# ── Session state ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    refresh_rate = st.slider("Refresh (s)", 1, 10, 3)
    max_history  = st.slider("History points", 30, 200, 60)
    auto_refresh = st.toggle("Auto refresh", value=True)

    st.divider()
    st.header("Test Prompt")
    test_prompt = st.text_area("Prompt", "Explain transformers briefly.")
    if st.button("Send", use_container_width=True):
        try:
            r = requests.post(f"{API_URL}/generate",
                              json={"prompt": test_prompt}, timeout=120)
            data = r.json()
            st.success(data.get("output", "")[:300])
        except Exception as e:
            st.error(str(e))

# ── Fetch stats ──────────────────────────────────────────────────────
def fetch_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=2)
        return r.json()
    except Exception:
        return None

stats = fetch_stats()

if stats is None:
    st.error("⚠️ Cannot connect to runtime. Start the C++ server first.")
else:
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({
        "time":       now,
        "queue_size": stats["queue"]["size"],
        "throughput": stats["batcher"]["throughput_rps"],
        "latency":    stats["batcher"]["avg_latency_ms"],
        "hit_rate":   stats["kvcache"]["hit_rate"] * 100,
        "cache_size": stats["kvcache"]["size"],
        "processed":  stats["queue"]["processed"],
    })
    if len(st.session_state.history) > max_history:
        st.session_state.history = st.session_state.history[-max_history:]

    df = pd.DataFrame(st.session_state.history)

    # ── KPIs ─────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Queue",        stats["queue"]["size"])
    k2.metric("Throughput",   f"{stats['batcher']['throughput_rps']:.2f} rps")
    k3.metric("Avg Latency",  f"{stats['batcher']['avg_latency_ms']:.0f} ms")
    k4.metric("Cache Hit%",   f"{stats['kvcache']['hit_rate']*100:.1f}%")
    k5.metric("Processed",    stats["queue"]["processed"])

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    def line_chart(col, y_col, title, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[y_col],
            mode="lines+markers",
            line=dict(color=color, width=2)
        ))
        fig.update_layout(title=title, height=260,
                          margin=dict(l=40, r=20, t=40, b=30))
        col.plotly_chart(fig, use_container_width=True)

    line_chart(c1, "throughput", "Throughput (req/s)", "#2196F3")
    line_chart(c2, "latency",    "Avg Latency (ms)",   "#FF9800")

    c3, c4 = st.columns(2)
    line_chart(c3, "queue_size", "Queue Length",        "#9C27B0")
    line_chart(c4, "hit_rate",   "KV Cache Hit Rate %", "#4CAF50")

    st.caption(f"Last updated: {now}")

# ── Auto refresh using Streamlit's rerun ─────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()