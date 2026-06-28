#!/bin/bash
set -e
export PATH="/home/ved_2511ai58/tools/cmake-4.1.1-linux-x86_64/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"
export PATH="/home/ved_2511ai58/tools/cmake-4.1.1-linux-x86_64/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "  LLM Inference Runtime - Setup Script"
echo "=========================================="

# ── 1. System packages ──────────────────────────────────────────────
echo "[1/7] Installing system packages..."
echo "[Skip] No sudo - packages already installed"
##sudo apt-get install -y \
##    build-essential \
##    cmake \
##    git \
##    curl \
##    wget \
##    python3 \
##    python3-pip \
##    python3-venv \
##    pkg-config \
##    htop \
##    screen \
##    ninja-build

# ── 2. Clone and build llama.cpp ────────────────────────────────────
echo "[2/7] Cloning llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
git pull

# ARM Oracle Cloud compatible build (no GPU, no march=native)
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DGGML_CUDA=ON \
    -G Ninja

cmake --build build --config Release -j$(nproc)
cd ..

# ── 3. Python virtual environment ───────────────────────────────────
echo "[3/7] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# ── 4. Python dependencies ──────────────────────────────────────────
echo "[4/7] Installing Python packages..."
pip install --upgrade pip
pip install \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    httpx==0.27.0 \
    aiohttp==3.9.5 \
    pydantic==2.7.1 \
    requests==2.31.0 \
    streamlit==1.35.0 \
    plotly==5.22.0 \
    pandas==2.2.2 \
    "numpy<2.0" \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    psutil==5.9.8

# ── 5. Download model via huggingface-cli ───────────────────────────
echo "[5/7] Downloading Qwen2.5 1.5B GGUF model..."
pip install huggingface_hub
mkdir -p models

python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    local_dir="models"
)
print(f"Model downloaded to: {path}")
PYEOF

# ── 6. Build project ────────────────────────────────────────────────
echo "[6/7] Building LLM Runtime..."
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=ON
cmake --build . -j$(nproc)
cd ..

# ── 7. Create output directories ────────────────────────────────────
echo "[7/7] Creating results directories..."
mkdir -p results/graphs
mkdir -p results/logs
mkdir -p results/benchmarks

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To run:"
echo "  source venv/bin/activate"
echo "  Terminal 1: ./build/llm_runtime"
echo "  Terminal 2: python api/main.py"
echo "  Terminal 3: streamlit run dashboard/app.py"
echo "  Terminal 4: python benchmark/load_test.py --users 1 5 10 20"
