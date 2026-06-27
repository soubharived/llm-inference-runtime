"""
FastAPI gateway — sits in front of the C++ runtime.
Handles HTTP from clients and forwards to C++ server.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
import time
import json
from pathlib import Path

app = FastAPI(title="LLM Inference Runtime API", version="1.0.0")

CPP_RUNTIME_URL = "http://localhost:8080"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Request / Response models ────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    request_id: int
    output: str
    tokens_generated: int
    latency_ms: float

# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{CPP_RUNTIME_URL}/health", timeout=2.0)
        return {"api": "ok", "runtime": r.json()}
    except Exception:
        return {"api": "ok", "runtime": "unreachable"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    async with httpx.AsyncClient() as client:
        # Submit to C++ runtime
        submit = await client.post(
            f"{CPP_RUNTIME_URL}/generate",
            json={"prompt": req.prompt},
            timeout=10.0
        )
        if submit.status_code != 200:
            raise HTTPException(status_code=502, detail="Runtime error")

        request_id = submit.json()["request_id"]

        # Poll for result (max 120 seconds)
        for _ in range(240):
            await asyncio.sleep(0.5)
            result_r = await client.get(
                f"{CPP_RUNTIME_URL}/result/{request_id}",
                timeout=5.0
            )
            data = result_r.json()
            if data.get("status") == "done":
                return GenerateResponse(
                    request_id=request_id,
                    output=data["output"],
                    tokens_generated=data.get("tokens", 0),
                    latency_ms=data.get("latency_ms", 0.0),
                )

        raise HTTPException(status_code=504, detail="Timeout waiting for result")


@app.get("/stats")
async def stats():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{CPP_RUNTIME_URL}/stats", timeout=5.0)
        return r.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)