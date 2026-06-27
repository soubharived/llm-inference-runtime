"""
Quick test client — send a single prompt and print the result.
Usage: python api/client.py "Tell me about transformers"
"""

import sys
import requests
import json
import time

API_URL = "http://localhost:8000"

def generate(prompt: str) -> dict:
    t0 = time.time()
    r = requests.post(f"{API_URL}/generate", json={"prompt": prompt}, timeout=120)
    r.raise_for_status()
    elapsed = (time.time() - t0) * 1000
    result = r.json()
    result["client_latency_ms"] = elapsed
    return result

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "What is machine learning?"
    print(f"Prompt: {prompt}\n")
    result = generate(prompt)
    print(f"Output:\n{result['output']}\n")
    print(f"Tokens generated : {result['tokens_generated']}")
    print(f"Runtime latency  : {result['latency_ms']:.1f} ms")
    print(f"Client latency   : {result['client_latency_ms']:.1f} ms")