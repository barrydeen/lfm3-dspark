"""Decode benchmark for LFM3-1.7B-Uncensored.

Two modes:
  greedy : temperature 0, top_k 1 (deterministic; identical text across modes)
  sample : the model's own generation_config defaults (temp 0.1, top_k 50,
           repetition_penalty 1.1) — the conditions the DSpark model card used

Token counts come from the usage field, not stream-chunk counts.
Usage: python bench.py [port] [greedy|sample]
"""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8110"
MODE = sys.argv[2] if len(sys.argv) > 2 else "greedy"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPT = (
    "Explain how a convolutional neural network learns features from images, "
    "step by step, including the role of pooling, fully connected layers and "
    "backpropagation. Then write a Python function that performs manual 2D "
    "convolution with a 3x3 kernel on a grayscale image, with padding. Finally, "
    "discuss the trade-offs of using larger kernels versus stacking smaller ones."
)

SAMPLES = {
    "greedy": {"temperature": 0, "top_k": 1},
    "sample": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1},
}


def run() -> None:
    body = json.dumps(
        {
            "model": "lfm3-1.7b-uncensored",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 512,
            "stream": True,
            "stream_options": {"include_usage": True},
            **SAMPLES[MODE],
        }
    ).encode()
    for i in range(3):
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        first = None
        usage = None
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if first is None and (chunk.get("choices") or chunk.get("usage")):
                    first = time.time()
                if chunk.get("usage"):
                    usage = chunk["usage"]
        dt = time.time() - t0
        n = usage["completion_tokens"] if usage else -1
        print(f"  req {i + 1}: out={n:4d}  ttfb={first - t0:6.2f}s  total={dt:6.2f}s  "
              f"decode={n / max(dt - (first - t0), 1e-6):7.1f} tok/s")


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"TOTAL {time.time() - t0:.1f}s across 3 x 512-token requests")
