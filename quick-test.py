"""Quick smoke test: a few requests through the OpenAI-compatible endpoint.

Usage: python quick-test.py [port]
"""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8110"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that computes the nth Fibonacci number.",
    "Solve: a number is 3 times the sum of its digits. Find all such 2-digit numbers.",
]


def main() -> None:
    for prompt in PROMPTS:
        body = json.dumps(
            {
                "model": "lfm3-1.7b-uncensored",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.1,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "stream": True,
            }
        ).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        first_tok = None
        n_out = 0
        text_parts: list[str] = []
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if first_tok is None:
                    first_tok = time.time()
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    n_out += 1
                    text_parts.append(piece)
        dt = time.time() - t0
        ttf = (first_tok - t0) if first_tok is None else (first_tok - t0)
        print(f"--- {prompt!r}")
        print(f"    out={n_out} tokens  ttfb={ttf:.2f}s  total={dt:.2f}s  "
              f"decoding={n_out / max(dt - ttf, 1e-6):.1f} tok/s")
        print("    " + "".join(text_parts)[:220].replace("\n", " \n    ") + ("..." if len("".join(text_parts)) > 220 else ""))
    print("OK")


if __name__ == "__main__":
    main()
