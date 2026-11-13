# LFM3-1.7B-Uncensored on vLLM with DSpark

A turnkey recipe for running
[OpenIntelligenceNet/LFM3-1.7B-Uncensored](https://huggingface.co/OpenIntelligenceNet/LFM3-1.7B-Uncensored)
with lossless [DSpark](https://huggingface.co/rasyosef/LFM3-1.7B-DSpark)
speculative decoding on a single NVIDIA GPU, behind an OpenAI-compatible API.

```
baseline (no drafter)  112 tok/s
+ DSpark (greedy)      175 tok/s   1.56x
+ DSpark (model's default sampling)  ~195-210 tok/s   ~1.8x
```

Measured on an RTX A4500 Laptop (16 GB, sm_86); see
[Benchmarks](#benchmarks). Larger GPUs should scale better.

## What you get

- `POST /v1/chat/completions`, `/v1/completions`, `/v1/models`, `/metrics`
  (OpenAI-compatible) on any interface you bind to
- Model served as `lfm3-1.7b-uncensored`, bf16, 262k-token context
- Lossless speculative decoding: drafted output is token-identical to plain
  decoding (the verifier only accepts what the target model would produce)

## Requirements

- **Linux** with an **NVIDIA GPU**, compute capability 7.5+ (Turing or newer).
  Tested: RTX A4500 Laptop, 16 GB VRAM, driver 610 (CUDA 13.3).
- **VRAM**: ~4.3 GB for weights + draft, plus KV cache. 8 GB works with
  default settings (smaller context window); 16 GB is comfortable.
- **Disk**: ~4.5 GB for both model repos (downloaded once into `./models`).
- **Python 3.10–3.12** (tested on 3.12.14). vLLM's prebuilt wheels do not
  cover 3.13+ in every configuration, so stick to 3.12.
- No CUDA toolkit needed: the install reuses vLLM's precompiled kernels.

## Quick start

```sh
git clone https://github.com/barrydeen/lfm3-dspark
cd lfm3-dspark

python3 -m venv venv
VLLM_USE_PRECOMPILED=1 ./venv/bin/pip install -r requirements.txt

./serve.sh                      # downloads models on first run, then serves
```

First-token within seconds of a request; the full load is ~1–2 minutes
(weight load + CUDA graph capture). Then:

```sh
curl http://127.0.0.1:8110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm3-1.7b-uncensored",
       "messages":[{"role":"user","content":"Say hello."}],
       "max_tokens":128,"temperature":0.1,"top_k":50}'
```

`uv` works too:

```sh
uv venv --python 3.12 venv
VLLM_USE_PRECOMPILED=1 uv pip install --python venv/bin/python -r requirements.txt
```

### Verify speculative decoding is active

```sh
# send a request, then:
curl -s 127.0.0.1:8110/metrics | grep spec_decode_num
# num_drafts / num_accepted_tokens grow with traffic
```

`bench.py` reports decode throughput; `quick-test.py` is a small smoke test.

## Options

`serve.sh` is driven by environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8110` | listen port |
| `HOST` | `0.0.0.0` | bind address (`127.0.0.1` for local-only) |
| `NSPEC` | `8` | speculative tokens per step (draft's block size) |
| `TARGET` | `OpenIntelligenceNet/LFM3-1.7B-Uncensored` | target model repo id or local dir |
| `DRAFT` | `rasyosef/LFM3-1.7B-DSpark` | DSpark draft repo id or local dir |
| `PY` | `./venv/bin/python` | python with vLLM installed |
| `$1` | `dspark` | `baseline` to serve without the drafter |

Examples:

```sh
./serve.sh baseline                          # A/B without the drafter
HOST=127.0.0.1 ./serve.sh                    # localhost only
TARGET=LiquidAI/LFM3-1.7B ./serve.sh         # the base instruct model
```

There is no API key by default — if you bind to a shared network, add
`--api-key <secret>` to the `vllm serve` invocation in `serve.sh`.

## Benchmarks

`bench.py` (3 x 512-token requests, fixed prompt, `stream_options` usage
counts — stream chunks are *not* tokens):

| Mode | RTX A4500 Laptop decode |
|---|---|
| baseline | 112 tok/s |
| DSpark, greedy (`temperature 0`) | ~175 tok/s |
| DSpark, sample (temp 0.1, top_k 50 — model defaults) | ~195–210 tok/s |

On an A100 40 GB the
[DSpark model card](https://huggingface.co/rasyosef/LFM3-1.7B-DSpark)
reports 220.9 → 636.6 tok/s (2.88x) with the same method; acceptance there is
2.83–4.1 tokens/step depending on the workload.

Expect **lower** speedups than the card under pure greedy: the drafter was
trained with sampled targets, and at `temperature 0` its argmax rarely matches
the target's argmax (acceptance collapses to ~1/step). Use the model's own
defaults (`temperature 0.1, top_k 50, repetition_penalty 1.1`) to get the
published behavior.

## How it works

- **vLLM PR [#66323](https://github.com/vllm-project/vllm/pull/66323)**
  (Python-only, hence `VLLM_USE_PRECOMPILED=1`) teaches
  `Lfm3ForCausalLM` to expose the auxiliary hidden states DSpark needs
  (layers 3/11/19/23). Without it the engine fails at load with
  `Model does not support EAGLE3 interface`.
- **The draft** (`rasyosef/LFM3-1.7B-DSpark`) is a 0.3 B, 4-layer parallel
  drafter trained with
  [speculators](https://github.com/vllm-project/speculators): one forward
  pass proposes all 8 block tokens, and the 1.7 B target verifies them in a
  single batched forward — on a memory-bound decode that costs roughly one
  extra step per ~4 accepted tokens.
- **The uncensored target** is an abliterated derivative of
  `LiquidAI/LFM3-1.7B`; its `config.json` is byte-identical, so vLLM's native
  LFM3 implementation loads it directly (no `trust_remote_code`). The draft
  was trained against the base instruct model, so acceptance on the
  abliterated target is slightly lower than the card's numbers, and outputs
  remain lossless regardless.
- CUDA graphs are fine for *serving* a trained drafter. The PR's known
  NaN-in-hidden-states issue only affects the `extract_hidden_states`
  training mode (`--enforce-eager` there).

## Troubleshooting

- **Port in use** — pick another: `PORT=8123 ./serve.sh`.
- **`pip` starts compiling vLLM** — you forgot `VLLM_USE_PRECOMPILED=1`.
  It must be in the environment *during install*.
- **Install fails on Python 3.13/3.14** — use 3.12:
  `python3.12 -m venv venv` (or `uv venv --python 3.12 venv`).
- **`Model does not support EAGLE3 interface`** — you installed stock vLLM
  instead of the PR head.
- **Not enough VRAM** — raise `--gpu-memory-utilization` lower or set
  `--max-model-len` smaller in `serve.sh`; quantized targets (e.g.
  `LiquidAI/LFM3-1.7B-NVFP4`) also work.
- **Slow model download** — set `HF_ENDPOINT` for mirrors, or pre-download
  with `hf download <repo> --local-dir models/<name>`.

## Alternatives

- **llama.cpp (GGUF)** — the DSpark card ships a drafter GGUF for
  `--spec-type draft-dspark`; pair
  [`rasyosef/LFM3-1.7B-DSpark-GGUF`](https://huggingface.co/rasyosef/LFM3-1.7B-DSpark-GGUF)
  with any LFM3-1.7B GGUF if you prefer llama.cpp (also the only route to
  sub-8 GB VRAM / CPU hybrids).
- **TensorFold** — as of Nov 2026 its LFM3 family is Metal/Intel-XPU only
  (CUDA is open PRs #1508/#1634, NVFP4/FP8/MLX checkpoints), and its
  DSpark support is the DeepSeek-V4.1/MiniMax-M3.1 drafter head, not the
  speculators draft format. vLLM + PR #66323 remains the reference route.

## Attribution & licenses

- Target model:
  [OpenIntelligenceNet/LFM3-1.7B-Uncensored](https://huggingface.co/OpenIntelligenceNet/LFM3-1.7B-Uncensored)
  (abliterated with
  [Heretic](https://github.com/p-e-w/heretic)) —
  [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B).
- Draft model:
  [rasyosef/LFM3-1.7B-DSpark](https://huggingface.co/rasyosef/LFM3-1.7B-DSpark)
  — Apache-2.0.
- Recipes, scripts in this repo — Apache-2.0.

## Layout

```
serve.sh         launcher (downloads models once, then serves)
bench.py         decode-throughput benchmark (greedy | sample)
quick-test.py    three-prompt smoke test
requirements.txt vLLM from PR #66323 + huggingface_hub
models/          (created) cached model repos, git-ignored
venv/            (created) python env, git-ignored
```
