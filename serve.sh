#!/usr/bin/env bash
# Serve OpenIntelligenceNet/LFM3-1.7B-Uncensored on vLLM with the
# rasyosef/LFM3-1.7B-DSpark draft model (speculative decoding, block size 8).
#
# vLLM must be installed from PR #66323 into ./venv (see README.md) — that
# PR is what adds DSpark/EAGLE-3 aux hidden state support for LFM3.
#
# Usage:
#   ./serve.sh             # DSpark speculative decoding (default)
#   ./serve.sh baseline    # no drafter, for A/B comparison
#
# Overrides (environment):
#   PORT, HOST, NSPEC       port (8110), bind address (0.0.0.0), spec tokens (8)
#   TARGET, DRAFT           repo id or local dir for target / draft model
#   PY                      python with vLLM installed (./venv/bin/python)
#   TOOL_PARSER             tool-call parser (qwen3_coder; empty disables)
#   REASONING_PARSER        reasoning parser (qwen3; empty disables)
#   THINKING                on|off, template default (on)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8110}"
HOST="${HOST:-0.0.0.0}"
NSPEC="${NSPEC:-8}"
# LFM3 emits <tool_call><function=name><parameter=key>value</parameter>,
# which is the Qwen3-Coder XML shape. Liquid's own vLLM guide prescribes
# qwen3_coder + the qwen3 reasoning parser; glm45/hermes/lfm2 do not match.
# Set TOOL_PARSER= (empty) to disable tool calling.
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# LFM3's template defaults enable_thinking=true. Keep it on and let the
# qwen3 reasoning parser separate `thinking` from the answer; THINKING=off
# forces direct answers, but this checkpoint then leaks its reasoning into
# content, so on is the better default. Clients can override per request.
THINKING="${THINKING:-on}"
TARGET="${TARGET:-OpenIntelligenceNet/LFM3-1.7B-Uncensored}"
DRAFT="${DRAFT:-rasyosef/LFM3-1.7B-DSpark}"
PY="${PY:-$PWD/venv/bin/python}"
MODE="${1:-dspark}"

command -v "$PY" >/dev/null 2>&1 || {
  echo "error: $PY not found. Create the env first: see README.md (python 3.10-3.12)." >&2
  exit 1
}
# vLLM ships no __main__, so use the console script next to the interpreter
VLLM="$(dirname "$PY")/vllm"
command -v "$VLLM" >/dev/null 2>&1 || {
  echo "error: $VLLM not found. Install first: see README.md." >&2
  exit 1
}

# Resolve a model id to a local directory, downloading it once into ./models
# so later starts are instant and work offline.
ensure_model() {
  local id="$1" dest
  if [[ "$id" != */* ]]; then   # already a local path
    dest="$id"
    [[ -d "$dest" ]] || { echo "error: local model dir '$dest' not found" >&2; exit 1; }
    echo "$dest"; return
  fi
  local name="${id##*/}"
  dest="models/$name"
  if [[ -d "$dest" ]]; then
    echo "$dest"; return
  fi
  echo "Downloading $id -> $dest"
  local hf=""
  if [[ -x venv/bin/hf ]]; then hf="venv/bin/hf"
  elif command -v hf >/dev/null 2>&1; then hf="hf"
  fi
  if [[ -n "$hf" ]]; then
    "$hf" download "$id" --local-dir "$dest"
  else
    "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$id', local_dir='$dest')"
  fi
  echo "$dest"
}

target="$(ensure_model "$TARGET")"

# OpenAI clients (OpenCode, etc.) send tool_choice:"auto" whenever tools are
# attached; vLLM rejects that unless auto tool choice + a parser are enabled.
TOOL_ARGS=()
if [[ -n "$TOOL_PARSER" ]]; then
  TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  TOOL_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "$THINKING" == "off" ]]; then
  TOOL_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi

if [[ "$MODE" == "baseline" ]]; then
  echo "Serving $target (no drafter) on $HOST:$PORT"
  exec "$VLLM" serve "$target" \
    --host "$HOST" --port "$PORT" \
    --served-model-name lfm3-1.7b-uncensored \
    "${TOOL_ARGS[@]}"
else
  draft="$(ensure_model "$DRAFT")"
  echo "Serving $target with DSpark draft $draft on $HOST:$PORT"
  exec "$VLLM" serve "$target" \
    --host "$HOST" --port "$PORT" \
    --served-model-name lfm3-1.7b-uncensored \
    --speculative-config "{\"model\": \"$draft\", \"num_speculative_tokens\": $NSPEC}" \
    "${TOOL_ARGS[@]}"
fi
