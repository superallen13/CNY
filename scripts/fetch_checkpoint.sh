#!/bin/bash
# Download a trained CNY checkpoint for evaluation.
#
#   bash scripts/fetch_checkpoint.sh <hf-repo-id> [target-dir]
#
# Serve the result behind an OpenAI-compatible endpoint (sglang or vllm), then
# point configs/eval.yaml at it via backend_url + model.
set -euo pipefail

REPO="${1:?usage: $0 <hf-repo-id> [target-dir]}"
TARGET="${2:-checkpoints/$(basename "$REPO")}"

command -v hf >/dev/null 2>&1 || { echo "ERROR: install huggingface_hub[cli]" >&2; exit 3; }

mkdir -p "$TARGET"
hf download "$REPO" --local-dir "$TARGET"

echo "[cny] checkpoint at $TARGET"
echo "[cny] serve it, e.g.:"
echo "  python -m sglang.launch_server --model-path $TARGET --port 30000"
