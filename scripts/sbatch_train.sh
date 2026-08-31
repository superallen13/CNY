#!/bin/bash
# CNY training launcher: Slurm + Apptainer, single node.
#
#   sbatch --account=<acct> --partition=<part> --gres=gpu:h100:2 \
#          scripts/sbatch_train.sh configs/train_qwen2.5-7b.yaml [KEY=VAL | --flag]...
#
# Required env:
#   SIF_IMAGE   path to the slime CUDA container (see slime/docker)
# Optional env:
#   WALKER_VENV       venv overlaid on the container (must contain PyYAML, ray, sglang)
#   CACHE_ROOT        writable scratch for HF/torch/triton caches (default: $PWD/.cache)
#   HF_LOCAL_DIR_HOST host dir for the base checkpoint (default: from the yaml)
#   HF_TOKEN, WANDB_API_KEY
#
# Multi-node runs are not launched from here; bring up a Ray cluster first and
# see README "Multi-node" for the exact commands.
#SBATCH --job-name=cny-train
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch_logs/cny_train_%j.log
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXP="${1:-configs/train_qwen2.5-7b.yaml}"
[ -f "$EXP" ] || { echo "ERROR: exp yaml not found: $EXP" >&2; exit 2; }
[ $# -gt 0 ] && shift
RUN_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then RUN_ARGS+=("$arg")
    elif [[ "$arg" == *=* ]]; then RUN_ARGS+=("--set" "$arg")
    else echo "ERROR: bad arg (expect KEY=VAL or --flag): $arg" >&2; exit 5
    fi
done

# --- python with PyYAML, used only to read the yaml on the host --------------
WALKER_VENV="${WALKER_VENV:-}"
LOADER_PY=""
for candidate in "$WALKER_VENV/bin/python" python3 python; do
    if [ -n "$candidate" ] && "$candidate" -c "import yaml" >/dev/null 2>&1; then
        LOADER_PY="$candidate"; break
    fi
done
[ -z "$LOADER_PY" ] && { echo "ERROR: no python with PyYAML on PATH" >&2; exit 4; }

cfgget() {
    PYTHONPATH="$REPO_ROOT" "$LOADER_PY" - "$EXP" "$1" <<'PY'
import sys
from walker.config import load_exp
print(getattr(load_exp(sys.argv[1]), sys.argv[2]) or "")
PY
}

SIF_IMAGE="${SIF_IMAGE:-$(cfgget sif_image)}"
[ -n "$SIF_IMAGE" ] || { echo "ERROR: set SIF_IMAGE (or sif_image in $EXP)" >&2; exit 3; }
[ -f "$SIF_IMAGE" ] || { echo "ERROR: SIF image not found: $SIF_IMAGE" >&2; exit 3; }
[ -n "$WALKER_VENV" ] || WALKER_VENV="$(cfgget walker_venv)"

HF_REPO="$(cfgget walker_hf_checkpoint)"
HF_LOCAL_DIR_HOST="${HF_LOCAL_DIR_HOST:-$(cfgget hf_local_dir_host)}"
HF_LOCAL_DIR_IN_SIF="$(cfgget hf_local_dir_in_sif)"
[ -n "$HF_LOCAL_DIR_HOST" ] || { echo "ERROR: set hf_local_dir_host in $EXP" >&2; exit 3; }
[ -n "$HF_LOCAL_DIR_IN_SIF" ] || { echo "ERROR: set hf_local_dir_in_sif in $EXP" >&2; exit 3; }

CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"
XDG_CACHE_HOST="$CACHE_ROOT/xdg"
LOG_DIR="$REPO_ROOT/sbatch_logs"
mkdir -p "$HF_LOCAL_DIR_HOST" "$XDG_CACHE_HOST" "$LOG_DIR"

JOB_ID="${SLURM_JOB_ID:-$$}"
INSTANCE_NAME="cny_${JOB_ID}"
RAY_PORT=$(( 6380 + (JOB_ID % 1000) ))
RAY_TEMP_DIR="${RAY_TEMP_DIR:-$CACHE_ROOT/ray/$JOB_ID}"
MASTER_ADDR=127.0.0.1
mkdir -p "$RAY_TEMP_DIR"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

echo "[cny] host=$(hostname) job=$JOB_ID exp=$EXP gpus=$NUM_GPUS"
echo "[cny] sif=$SIF_IMAGE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Rollout knobs are read from the environment by the Ray actors, which inherit
# from the raylet started below, not from the walker.run driver. Export them
# before ray starts, straight from the yaml, so there is a single source.
eval "$(PYTHONPATH="$REPO_ROOT" "$LOADER_PY" -m walker.run --exp "$EXP" \
        "${RUN_ARGS[@]}" --print-walker-env)"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

VENV_BIND=""; VENV_PYTHONPATH=""; VENV_PATH=""
if [ -n "$WALKER_VENV" ] && [ -d "$WALKER_VENV/lib/python3.12/site-packages" ]; then
    VENV_BIND="--bind $WALKER_VENV:$WALKER_VENV:ro"
    VENV_PYTHONPATH="$WALKER_VENV/lib/python3.12/site-packages"
    VENV_PATH="$WALKER_VENV/bin"
    echo "[cny] venv overlay: $WALKER_VENV"
fi

cleanup() {
    echo "[cny] cleanup..."
    apptainer exec instance://"$INSTANCE_NAME" pkill -f sglang.launch_server 2>/dev/null || true
    apptainer exec instance://"$INSTANCE_NAME" ray stop 2>/dev/null || true
    apptainer instance stop "$INSTANCE_NAME" 2>/dev/null || true
}
trap cleanup EXIT

echo "[cny] starting apptainer instance: $INSTANCE_NAME"
apptainer instance start --nv \
    --bind "$REPO_ROOT:/walker" \
    --bind "$XDG_CACHE_HOST:/walker_xdg" \
    --bind "$HF_LOCAL_DIR_HOST:$HF_LOCAL_DIR_IN_SIF" \
    $VENV_BIND \
    "$SIF_IMAGE" "$INSTANCE_NAME"

if ! apptainer instance list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$INSTANCE_NAME"; then
    echo "[cny] ERROR: apptainer instance '$INSTANCE_NAME' failed to start" >&2
    exit 1
fi

ainst() {
    local _PP="/walker/slime:/walker:/workspace/Megatron-LM:/root/Megatron-LM"
    [ -n "$VENV_PYTHONPATH" ] && _PP="${VENV_PYTHONPATH}:${_PP}"
    local _PATH_PREFIX=""
    [ -n "$VENV_PATH" ] && _PATH_PREFIX="${VENV_PATH}:"
    local WALKER_ENVS=()
    while IFS='=' read -r _k _v; do
        WALKER_ENVS+=(--env "${_k}=${_v}")
    done < <(env | grep -E "^WALKER_" || true)
    apptainer exec \
        "${WALKER_ENVS[@]}" \
        --pwd /walker \
        --env PYTHONPATH="${_PP}" \
        --env PATH="${_PATH_PREFIX}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        --env VIRTUAL_ENV="${WALKER_VENV}" \
        --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
        --env WALKER_IN_SIF=1 \
        --env HF_HOME=/walker_xdg/hf \
        --env HF_HUB_CACHE=/walker_xdg/hf/hub \
        --env TORCH_HOME=/walker_xdg/torch \
        --env XDG_CACHE_HOME=/walker_xdg/xdg \
        --env TRITON_CACHE_DIR=/walker_xdg/triton \
        --env HF_TOKEN="${HF_TOKEN}" \
        --env RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}" \
        --env RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
        --env NCCL_BLOCKING_WAIT=1 \
        --env NCCL_ASYNC_ERROR_HANDLING=1 \
        --env WANDB_API_KEY="${WANDB_API_KEY}" \
        instance://"$INSTANCE_NAME" "$@"
}

if [ ! -f "$HF_LOCAL_DIR_HOST/model.safetensors.index.json" ] && \
   [ ! -f "$HF_LOCAL_DIR_HOST/model.safetensors" ]; then
    echo "[cny] base checkpoint missing; downloading $HF_REPO"
    ainst hf download "$HF_REPO" --local-dir "$HF_LOCAL_DIR_IN_SIF"
fi

PY="python"
[ -n "$VENV_PATH" ] && PY="${VENV_PATH}/python"

echo "[cny] starting ray head on ${MASTER_ADDR}:${RAY_PORT}"
ainst "$PY" -m ray.scripts.scripts start --head \
    --node-ip-address "$MASTER_ADDR" --port "$RAY_PORT" \
    --min-worker-port 10002 --max-worker-port 19999 \
    --temp-dir="$RAY_TEMP_DIR" --num-gpus "$NUM_GPUS" \
    --object-store-memory 16000000000 --disable-usage-stats --include-dashboard=false

echo "[cny] handing off to walker.run"
ainst env NUM_GPUS="$NUM_GPUS" \
    "$PY" -m walker.run --exp "$EXP" "${RUN_ARGS[@]}"

echo "[cny] done $(date)"
