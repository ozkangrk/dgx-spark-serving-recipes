#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${NAME:-qwen38-r0b0-sglang-dflash2-256k}"
IMAGE="${IMAGE:-qwen38-r0b0-sglang-dflash2-sm121:0.2.0}"
PORT="${PORT:-8888}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-20}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-4}"
DRAFT_TOKENS="${DRAFT_TOKENS:-10}"
TARGET="$ROOT/models/target"
DRAFT="$ROOT/models/draft"
CACHE="$ROOT/runtime/triton-cache"
NETWORK="qwen38-r0b0-loopback"
for f in "$TARGET/config.json" "$TARGET/model-00001-of-00004.safetensors" "$TARGET/model-00002-of-00004.safetensors" "$TARGET/model-00003-of-00004.safetensors" "$TARGET/model-00004-of-00004.safetensors" "$DRAFT/config.json" "$DRAFT/model.safetensors"; do
  [[ -f "$f" ]] || { echo "missing required artifact: $f" >&2; exit 2; }
done
mkdir -p "$CACHE"
chmod 1777 "$CACHE"
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME already running"; exit 0
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart no \
  --gpus all \
  --network "$NETWORK" \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --cpus 14 \
  --memory 112g --memory-swap 112g \
  --pids-limit 4096 \
  --shm-size 32g \
  -p "127.0.0.1:${PORT}:30000" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$TARGET:/model:ro" \
  -v "$DRAFT:/draft:ro" \
  -v "$CACHE:/root/.triton:rw" \
  --label qwen38.bundle=r0b0-sglang-dflash2-256k \
  --label qwen38.source-commit=535e580eea3e777306629746c9ab45153b653a1a \
  --label qwen38.target-revision=36f717a22990e82c54c1d48ee77c491b87825680 \
  --label qwen38.draft-revision=50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  "$IMAGE" \
  sglang serve \
    --trust-remote-code \
    --model-path /model \
    --served-model-name qwen38-27b-dflash2-256k \
    --host 0.0.0.0 --port 30000 \
    --attention-backend flashinfer \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 8192 \
    --context-length 262144 \
    --mem-fraction-static "$MEM_FRACTION" \
    --disable-prefill-cuda-graph \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-full-memory-ratio 4.59 \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
    --max-total-tokens 1048576 \
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /draft \
    --speculative-num-draft-tokens "$DRAFT_TOKENS" \
    --enable-metrics --enable-cache-report --sleep-on-idle
printf 'launched %s on http://127.0.0.1:%s/v1 (restart=no until validation)\n' "$NAME" "$PORT"
