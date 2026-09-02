#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${NAME:-qwen38-radixark-dflash2-4x128k}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b-dflash2}"
PORT="${PORT:-8888}"
TARGET="${TARGET:-$HOME/Documents/Models/RadixArk-Qwen3.8-27B-NVFP4}"
DRAFT="${DRAFT:-$ROOT/models/draft}"
CACHE="${CACHE:-$ROOT/runtime/triton-cache-radixark}"
NETWORK="qwen38-radixark-loopback"

[[ -f "$TARGET/config.json" ]] || { echo "missing target config" >&2; exit 2; }
[[ $(find "$TARGET" -maxdepth 1 -name '*.safetensors' -type f | wc -l) -ge 3 ]] || { echo "missing target shards" >&2; exit 2; }
[[ -f "$DRAFT/config.json" && -f "$DRAFT/model.safetensors" ]] || { echo "missing DFlash2 draft" >&2; exit 2; }
mkdir -p "$CACHE"
chmod 1777 "$CACHE"
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
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
  --label qwen38.profile=radixark-dflash2-4x128k \
  --label qwen38.target=RadixArk-Qwen3.8-27B-NVFP4 \
  --label qwen38.draft=z-lab-Qwen3.8-27B-DFlash2 \
  "$IMAGE" \
  sglang serve \
    --trust-remote-code \
    --model-path /model \
    --served-model-name qwen38-27b-radixark-dflash2 \
    --host 0.0.0.0 --port 30000 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 8192 \
    --context-length 262144 \
    --mem-fraction-static 0.85 \
    --disable-prefill-cuda-graph \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-full-memory-ratio 4.59 \
    --max-running-requests 4 \
    --max-mamba-cache-size 20 \
    --max-total-tokens 1048576 \
    --cuda-graph-max-bs-decode 4 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /draft \
    --speculative-num-draft-tokens 10 \
    --enable-metrics --enable-cache-report --sleep-on-idle

printf 'launched %s on http://127.0.0.1:%s/v1\n' "$NAME" "$PORT"
