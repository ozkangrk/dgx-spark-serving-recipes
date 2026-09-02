#!/usr/bin/env bash
set -euo pipefail

NAME="qwen38-radixark-dflash2-c3-fastlane"
IMAGE="lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe"
TARGET="${TARGET:-$HOME/Documents/Models/RadixArk-Qwen3.8-27B-NVFP4}"
DRAFT="${DRAFT:-$HOME/Documents/Models/Qwen3.8-27B-DFlash2-SGLang-256K/models/draft}"
CACHE="${CACHE:-$HOME/.cache/gb10/sglang-cache-c3-d16-bf16kv}"
NETWORK="qwen38-radixark-loopback"

[[ -f "$TARGET/config.json" ]] || { echo "missing target" >&2; exit 2; }
[[ -f "$DRAFT/config.json" && -f "$DRAFT/model.safetensors" ]] || { echo "missing draft" >&2; exit 2; }
mkdir -p "$CACHE"
chmod 1777 "$CACHE"
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container name already exists: $NAME" >&2
  exit 3
fi

docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --gpus all \
  --network "$NETWORK" \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --cpuset-cpus 5-9,15-19 \
  --memory 112g --memory-swap 112g \
  --pids-limit 4096 \
  --shm-size 32g \
  -p 127.0.0.1:8888:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_CACHE_DIR=/root/.cache/sglang \
  -v "$TARGET:/model:ro" \
  -v "$DRAFT:/draft:ro" \
  -v "$CACHE:/root/.cache/sglang:rw" \
  --label qwen38.profile=c3-draft16-bf16kv-fastlane \
  --label qwen38.repo_commit=2db1ecda01d9cab104aff9706083d1352a651a1a \
  --label qwen38.target_revision=554ebba9b5f1b79dc11246341960360e6ef05ef4 \
  --label qwen38.draft_revision=50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --label qwen38.max_agents=3 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /model \
    --served-model-name qwen38-27b-radixark-dflash2 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --kv-cache-dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --mamba-full-memory-ratio 4.21 \
    --mamba-radix-cache-strategy extra_buffer \
    --max-mamba-cache-size 15 \
    --max-running-requests 3 \
    --max-total-tokens 524288 \
    --cuda-graph-max-bs-decode 4 \
    --context-length 262144 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /draft \
    --speculative-num-draft-tokens 16 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --sampling-defaults model \
    --enable-metrics \
    --enable-cache-report
