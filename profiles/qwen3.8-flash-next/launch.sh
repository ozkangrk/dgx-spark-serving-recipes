#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${NAME:-qwen38-flash-next-sglang-64k}"
IMAGE="${IMAGE:-qwen38-flash-next-sglang-sm121:0.1.0}"
PORT="${PORT:-8888}"
MODEL="$ROOT/models/target"
PLE="$ROOT/runtime/ple"
CACHE="$ROOT/runtime/cache"
NETWORK="qwen38-flash-next-loopback"

for f in "$MODEL/config.json" "$MODEL/model.safetensors.index.json"; do
  [[ -f "$f" ]] || { echo "missing model artifact: $f" >&2; exit 2; }
done
shopt -s nullglob
shards=("$MODEL"/*.safetensors)
(( ${#shards[@]} >= 100 )) || { echo "incomplete model: expected many safetensors shards, found ${#shards[@]}" >&2; exit 2; }
mkdir -p "$PLE" "$CACHE"
chmod 1777 "$PLE" "$CACHE"
# A boot interrupted while materializing the 48 GiB PLE mmap file can leave a
# sparse-but-poisoned backing store. Rebuild it after any boot that never
# reached readiness; wait-ready.sh removes this marker only after /v1/models.
if [[ -f "$PLE/.loading" ]]; then
  echo "previous boot did not reach readiness; rebuilding PLE backing file"
  rm -f "$PLE"/ple_table_*.bin
fi
touch "$PLE/.loading"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "missing derived image: $IMAGE; run runtime/build-image.sh" >&2; exit 2; }
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME already running"; exit 0
fi
# Refuse to co-run another inference workload.
if docker ps --format '{{.Names}}' | grep -Eiq '(qwen|vllm|sglang|ds4|llama)'; then
  echo "another inference container is running; stop it explicitly before launch" >&2
  docker ps --format '{{.Names}}\t{{.Status}}' >&2
  exit 3
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null

docker run -d \
  --name "$NAME" \
  --restart no \
  --gpus all \
  --network "$NETWORK" \
  --ipc host \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --cpus 16 \
  --memory 110g --memory-swap 110g \
  --pids-limit 4096 \
  --shm-size 16g \
  -p "127.0.0.1:${PORT}:30000" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_QWEN4_PLE_MMAP_DIR=/ple \
  -e SGLANG_CACHE_DIR=/cache \
  -v "$MODEL:/model:ro" \
  -v "$PLE:/ple:rw" \
  -v "$CACHE:/cache:rw" \
  --label flashnext.bundle=nvfp4-sglang-262k \
  --label flashnext.model-revision=7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --label flashnext.source-commit=e8d6c6cc9adfd3cf174e293fcd89e705e2e60ad1 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model --tp-size 1 \
    --served-model-name qwen38-flash-next-nvfp4-sglang-64k \
    --quantization modelopt_fp4 \
    --prefill-attention-backend triton \
    --decode-attention-backend trtllm_mha \
    --ple-offload-embedding \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 5 \
    --mem-fraction-static 0.85 \
    --context-length 65536 \
    --chunked-prefill-size 1024 \
    --max-running-requests 1 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 30000 \
    --enable-metrics --sleep-on-idle
printf 'launched %s; endpoint http://127.0.0.1:%s/v1\n' "$NAME" "$PORT"
