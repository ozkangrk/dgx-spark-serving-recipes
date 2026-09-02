#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8888}"
NAME="${NAME:-qwen38-r0b0-sglang-dflash2-256k}"
for i in $(seq 1 120); do
  if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "ready probe=$i"; exit 0
  fi
  if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -qx true; then
    docker logs --tail 100 "$NAME" >&2 || true; exit 2
  fi
  sleep 10
done
docker logs --tail 100 "$NAME" >&2 || true
exit 3
