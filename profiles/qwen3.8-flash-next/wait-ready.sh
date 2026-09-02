#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8888}"
NAME="${NAME:-qwen38-flash-next-sglang-128k-kda}"
TIMEOUT="${TIMEOUT:-3600}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
start=$(date +%s)
while (( $(date +%s) - start < TIMEOUT )); do
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/tmp/flashnext-models.json 2>/dev/null; then
    rm -f "$ROOT/runtime/ple/.loading"
    touch "$ROOT/runtime/ple/.ready"
    cat /tmp/flashnext-models.json
    echo
    echo "READY after $(( $(date +%s) - start )) seconds"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited before readiness" >&2
    docker logs "$NAME" 2>&1 | tail -120 >&2 || true
    exit 1
  fi
  sleep 10
done
echo "readiness timeout after ${TIMEOUT}s" >&2
docker logs "$NAME" 2>&1 | tail -120 >&2 || true
exit 1
