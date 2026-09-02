#!/usr/bin/env bash
set -euo pipefail
NAME="${NAME:-qwen38-flash-next-sglang-128k-kda}"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker stop -t 30 "$NAME"
else
  echo "$NAME is not running"
fi
