#!/usr/bin/env bash
set -euo pipefail
NAME="${NAME:-qwen38-r0b0-sglang-dflash2-256k}"
docker stop "$NAME"
