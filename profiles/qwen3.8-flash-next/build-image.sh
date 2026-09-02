#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVERLAY="$ROOT/runtime/overlay"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}"
TAG="${TAG:-qwen38-flash-next-sglang-sm121:0.1.0}"
(cd "$OVERLAY" && sha256sum -c MANIFEST.sha256)
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$OVERLAY/qwen4_exp.py" "$OVERLAY/qwen_sparse_attn_backend.py" "$STAGE/"
cat > "$STAGE/Dockerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG SGL=/sgl-workspace/sglang/python/sglang
COPY qwen4_exp.py ${SGL}/srt/models/qwen4_exp.py
COPY qwen_sparse_attn_backend.py ${SGL}/srt/layers/attention/qwen_sparse_attn_backend.py
RUN python3 - <<'PY'
import ast
sgl = "/sgl-workspace/sglang/python/sglang"
model = open(f"{sgl}/srt/models/qwen4_exp.py").read()
qsa = open(f"{sgl}/srt/layers/attention/qwen_sparse_attn_backend.py").read()
ast.parse(model); ast.parse(qsa)
assert "SGLANG_QWEN4_PLE_MMAP_DIR" in model
assert "is_sm100_supported() or is_sm120_supported()" in qsa
assert "flash_attention_v4" in qsa
print("verified Flash-Next PLE mmap + SM121 QSA overlay")
PY
EOF
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$TAG" "$STAGE"
docker image inspect "$TAG" --format '{{.Id}}'
