#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVERLAY="$ROOT/runtime/overlay-ple-fast"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}"
TAG="${TAG:-qwen38-flash-next-sglang-sm121:0.1.1-ple-fast}"
for f in qwen4_exp.py qwen_sparse_attn_backend.py; do
  [[ -f "$OVERLAY/$f" ]] || { echo "missing $OVERLAY/$f" >&2; exit 2; }
done
python3 - "$OVERLAY" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
model = (root / "qwen4_exp.py").read_text()
qsa = (root / "qwen_sparse_attn_backend.py").read_text()
compile(model, "qwen4_exp.py", "exec")
compile(qsa, "qwen_sparse_attn_backend.py", "exec")
assert "PLE_LIFECYCLE_V2_SEQUENTIAL_THEN_RANDOM" in model
assert '_madvise_ple_storage(storage, nbytes, "sequential")' in model
assert "_finalize_ple_mmap_for_random_access(self)" in model
# A deliberately preserves the baseline QSA path; B+C changes it later.
assert "is_sm100_supported() or is_sm120_supported()" in qsa
print("verified A overlay: sequential PLE load, random serving advice, baseline QSA")
PY
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
assert "PLE_LIFECYCLE_V2_SEQUENTIAL_THEN_RANDOM" in model
assert "is_sm100_supported() or is_sm120_supported()" in qsa
print("verified A image")
PY
EOF
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$TAG" "$STAGE"
docker image inspect "$TAG" --format '{{.Id}}'
