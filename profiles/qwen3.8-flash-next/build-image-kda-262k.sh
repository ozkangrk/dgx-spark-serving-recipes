#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVERLAY="$ROOT/runtime/overlay-kda-262k"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}"
TAG="${TAG:-qwen38-flash-next-sglang-sm121-kda:0.2.0}"
EXPECTED_INIT=d9e9bb916c5517d0d0ba83f28670fa0e4ddf92cbd4e64e25d6649c116f489737
EXPECTED_KERNEL=a0fd72586b341a1405cf5a4aa98686cabf3965270ae1e5f70a93ff7db0b104d1
printf '%s  %s\n' "$EXPECTED_INIT" "$OVERLAY/kda_qwen38_qsa_sm121/__init__.py" | sha256sum -c -
printf '%s  %s\n' "$EXPECTED_KERNEL" "$OVERLAY/kda_qwen38_qsa_sm121/kernel.py" | sha256sum -c -
python3 - "$OVERLAY" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
for rel in ["qwen4_exp.py","qwen_sparse_attn_backend.py","sm121_kda_varlen.py","kda_qwen38_qsa_sm121/__init__.py","kda_qwen38_qsa_sm121/kernel.py"]:
    compile((root/rel).read_text(),rel,"exec")
qsa=(root/"qwen_sparse_attn_backend.py").read_text()
assert "is_sm100_supported() or is_sm120_supported()" not in qsa
assert "qsa.sm121_kda_varlen" in qsa
assert "PLE_LIFECYCLE_V2_SEQUENTIAL_THEN_RANDOM" in (root/"qwen4_exp.py").read_text()
print("verified B+C overlay: A PLE lifecycle + exact SM121 KDA route")
PY
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
cp -a "$OVERLAY/." "$STAGE/"
cat > "$STAGE/Dockerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG SGL=/sgl-workspace/sglang/python/sglang
COPY qwen4_exp.py ${SGL}/srt/models/qwen4_exp.py
COPY qwen_sparse_attn_backend.py ${SGL}/srt/layers/attention/qwen_sparse_attn_backend.py
COPY sm121_kda_varlen.py ${SGL}/srt/layers/attention/qsa/sm121_kda_varlen.py
COPY kda_package_init.py ${SGL}/kernels/kda_kernels/__init__.py
COPY kda_qwen38_qsa_sm121/ ${SGL}/kernels/kda_kernels/qwen38_qsa_sm121/
RUN python3 - <<'PY'
import ast
from pathlib import Path
sgl=Path('/sgl-workspace/sglang/python/sglang')
files=[sgl/'srt/models/qwen4_exp.py',sgl/'srt/layers/attention/qwen_sparse_attn_backend.py',sgl/'srt/layers/attention/qsa/sm121_kda_varlen.py',sgl/'kernels/kda_kernels/qwen38_qsa_sm121/__init__.py',sgl/'kernels/kda_kernels/qwen38_qsa_sm121/kernel.py']
for p in files: ast.parse(p.read_text())
assert 'PLE_LIFECYCLE_V2_SEQUENTIAL_THEN_RANDOM' in files[0].read_text()
assert 'qsa.sm121_kda_varlen' in files[1].read_text()
print('verified B+C image')
PY
EOF
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$TAG" "$STAGE"
docker image inspect "$TAG" --format '{{.Id}}'
