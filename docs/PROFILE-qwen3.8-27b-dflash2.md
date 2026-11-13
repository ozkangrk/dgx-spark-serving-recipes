# Qwen3.8-27B on DGX Spark (GB10)

> **Adapted from darkdatter's README.** This page is the README of
> [darkdatter/gb10-repo](https://github.com/darkdatter/gb10-repo) at commit
> `2db1ecd` (MIT licence, copy in
> [`LICENSES/darkdatter-gb10-repo-MIT.txt`](../LICENSES/darkdatter-gb10-repo-MIT.txt)).
> The recipe is **darkdatter's**, and so are the numbers: darkdatter measured
> them on an ASUS DGX Spark, and "I" and "we" below mean darkdatter. Figures
> attributed to "a second machine" or "an independent run" are kosztyua's
> reproduction ([darkdatter/gb10-repo#1](https://github.com/darkdatter/gb10-repo/pull/1)).
> Edits in this copy: links repointed to this repo; notes that darkdatter's
> `scripts/` and `patches/` are not included here; the toolkit credit
> paragraph below the headline table and the toolkit-checkout note in step 3;
> the corrected event-counting trap; the layout and licence lines at the end.
> This repo author's own Qwen3.8-27B work (a rerun of this recipe on an NVIDIA
> DGX Spark on 2026-08-30, and the 2026-09-02 fast-lane A/B evidence in
> [`profiles/qwen3.8-27b-dflash2/evidence/`](../profiles/qwen3.8-27b-dflash2/evidence/))
> is listed in the [main README](../README.md#provenance).

A measured recipe for the fastest Qwen3.8-27B setup I could get on a DGX Spark:
**SGLang + NVFP4 + DFlash2 speculative decoding**, managed by
[SparkStation](https://github.com/kshetrajna12/sparkstation).

| Single-stream | Peak aggregate | TTFT | HumanEval pass@1 |
|---:|---:|---:|---:|
| **78.6 tok/s** | **480.7 tok/s** (16 streams) | **190 ms** | **97.0%** |

Those two throughput figures come from different settings: single-stream is at
`--speculative-num-draft-tokens 16`, peak aggregate at 8. The optima diverge —
see step 6.

**NVFP4 beats FP8, and SGLang beats vLLM, on this hardware.** The widely-shared
"FP8 on vLLM at ~32 tok/s" recipe is roughly half this speed.

The recipe is built on Mia AI Lab's
[Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
toolkit (`start-dflash.sh`, SGLang overlay, NVFP4 head patch), z-lab's DFlash2
draft and RadixArk's NVFP4 target; pins are in the table below.

Reproduce with [`bench/`](../bench/). Full data in
[`RESULTS.md`](RESULTS.md). The exact
build inputs behind those numbers are in
[`BUILD-MANIFEST.md`](BUILD-MANIFEST.md).

The helper scripts (`scripts/`) and patch notes (`patches/`) that accompanied
the original write-up are not part of this repository; the steps that used
them say so.

## Ingredients

| Component | Pin |
|---|---|
| Target | `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9` |
| Draft | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c` |
| Image | `lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2` |
| Toolkit | [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) @ `c90d8c34` |
| Disk | ~110 GB |

All model repos are public; no token needed. Verified on Ubuntu 24.04,
kernel 6.17-nvidia, driver 580.173.02 / CUDA 13.0, Docker 29.2.1, 128 GB unified.

---

## 1. Verify GPU passthrough

```bash
./scripts/00-verify-gpu.sh          # helper not included in this repo
```

DGX OS ships CDI, not a registered Docker runtime — `docker info` showing only
`runc` is normal and `--gpus all` still works.

## 2. Pull the image, fetch the weights

```bash
./scripts/01-build-and-fetch.sh     # pulls the official image + weights (helper not included in this repo)
```

Without the helper, pull the image and download the weights at the revisions
in the Ingredients table and [`BUILD-MANIFEST.md`](BUILD-MANIFEST.md).

LMSYS published official DFlash2 images on 2026-08-22: `dev-cu13-qwen38-27b-dflash2`,
`dev-qwen38-27b-dflash2`, `dev-cu12-qwen38-27b-dflash2`, arm64 included. Only the
`qwen38-27b-dflash2` tag name is absent. The official image needs no `lm_head` patch.

## 3. Run standalone, get a baseline

Get a number before adding orchestration. Run this from a checkout of Mia AI
Lab's toolkit at `c90d8c34`.

```bash
cd ~/spark/Qwen3.8-27B-SGLang-DGX-Spark   # required: start.sh uses WORK_DIR="$(pwd)"
cp -n .env.sample .env
IMAGE=lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2 \
  DF_EXTRA="--mem-fraction-static 0.85" ./start-dflash.sh
```

Serves on **:8888**, OpenAI- and Anthropic-compatible. First boot ≈3 min.

`IMAGE=` is required: `start-dflash.sh` defaults to the locally built tag and
will build it if absent, which is exactly what step 2 avoids.

**Use 0.85 on a first boot.** The toolkit defaults NVFP4 to `0.90` and generic
`start.sh` to `0.95`; both this project and the toolkit record machines
unrecoverably wedged during weight load at higher fractions — GB10 unified
memory starves the OS.

```bash
GB10_BASE_URL=http://127.0.0.1:8888/v1 GB10_MODEL=qwen3.8-27b-sglang \
  python3 bench/perf.py
```

## 4. Add SparkStation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --depth 1 https://github.com/kshetrajna12/sparkstation.git
cd sparkstation && uv sync
cp .env.example .env      # TOTAL_UNIFIED_MEMORY_GB=128, MEMORY_HARD_LIMIT_GB=110
```

One required `models.yaml` edit — the entry ships pinned to a second machine:

```diff
-    host: worker1
+    host: primary
```

```bash
uv run sparkstation start -d --profile generic
```

Gateway on **:8000**. The model container is bridge-networked, so its internal
:8000 maps to host **:8001** — no collision despite both reporting 8000.
Gateway overhead measured at zero.

## 5. Unlock concurrency

The shipped config caps you at 4 concurrent requests. **Three flags are
co-limiting** — raising only the obvious one measures nothing.

```diff
- "--max-running-requests"      "4"  →  "16"
- "--max-mamba-cache-size"     "20"  →  "80"    # 16 requests × 5 slots
- "--cuda-graph-max-bs-decode"  "4"  →  "16"    # else batches >4 fall back to eager
```

| Streams | cap 4 | cap 16 |
|---:|---:|---:|
| 1 | 58.3 | 52.9 |
| 8 | 190.1 | 315.9 |
| **16** | — | **480.7** |
| 32 | — | 469.8 |

16 is the cap, not a hardware optimum. SGLang clamps
`max_running_requests = max_mamba_cache_size / 5`, so the peak lands wherever the
pool allows and the 24/32 rows are queueing against it.

A larger pool does raise aggregate — we measured **474.0 tok/s at 32 streams**
with pool 160, against 362.7 at pool 80. [An independent run](REPRODUCTION.md)
reports higher still (606.4 at pool 160, 640.9 at pool 240 with cap 48); we did
not reproduce those magnitudes, see that file for why.

Aggregate is bought with KV, so these are short-prompt figures. Pool 240 needs
~47 GB of GDN state, which on 128 GB means either a mem-fraction we measured as
harmful or a KV pool too small to be useful — it failed to boot here. Treat
pool 160 as the practical ceiling.

Leave `--max-total-tokens 1048576` in place: uncapping it grows the KV pool but
nothing uses the space, and the lost headroom cost 18% at 32 streams.

Costs ~5% single-stream and 11.8 GB of GDN state. (The patch notes for this
`models.yaml` edit are not included in this repo; the diff above is the change.)

## 6. Tune draft tokens — the largest single-stream lever

`--speculative-num-draft-tokens` ships at 8. The measured optimum is ~16,
worth **+28%** single-stream. A second machine measured +41.6%; our 78.6 is
probably the low end of the spread. See [RESULTS](RESULTS.md#draft-tokens).

| draft | single | agg @16 | accept_len | accept_rate |
|---:|---:|---:|---:|---:|
| 6 | 49.5 | 404.2 | 5.25 | 0.85 |
| 8 (default) | 61.3 | 429.2 | 6.63 | 0.80 |
| 10 | 65.2 | **435.1** | 7.18 | 0.69 |
| 12 | 75.1 | 402.8 | 8.27 | 0.66 |
| 16 | **78.6** | 384.8 | 9.52 | 0.57 |
| 20 | 72.0 | 330.7 | 8.48 | 0.40 |
| 24 | 69.0 | 284.5 | 8.40 | 0.32 |

**The optima diverge — pick one:** 16 for interactive/single-stream, **10 for
concurrent serving** (435 vs 385 aggregate). You cannot have both.

Past 16, `accept_len` *falls* even though more tokens are drafted — the drafter
can't sustain longer correct runs, so you pay draft compute for tokens that get
rejected. `accept_rate` collapses from 0.80 to 0.32.

## 7. Benchmark quality

```bash
./scripts/run-humaneval.sh          # thinking off, ~5 min
./scripts/run-humaneval.sh think    # thinking on,  ~20 min
```

That wrapper is not included in this repo. It chained
[`bench/humaneval/generate.py`](../bench/humaneval/generate.py),
[`execute.py`](../bench/humaneval/execute.py) (inside a `--network none`
container) and [`report.py`](../bench/humaneval/report.py); each file's
docstring gives the command to run it on its own.

164 problems at temperature 0, each executed against its real unit tests in a
`--network none` container. Real pass@1, not self-judged.

| Mode | pass@1 | Tokens/problem | Wall |
|---|---:|---:|---:|
| Thinking off | 93.9% | ~200 | 3 min |
| Thinking on | **97.0%** | ~945 | 19 min |

Default to thinking off for well-specified functions — 93.9% at a fifth of the
tokens. Switch it on for hard cases.

---

## Traps

Each of these cost real time.

- **Pin the target revision, not just the draft.** `hf download` without
  `--revision` takes the repo's mutable default, and SGLang needs `--revision`
  separately — downloading the right checkpoint does not make the server load
  it. SparkStation's `models.yaml` already carries it; the standalone path does
  not.
- **Fully-offline first start still touches the network.** SGLang's early
  speculative-algorithm probe resolves the draft config without a revision, so a
  cache holding only the pinned snapshot can still reach for that repo's default
  ref. Prime the cache while online.
- **A working CDI fallback is not automatically usable.** If `--gpus all` fails
  and `--device nvidia.com/gpu=all` works, neither launcher picks that up — both
  hardcode `--gpus all`. You have to edit them.
- **SparkStation restarts healthy models under load.** `HEALTH_CHECK_TIMEOUT_SECONDS=5`
  x 3 failures: a model saturated on long prefill cannot answer a 5s probe, so the
  supervisor kills it mid-job and clients get 503s. Raise it to 30.
- **mem-fraction is not a perf lever here.** 0.82 / 0.85 / 0.90 all measure the
  same. Concurrency is bound by mamba slots, long context by prefill — never by
  memory capacity.
- **The DFlash2 image now exists upstream** (`dev-cu13-qwen38-27b-dflash2`).
- **HF cache symlinks inside the container mount break everything.** The
  container bind-mounts `~/.cache/huggingface`; symlinks *within* it point at
  host-only paths, so it re-downloads and dies with
  `OSError: I/O error: File exists (os error 17)`. Keep real directories there.
- **`host: worker1` assumes a second Spark.** Use `primary`, and check any
  profile you use — a profile-level `host:` wins.
- **`start.sh` uses `WORK_DIR="$(pwd)"`**, not the script dir. Run it from
  inside the repo or your caches land elsewhere.
- **5 mamba slots per request, not 4.** DFlash2's verify needs an extra. SGLang
  clamps `max_running_requests = pool / 5` and logs it. Size the pool at 5×
  target.
- **uv installs `cli.py` into site-packages.** Editing the repo copy does
  nothing — the console script imports from `.venv/bin`. Copy it across too.
- **Gateway reports "not healthy" while working.** `_gateway_healthy()`
  hardcodes `Bearer dummy-key`, so a custom `LITELLM_MASTER_KEY` gets HTTP 400.
  Patch it to send the configured key (the patch is not included in this repo).
- **Only the FLUX launcher forwards `HF_TOKEN`.** A gated model can't
  authenticate its own download; pre-pull on the host.
- **Never `docker rm -f` a managed container.** The supervisor's state goes
  stale; `sparkstation stop && start` reconciles.
- **Boot-to-boot variance is ~8% on single-stream**, against <2% run-to-run
  within one server instance. Size A/B deltas against 8%, and re-measure on a
  fresh boot before believing a small win.
- **Greedy is not bitwise deterministic.** Temperature 0 still flips 2–3
  HumanEval problems between runs. Don't read a sub-2% delta as a regression.
- **Thinking is on by default**, and a small `max_tokens` returns empty
  `content` with everything in `reasoning_content`. Always pair a token cap with
  a `finish_reason` check — truncation otherwise reads as a quality regression.
  That mistake made a 97.0% run score 90.9%.
- **Count `completion_tokens`, not stream events.** DFlash2 emits ~3.75 tokens
  per event; event-counting under-reads throughput ~3.75×. *(Correction in this
  copy: darkdatter's original says event-counting "inflates throughput ~4×".
  Counting each event as one token under-counts tokens, so it under-reads, as
  Mia AI Lab's toolkit README says.)*

## Layout

In this repository:

```
bench/     common.py · perf.py · longctx.py · humaneval/{generate,execute,report}.py
docs/      RESULTS.md — all measurements
           BUILD-MANIFEST.md — resolved build inputs for those numbers
           REPRODUCTION.md — the second-machine reproduction
profiles/qwen3.8-27b-dflash2/   not from gb10-repo: this repo author's deployment
                                (launchers · gates · bench-results) and 2026-09-02
                                fast-lane A/B evidence
```

The `scripts/` and `patches/` directories of the original write-up are not
included.

The bench scripts read `GB10_BASE_URL`, `GB10_API_KEY`, `GB10_MODEL` from the
environment (see the docstring in [`bench/common.py`](../bench/common.py));
nothing is baked in.

## Caveats

Throughput figures are code generation at temperature 0 on short prompts.
Long-context behaves very differently — a ~120K-token prompt costs ~95 s before
the first token (full prefill curve in
[`RESULTS.md`](RESULTS.md)). Single machine; treat the ordering
of findings as durable and absolute numbers as indicative.

MIT (darkdatter's licence) — see
[`LICENSES/darkdatter-gb10-repo-MIT.txt`](../LICENSES/darkdatter-gb10-repo-MIT.txt).
