# DGX Spark Serving Recipes

Measured, reproducible serving recipes for large open-weight models on a
single NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X, aarch64, SM121) —
and notes on where two-node clusters become necessary.

Every profile under [`profiles/`](profiles/) is a **working deployment**:
launch scripts, health/wait gates, quality gates, and the benchmark data
that justifies the chosen operating point. Not every profile is my own
work and not every number is my measurement: see [Provenance](#provenance)
and [Credits](#credits) for who built and measured what. Figures from
tweets, forums or model cards are labelled as third-party where they appear.

## Hardware reality (why these recipes look the way they do)

GB10 is a **capacity machine, not a bandwidth machine**: 128 GB unified
memory at ~273 GB/s. Decode throughput is bandwidth-bound
(`active_params × bytes_per_weight / 273 GB/s`), prefill and batched
serving are compute-bound and scale well. Consequences:

- MoE models with low active parameters are the sweet spot.
- Dense models >15B are painful interactively, fine for batch.
- Speculative decoding (DFlash/DSpark/MTP) is the highest-leverage knob
  for single-stream latency.
- Batch amortizes the bandwidth tax: the same engine that gives 20 tok/s
  at batch 1 can serve hundreds of tok/s aggregate at batch 16+.

See [`docs/nvfp4-serving-guide.md`](docs/nvfp4-serving-guide.md) for a
per-model decision table (which model fits one Spark, expected decode
rates, quantization notes). It is desk research in Turkish, compiled from
model cards and community reports: its figures are third-party, not
measurements from this repo.

## Profiles

| Profile | Engine | Quant | Context | Why it exists |
|---|---|---|---|---|
| [`qwen3.8-27b-dflash2`](profiles/qwen3.8-27b-dflash2/) | SGLang + DFlash2 spec-decode | NVFP4 target + BF16 draft | 256K | Flagship: the fastest Qwen3.8-27B setup I measured on GB10 (78.6 tok/s single-stream at draft 16, 480.7 tok/s @16 streams at draft 8, HumanEval 97.0%). Includes frozen A/B evidence for draft-width/KV-dtype operating points. My measurements, made with Mia AI Lab's toolkit, z-lab's DFlash2 draft and RadixArk's NVFP4 target. |
| [`deepseek-v4-flash-exl3`](profiles/deepseek-v4-flash-exl3/) | ExLlamaV3 (EXL3) | NVFP4 KV records | 384K single request | **By Mia AI Lab** (recipe and numbers are theirs; see its README). Deep-context: 439K-token pool on ONE Spark, auto-download launcher, TP1 coalesced checkpoint. The 157 GB model that does not fit one box, made to fit. |
| [`qwen3.8-flash-next`](profiles/qwen3.8-flash-next/) | SGLang (derived image) | NVFP4 | 64K/128K/256K variants | Launch matrix for PLE/KDA attention variants incl. FP8-KV trade study; boot-interrupt-safe PLE materialization. Adapted from hasso5703's SGLang recipe; serves RadixArk's NVFP4 checkpoint. |

## Shared benchmark harness

[`bench/`](bench/) is engine-agnostic (OpenAI-compatible endpoint via env
vars, no baked-in hosts or keys):

```bash
export GB10_BASE_URL=http://127.0.0.1:8888/v1
export GB10_MODEL=qwen38-27b-radixark-dflash2
python3 bench/perf.py        # TTFT, single-stream, concurrency ladder, prefill
python3 bench/longctx.py     # concurrent long-context scaling (cache-dedup safe)
python3 bench/humaneval/     # pass@1 with sandboxed execution
```

Measurement notes that matter on this box: count `completion_tokens` over
wall time, never SSE events (speculative decoders emit multi-token
events — event counting inflates rates ~4x); make long-context prompts
unique per stream or the radix cache measures itself; assert
`cached_tokens == 0` where you need cold-cache numbers.

## Provenance

- **Qwen3.8-27B** (`profiles/qwen3.8-27b-dflash2/`, `docs/RESULTS.md`,
  `docs/PROFILE-qwen3.8-27b-dflash2.md`): my own runs on one DGX Spark,
  with pinned model revisions and container digests recorded in the
  profile and in [`docs/BUILD-MANIFEST.md`](docs/BUILD-MANIFEST.md). The
  numbers in `docs/RESULTS.md` were produced with Mia AI Lab's toolkit,
  z-lab's DFlash2 draft and RadixArk's NVFP4 target; some files in the
  profile's `bench-results/` come from an earlier deployment built on
  r0b0tlab's bundle (see [Credits](#credits)).
- **DeepSeek-V4-Flash** (`profiles/deepseek-v4-flash-exl3/`): Mia AI
  Lab's recipe, included with its own README. Its measured results are
  Mia AI Lab's, not mine.
- **Qwen3.8-Flash-Next** (`profiles/qwen3.8-flash-next/`): launch scripts
  adapted from hasso5703's SGLang recipe (`hasso5703/dgx-spark-qwen38` at
  `e8d6c6cc`, the `source-commit` label in `launch.sh`), serving RadixArk's
  NVFP4 checkpoint (the `model-revision` label). The Flash-Next
  figures on the one-pager are my measurements.
- [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) reports a reproduction of
  the Qwen3.8-27B numbers on a second DGX Spark.
- Frozen A/B evidence (`profiles/qwen3.8-27b-dflash2/evidence/`) keeps
  raw JSON plus the harness scripts that produced it; known-invalid runs
  are excluded and the exclusion is documented in its `REPORT.md`.
- Engine comparison (SGLang vs vLLM vs TensorRT-LLM) and the failed
  attempts are written up in [`docs/RESULTS.md`](docs/RESULTS.md) and
  [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Requirements

- DGX Spark / GB10 (works on other CUDA 13 aarch64 boxes; untested
  elsewhere), DGX OS 7 / Ubuntu 24.04, Docker with CDI GPU passthrough
- ~25–200 GB disk depending on profile (model weights are NOT in this
  repo; launchers fetch or reference them locally)
- No cloud account, no API keys: everything runs locally.

## Credits

This repo builds on other people's work:

- **Mia AI Lab** ([@MiaAI_lab](https://x.com/MiaAI_lab)): the whole
  DeepSeek-V4-Flash EXL3 profile (recipe, launcher, image patches and its
  measured results; its README lists its own credits), and the
  [Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
  toolkit at commit `c90d8c34` (`start-dflash.sh`, SGLang overlay, NVFP4
  head patch), with which my Qwen3.8-27B numbers were produced.
- **z-lab**: the DFlash2 draft model `z-lab/Qwen3.8-27B-DFlash2`.
- **RadixArk**: the NVFP4 checkpoints `RadixArk/Qwen3.8-27B-NVFP4` (the
  Qwen3.8-27B target) and the Qwen3.8-Flash-Next checkpoint served by
  `profiles/qwen3.8-flash-next/`.
- **LMSYS / SGLang**: the serving engine and the official DFlash2 images.
- **hasso5703**: the SGLang Qwen3.8 recipe the Flash-Next launch matrix is
  adapted from, and a reference figure in `docs/RESULTS.md`.
- **r0b0tlab**: the SGLang bundle and `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`
  target behind the earlier deployment documented in
  `profiles/qwen3.8-27b-dflash2/docs/`.
- **SparkStation** ([kshetrajna12/sparkstation](https://github.com/kshetrajna12/sparkstation)):
  the orchestration layer in the Qwen3.8-27B write-up.

## License

MIT — see [LICENSE](LICENSE). Third-party material in this repo (see
[Credits](#credits)) stays under its authors' terms.
