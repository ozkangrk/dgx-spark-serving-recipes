# DGX Spark Serving Recipes

Measured, reproducible serving recipes for large open-weight models on a
single NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X, aarch64, SM121) —
and notes on where two-node clusters become necessary.

Every profile under [`profiles/`](profiles/) is a **working deployment**:
launch scripts, health/wait gates, quality gates, and the benchmark data
that justifies the chosen operating point. Numbers are first-party
measurements on the hardware listed in each profile; nothing is quoted
from tweets or model cards without an independent local reproduction.

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

See [`docs/nvfp4-serving-guide.md`](docs/nvfp4-serving-guide.md) for the
full per-model decision table (which model fits one Spark, expected
decode rates, quantization notes).

## Profiles

| Profile | Engine | Quant | Context | Why it exists |
|---|---|---|---|---|
| [`qwen3.8-27b-dflash2`](profiles/qwen3.8-27b-dflash2/) | SGLang + DFlash2 spec-decode | NVFP4 target + BF16 draft | 256K | Flagship: fastest measured Qwen3.8-27B setup on GB10 (78.6 tok/s single-stream, 480.7 tok/s @16 streams, HumanEval 97.0%). Includes frozen A/B evidence for draft-width/KV-dtype operating points. |
| [`deepseek-v4-flash-exl3`](profiles/deepseek-v4-flash-exl3/) | ExLlamaV3 (EXL3) | NVFP4 KV records | 384K single request | Deep-context: 439K-token pool on ONE Spark, auto-download launcher, TP1 coalesced checkpoint. The 157 GB model that does not fit one box, made to fit. |
| [`qwen3.8-flash-next`](profiles/qwen3.8-flash-next/) | SGLang (derived image) | NVFP4 | 64K/128K/256K variants | Launch matrix for PLE/KDA attention variants incl. FP8-KV trade study; boot-interrupt-safe PLE materialization. |

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

- All figures in `profiles/*/bench-results` and `profiles/*/evidence`
  are first-party runs on DGX Spark hardware by the repo author, with
  pinned model revisions and container digests recorded in each profile.
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

## License

MIT — see [LICENSE](LICENSE).
