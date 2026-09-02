# Benchmarking Guide

The benchmark suite creates one immutable directory per UTC timestamp and profile/config:

```text
benchmarks/runs/YYYYMMDDTHHMMSSZ_dflash2-256k_mem085/
  config.json
  summary.json
  MANIFEST.sha256
  raw/*.json
```

Every request retains raw SSE events, usage token counts, TTFT, end-to-end speed and post-first-token decode speed. Requests use unique nonces unless a cache experiment explicitly says otherwise.

## Standard matrix

```bash
cd ~/Documents/Models/Qwen3.8-27B-DFlash2-SGLang-256K
python3 benchmarks/run_matrix.py \
  --output-limits 128,256,512 \
  --reps 3 \
  --context-targets 8192,32768,65536 \
  --concurrency 1,2,4
```

Covered features:

- fresh code generation
- code editing/refactoring
- agentic coding response
- structured tool calling
- increasing prompt/context sizes
- output limits 128/256/512
- concurrency 1/2/4 with common-wall aggregate throughput

## Custom run examples

Longer output limits:

```bash
python3 benchmarks/run_matrix.py --output-limits 256,512,1024 --reps 5
```

Context-focused run:

```bash
python3 benchmarks/run_matrix.py --output-limits 128 --reps 2 \
  --context-targets 32768,65536,131072,235000 --concurrency 1
```

The context target is approximate input construction; use each raw response's `prompt_tokens` as the authoritative token count.

## Interpretation

- `decode_tok_s`: completion tokens divided by time from first visible token to stream end.
- `e2e_tok_s`: completion tokens divided by full request wall time, including prefill/TTFT.
- `aggregate_tok_s`: sum of completion tokens divided by a shared concurrency-group wall interval.
- Never compare aggregate throughput with single-stream decode as if they were the same metric.
- A configured 262K maximum does not prove 50 tok/s at a fully occupied 262K prompt. Long-context TTFT/prefill and decode must be reported separately.
- Retain failed requests. Missing usage or a failed concurrent stream makes the run fail closed.

## Exact long-context retrieval gate

The local Hermes Python environment has an incompatible `regex` package for this checkout's Transformers build. Use the verified `~/train-env` interpreter:

```bash
~/train-env/bin/python repo/scripts/run_niah.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model qwen38-27b-dflash2-256k \
  --tokenizer models/target \
  --depths 235000 --max-tokens 64 --needle-fraction 0.5 \
  --protocol-label LOCAL_DFLASH2_235K \
  --output runtime/niah-235k.json
```

## Reproducibility

Before citing a run, verify:

```bash
cd benchmarks/runs/<RUN_ID>
sha256sum -c MANIFEST.sha256
```

Also preserve container image ID, source/model/draft revisions and full launch command in the associated report.
