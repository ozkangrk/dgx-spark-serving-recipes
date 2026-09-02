# Local Deployment Report — Qwen3.8-27B DFlash2 SGLang 256K

**Date:** 2026-08-21  
**Status:** Hosted and healthy; strict “≥50 tok/s for every coding workload” gate is **PARTIAL**, not universally met.

## Active service

- Endpoint: `http://127.0.0.1:8888/v1`
- Model: `qwen38-27b-dflash2-256k`
- Context advertised: `262144`
- Engine: SGLang, native DFLASH K8
- Memory fraction: `0.85`
- Max running requests: `4`
- Target/draft KV budget: `1,048,576` tokens
- Restart policy: `unless-stopped`
- Security: loopback-only host publication, no privileged mode, all Linux capabilities dropped, `no-new-privileges`, read-only model mounts

## Provenance

- Source commit: `535e580eea3e777306629746c9ab45153b653a1a`
- Target revision: `36f717a22990e82c54c1d48ee77c491b87825680`
- Draft revision: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
- Draft weight SHA-256: `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`
- Image ID: `sha256:8e9ef8511c7cf943235206894523e8e1fe5bcdb7982a4d97dc5d90ad987020b7`
- Container ID: `6dbccccb03140696737c273acd9d3e06a66d3202fbfec25f25470947946d8c28`
- All four target shard hashes passed `runtime/target-shards.sha256`.

## Correctness and stability gates

- Semantic gate: **PASS 9/9**
- Sanity suite: **PASS 8/8**
  - thinking on/off arithmetic
  - exact-string and word problem
  - code shape
  - structured `get_weather(Tokyo)` tool call
  - 2,777-token long generation
  - deterministic repeat
- Exact 235,000-token NIAH retrieval: **PASS 1/1**
  - code returned exactly: `QWEN38-NIAH-9X4K`
  - HTTP 200
  - elapsed: `386.42 s`
- Stability monitor: **PASS 12/12** over 165 seconds
- OOM: false
- Container restarts: 0

## Local benchmark — standard matrix

Run: `benchmarks/runs/20260821T074456Z_dflash2-256k_mem085/`

All rows completed without request errors. Values below are medians of 3 runs unless marked single-run.

| Feature | Output limit | Decode tok/s | E2E tok/s | TTFT |
|---|---:|---:|---:|---:|
| Code generation | 128 | 46.77 | 42.47 | 0.28 s |
| Code generation | 256 | 49.76 | 47.22 | 0.28 s |
| Code generation | 512 | **55.95** | **54.31** | 0.28 s |
| Code edit/refactor | 128 | **53.31** | 47.72 | 0.28 s |
| Code edit/refactor | 256 | **54.53** | **51.43** | 0.28 s |
| Code edit/refactor | 512 | **53.65** | **50.75** | 0.28 s |
| Agentic coding | 128 | 28.67 | 26.99 | 0.28 s |
| Agentic coding | 256 | 35.57 | 34.25 | 0.28 s |
| Agentic coding | 512 | 34.73 | 34.08 | 0.28 s |
| Tool call | 128 | 116.51 | 27.76 | 0.74 s |

### Concurrency

| Concurrent requests | Aggregate tok/s | Success |
|---:|---:|---:|
| 1 | 50.74 | 1/1 |
| 2 | 71.03 | 2/2 |
| 4 | 136.62 | 4/4 |

## Context scaling

Each row completed successfully with 128 output tokens. Prompt count is the server-reported authoritative count.

| Prompt tokens | TTFT | Decode tok/s | E2E tok/s |
|---:|---:|---:|---:|
| 8,259 | 7.19 s | 55.11 | 13.45 |
| 32,824 | 22.06 s | 56.17 | 5.26 |
| 65,587 | 44.75 s | 51.26 | 2.71 |
| 131,124 | 162.93 s | 50.18 | 0.77 |
| 235,052 | 238.15 s | 45.01 | 0.53 |

Separate exact NIAH at 235,000 prompt tokens passed, but took 386.42 seconds end-to-end because that test constructs and evaluates a full retrieval request.

## Verdict

The deployment satisfies:

- DFlash2 K8: **yes**
- native 256K-class context: **yes**, advertised 262,144 and locally exercised at 235K
- coding around/above 50 tok/s: **yes for code edits and 512-token code generation**
- guaranteed minimum 50 tok/s for every coding/agentic request: **no**

Short code generation was 46.77–49.76 tok/s, and agentic coding was 28.67–35.57 tok/s. At 235K active context, decode was 45.01 tok/s and TTFT was 238 seconds. Therefore the strict universal minimum-50 target is not proven and is explicitly classified **PARTIAL**.

## Post-deployment incident: LLM ServeVerdict quick run

On 2026-08-21 at 08:26 UTC, a later LLM ServeVerdict quick benchmark completed 14/15 requests, then the tool-call request returned HTTP 500 while SGLang terminated with `torch.distributed.DistBackendError` / `nvmlInit_v2() failed: Unknown Error`, SIGQUIT and `kill_process_tree`. Docker restarted the container once; it returned healthy at 08:30:26 UTC with OOM=false and the same model/context identity. The sealed run is `~/Desktop/Inference-Engineering-Studio/data/qwen38-dflash2-256k/quick-20260821T082648Z.json` and is correctly classified `degraded`.

This supersedes the earlier restart=0 observation as a lifetime stability claim. Current promotion verdict is **REJECT / keep opt-in for further root-cause work**, despite strong pre-failure throughput.

## Artifacts

- Hosting instructions: `docs/HOSTING.md`
- Benchmark instructions: `docs/BENCHMARKING.md`
- Semantic result: `runtime/semantic-gate.json`
- Sanity result: `runtime/sanity-suite.json`
- NIAH result: `runtime/niah-235k.json`
- Stability result: `runtime/stability-20260821T075739Z.json`
- Raw benchmarks: `benchmarks/runs/*/raw/`
- Per-run manifests: `benchmarks/runs/*/MANIFEST.sha256`
