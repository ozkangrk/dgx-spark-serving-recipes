# Qwen3.8-27B C3 fast-lane deployment — 2026-09-02

## Verdict

**PROMOTE_TRIAL** for the user's normal C1/C2 usage, with a hard maximum of three concurrent requests. The candidate remains live at `127.0.0.1:8888`; the incumbent is stopped intact with restart disabled.

This is not the tweet's unverified “4-bit drafter” identity. The available pinned public SGLang recipe uses the same RadixArk NVFP4 target and the public **BF16** `z-lab/Qwen3.8-27B-DFlash2` draft. We changed only the serving operating point: DFlash draft 10 → 16, FP8 KV → BF16 KV, cap 4 → 3, and token pool 1,048,576 → 524,288.

## Exact active runtime

- Container: `qwen38-radixark-dflash2-c3-fastlane`
- Image: `sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`
- Image digest: `sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`
- Target revision: `554ebba9b5f1b79dc11246341960360e6ef05ef4`
- Draft revision: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
- Context per request: 262,144; shared token pool: 524,288
- Request cap: 3; Mamba slots: 15; verify CUDA graphs captured for C1/C2/C3
- KV: BF16; DFlash width: 16; thinking disabled by Hermes provider config
- Restart policy: `unless-stopped`

## Frozen API C1/C2/C3 A/B

Same unique-leading-nonce code prompts, 512 output tokens, temperature 0, three groups per concurrency:

| Concurrency | Baseline draft10/FP8 KV | Candidate draft16/BF16 KV | Delta |
|---:|---:|---:|---:|
| C1 aggregate | 55.69 tok/s | **60.08 tok/s** | **+7.9%** |
| C2 aggregate | 83.00 tok/s | **97.33 tok/s** | **+17.3%** |
| C3 aggregate | **136.61 tok/s** | 113.56 tok/s | **−16.9%** |

The candidate is selected for C1/C2 latency/throughput, not fleet throughput. C3 is an admission ceiling, not the preferred operating point.

## Context ladder (candidate; median of four)

| Prompt | Decode | TTFT | Effective prefill |
|---:|---:|---:|---:|
| 419 tokens | 71.50 tok/s | 0.290 s | 1,426 tok/s |
| 8,483 tokens | 69.94 tok/s | 4.261 s | 1,991 tok/s |
| 68,268 tokens | 54.06 tok/s | 44.024 s | 1,551 tok/s |

The local 68K result is 19.3% below the pasted tweet's 67 tok/s at ~65K. It does not reproduce that headline. An exact 68,317-token repeat reduced TTFT from 44.07 s to 0.65 s (68.2×), confirming useful radix-prefix reuse.

## Correctness and real Hermes gates

- Direct arithmetic + pinned structured tool call: **20/20 passed**.
- Real Hermes file/tool tasks: C1 **1/1**, C2 **2/2**, C3 **3/3** passed.
- Group wall: C1 12.3 s; C2 26.3 s; C3 90.3 s.
- After tests: endpoint healthy, queue 0, restart 0, OOM false.
- Open warning: SGLang logs `DFLASH block size mismatch` because the public draft config declares block 8 while runtime verifies 16. Public sweeps use this mode, and exercised gates passed, but it remains a quality-contract warning.

The first Hermes harness run was invalid because profile terminal CWD overrode `--in`; agents wrote `/home/ozkangu/result.json`. That file was removed. The corrected v2 harness included explicit absolute paths and is the only Hermes result cited above.

## Rollback

```bash
docker update --restart=no qwen38-radixark-dflash2-c3-fastlane
docker stop -t 60 qwen38-radixark-dflash2-c3-fastlane
docker update --restart=unless-stopped qwen38-radixark-dflash2-4x128k
docker start qwen38-radixark-dflash2-4x128k
curl -fsS http://127.0.0.1:8888/v1/models
```

## Evidence

- `baseline.json`
- `baseline-c123.json`
- `candidate-c123.json`
- `candidate-context-ladder.json`
- `candidate-quality-smoke.json`
- `candidate-hermes-c123-v2.json`
- `launch_c3_draft16_bf16kv.sh`
