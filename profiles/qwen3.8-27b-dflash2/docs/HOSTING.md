# Qwen3.8-27B DFlash2 SGLang 256K — Hosting Runbook

## Frozen bundle

- Source: `r0b0tlab/qwen38-27b-nvfp4-sm121-sglang`
- Source commit: `535e580eea3e777306629746c9ab45153b653a1a`
- Target: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`
- Target revision: `36f717a22990e82c54c1d48ee77c491b87825680`
- Draft: `z-lab/Qwen3.8-27B-DFlash2`
- Draft revision: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
- Local image: `qwen38-r0b0-sglang-dflash2-sm121:0.2.0`
- Profile: DFlash2 K8, native context `262144`, memory fraction `0.85`
- Endpoint: loopback only, `http://127.0.0.1:8888/v1`
- Served model: `qwen38-27b-dflash2-256k`

## Layout

- `models/target/`: exact target checkpoint
- `models/draft/`: exact DFlash2 draft
- `repo/`: detached frozen source
- `runtime/`: hardened launch/stop/readiness scripts and runtime evidence
- `benchmarks/`: reproducible benchmark runner and dated run artifacts
- `docs/`: operational documentation and reports

## Start

```bash
cd ~/Documents/Models/Qwen3.8-27B-DFlash2-SGLang-256K
./runtime/launch.sh
./runtime/wait-ready.sh
```

The launcher starts with Docker restart policy `no`. Only after health, arithmetic, tool-call, benchmark and stability gates pass should persistence be enabled:

```bash
docker update --restart unless-stopped qwen38-r0b0-sglang-dflash2-256k
```

## Stop

```bash
./runtime/stop.sh
```

## Health and model identity

```bash
curl -fsS http://127.0.0.1:8888/health
curl -fsS http://127.0.0.1:8888/v1/models
```

## Security boundary

The service is published only on host loopback (`127.0.0.1:8888`). The container uses a dedicated bridge, read-only model mounts, dropped Linux capabilities, `no-new-privileges`, bounded memory/CPU/PIDs, and offline Hugging Face mode. Do not change the host bind to `0.0.0.0` without an authenticated reverse proxy or verified Tailscale ACL boundary.

## Rollback

The prior container `qwen38-kearuga-sglang-dflash2-candidate` is preserved stopped. To roll back:

```bash
./runtime/stop.sh || true
docker start qwen38-kearuga-sglang-dflash2-candidate
curl -fsS http://127.0.0.1:8888/health
```

Never run both model containers concurrently on the single GB10.
