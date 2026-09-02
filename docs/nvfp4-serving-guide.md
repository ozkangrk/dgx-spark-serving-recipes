# DGX Spark (GB10) NVFP4 Serving Rehberi
**Tarih:** 2026-07-22 · **Donanım:** NVIDIA GB10 Grace-Blackwell, 128GB unified, ARM64/aarch64, SM121, ~273 GB/s bellek bant genişliği
**Kapsam:** `~/Documents/Models/` altındaki 6 NVFP4 model
**Kaynak:** resmi (NVIDIA/vLLM/HF model card) + topluluk (forum/blog/X) — her rakamın yanında etiketli. Rakamlar Temmuz 2026 itibarıyla.

> ⚠️ **Altın kural (senin CLAUDE.md'inle uyumlu):** GB10 tek-stream'de **bellek bant genişliğine bağlı (bandwidth-bound)**, hesaba değil. Yani tek kullanıcı decode hızını asıl belirleyen **aktif param × 2B / 273 GB/s**. MoE modeller (az aktif param) tek-box'ın tatlı noktası; dense modeller (Gemma-31B, Qwen-27B) aynı boyutta çok daha yavaş.

---

## 0. Hızlı Karar Tablosu — hangi model tek Spark'a sığar?

| Model | Tip | Ağırlık | Tek Spark? | Tek-stream decode | Not |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B-NVFP4** | MoE (3B aktif) | 24G | ✅ | ~55–77 → **97–110 (MTP-3)** tok/s | Daily driver. **Senin hibrit-mamba tuzağın geçerli** (aşağı bak) |
| **Qwen3.6-27B-NVFP4** | **Dense** | 22G | ✅ | ~24–33 tok/s | Dense → yavaş; kalite/kararlılık için |
| **Gemma-4-31B-IT-NVFP4** | **Dense** | 31G | ✅ | **~7** → 16–20 (MTP) tok/s | En yavaşı (dense + head_dim 256 KV) |
| **Laguna-S-2.1-NVFP4** | MoE (8.5B aktif) | 67G | ✅ | 15 (metin) / **22–24 (kod)** tok/s | Coding uzmanı; DFlash spec-decode |
| **Nemotron-3-Super-120B-A12B-NVFP4** | MoE (12B aktif) | 75G | ✅ | **22.7–23.7 tok/s** (resmi) | Reasoning tavanı; 128K test |
| **DeepSeek-V4-Flash-NVFP4** | MoE (13B aktif) | 157G | ❌ **>128GB** | ~50–67 tok/s (2× Spark) | **Tek kutuya SIĞMAZ → 2× Spark TP=2 şart** |

**🔴 Kritik:** DeepSeek-V4-Flash ağırlığı (168GB) tek Spark'ın 128GB'ını aşıyor. Tek kutuda serve **edilemez**; doğrulanmış tüm reçeteler 2× DGX Spark (TP=2, RoCE/IB fabric) üzerinde. İndirdik ama tek box'ta çalıştıramazsın — ya ikinci Spark ya da daha küçük bir modele düş.

---

## 1. Serving Stack Seçimi

| Engine | Ne zaman | NVFP4 durumu |
|---|---|---|
| **vLLM** ⭐ | Varsayılan. Concurrency, OpenAI API, PagedAttention, KV knob'ları | En olgun NVFP4 yolu. **`vllm/vllm-openai:cu130-nightly` (aarch64) şart** — stok NGC image NVFP4/Blackwell MoE backend'i desteklemez |
| **SGLang** | Paylaşımlı prefix / agent döngüleri / kısıtlı JSON (RadixAttention) | sm_121-native MTP fork'ları var; prefix örtüşmesi yoksa marjinal |
| **TensorRT-LLM** | ❌ Spark'ta **önerilmez** | Datacenter Blackwell'e göre; GB10 SM12x farklı. "Deneysel" say |
| **llama.cpp / Ollama** | Basitlik / tek-box fallback | NVFP4 tensor-core yolu DEĞİL (GGUF Q4) ama bazen tek çalışan yol |

**Quantization flag'i (checkpoint kaynağına göre):**
- `nvidia/*-NVFP4` (ModelOpt): **`--quantization modelopt`** (veya `modelopt_fp4`)
- `RedHatAI/*`, `unsloth/*` (llm-compressor): **`--quantization compressed-tensors`** *veya flag'i hiç verme* → vLLM `hf_quant_config.json`'dan otomatik algılar
- `poolside/Laguna`: **flag verme** — otomatik algılanır
- Genel kural: pre-quantized NVFP4'te çoğu zaman `--quantization`'ı boş bırak, vLLM algılar.

Kaynaklar: [vLLM DGX Spark blog](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) · [Medium: 4 engine](https://medium.com/@michael.hannecke/four-inference-engines-one-box-when-to-use-which-on-the-dgx-spark-6b32a53db768) · [vLLM ModelOpt docs](https://docs.vllm.ai/en/stable/features/quantization/modelopt/)

---

## 2. SM121 / GB10 Tuzakları (deploy ÖNCESİ oku)

1. **MoE backend flag'i ÇELİŞKİLİ** (en tartışmalı ayar, sürüme bağlı — kendi vLLM'inde ampirik test et):
   - `--moe-backend flashinfer_b12x` + `export CUTE_DSL_ARCH=sm_121a` → Unsloth'un önerdiği hızlı yol (vLLM ≥0.25 kernel şart; bazı GB10'larda kernel yok → Marlin'e düşüyor)
   - `--moe-backend flashinfer_cutlass` → RedHatAI/stevescargall cu130-nightly yolu
   - `--moe-backend marlin` + `VLLM_USE_FLASHINFER_MOE_FP4=0` → "CUTLASS FP4 SM121'de sessizce yanlış" kampı (Unsloth Marlin'i ~2.5x yavaş sayıyor)
   - **Doğrulama:** throughput'a güvenmeden önce çıktının bozuk (`!!!!!`) olmadığını kontrol et.
2. **Mamba-2 / hibrit (GatedDeltaNet) katmanları SM121'de bozuk** → "illegal instruction" / 9–21 tok/s'e düşme. **Saf MoE / saf transformer tercih et.**
3. **Prefix caching:** saf `qwen3_moe`'da çalışır; **hibrit Mamba/GatedDeltaNet'te bozuk (0% hit / hang)** — senin CLAUDE.md'deki Qwen3.6-35B-A3B notunla birebir.
4. **`--enforce-eager`:** hibrit-mamba modelde ŞART (correctness); saf-MoE NVFP4'te ~%55 throughput cezası. **Mimariye göre seç.**
5. **İlk istek ~25s JIT derlemesi** → prod öncesi **warmup** yap (senin MEMORY kuralın).
6. Driver 580.x'te kal (590.x CUDAGraph deadlock — anekdot); `--load-format fastsafetensors` NGC'de ImportError.

---

## 3. Global Optimizasyon Knob'ları

- **KV bellek:** Tüm Spark reçeteleri `--gpu-memory-utilization` (0.85–0.87; **~0.90 tavan**, OS page-cache ile yarışıyor) kullanıyor. Senin `--kv-cache-memory-bytes` yaklaşımın daha temiz ama hiçbir Spark reçetesinde görülmedi — küçük modellerde util-fraction savurgan (senin kuralın: 8–12B'de 0.35–0.45). **`--kv-cache-dtype fp8` KV'yi ~yarıya indirir** (uzun bağlam/concurrency için).
- **`--max-num-seqs`:** İki okul → **4** (tek kullanıcı/interaktif; ~273 GB/s'de 4 decode stream üstünde bant vergisi kazancı yer) vs **32** (aggregate throughput, near-linear ölçekleniyor). İş yüküne göre.
- **MTP (Multi-Token Prediction / spec-decode):** ETKİLİ. `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. 35B-A3B'de single'ı 77→108 tok/s'e çıkarıyor, accept %85–93.
- **Bağlam:** NVFP4 checkpoint'ler genelde 262144 (256K) destekler; headroom için 131072 (128K) koş.

---

## 4. Model Bazında Reçeteler + Performans

### 4.1 Qwen3.6-35B-A3B-NVFP4 ⭐ (daily driver)
> ⚠️ **DİKKAT — senin özel durumun:** İndirdiğimiz `RedHatAI/Qwen3.6-35B-A3B-NVFP4`'ün mimarisi `Qwen3_5MoeForConditionalGeneration` (multimodal + **hibrit mamba**). Topluluk reçeteleri bunu "saf MoE" varsayıp prefix-caching + CUDAGraph AÇIK koşuyor. **Senin CLAUDE.md'deki kanıtlı config kazanır:** ağır/concurrent yükte `--no-enable-prefix-caching --enforce-eager` + warmup + taze server. Aşağıdaki generic reçeteyi tek-istek/interaktifte kullan, agentic batch'te KENDİ config'ine dön.

**Generic hızlı reçete (NVIDIA variant, tek-istek/interaktif):**
```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 8000 \
  --quantization modelopt --kv-cache-dtype fp8 \
  --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```
**Performans (GB10):**
- NVIDIA NVFP4: single **76.7** → MTP-2 **98.0** → MTP-3 **108.3 tok/s**; 8-paralel **298.7 tok/s** — [dev.classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-qwen3-6-35b-a3b-nvfp4-new-champion/) *(topluluk)*
- RedHatAI: single **55.9** (TTFT 166ms), c=32 aggregate **433.4 tok/s** (~7.8× ölçek), MTP accept %85–93 — [stevescargall](https://stevescargall.com/blog/2026/04/vllm-recipe-redhatai/qwen3.6-35b-a3b-nvfp4-on-dgx-spark/) *(vendor recipe)*
- Unsloth NVFP4: NVIDIA'nın %12–20 gerisinde (single 67.5, MTP-3 89.9) — aynı kaynak
- **Konsensüs: ~55–77 single, ~97–110 MTP-3, ~300–430 aggregate (8–32 concurrent).** NVIDIA modelopt varyantı en hızlısı.

### 4.2 Qwen3.6-27B-NVFP4 (DENSE)
```bash
export CUTE_DSL_ARCH=sm_121a
vllm serve unsloth/Qwen3.6-27B-NVFP4 --moe-backend flashinfer_b12x \
  --max-model-len 262144 --gpu-memory-utilization 0.78 \
  --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill
```
**Performans:**
- Doğruluk: NVFP4 MMLU **84.46%** vs FP16 84.51% (~kayıpsız) — [NVIDIA forum](https://forums.developer.nvidia.com/t/benchmark-report-qwen-qwen3-6-27b-vs-nvidia-qwen3-6-27b-nvfp4/375094) *(resmi-forum)*
- Single-stream decode: **~24–33 tok/s** (dense → 35B-A3B MoE'den yavaş); concurrency'de ~80–98, bir toplulukta c=16 aggregate ~144 tok/s — aynı forum + [unsloth disc](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4/discussions/11) *(anekdot)*
- ⚠️ Unsloth'un "125 tok/s / 6863 total" rakamı **B200, GB10 DEĞİL** — atfetme.

### 4.3 Gemma-4-31B-IT-NVFP4 (DENSE, en yavaş)
```bash
vllm serve nvidia/Gemma-4-31B-IT-NVFP4 --host 0.0.0.0 --port 8000 \
  --quantization modelopt --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 --max-model-len 32768
# Küçük ağırlık (~16-21GB): interaktifte util 0.35-0.45 yeter (senin kuralın); 256K bağlam istersen 0.85.
```
**Performans:**
- Decode **~7.0 tok/s** (bandwidth-walled, dense; 3 koşuda ±0) — [ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-gemma4-31b-dense-7-toks) *(anekdot)*
- **MTP ile: 16.3–19.8 tok/s** (~5.25×) `--speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'` — [note.com](https://note.com/gb10_tsurumitsu/n/n4b0019c97d1c?hl=en) *(anekdot)*
- Doğruluk (resmi): GPQA 85.35 vs 85.80, AIME 87.6 vs 87.9 (~kayıpsız, 256K korundu) — [HF card](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
- 💡 Throughput önemliyse **Gemma-4-26B-A4B-NVFP4 (MoE) ~52 tok/s** — dense-31B yerine düşün.

### 4.4 Laguna-S-2.1-NVFP4 (coding uzmanı, MoE 8.5B aktif)
```bash
vllm serve poolside/Laguna-S-2.1-NVFP4 --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --speculative-config '{"model":"poolside/Laguna-S-2.1-DFlash-NVFP4","num_speculative_tokens":7}' \
  --enable-auto-tool-choice --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 \
  --max-num-seqs 4 --max-model-len 262144 --gpu-memory-utilization 0.85
# vLLM >=0.25 şart. Quant flag verme (auto). DFlash draft modelini de indir (poolside/Laguna-S-2.1-DFlash-NVFP4).
```
**⚠️ Tuzaklar:** `--max-num-seqs` default 256 **DFlash ile crash** → **4 ile başla** (senin worker-flood geçmişinle uyumlu). `num_speculative_tokens` kartta 15, forumda 7 daha iyi. DFlash accept Spark'ta **zayıf** raporlandı (~2.29 mean, %18.5) — spec kazancı iş-yükü bağımlı.
**Performans:**
- Resmi kart (GB10): prefill 600–800, decode **15 (metin) / 22–24 (kod) tok/s**, DFlash 2.9–3.1 tok/step — [HF card](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4)
- Forum (TP=2): prefill 5488'e kadar, decode **20–47 tok/s**, ama DFlash accept "abysmal" — [forum](https://forums.developer.nvidia.com/t/laguna-s-2-1-config-benchmarks/377663) *(anekdot)*
- vLLM recipe: https://recipes.vllm.ai/poolside/Laguna-S-2.1

### 4.5 Nemotron-3-Super-120B-A12B-NVFP4 (reasoning tavanı, MoE 12B aktif)
```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --served-model-name nemotron-3-super \
  --trust-remote-code --max-model-len 131072 \
  --gpu-memory-utilization 0.75 --max-num-seqs 4 \
  --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
# Image: vllm/vllm-openai:cu130-nightly (aarch64/sm_121). util 0.75: 75GB ağırlık+KV; 0.85 savurgan.
```
**Performans (resmi vLLM blog, tek Spark, NVFP4 native):**
- **Decode 22.7–23.7 tok/s** (tutarlı) — senin duyduğun "22–24" doğrulandı
- Prefill 140 (kısa) → **1,884 tok/s** (7K token uzun prompt); 128K test, tek-user KV util <%5 — [vLLM blog](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) *(resmi)*
- ⚠️ Anekdot: stok NGC image aarch64'te NVFP4'ü reddedebiliyor ("No sm_121") → bazıları GGUF Q4'e (19–22 tok/s) düşmüş. **cu130-nightly aarch64 image kullan.**

### 4.6 DeepSeek-V4-Flash-NVFP4 (🔴 tek Spark'a SIĞMAZ)
```bash
# SADECE 2× DGX Spark, TP=2 (RoCE/InfiniBand fabric). Tek box İMKANSIZ (168GB > 128GB).
vllm serve fraserprice/DeepSeek-V4-Flash-DSpark \
  --tensor-parallel-size 2 --distributed-executor-backend mp --nnodes 2 \
  --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
  --gpu-memory-utilization 0.80 --max-model-len 1048576 \
  --max-num-seqs 6 --max-cudagraph-capture-size 6 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3}' \
  --async-scheduling --enable-chunked-prefill
# Env: VLLM_USE_B12X_MOE=1 (50-60 tok/s anahtarı), VLLM_USE_FLASHINFER_SAMPLER=1
```
**Performans (2× Spark, anekdot):** single decode ~50–67 tok/s; 1M bağlamda 50–65 (NVFP4 KV havuzu ~2.04M token'a katlanıyor); C12@1M aggregate ~230; C16@200K ~315 tok/s. ⚠️ Tam-dolu 1M'de bazıları <10–20 tok/s'e düşüyor. — [1M NVFP4-KV repo](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark) · [forum](https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846)

---

## 5. Senin Kurulumun İçin Pratik Öneriler

- **Daily-driver coding:** Qwen3.6-35B-A3B-NVFP4 + MTP-3. Ama SENİN hibrit-mamba checkpoint'inde ağır agentic yükte CLAUDE.md config'in (`--no-enable-prefix-caching --enforce-eager` + warmup + taze server) kazanır — generic prefix-caching reçetesine kanma.
- **Coding alternatifi denemesi:** Laguna-S-2.1 (kod decode 22–24, DFlash). `--max-num-seqs 4` ile başla.
- **Reasoning ağır iş:** Nemotron-120B (23 tok/s ama 12B-aktif kalite). util 0.75.
- **Hız değil kalite/multilingual (TR):** Gemma-31B yavaş (7 tok/s) → MTP şart veya 26B-A4B MoE'ye geç.
- **DeepSeek-V4-Flash:** 2. Spark almadan raf modeli. Tek box'ta serve planı yapma.
- **Genel:** hepsi bandwidth-bound → MoE'ler (35B-A3B, Laguna, Nemotron) tek-box'ın verimli noktası; dense'ler (27B, Gemma-31B) MTP olmadan yavaş.

**Tekrarlayan doğrulama gereken (ampirik test et):** MoE-backend flag (b12x/cutlass/marlin), Laguna max-num-seqs (4 vs 32) + spec-tokens (7 vs 15), enforce-eager (mimariye göre).
