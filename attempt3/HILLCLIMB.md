# Overnight hill climb log (2026-09-23)

Plan: `C:\Users\amazi\.claude\plans\i-want-you-to-enumerated-candle.md`.
- **Goal:** a better quantized fast mode (closer to FP32 at the same speed), plus speed wins in both modes.
- Every attempt is listed, accepted or not.
- **Fidelity metric:** `falcon-ocr-attempt agree`. Each page is teacher-forced with the FP32 calibration tokens (`artifacts/phase4/checks/calibration-reference.json`), and we count the steps where the profile's own greedy choice differs.
  - Reported as flips per 1,000 forced steps.
  - Screening set: `artifacts/phase4/agree/screen-pages.txt`, 24 pages (3 per category), at most 768 steps each.
- **Sanity:** FP32 with the fast exp has 0 flips in 512 steps. W8 + Q8 has 4.

## Weight error of W8 variants (offline, mean relative Frobenius error over the 88 body matrices)

| Variant | Mean | Max |
|---|---:|---:|
| RTN G64 (current `body.w8`) | 0.600% | 0.659% |
| RTN G64, MSE-optimal clip | 0.592% | 0.652% |
| RTN G32 | 0.538% | 0.568% |

This is close to the noise floor of 8-bit rounding of bell-shaped weights (about 0.55% at G64). So there are no big within-group outliers, and clipping and group size can only give small gains. Output-error methods (GPTQ, mixed precision) are the larger levers.

## Fidelity decomposition (screening set, 768 steps)

| Arm | Weights | KV | Flips / 1,000 steps |
|---|---|---|---:|
| w8q8 | W8 RTN G64 | Q8 | 8.62 (123 / 14,263) |
| w8f32 | W8 RTN G64 | FP32 | 8.62 (123 / 14,263) |
| kvq8 | FP32 | Q8 | 0.77 (11 / 14,263) |

- W8 + Q8 and W8 + FP32 KV have the same total, but different flips on 6 of 24 pages. Q8 KV adds and removes a few flips in about equal numbers.
- **The W8 weights cause fast mode's drift.** KV alone is about 9% of it.
- The BF16-KV arms were stopped as unneeded.

## Captured Grams (`capture-gram`, 12 calibration pages disjoint from the screening set, 82,836 rows per matrix)

Projection inputs are extremely anisotropic. The largest diagonal entry over the median:

| Matrix | Max / median |
|---|---:|
| layer 10 W2 (squared-ReLU outputs) | 614,663 |
| layer 21 W2 | 2,435 |
| layer 10 WO | 273 |
| layer 0 QKV | 41 |
| layer 10 W13 | 30 |

RTN error in a few dominant input channels therefore dominates the output error. Activation-aware quantization (GPTQ, weighted clipping) targets exactly that.

## Speed attempts

| # | Change | Evidence | Decision |
|---|---|---|---|
| S1 | `--threads 32` against 16 | Journal A/B at 01:24, 3 rounds, all arms token-identical (`artifacts/phase4/ab-s1s3`). Fast mode: 16.29 → 20.85 s (prefill 6.20 → **4.82 s**, decode 8.79 → 14.00 ms/tok). Exact mode: 42.80 → 56.82 s (prefill 6.01 → 4.65 s, decode 32.19 → 45.70 ms/tok). | **Rejected as a single setting.** SMT helps compute-bound prefill (−22%) but hurts memory-bound decode (+50–60%). Follow-up: 32-thread prefill with a 16-thread decode team. |
| S3 | Exact mode on the split FP32 cache (`split-f32`, bitwise equal to compact) | Same run: 42.80 → **39.64 s** (decode 32.19 → 29.31 ms/tok), tokens identical. | **Accepted: −7.4%.** |
| note | Host quietness | These quiet-host numbers are about 15–20% faster than the 20:48 run. Earlier runs overlapped other load, so compare only within one interleaved run. | |
| S2 | gemm AVX-512 kernels (`--features gemm-avx512`) | Journal A/B from 20:48 on a quiet host, 3 rounds (`artifacts/phase4/ab-stage5`). Prefill 7.62 vs 7.54 s without the fast exp, and 6.59 vs 6.60 s with it. Projections about −0.1 s (W13 1.06 → 0.99 s, W2 0.54 → 0.48 s), attention +0.05 s. | **Rejected**: under 2%. Zen 4 runs AVX-512 as two 256-bit halves. Worth re-testing on an Intel AVX-512 host. |

## Offline output-error proxy (`attempt3/w8_output_error.py`)

Relative output error is sqrt(tr(dW G dWᵀ) / tr(W G Wᵀ)), using the captured input Grams. These are in-sample for GPTQ, so the screening-set agreement runs decide.

| Overlay | Mean | Max | Worst matrices |
|---|---:|---:|---|
| RTN G64 (current) | 0.646% | 2.160% | W13 of layers 0–11 (1–2%) |
| RTN G32 | 0.582% | 1.938% | the same |
| MSE-clip G32 | 0.578% | 1.944% | the same |
| **GPTQ G64** | **0.139%** | **0.380%** | W2 of layers 9–19 |

GPTQ removes the early-layer W13 hot spot. It reduces the proxy 4.6× on average and 5.7× on the worst matrix, at identical storage (W8G64, the same kernels).

## Weight variants: teacher-forced agreement (screening set, fast-mode config W8 + Q8 KV, 768 steps)

| Overlay | Flips / 1,000 steps | vs RTN G64 | Decision |
|---|---:|---:|---|
| RTN G64 (current) | 8.62 (123) | | baseline |
| **GPTQ G64** (12-page Gram, damp 0.01) | **3.44 (49)** | **−60%** | **accepted as candidate**; same bytes and kernels |
| RTN G32 | 9.47 (135) | +10% | **rejected**: within counting noise of G64 (±11 flips), +6% weight bytes. The 10% proxy gain did not carry over. |
| GPTQ G32 | 2.66 (38) | −69% (−22% vs GPTQ G64) | candidate. About +2% total decode bytes, just outside counting noise (±7) of GPTQ G64. |
| **GPTQ G64 act-order** (static groups) | **2.73 (39)** | **−68%** (−20% vs GPTQ G64) | **preferred G64 candidate**: the same flips as G32 with no extra bytes. Proxy 0.122%. |
| GPTQ G32 act-order | 2.73 (39) | −68% | **rejected**: no gain over G64 act-order. Proxy 0.110%. |

**Weights winner: GPTQ G64 act-order** (`artifacts/phase4/w8/gptq64ao.safetensors`).

By category (flips, RTN → GPTQ):

| Category | RTN | GPTQ |
|---|---:|---:|
| degraded | 52 | 14 |
| tables | 23 | 8 |
| handwriting | 14 | 8 |
| multi_column | 11 | 6 |
| formulas | 9 | 4 |
| ordinary | 8 | 4 |
| tiny_text | 6 | 4 |
| full-page | 0 | 1 |

Pages: 15 better, 4 worse, 5 the same. The capture pages are disjoint from the screening set.

## KV variants (screening set, 768 steps)

| Arm | Flips / 1,000 steps | vs Q8 | Decision |
|---|---:|---:|---|
| GPTQ G64 act-order weights + Q8 KV | 2.73 (39) | | |
| GPTQ G64 act-order weights + **per-channel key scales** (`w8-body-kv-q8-kc`) | 2.66 (38) | none | Flip-neutral. Adopt only if faster (about 6% fewer KV bytes). |
| FP32 weights + Q8 KV | 0.77 (11) | | |
| FP32 weights + per-channel key scales (`kv-q8-kc`) | 0.77 (11) | none | Key-channel outliers do not drive the KV flips. |

| # | Change | Evidence | Decision |
|---|---|---|---|
| S5 | Hybrid threads: 32-thread prefill pool, 16-thread decode team (`--threads 32 --decode-threads 16`) | Journal A/B at 01:48, 3 rounds, tokens identical (`artifacts/phase4/ab-hybrid`). Fast (GPTQ weights) 16.38 → **14.85 s** (prefill 6.22 → 4.74 s, decode unchanged at 8.8 ms/tok). Exact split 39.77 → **38.35 s**. | **Accepted: −9.3% fast, −3.6% exact.** Main-binary defaults: `--threads` = logical CPUs, `--decode-threads` = physical cores (`src/cpu.rs`, which detects 32/16 here). |
| S6 | Per-channel key scales (`w8-body-kv-q8-kc`) for speed | Journal A/B, 4 rounds (`artifacts/phase4/ab-kc`): 14.97 → 14.79 s (−1.2%; attention 4.32 → 4.15 ms). | **Not adopted**: under 2% and flip-neutral. It stays available as a profile. |

## Confirmation: free-running on the 55 calibration pages outside the Gram capture set (`artifacts/phase4/checks/eval55-*.json`)

| Fast mode | Identical pages | EOS pages identical | Token edits (EOS pages) | Stop changes | CER vs truth on the 45 both-EOS pages |
|---|---:|---:|---:|---|---|
| RTN W8 + Q8 | 15/55 | 15/46 | 4,282 | 1 (df13 EOS → loop) | 18.40% → 17.65% |
| **GPTQ G64 act-order + Q8** | **25/55** | **24/46** | **2,996 (−30%)** | 2 (df13 EOS → loop; 3ae0 loop → EOS, CER 2.88 → 0.13) | 18.40% → 18.38% |
| FP32 weights + Q8 (reference point) | 39/55 | 32/46 | 286 | 0 | |

- With `--stop-repetition`, the simulated stop fires on 8 GPTQ pages, none of them EOS pages (so df13 is caught). Tokens drop 34.7%; micro CER vs truth over all 55 pages goes 62.6% → 40.2%.
- The 55 pages took 1,005 s with hybrid threads on a quiet host.

**Decision:** GPTQ G64 act-order is the fast-mode weights. It is installed as `artifacts/model/w8-gptq.safetensors` (sha256 `60808bbf…`), which `--mode fast` loads by default.
| S7 | Multi-page batching in fast mode (`--batch-size`), 8 EOS calibration pages of 316–631 tokens | `artifacts/phase4/checks/batch-b*.json`: batch 1 → 390 pages/h, 2 → 419, 4 → 439, 8 → 444. Tokens identical to batch 1 on 8/8 pages at every size. | Works and is exact, but only +14%. On these pages prefill (about 4.8 s each, sequential) is about half the wall time, and batching shares only decode weight reads. Next lever for corpus throughput: prefill speed, or overlapping prefill with decode. |
| S8 | AVX-512F prefill QK/PV tiles (`kernels/prefill64.rs` `wide`), exact | First version: 16 queries × 24-key blocks was 2.4× *slower* (attention 6.45 s). Key rows are 4 KB apart, so 24 rows in one L1 set thrash its 8 ways. Fixed: 32 queries (two zmm) × 8-key blocks, and PV as 6 rows × the whole 64-wide row. Journal A/B, 2 rounds (`artifacts/phase4/ab-wide`): attention 2.64 → **2.41 s** (−9%), prefill 4.74 → 4.61 s. Tokens identical; smoke trace byte-exact under `auto` too. | **Kept, but not a hill-climb win here**: under 1% end to end on Zen 4, which runs AVX-512 as two 256-bit halves. Enabled for backend `auto`/`avx512` on AVX-512F hosts because outputs are bitwise identical. Intel hosts with full-width AVX-512 should gain more (unmeasured). |
| S9 | 4-bit screened head (26.7 MB instead of 53.5 MB), still exact | Journal A/B, 3 rounds (`artifacts/phase4/ab-i4`). The 4-bit screen fell back to the full FP32 head on **1,140 of 1,140 steps**: the candidate count exceeded 1,024 every time, versus 1.01 candidates per step for the 8-bit screen. Decode 8.91 → 12.29 ms/tok. Tokens identical, because the fallback is exact. | **Rejected and reverted.** The rigorous worst-case bound, half a quantization step times the full `|x|` mass, is about 18× wider at 4 bits and covers thousands of rows even when the true margin is clear. |

# Phase 5: kernel and data-type work, anchored on FP32 (2026-09-23)

Order agreed with the user: thread autotuning, INT16 exact mode, speculative decoding, own prefill GEMM, prefill attention tiles. Fidelity anchor: FP32.

## Storage types against FP32 (GPU harness, 55 calibration pages outside the Gram set, 24,262 teacher-forced steps, FP32 KV)

| Body weights | Bytes/weight | KL vs FP32 | Flips |
|---|---:|---:|---:|
| INT16, absmax scale per 64 inputs | 2.06 | 1.7e-7 | 1 |
| FP16 | 2.00 | 7.1e-6 | 11 |
| BF16 | 2.00 | 2.2e-4 | 51 |
| GPTQ INT8 G64 act-order (fast mode) | 1.06 | 2.8e-4 | 63 |
| BF16-source GPTQ + BF16-rounded weights | 1.06 | 6.8e-4 | 97 |
| INT8 round-to-nearest G64 | 1.06 | 3.4e-3 | 168 |

Offline activation-weighted output error (`attempt3/dtype_proxy.py`) agrees: INT16 0.0025%, FP16 0.024%, BF16 0.19%, GPTQ INT8 0.12%, INT4 10–12%. BF16 is dominated as a storage type; 4-bit is out for this model.

| # | Change | Evidence | Decision |
|---|---|---|---|
| T1 | Decode thread count sweep, fast mode | Journal, 2 rounds (`artifacts/phase4/ab-threads`): 8 → 9.70, 10 → 9.21, 12 → 9.35, 14 → 9.47, 16 → 9.43, 20 → 9.85 ms/token (a noisier host period than other runs). Flat from 10 to 16. | Informs T2. |
| T2 | `--decode-threads auto` (`src/tune.rs`): time team sizes {P/2, 3P/4, P, (P+L)/2} round-robin on the first decode steps, keep the smallest within 2% of the fastest | Journal A/B, 3 rounds (`artifacts/phase4/ab-tune`): auto chose 12 each time (medians 8:8.6, 12:8.3, 16:8.7, 24:8.9 ms/step); decode 8.81 → 8.54 ms/token, total 14.55 → 14.36 s. Tokens identical. | **Accepted as the main binary's default** (−3% decode here). Its real purpose is other hosts (SMT, efficiency cores, higher bandwidth). |
| T3 | 16-bit body weights (`w16-body`: absmax scale per 64 inputs, quantized at load; decode kernel = FP32 math on the dequantized weights, bitwise, tested) | Journal A/B, 3 rounds (`artifacts/phase4/ab-w16`): exact 36.56 → **30.18 s** (decode 28.01 → 22.35 ms/token), tokens identical to FP32. Load +0.5 s. | Kept as a profile; superseded by T5. |
| T4 | FP32 cache split into 4 position chunks (`FALCON_OCR_SPLIT_CHUNKS=4`) | `artifacts/phase4/ab-w16c4`: w16 22.34 → 22.25 ms/token, exact unchanged. The single-chunk FP32 scan already saturates bandwidth with 8 threads. | Rejected. |
| T5 | 16-bit KV cache (`SplitQ16`: the Q8 record layout with [-32767, 32767] codes, scale per 32 values) + 16-bit weights = `w16-body-kv-q16`, main binary `--mode near-exact` | Journal A/B, 3 rounds (`artifacts/phase4/ab-w16kv`): exact 38.00 → **21.94 s** (−42%; decode 29.16 → 15.16 ms/token), tokens identical to FP32. Teacher-forced on the 55 calibration pages against the FP32 anchor (CPU kernels, 24,262 steps): **KL 9.2e-8, 1 flip**. | **Accepted** as `--mode near-exact`. `--mode exact` stays bitwise FP32. |
| T6 | Speculative decoding, n-gram drafts (`--speculate 4`), verified in one multi-row step (bitwise single-row attention per row, multi-row screened head, cache truncation) | Tokens identical on 12 fast and near-exact runs. Clean A/B, 5 pages, 2 rounds (`artifacts/phase4/ab-spec`): loop page 1306da 42.4 → **23.9 s** fast, 71.1 → **28.6 s** near-exact; short page 4a75 −5% / −17%; but normal text **slower** in fast mode (journal 14.4 → 15.6 s, ebac 24.0 → 27.1 s) and neutral to +5% in near-exact. A 5-row verify step costs 23.3 ms vs 8.8 ms single (fast), 26.6 vs 15.1 ms (near-exact): decode attention is compute-heavy per row, so extra rows cost 2.9–3.6 ms each, while normal text accepts only 10–20% of drafts (loops 99%). | Kept, off by default; adaptive policy in T7. |
| T7 | Adaptive drafting (`draft::DraftPolicy`): draft only while the running acceptance rate beats the measured break-even `extra_ms / single_ms`, with a probe every 16 steps | 5 pages, 2 rounds (`artifacts/phase4/ab-spec2`), tokens identical in every round. Fast: journal 14.61 → 14.59 s, ebac 24.37 → 24.54, df13 44.94 → **34.90**, 4a75 9.31 → 8.89, loop 1306da 44.05 → **25.88**. Near-exact: 21.78 → 21.33, 39.35 → 38.34, 34.15 → 32.85, 12.55 → **10.89**, 72.14 → **31.12**. Normal text drafts on 2–5% of steps. | **Accepted**: `--speculate 4` is the main binary's default. |
| T8 | Panel GEMM for quantized prefill projections (`kernels/panel_gemm.rs`): weights dequantized once per call into 16-column FP32 panels, AVX2 6x16 register tile over the whole reduction, squared-ReLU gate fused into the W13 epilogue | Probe at 6,544 rows vs the `gemm` crate: qkv 16.2 → 11.4 ms, wo 6.6 → 6.0, w13 30.9 → 27.2, w2 17.5 → 12.4 (1.27–1.56 → 1.70–1.86 TFLOP/s). Journal A/B, 3 rounds (`artifacts/phase4/ab-panel`): projections 1.73 → 1.39 s, prefill 4.66 → **4.28 s**; near-exact 21.91 → 21.60 s, fast 14.54 → 14.29 s. Tokens identical to FP32 in both modes. | **Accepted** for the quantized profiles (`FALCON_OCR_PANEL_GEMM=0` restores the old path). Exact mode keeps the `gemm` crate (bitwise FP32). |
| T9 | Key-transposed decode attention layout (proposed) | Rechecked the premise first. Decode attention is at the bandwidth floor at one row: probe `attempt::prefix::probe` over 22 layers' caches: Q8 207 MB 3.85 ms (54 GB/s at 8 threads), Q16 392 MB 7.23 ms, BF16 369 MB 7.19 ms, FP32 737 MB 13.97 ms; 12–16 threads are slower than 8. | **Dropped**: it would only cheapen verification rows. Bytes are what matter. |
| — | Build hygiene | The system clock jumped back ~3 h; cargo skipped rebuilds (sources older than artifacts), which voided one A/B. `cargo clean -p falcon-ocr` fixed it; compile lines are now checked before every A/B. | |
| T10 | BF16 scales for the Q8/Q16 caches (rounded up; codes computed against the stored scale): 180 → 170 and 340 → 330 bytes per record | Journal A/B, 3 rounds (`artifacts/phase4/ab-bf16s`): fast decode 8.66 → 8.45 ms/token (total 14.13 → 13.92 s), near-exact 15.13 → 14.94 (21.54 → 21.35 s); tokens identical to FP32. Against the FP32 anchor (55 pages): fast KL 2.76e-4 → 2.80e-4, flips 61 → 64 (noise); near-exact 9.16e-8 → 9.11e-8, 1 flip. | **Accepted.** |
| T11 | Attention on 8 tasks (`FALCON_OCR_SPLIT_CHUNKS=1`) with the 12-thread team | Decode 8.45 → 9.02 ms/token. | Rejected. |
| T12 | Kernel-ready model files (`falcon-ocr --mode near-exact|fast pack --output F`, `--model-file F`): one safetensors file with every tensor in the layout the kernels read (FP32 embedding/head/norms/projector/sinks, 8/16-bit body codes + FP32 scales, INT8 head screen + bound constants, recipe and a tensor digest in metadata), mapped and used in place (`buf::Buf`) | Files: near-exact 806 MB, fast 638 MB (`artifacts/packed/`). Journal, first 300 tokens: tokens identical to the normal loaders in both modes. Load 2,242 → **10 ms** (near-exact), 2,072 → **9.6 ms** (fast), plus no head-screen build. Peak resident 2.89 → **1.81 GB** / 2.72 → **1.63 GB**; resident after the page 1.53 → 0.45 / 1.36 → 0.28 GB. | **Accepted.** No full-file hash by default (`--verify-model-file` checks every tensor). |
| T13 | IEEE FP16 KV cache for near-exact (`w16-body-kv-f16`; 320 vs 330 bytes per record) | Journal A/B, 3 rounds (`artifacts/phase4/ab-f16kv`): attention 7.43 → 7.21 ms, total 21.27 → 21.02 s (−1.2%). Against the FP32 anchor: KL 9.1e-8 → **1.6e-7**, flips 1 → **4** (FP32 weights + FP16 KV alone: 9.9e-8, 1 flip, vs 2.6e-8, 0 for Q16). | **Rejected** for near-exact (fidelity is its purpose); profile kept. |
| T14 | Image resolution (`--max-dimension`), near-exact on the 64 calibration pages with speculation and the repetition stop (`artifacts/phase4/resolution/`) | 1536: median 6,544 image tokens, 1,106 s (prefill 278, decode 819). **1280**: 4,576 tokens, **887 s (−20%)**, prefill −41%, decode −13%. **1024**: 2,896 tokens, **735 s (−34%)**, prefill −68%, decode −22%. Micro CER vs truth on the 48 pages all three end at EOS: 16.42% → **15.69%** → 16.82% (tiny_text 6.96 → 6.46 → 6.20; handwriting 67.0 → 67.6 → 73.4; tables 8.4 → 7.5 → 9.1). On the 55 pages 1536 ends at EOS: 19.10 → 20.67 → 21.75% (more loops at lower resolution: repetition stops 7 / 9 / 4). | 1280 looks accuracy-neutral on this small set; 1024 costs handwriting/tables. Default stays 1536 (production's size); confirm 1280 on the 200 held-out pages + judge before recommending it. |
| T15 | Fused prefill QKV tail (`fused_prefix_rows`): split, per-head RMS norms, RoPE and the compact-cache append in one parallel pass per row, keys and unique values written straight into the cache | Teacher-forced top-32 dumps on 2 calibration pages × 64 steps are **bitwise identical** to the unfused binary (split-f32 and near-exact). Journal A/B, 3 rounds (`artifacts/phase4/ab-fuse`): cache_append 106 → 0 ms; prefill 4.27 → 4.17 s (near-exact), 4.61 → 4.54 s (exact). The fused pass itself still takes ~200 ms (scalar per-head norm and RoPE). | **Accepted** (all modes). Follow-up: vectorize the norm/RoPE with the same operation order (~0.15 s). |
| T16 | Prefill fusions, round 2: AVX2 per-head norm + RoPE in the fused pass (same pairwise tree and products), no zero-fill of the cache rows; RMS norms folded into the panel GEMM's row packing (`row_scale`) and residual adds into its epilogue (`Epilogue::Add`) for quantized profiles | Unit tests: AVX2 row bitwise equal to the portable row (50 randomized rows); folded scale = pre-scaled rows and Add = store then `+=`, bitwise. Teacher-forced top-32 dumps bitwise identical to the previous binary in exact, near-exact and fast. Journal A/B, 3 rounds (`artifacts/phase4/ab-fuse2`): prefill 4.25 → **4.08 s** (norm+RoPE 196 → 112 ms, wo+residual 176 → 145, w2+residual 310 → 271); decode unchanged within noise. | **Accepted.** |
