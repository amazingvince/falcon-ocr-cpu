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
