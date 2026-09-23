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
