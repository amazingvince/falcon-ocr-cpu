# Phase 4: calibration verdict, generated-tail compression, loop stop (Stage 5)

**Status:** measured 2026-09-22.
- Quality evidence is the 67-page v3 calibration set (64 locked pages plus the 3 smoke full pages).
- Speed evidence is the journal page.
- The 200 held-out pages have not been run with any lossy profile.

## Setup

- **Host:** Ryzen 9 7950X, DDR5, Windows 11. All runs use `--threads 16 --backend avx2 --head screened`.
- **Workload:** max-dim 1536, 4,096-token budget.
- **Binaries:**

  | Name | Built from | Contents |
  |---|---|---|
  | `check` | `1ceea6a` | earlier head |
  | `stage5` | `0a1cc07` | Q8/BF16 generated tail, `FALCON_OCR_EXP=fast`, cargo feature `gemm-avx512` |
  | `stage6` | `8b9776b` | adds `--stop-repetition` |

  Copies are in `artifacts/phase4/bin/`.
- **Harnesses:**
  - Calibration: one process per profile over all pages, compared with `research/phase4-hillclimb/attempt3/compare_profiles.py`. Raw data is in `artifacts/phase4/checks/`.
  - Journal A/B: `research/phase4-hillclimb/attempt3/ab.py`, interleaved fresh processes with `FALCON_OCR_PHASES=1`. Raw data is in `artifacts/phase4/ab-stage5/`.
  - Loop-stop simulation: `research/phase4-hillclimb/attempt3/loop_stop_sim.py`.

## Quality: 67 calibration pages against FP32

| Profile | Identical pages | EOS pages identical | Token edits on EOS pages | CER vs truth, EOS pages (FP32 → cand) | EOS → runaway | Corpus time |
|---|---:|---:|---:|---|---:|---:|
| FP32 (`reference`) | 67/67 | 58/58 | 0 | 18.54% | 0 | 4,573 s |
| FP32 weights + Q8 KV + fast exp (`kv-q8`) | 48/67 | 41/58 | 316 / 56,385 (0.56%) | 18.54% → 18.52% | 0 | 2,591 s |
| W8 body + Q8 KV (`w8-body-kv-q8`) | 19/67 | 19/58 | 8,676 / 56,385 | 18.54% → 23.73% | 2 | 2,089 s |
| W8 body + BF16 KV (`w8-body-kv-bf16`) | 18/67 | 18/58 | 8,658 / 56,385 | 18.54% → 23.73% | 2 | 2,535 s |

- **Pages where both runs stop at EOS (56 pages).** W8 + Q8 has 2,361 token edits in 54,011 tokens (4.4%). Against ground truth: 17 pages better, 18 worse, 21 unchanged. Micro CER goes 17.71% → 17.20%. The EOS-row increase above (18.54% → 23.73%) comes entirely from the 2 pages that W8 sent into repetition loops (278259ab4fff8ac3, df13ca79dd59b21e).
- **Weights, not KV format, cause most of the W8 drift.** W8 + Q8 and W8 + BF16 agree with each other on 47/67 pages, and each agrees with FP32 on only 18–19.
- **Divergences start at near-ties.**
  - The median first divergence is at token ~116.
  - After the first flip, the page continues as a different but plausible transcription.
  - FP32 weights with Q8 KV flip far less often, and mostly by 1–7 tokens.
- **Noise floor.** Rounding-level differences essentially never flip tokens on this model. GPU FP32 against CPU FP32 is 24/24 pages and 29,205 tokens identical (`artifacts/cpu/linux-v3-gpu-comparison-complete-v1.json`). Q8 KV and W8 perturb logits well above rounding level.
- **FP32 + fast exp alone** (`reference`, `FALCON_OCR_EXP=fast`, `stage6`) is identical to FP32 on 67/67 pages: 0 edits in 93,249 tokens, the same stops.
  - So the fast exp is token-safe, and the 19 pages that differ under `kv-q8` come from Q8 KV.
  - That run's wall time is not comparable. The host ran games and chat clients from about page 15 onward, and decode was about 30% slower from then on.
  - Tokens do not depend on host load.

## Speed: journal page, W8 + Q8 KV, 3 interleaved rounds (medians)

| Arm | Prefill | Decode | Total | Tokens vs FP32 |
|---|---:|---:|---:|---|
| `check` (before) | 7.54 s | 13.53 ms/tok | 23.11 s | identical |
| `stage5` Q8 generated tail | 7.54 s | **11.07 ms/tok** | 20.33 s | identical |
| + `FALCON_OCR_EXP=fast` | **6.60 s** | 11.17 ms/tok | 19.40 s | identical |
| `gemm-avx512` build | 7.62 s | 10.87 ms/tok | 20.08 s | identical |
| `gemm-avx512` + fast exp | 6.59 s | 10.70 ms/tok | **18.84 s** | identical |

- **Q8 generated tail.** Decode attention falls from 7.2 to 5.4 ms/step. The FP32 tail (512 B per position and group) was read with a 2 KB stride. On a page that runs to 4,096 tokens, the tail grows to about 370 MB/token, more than the whole Q8 image cache.
- **Fast exp** saves 0.95 s of prefill attention (4.71 → 3.85 s). On the exact FP32 path it saves 7.29 → 6.30 s and stays token-identical on this page.
- **gemm AVX-512** on Zen 4 (double-pumped AVX-512) trims projection time by about 0.1 s. Its decode difference is within noise, because decode does not use gemm. It needs a bracket before it counts.
- **Decode phases per step (`stage5`):**

  | Phase | Time |
  |---|---:|
  | attention | 5.41 ms |
  | W13 + gate | 1.96 ms |
  | head | 1.14 ms |
  | W2 | 0.99 ms |
  | QKV | 0.90 ms |
  | WO | 0.48 ms |
  | small phases | 0.2 ms |

  These sum to the whole 11.07 ms step: there is no hidden per-step overhead. Streaming about 445 MB/token at 11.07 ms is 81% of the measured 49.5 GB/s floor.
- **No W8 prefill penalty.** W8 and FP32 transformer prefill are equal on a single page: 7.2 s each on 4a756837631f4f69, and within 20 ms on the journal page. The +0.9 s per page seen in the W8 corpus runs came from other activity on the host during those runs.

## Repetition stop (`--stop-repetition`, opt-in)

- **Rule:** stop once the newest tokens repeat a cycle of p ≤ 128 tokens for at least max(256, 4p) tokens. The finish reason is `repetition`, and output up to that step is unchanged. Implemented in `src/repetition.rs`.
- **Calibration outputs:** 9 of 67 FP32 pages run to 4,096 tokens, which is 40% of all decode steps. Seven of the nine are exact loops, with periods of 1–38 tokens and periodic tails of 2,675–4,072 tokens. The longest periodic window on any page that ends at EOS is 74 tokens.
- **Offline simulation** (CER uses proportional text truncation):

  | Outputs | Pages fired on | Fired on an EOS page | Decode tokens | Micro CER vs truth |
  |---|---:|---:|---:|---|
  | FP32 | 7 | 0 | −26.2% | 53.6% → 37.5% |
  | W8 + Q8 | 9 (both W8-induced loops included) | 0 | −29.2% | 56.5% → 37.3% |

- **End to end:** FP32 on 1306da2487430257 stops at token 294 with `finish_reason: "repetition"`. Those 294 tokens are identical to the FP32 run.

## Modes (user decision 2026-09-22: offer both)

| Mode | Contents | Journal page | Calibration tokens vs FP32 | Ground truth |
|---|---|---:|---|---|
| Exact | FP32 weights and KV, fast exp, screened head | ~53 s | 67/67 identical | reference |
| Fast | W8 body + Q8 KV + Q8 tail + fast exp (+ `gemm-avx512`), with `--stop-repetition` recommended | ~19 s | 19/67 identical | neutral on EOS pages; loops caught by the stop |

- **Exposed as `falcon-ocr --mode exact|fast`.**
  - Default head: screened. The fast exp is used for `run`; `trace` keeps the platform exp.
  - `--mode fast` quantizes at load in about 0.5 s, or reads `--w8-artifact`.
  - Both modes are token-identical to FP32 on the journal page.
- **FP32 weights + Q8 KV** runs at about 31 s on the journal page and is much closer to FP32 (0.56% token edits on EOS pages). It flips tokens on 19 pages, so it is not an exact mode.
- **BF16 storage of the FP32 weights would not be lossless.** Only about 1 in 65,536 values is exactly representable, so the checkpoint is genuine FP32.
