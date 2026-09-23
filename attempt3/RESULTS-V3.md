# A better quantized Falcon-OCR v1.5, and faster exact and fast modes (2026-09-23)

**Status:** measured overnight on a quiet Ryzen 9 7950X (16 cores / 32 threads, DDR5, Windows 11).
- Quality evidence: the v3 calibration pages. The 200 held-out pages have not been run with any lossy setting.
- Every attempt, including rejected ones, is in [HILLCLIMB.md](HILLCLIMB.md).

## Summary

- **Hugging Face has no better quantized v1.5 model.** So we built one: a GPTQ (act-order) W8 overlay.
  - Its per-step flips against FP32 fall from 8.6 to 2.7 per 1,000 teacher-forced steps.
  - It keeps FP32 tokens on 25 of 55 held-back calibration pages (plain rounding: 15).
  - It has the same storage, kernels and speed as plain rounding.
- **Exact mode is 40% faster and still token-identical** (journal page 63.8 → 38.4 s). The gains come from:
  - the split FP32 cache;
  - the fast exp;
  - the screened head;
  - hybrid thread counts.
- **Fast mode takes the journal page from 63.8 to 14.7 s** and is ground-truth neutral on pages that end normally. `--stop-repetition` catches runaway loops.
- **With default flags, both modes are token-identical to FP32 on all 4 full benchmark pages.**

| Page (1536 px, 4,096-token budget) | FP32 before Phase 4 | Exact mode now | Fast mode now |
|---|---:|---:|---:|
| journal `3f294b5e60a0c2d4` (1,140 tokens) | 63.8 s | **38.4 s** | **14.7 s** |
| `ebac2ad1cac11a99` (2,237) | | 71.2 s | 24.8 s |
| `a1336e3bc391f255` (2,195) | | 74.7 s | 25.8 s |
| `bc2882dcec9a3e02` (2,428) | | 81.9 s | 27.3 s |
| Tokens vs FP32 (these 4 pages) | | identical | identical |

## 1. Hugging Face survey

| Repository | Weights | What it is |
|---|---|---|
| `Nicolassuez/Falcon-OCR-int8-openvino` | **v1.5** (rev `0c9f85d2`, weights sha `3df91e40…`, the same as ours) | OpenVINO IR. NNCF INT8_ASYM per-channel weights without calibration, dynamic INT8 activations. On its own 4-page check, 3 pages were bit-identical to FP32. 30.8 ms/token at 350 cached positions (i7-13700H, 6 threads). |
| `beaupi/Falcon-OCR-oQ6`, `-oQ8` | pre-1.5 | 6/8-bit repacks |
| `mlx-community/Falcon-OCR-bf16` | pre-1.5 | MLX BF16 |
| `ningpp/Falcon-OCR-ONNX` | pre-1.5 | FP32 ONNX |
| `Remidesbois/Falcon-OCR-Poneglyph` | pre-1.5 | manga fine-tune |

There is no calibrated (GPTQ/AWQ) and no 4-bit v1.5 model.

## 2. Measuring closeness to FP32

- **Metric: `falcon-ocr-attempt agree`.** Each page is teacher-forced along the FP32 tokens, and we count the steps where the model's own greedy choice differs.
  - A flip at step 30 of a free-running page turns everything after it into edits. Per-step agreement is not inflated that way.
  - Screening costs about 6 minutes on 24 pages (3 per category, 14,263 steps).
- **Where fast mode's drift came from** (screening set):

  | Weights | KV | Flips / 1,000 steps |
  |---|---|---:|
  | W8 round-to-nearest (old fast mode) | Q8 | 8.62 |
  | W8 round-to-nearest | FP32 | 8.62 |
  | FP32 | Q8 | 0.77 |
  | FP32, fast exp | FP32 | 0 |

- **The 8-bit weights cause the drift.** The projection inputs are extremely anisotropic: in the layer-10 W2 input (squared-ReLU outputs), one channel carries 614,663× the median energy. Plain absmax rounding leaves the dominant channels' error uncorrected.
- The largest output error was in W13 of layers 0–11 (1–2% relative).

## 3. The quantized model: GPTQ W8, act-order, static G64 groups

**Build.** `bash attempt3/make_gptq_overlay.sh` takes about 20 minutes and writes `artifacts/model/w8-gptq.safetensors`, which `--mode fast` loads automatically.
1. It captures the mean XᵀX of every projection input on 12 calibration pages (`attempt3/gptq-calibration-pages.txt`), running the FP32 model with `falcon-ocr-attempt capture-gram`.
2. It GPTQ-quantizes each of the 88 body matrices (`attempt3/w8_variants.py --method gptq --act-order`).
3. The output is the same W8G64 format (int8 codes, FP32 scale per 64 inputs) and uses the same kernels.

The overnight build has sha256 `60808bbf…`.

| Variant (screening set, W8 + Q8 KV) | Flips / 1,000 steps | Output-error proxy | Decision |
|---|---:|---:|---|
| RTN G64 (old fast mode) | 8.62 | 0.646% | baseline |
| RTN G32 | 9.47 | 0.582% | rejected (noise, +6% bytes) |
| RTN, MSE / activation-weighted clip | | 0.59% / 0.57% | proxy only: no real gain |
| GPTQ G64 | 3.44 | 0.139% | |
| GPTQ G32 | 2.66 | 0.124% | same as act-order, +6% bytes |
| **GPTQ G64 act-order** | **2.73** | **0.122%** | **chosen** |
| GPTQ G32 act-order | 2.73 | 0.110% | no gain |
| GPTQ G64 act-order + per-channel Q8 key scales | 2.66 | | KV variant flip-neutral, 1.2% faster: not adopted |

**Free-running confirmation** on the 55 calibration pages outside the Gram capture set:

| | Identical to FP32 | EOS pages identical | Token edits (EOS pages) | CER vs truth, 45 both-EOS pages | Stop changes |
|---|---:|---:|---:|---|---|
| RTN W8 (old fast) | 15/55 | 15/46 | 4,282 | 18.40% → 17.65% | 1 EOS → loop |
| **GPTQ act-order (new fast)** | **25/55** | **24/46** | **2,996** | 18.40% → 18.38% | 1 EOS → loop, 1 loop → EOS (CER 2.88 → 0.13) |
| FP32 weights + Q8 KV (reference point) | 39/55 | 32/46 | 286 | | 0 |

- With `--stop-repetition`, the stop fires on 8 GPTQ pages, none that ended at EOS. It catches the one EOS → loop page. Decode tokens drop 35%, and micro CER vs truth over all 55 pages goes 62.6% → 40.2%.

## 4. Speed

All A/Bs are on the journal page, interleaved fresh processes, on a quiet host.

| Change | Result | Decision |
|---|---|---|
| Split FP32 cache for exact mode (bit-identical to compact) | 42.8 → 39.6 s | **default for `--mode exact run`** |
| 32 threads for everything | prefill −22%, decode +50–60% | rejected as a single setting |
| **Hybrid threads**: prefill on all logical CPUs, decode on one per physical core (`src/cpu.rs`) | fast 16.4 → 14.9 s, exact 39.8 → 38.4 s | **default** |
| AVX-512F prefill tiles (bitwise identical to AVX2) | attention 2.64 → 2.41 s (under 1% total on Zen 4) | on for AVX-512 hosts; Intel should gain more |
| gemm AVX-512 kernels | under 2% | rejected |
| 4-bit screened head | fell back to the full head on every step (bound too wide) | reverted |
| Multi-page batching in fast mode | +14% pages/hour, bit-exact | available (`--batch-size`); prefill dominates short pages |

**Where the time goes now** (journal page, fast mode):
- **Decode: 8.8 ms/token.** That is about 445 MB/token at roughly 50 GB/s, the measured bandwidth floor.
  - Phases: attention 4.2, W13 1.5, head 1.0, W2 0.8, QKV 0.7, WO 0.4 ms.
- **Prefill: 4.6 s**, at about 60–70% of FP32 peak.
  - attention 2.4 s;
  - projections 1.6 s;
  - split, RoPE and appends 0.4 s.
- Further decode gains need fewer bytes (4-bit weights or KV), which the proxy says would cost much more fidelity. Prefill is the remaining frontier: BF16/INT8 compute (lossy), or overlapping one page's prefill with another page's decode.

## 5. How to run

```sh
falcon-ocr run page.png                                   # exact: tokens identical to FP32
falcon-ocr --mode fast run page.png                       # fast: GPTQ W8 + Q8 KV
falcon-ocr --mode fast --stop-repetition run page.png     # fast, runaway loops stopped
bash attempt3/make_gptq_overlay.sh                        # (re)build artifacts/model/w8-gptq.safetensors
falcon-ocr doctor                                          # shows logical CPUs / physical cores
```

- `--threads` defaults to all logical CPUs and `--decode-threads` to the physical cores. Override either explicitly.
- Without the overlay file, `--mode fast` falls back to round-to-nearest W8 at load, and says so on stderr.

## 6. Next: held-out qualification (needs the budget confirmed)

**Proposed budget**, on held-out pages where FP32 ends at EOS:
- micro CER vs ground truth no more than 0.25 pt worse than FP32 overall;
- no more than 1 pt worse in any category;
- `--stop-repetition` fires on none of those pages.

**Run**, about 1 hour. The FP32 held-out tokens already exist in `artifacts/cpu/corpus-v3-fp32-4096` and need a small format adapter for `compare_profiles.py`.

```sh
mapfile -t P < <(python -c "import json;print('\n'.join(p['canonical_path'] for p in json.load(open('reference/corpus-v3-evaluation-lock.json'))['pages']))")
FALCON_OCR_EXP=fast target/release/falcon-ocr-attempt.exe --threads 32 --decode-threads 16 \
  --profile w8-body-kv-q8 --head screened --w8-artifact artifacts/model/w8-gptq.safetensors \
  --stop-repetition bench "${P[@]}" --warmup 0 --samples 1 \
  --report artifacts/phase4/checks/heldout200-fast-gptq.json
```

Exact mode needs no budget. Its tokens can be checked against the same stored FP32 outputs (about 3 hours).
