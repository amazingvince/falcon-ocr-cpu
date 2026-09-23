# A better quantized Falcon-OCR v1.5: GPTQ fast mode (overnight, 2026-09-23)

**Status:** in progress. Every attempt, including rejected ones, is in [HILLCLIMB.md](HILLCLIMB.md).

## 1. Is there a better quantized model on Hugging Face?

No. The survey was done on 2026-09-22.

| Repository | Base | What it is |
|---|---|---|
| `Nicolassuez/Falcon-OCR-int8-openvino` | **v1.5** (rev `0c9f85d2`, weights sha `3df91e40…`, same as ours) | OpenVINO IR. NNCF INT8_ASYM per-channel weights without calibration, dynamic INT8 activations. On its own 4-page check, 3 pages were bit-identical to FP32. Decode is 30.8 ms/token at 350 cached positions (i7-13700H, 6 threads). |
| `beaupi/Falcon-OCR-oQ6`, `-oQ8` | pre-1.5 (April) | 6/8-bit repacks of the old weights |
| `mlx-community/Falcon-OCR-bf16` | pre-1.5 | MLX BF16 |
| `ningpp/Falcon-OCR-ONNX` | pre-1.5 | FP32 ONNX (1.08 GB) |
| `Remidesbois/Falcon-OCR-Poneglyph` | pre-1.5 | manga fine-tune |

There is no calibrated (GPTQ/AWQ) and no 4-bit v1.5 model. So we built one.

## 2. Where fast mode's drift came from

- **Metric.** Teacher-forced agreement (`falcon-ocr-attempt agree`). Each page is forced along the FP32 tokens, and we count the steps where the quantized model's own greedy choice differs.
- **Screening set.** 24 calibration pages (3 per category) × up to 768 steps, 14,263 forced steps in total.

| Weights | KV | Flips / 1,000 steps |
|---|---|---:|
| W8 RTN (fast mode until now) | Q8 | 8.62 |
| W8 RTN | FP32 | 8.62 |
| FP32 | Q8 | 0.77 |
| FP32 (fast exp) | FP32 | 0 |

The 8-bit **weights** cause the drift. The projection inputs are extremely anisotropic: in the layer-10 W2 input, one channel has 614,663× the median energy. Plain absmax rounding therefore wastes precision where the activations are small, and leaves the dominant channels' error uncorrected.

## 3. GPTQ W8 (same format, same kernels, same speed)

- **Build.** `attempt3/make_gptq_overlay.sh`.
  - Gram capture: 12 calibration pages (`attempt3/gptq-calibration-pages.txt`), FP32 model, 82,836 rows per matrix.
  - Then GPTQ per matrix, stored as W8G64 codes plus FP32 scales.
- **Result on the screening set** (disjoint from the capture pages):
  - **3.44 flips per 1,000 steps, against 8.62 (−60%)**.
  - Fewer flips in 7 of 8 categories; the largest drops are degraded scans (52 → 14 flips) and tables (23 → 8).
- **Offline output error** (activation-weighted, relative): 0.139% against 0.646% for RTN.

(Section to be completed: variant results, 55-page free-running confirmation, speed results, how to run.)
