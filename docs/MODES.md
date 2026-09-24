# Modes

Status: current as of 2026-09-24. Measurements: Ryzen 9 7950X (16 cores,
32 threads, DDR5), Windows 11, the journal benchmark page (6,544 image tokens).

## The metric

Closeness to FP32 is measured with `falcon-ocr-eval agree`: every calibration
page is teacher-forced along the FP32 tokens, and the steps where the
configuration's own greedy choice differs are counted (flips), together with
the KL divergence of its distribution. A flip at step 30 of a free-running
page turns everything after it into edits, so free-running comparisons
understate agreement; per-step agreement does not. The anchor set is the 55
calibration pages outside the GPTQ capture set, 24,262 steps
(`artifacts/phase4/checks/calibration-reference.json`).

## The three modes

| | `exact` | `near-exact` (default) | `fast` |
|---|---|---|---|
| Body weights | FP32 | 16-bit codes, FP32 absmax scale per 64 inputs, quantized at load | 8-bit GPTQ act-order codes, FP32 scale per 64 inputs (`w8-gptq.safetensors`) |
| KV cache after sealing | FP32 split records | 16-bit codes, BF16 scale per 32 values | 8-bit codes, BF16 scale per 32 values |
| Prefill projections | `gemm` crate, FP32 | FP32 panel GEMM | FP32 panel GEMM (`--tune prefill-bf16=all`: BF16) |
| Prefill attention | FP32 | FP32 | BF16 products on AVX512-BF16 CPUs, FP32 elsewhere |
| Bytes per token (weights + head) | 876 MB | 401 MB | 233 MB |
| KV bytes per position | 113 KB | 58 KB | 30 KB |
| Against FP32, 24,262 steps | bitwise | KL 9e-8, **1 flip** | KL 2.7e-4, **63 flips** |
| Journal page | 38 s | 21.5 s | 12.6 s (14 s without AVX512-BF16) |
| Profile label | `split-f32` (`reference` for traces and batches) | `w16-body-kv-q16` | `w8-body-kv-q8` |

Every mode uses the exact screened head and the portable vector exp in
prefill attention (token-identical to the platform exp on all calibration
pages; traces use the platform exp). Exact mode is bit-identical to the FP32
reference: the smoke trace and the GPU parity test hold under both the
reference and the automatic configuration.

Storage types were chosen against FP32 on the anchor set (GPU harness):
INT16 with a scale per 64 inputs (1 flip, KL 1.7e-7) beat FP16 (11 flips)
and BF16 (51 flips) at the same bytes; GPTQ INT8 (63 flips) beat
round-to-nearest INT8 (168 flips); 4-bit weights cost 10–12% output error
and are out. An IEEE FP16 KV cache was 1.2% faster than Q16 but raised
near-exact's flips from 1 to 4, so it was rejected.

## Fast mode's GPTQ overlay

`tools/make_gptq_overlay.sh` builds `artifacts/model/w8-gptq.safetensors` in
about 20 minutes: it captures the mean XᵀX of every projection input on 12
calibration pages (`tools/gptq-calibration-pages.txt`, `falcon-ocr-eval
capture-gram`), then GPTQ-quantizes the 88 body matrices with act-order
(`tools/w8_variants.py --method gptq --act-order`). The projection inputs are
extremely anisotropic (one channel of a layer-10 W2 input carries 614,663×
the median energy), which is why plain rounding drifts three times as much.
The published overlay has SHA-256 `60808bbf…`; the published fast packed file
contains it. Without an overlay, `--allow-rtn` quantizes round-to-nearest at
load (2.7 → 8.6 flips per 1,000 steps).

## Fast mode's held-out result

Fast mode ran once on the 200 held-out pages against a pre-registered budget
(`reference/phase4-quality-budget.json`). It was within budget overall
(micro CER −0.16 pt) and on 5 of 7 categories, and **failed** on handwriting
(+5.4 pt, three pages sent into loops by the 8-bit weights) and degraded
scans (+1.4 pt of spread drift). The repetition stop fired on five pages
where FP32 ends at EOS; four are 8-bit-induced loops, and on the fifth FP32
itself emits a 2,232-token hallucination. Fast mode is therefore **not
qualified** for handwritten or degraded pages; use `near-exact` there. A
post-hoc comparison with production BF16 serving found that FP32 itself
would fail the same per-category budget against production, and a blinded
LLM judge rated the four systems' content equal except on loop pages.
Details: `research/phase4-hillclimb/attempt3/RESULTS-V3.md` §6 and §8.

## Loops and the repetition stop

The stop (`--stop-repetition`, on by default) ends a page once it repeats a
cycle of at most 128 tokens for at least `max(256, 4 × cycle)` tokens, with
`finish_reason: "repetition"`; output before the stop is unchanged. On the
calibration pages it never fired on a page that ends normally and cut decode
work by about a third; it also ends genuine FP32 loops. Legitimately periodic
content longer than that (a 256-token page of identical lines) would be cut:
pass `--stop-repetition=false`.

## Packed model files

`falcon-ocr --mode near-exact|fast pack --output F` writes one safetensors
file holding every tensor in the layout the kernels read (FP32 embedding,
head, norms, projector and sinks; body codes and scales; the INT8 head screen
and its bounds), with the recipe and a tensor digest in its metadata.
`--model-file F` (or a `falcon-ocr-v1.5-<mode>.safetensors` in `--model`)
maps it and uses it in place: load 2.2 s → 10 ms, peak resident memory
2.9 → 1.8 GB, tokens identical to the checkpoint loader (gated by
`tests/modes.rs`). The published files are on Hugging Face at
`amazingvince/falcon-ocr-v1.5-cpu`.

## Research profiles

`falcon-ocr-eval --profile` takes any `Weights × Kv` pair: `reference`,
`split-f32`, `kv-q16`, `kv-q8`, `w16-body-compact`, `w16-body`,
`w16-body-kv-q16`, `w16-body-kv-q8`, `w8-body`, `w8-body-split-f32`,
`w8-body-kv-q16`, `w8-body-kv-q8`. Only the three above are modes of the
CLI.
