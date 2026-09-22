# Head-to-head: this runner versus `focr` (pszemraj/falcon-ocr.rs)

This document records a matched CPU comparison between this repository's
`falcon-ocr` runner and Peter Szemraj's `focr` (`pszemraj/falcon-ocr.rs`, a
Rust runtime whose numerical kernels are the vendored GGML CPU backend). Both
implementations run the same pinned Falcon-OCR v1.5 FP32 weights on one machine
with identical preprocessing settings and generation budgets. The driver is
`scripts/compare_focr.py`; its unit tests are `scripts/test_compare_focr.py`.
Raw evidence lives under `artifacts/benchmarks/focr-comparison-v1/` (ignored by
Git); sanitized reports (paths relativized, OCR text removed) are copied to
`reference/benchmarks/focr-comparison-v1-<lane>.json`.

## Machine and environment

- AMD Ryzen 9 7950X (16 cores / 32 threads, two CCDs, 64 MiB L3), 64 GB DDR5-4800,
  Windows 11 Pro 26200, native (not WSL).
- Rust 1.94.0 MSVC for both sides (his checkout sits under this tree, so
  `rust-toolchain.toml` applies to it too). CMake 4.2.1, Visual Studio 2022.
- Every measured recognition runs in a fresh process; alternatives run
  sequentially on an otherwise idle machine with an operator attestation
  recorded in each report.

## Builds under test

### Ours

`falcon-ocr` and `examples/ocr_bench.rs`, release profile, `--locked`, FP32,
`--backend avx2` (what `auto` resolves to), `--weight-layout unpacked`.
Two cache layouts are reported: `compact` (the default after the promotion in
`docs/PERFORMANCE.md`) and `expanded`. Commit and binary hashes are in each
report's `implementations.ours` block.

### focr, unmodified (primary lane)

Upstream `main` at commit `331be9ef3e13156208c76ca0bbd6e77ead823e07`
(2026-07-29, crate `falcon-ocr-rs` 0.1.8) cloned with its `vendor/ggml`
submodule at `707321c4cf6d21cb4bc831aa8b687dbf01a521ce` (GGML 0.15.2), both
trees clean, built with the default cargo features (`ggml-cpu`, `turbojpeg`)
under MSVC. The build was verified to contain the GGML backend:
`focr inspect --backend ggml-linear` reports `linear=ggml-cpu ffn=ggml-cpu
attention=ggml-cpu final_norm=ggml-cpu ... ggml_commit=707321c`, and the GGML
CMake cache records `GGML_NATIVE=OFF GGML_AVX=ON GGML_AVX2=ON GGML_AVX512=OFF
GGML_CPU_REPACK=OFF GGML_OPENMP=OFF` (the author pins these for reproducible
numerics). Runtime flags: `--backend ggml-linear --weight-mode resident-f32
--activation-dtype f32 --prompt-mode hf-split --category plain`.

The earlier snapshot in `.tmp/friend-falcon-ocr` had been built with
`--no-default-features` (its binary contained no GGML symbols); it measured his
scalar comparison backend and was discarded.

### focr, modified AVX-512 lane (secondary, clearly separated)

The same source in a separate worktree with exactly two lines changed
(`artifacts/benchmarks/focr-comparison-v1/focr-avx512-modified.patch`):
`-DGGML_AVX512=ON` added to the GGML configure arguments, and the CMake config
stamp renamed so the build cannot reuse the baseline cache. Everything else,
including `GGML_NATIVE=OFF`, is unchanged. This lane answers "what if his build
used the 7950X's AVX-512", it is not his released configuration, and it is never
placed in the same table as the unmodified build.

## Matched inputs

- Same PNG files (`artifacts/corpus/...`), same `--min-dimension 64`, same
  `--max-dimension` on both sides (1024 is his default, 1536 is ours).
- `focr` defaults to a 3,920-token image budget that silently shrinks large
  pages; every run passes `--max-image-tokens 39200`, which matches our
  10,035,200-pixel ceiling. Pre-flight `focr preprocess-dump` on the journal
  page gives 1088x1536 / 6,528 patches / 6,544 prompt tokens at 1536 and
  720x1024 / 2,880 / 2,896 at 1024, identical to ours (`input_tokens - 16`).
  The driver checks this per page and fails the page on any mismatch.
- `focr` validates the checkpoint config against a fixed `max_seq_len` of 8192
  and refuses `prompt + max_new_tokens > 8192`, so `.tmp/compare-model` holds our
  v1.5 config patched to 8192 (plus the `w13_layout` sidecar it requires) over
  hard links to the same `model.safetensors` and `tokenizer.json`. Where the
  requested generation budget does not fit, the driver clamps
  `max_new_tokens` to `8192 - prompt` on **both** sides and records it.
- `--max-length 8192` for `focr` (its default 4096 rejects full pages).

## Measurement

- **Warm latency** is each side's own warm harness on a fresh process:
  `ocr_bench --warmup 2 --repetitions 3` for ours and `focr bench --repeats 5`
  with the first two repeats dropped (his bench has no warmup; the first repeat
  also pays GGML graph-workspace creation). Two processes per side in A B B A
  order; the report shows each process median, the drift between them, and the
  better one.
- **`model`** = prefill + decode as each implementation reports them; both
  exclude file decode and image preprocessing and are the primary comparable.
  `e2e` differs by definition (ours: warm RGB-buffer wall time; his
  `elapsed_ms` includes PNG decode) and is footnoted only.
- **Cold start**: one `run` per side in a fresh process, recording process wall
  time, model load (ours) / backend preparation (his), and the generated token
  IDs.
- **Output agreement** from the cold runs with stop tokens (11, 263) stripped:
  exact ID match, first divergence index, common-prefix fraction, normalized
  token and text Levenshtein distance. Ours is bit-exact against the GPU FP32
  reference on 200 pages (`docs/STATUS.md`); his is validated against the
  official PyTorch model with thresholds, so small divergences are expected.
- **Thread counts**: a sweep on the journal page at 1024 / 256 tokens
  (his 8, 16, 24, 32; ours 8, 16, 32) selects each side's best by median
  `model` time, and the main lanes use those values explicitly.

## Lanes

| Lane | Pages | max-dim | max-new-tokens | Output |
|---|---|---|---|---|
| a | 8 smoke pages | 256 | 32 | `smoke/` (parity and harness validation) |
| b | journal page | 1024 | 256 | `sweep/` (thread selection) |
| c | journal + 3 full pages | 1024 | 4096 (to EOS) | `dim1024/`, `dim1024-ours-expanded/` |
| d | journal + 3 full pages | 1536 | 1536, clamped per page | `dim1536/`, `dim1536-ours-expanded/` |
| e | journal page | 1024, 1536 | as c/d | `avx512-modified-*/` (modified build only) |

Pages: journal `artifacts/corpus/v3/3f294b5e60a0c2d4` (1653x2339), full pages
`artifacts/corpus/smoke/{ebac2ad1cac11a99,a1336e3bc391f255,bc2882dcec9a3e02}`,
smoke set `artifacts/corpus/smoke/{02efc5e800fa8102,15ef968b6700f557,
23aab39cd5bf7391,2481e823cf3d165a,417692fb6e2a625b,4a221661204e2a92,
4bf114e43e8dcfed,7442dc2baaeb00ff}`.

## Results

### Headline

Model time (prefill + decode, medians, best of two fresh processes per side,
each side at its best thread count: ours 16, focr 8), identical generated tokens
on every page:

| max-dim | pages | ours (compact default) | focr (unmodified, ggml-linear) | focr / ours |
|---|---|---:|---:|---:|
| 1024 | journal + 3 full pages, to EOS (1117–2423 tokens) | 40.9 – 90.9 s | 215.3 – 520.3 s | 5.3x – 5.8x |
| 1536 | journal, `bc2882dcec9a3e02` (1140 / 1264 tokens) | 63.8 / 72.6 s | 436.2 / 507.8 s | 6.8x / 7.0x |
| 1536 | `ebac2ad1cac11a99`, `a1336e3bc391f255` (1536 / 1168 tokens, clean rerun) | 79.6 / 69.5 s | 543.3 / 478.9 s | 6.8x / 6.9x |

Decode throughput: ours 21–30 tok/s, focr 2.2–5.7 tok/s. Prefill: ours
2.9–3.5 s at 1024 and 12.3–13.7 s at 1536; focr 16–21 s and 75–105 s. Peak
resident memory: ours 1.8–2.7 GB, focr 11.7–25.5 GB. Enabling AVX-512 in his
GGML build (modified lane) changes his numbers by +1.2–1.5%, within drift.

Build identities recorded in every report: ours `falcon-ocr` commit `95daabc0`
(lane a used the pre-promotion control binary from commit `5275f467`, expanded
cache); `focr` binary `7b7442d6…` from commit `331be9ef`, GGML `707321c4`, clean
tree, rustc 1.94.0 on both sides. All numbers are medians of three measured
recognitions per fresh process, best of two processes per side, on the idle
host; drift between the two processes of a side is listed where it exceeded 3%.

### Lane a: 8 smoke pages, max-dimension 256, 32 tokens, 16 threads each

All eight pages match token for token (32 tokens, or 3 on the page that stops
early), geometry matches (176–208 x 256, 176–208 patches), both sides are
deterministic across processes and against their cold runs. Model time
(prefill + decode): ours 0.84–0.92 s, focr 3.78–3.98 s, ratio 4.3–4.6x (3.6x on
the 3-token page). Decode: ours ~45 tok/s, focr ~9.5 tok/s. This lane validates
the harness; it is not the latency headline.

### Lane b: thread sweep, journal page, max-dimension 1024, 256 tokens

| threads | ours model s | focr model s |
|---:|---:|---:|
| 8 | 13.82 | **60.18** |
| 16 | **11.96** | 63.84 |
| 24 | | 72.23 |
| 32 | 13.26 | 144.68 |

focr's decode grows from 41 s at 8 threads to 135 s at 32 (its GGML backend
starts a disposable thread pool per graph compute, about 89 per token); its
prefill improves with threads. Each side runs the main lanes at its own best:
ours 16, focr 8.

### Lane c: four full pages at max-dimension 1024 (focr's default), to EOS

Our default (compact cache; `dim1024-ours-compact/`, measured in a later clean
window with 0.1–0.3% drift) against focr (`dim1024/`):

| page | dims / patches | out tokens | ours prefill s | ours decode s | ours model s | focr prefill s | focr decode s | focr model s | focr / ours |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3f294b5e60a0c2d4 (journal) | 720x1024 / 2880 | 1117 | 3.23 | 37.63 | 40.86 | 18.20 | 197.13 | 215.33 | 5.27x |
| ebac2ad1cac11a99 | 672x1024 / 2688 | 2206 | 2.90 | 76.02 | 78.92 | 16.03 | 416.17 | 432.40 | 5.48x |
| a1336e3bc391f255 | 768x1024 / 3072 | 2014 | 3.53 | 71.29 | 74.79 | 20.99 | 409.11 | 430.10 | 5.75x |
| bc2882dcec9a3e02 | 768x1024 / 3072 | 2423 | 3.54 | 87.32 | 90.88 | 21.13 | 499.14 | 520.25 | 5.72x |

Decode throughput: ours 27.8–29.7 tok/s, focr 4.9–5.7 tok/s. Every page's
generated IDs are identical between the implementations (1116–2422 tokens after
stripping the stop token) and all runs stopped at EOS. Peak resident memory:
ours 1.8–2.2 GB, focr 11.7–12.9 GB (`--weight-mode resident-f32` keeps Rust and
GGML copies of the weights plus a fully preallocated F32 KV cache). focr control
drift 0.7–2.0%. Cold start (fresh process, includes model load): ours 0.75 s
load, focr 0.19–0.22 s backend preparation, otherwise the same latencies.

The same pages with our previous `expanded` layout (fixed-width kernels on,
`dim1024/` ours rows and `dim1024-ours-expanded/`, same window as the focr
runs): 43.53 / 88.01 / 81.71 / 98.70 s and 43.51 / 85.24 / 81.03 / 98.81 s model
time. Compact is 6–10% faster even at a 2.7–3.1k-token prefix.

### Lane d: four full pages at max-dimension 1536 (our default dimension)

First pass (`dim1536/`). Our rows here ran with `--cache-layout expanded` (the
driver's default still named the previous layout; corrected afterwards), so they
measure the fixed-width kernels on expanded caches, not the compact default.
`max-new-tokens` was requested as 1536 and clamped to `8192 - prompt` on both
sides where the prompt was longer: 1168 tokens on `a1336e3bc391f255`, 1264 on
`bc2882dcec9a3e02`.

| page | dims / patches | out tokens (stop) | ours prefill s | ours decode s | ours model s | focr prefill s | focr decode s | focr model s | focr / ours |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3f294b5e60a0c2d4 (journal) | 1088x1536 / 6528 | 1140 (eos) | 12.76 | 56.50 | 69.24 | 80.38 | 355.23 | 436.24 | 6.30x |
| ebac2ad1cac11a99 † | 1024x1536 / 6144 | 1536 (cap) | 11.53 | 76.02 | 87.61 | 74.95 | 470.77 | 545.83 | 6.23x |
| a1336e3bc391f255 † | 1168x1536 / 7008 | 1168 (clamped cap) | 16.75 | 83.58 | 100.19 | 105.46 | 520.50 | 626.39 | 6.25x |
| bc2882dcec9a3e02 | 1152x1536 / 6912 | 1264 (clamped cap) | 14.11 | 65.29 | 79.43 | 95.38 | 412.56 | 507.82 | 6.39x |

† Disturbed windows: on `ebac2ad1cac11a99` the two processes of each side
disagreed by 37.9% (ours) and 28.6% (focr), and on `a1336e3bc391f255` our
expanded-cache runs were 33% slower than the same configuration measured later
(`dim1536-ours-expanded/`: 75.14 s). The table shows the better process; both
pages are being rerun in a clean window (`dim1536-rerun/`) and this section will
be updated. The journal and `bc2882dcec9a3e02` rows had 0.2–1.1% drift on both
sides and are considered clean.

Every page's generated IDs are identical between the implementations
(1139–1536 tokens). Decode throughput: ours 14–20 tok/s (expanded cache),
focr 2.2–3.3 tok/s. Prefill at a 6.1–7.0k-token prefix: ours 11.5–16.7 s,
focr 75–105 s. Peak resident memory: ours 3.0–3.2 GB, focr 22.8–25.5 GB.

Our compact default at 1536 (`dim1536-ours-compact/`, clean window, 0.05–0.11%
drift) on the two undisturbed pages, against the focr rows above:

| page | out tokens | ours prefill s | ours decode s | ours model s | focr model s | focr / ours |
|---|---:|---:|---:|---:|---:|---:|
| 3f294b5e60a0c2d4 (journal) | 1140 | 12.29 | 51.44 | 63.77 | 436.24 | 6.84x |
| bc2882dcec9a3e02 | 1264 | 13.69 | 58.89 | 72.59 | 507.82 | 7.00x |

The journal figure agrees with the promotion bracket's 62.7 s
(`docs/PERFORMANCE.md`); expanded caches measured 69.2 s here. Peak resident
memory with the compact cache: 2.6–2.7 GB.

Clean rerun of the two disturbed pages (`dim1536-rerun/`, ours compact and focr
in one window, same ABBA order). The host stopped this lane for low system
memory after 13 of its 14 processes (focr peaks near 25 GB resident at 1536),
so no `report.json` exists; the numbers below are assembled from the completed
per-process files (`reference/benchmarks/focr-comparison-v1-dim1536-rerun-partial.json`).

| page | out tokens | ours prefill s | ours decode s | ours model s | focr prefill s | focr decode s | focr model s | focr / ours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ebac2ad1cac11a99 (two processes per side, drift ours 2.8%, focr 0.02%) | 1536 (cap) | 11.23 | 68.4 | 79.6 | 76.13 | 467.0 | 543.3 | 6.83x |
| a1336e3bc391f255 (one completed process per side) | 1168 (clamped cap) | 14.37 | 55.0 | 69.5 | 95.09 | 383.4 | 478.9 | 6.89x |

Generated IDs are again identical on both pages. These replace the disturbed
first-pass cells above: with the first-pass expanded rows the ratios were 6.2x
and 6.3x; with the compact default in a clean window they are 6.8x and 6.9x, in
line with the journal page and `bc2882dcec9a3e02`.

### Lane e: modified AVX-512 focr build (separate lane, not his released build)

Same source with `-DGGML_AVX512=ON` (patch recorded; `GGML_NATIVE` still OFF),
binary `a1d12c23…`, thread sweep repeated for this build (best again 8 threads:
61.57 s at 8, 65.07 at 16, 73.49 at 24, 148.14 at 32 on the 1024/256-token
sweep, versus 60.18 s for the unmodified build).

| page / dims | out tokens | focr AVX-512 prefill s | decode s | model s | unmodified focr model s (lane c/d) | same-window ours (expanded) model s |
|---|---:|---:|---:|---:|---:|---:|
| journal, 1024 | 1117 | 19.29 | 199.18 | 218.48 | 215.33 | 44.10 |
| journal, 1536 | 1140 | 85.30 | 356.09 | 441.38 | 436.24 | 69.61 |

Enabling AVX-512 in GGML does not help his build on this CPU: within drift of
the unmodified numbers (+1.5% / +1.2%). Outputs remain identical to ours.
