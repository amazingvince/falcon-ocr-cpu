# Head-to-head: this runner versus `focr` (pszemraj/falcon-ocr.rs)

This document records a matched CPU comparison between this repository's
`falcon-ocr` runner and Peter Szemraj's `focr` (`pszemraj/falcon-ocr.rs`, a
Rust runtime whose numerical kernels are the vendored GGML CPU backend). Both
implementations run the same pinned Falcon-OCR v1.5 FP32 weights on one machine
with identical preprocessing settings and generation budgets. The driver is
`scripts/compare_focr.py`; its unit tests are `scripts/test_compare_focr.py`.
Raw evidence lives under `artifacts/benchmarks/focr-comparison-v1/` (ignored by
Git); sanitized reports are copied to `reference/focr-comparison-v1-*.json`.

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

(Filled in from the lane reports; see the sections below.)
