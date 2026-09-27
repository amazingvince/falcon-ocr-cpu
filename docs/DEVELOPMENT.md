# Development

Status: current as of 2026-09-24.

## Toolchain

`rust-toolchain.toml` pins Rust 1.94.0 with rustfmt and clippy. Windows
needs the MSVC C++ build tools; CMake and NASM build the vendored SIMD
libjpeg-turbo, and the tokenizer builds Oniguruma with the C compiler.
`tools/build_windows.ps1 <cargo args>` and `bash tools/build_linux.sh <cargo
args>` find installed tools or fetch checksum-verified local copies of NASM
and CMake into `artifacts/tools/` (no global changes); without NASM on
`PATH`, set `ASM_NASM` to the executable. Linux also needs a C compiler and
GNU make. `cargo build --no-default-features` drops libjpeg-turbo (JPEG then
decodes through the `image` crate, not Pillow-exact).

Cargo features: `turbojpeg` (default on); `gemm-avx512` (the `gemm` crate's
AVX-512 kernels for FP32 projections: token-checked, under 2% on Zen 4).

## The check

`tools/check.sh` (`tools/check.ps1` on Windows) is the gate every change
passes: `cargo fmt --check`, `cargo clippy --all-targets --locked -D
warnings`, `cargo test --release --lib --locked`; with `--smoke` it also
rebuilds the binary and checks the exact-mode smoke trace hash
(`e2dad223…`). Lints: `Cargo.toml` sets `clippy::all` and
`unsafe_op_in_unsafe_fn` to warn (denied in the check), `clippy.toml` allows
eight arguments, `rustfmt.toml` sets a 120-column width.

## Tests

- **Library tests** (`cargo test --release --lib`) are bitwise oracles: every
  vector kernel against its portable instantiation or the scalar loop, the
  compact cache against the expanded one, quantized dots against the
  dequantized products, fast exp against the platform exp over the softmax
  domain, the split cache against the compact kernel over its decoded values.
  A fast path without such a test does not ship.
- **Integration tests that need no weights** run in CI
  (`cargo test --release --locked --tests`): negative inputs, preprocessing
  fixtures, tokenizer prompts.
- **Ignored gates** need `artifacts/` (the pinned checkpoint, the GPTQ
  overlay, the packed files, the GPU smoke reference and the corpus):

  ```sh
  cargo test --release --locked --test modes --test gpu_parity --test cache_layout \
    --test batch_parity --test batch_trace --test decode_allocations --test head_screen -- --ignored
  ```

  `tests/modes.rs` is the mode gate: exact mode under the automatic
  configuration reproduces the GPU smoke tokens; near-exact and fast reproduce
  `tests/fixtures/modes/*.json`; packing round-trips the checkpoint loader
  and matches the published files; speculation, the decode team, drafts, the
  head and the explicit AVX2 backend never change tokens; the repetition stop
  yields a prefix; the three trace hashes are pinned.
- **Draft head** (needs an exported head file and `artifacts/model`):
  `FALCON_OCR_DRAFT_HEAD=<head>.safetensors cargo test --release --lib
  draft_head -- --ignored --nocapture` checks the Rust chain against the
  PyTorch fixture written by `export_head.py` (equal tokens up to the first
  near tie) with FP32 and INT8 drafter caches, and prints the draft-step
  timing probe (warm and cold).
- **Token agreement** against the FP32 anchor: `bash
  tools/agree_queue.sh <out> tools/gptq-calibration-pages.txt 8192
  near-exact=w16-body-kv-q16 fast=w8-body-kv-q8=artifacts/model/w8-gptq.safetensors`
  (expect 1 and about 63 flips of 24,262). The CI `weights` job runs the
  gates and this queue on a self-hosted runner labelled `falcon-weights`.
- **Python**: `python -m unittest discover -s tests -p "test_*.py"`; `ruff
  check tools` (`pyproject.toml`).

Fixtures: `tests/fixtures/README.md` says how the preprocessing and decode
fixtures are regenerated from the pinned Pillow/PyTorch environment;
`tests/fixtures/modes/README.md` how the mode fixtures are recorded with the
CLI. Regenerate a fixture only when the numerics change on purpose.

## Adding an instruction set

Implement `simd::Simd` for it (every method, including `sum_tree` and the
pair operations), add the `#[target_feature]` entry wrappers next to the
existing ones (`kernels/attention/decode64.rs`, `panels/panel.rs`,
`model/fused.rs`, `quant/linear.rs`), extend `kernels::Simd` and `Backend`,
and run the library tests on that machine: they compare the new
instantiation with `Portable` bit for bit. Then run `tools/m4_check.sh`-style
checks with weights: smoke trace, token agreement, a speed report.

## Knobs for experiments

`--exp exact|fast` selects the prefill exp; `--tune key=value` (repeatable)
takes `prefill-bf16=off|attention|all`, `split-chunks=1..4` (position chunks
of the split cache scan), `phases=1` (phase split of forwards on stderr) and
`prefill-profile=1` (prefill attention stage cycles). Both flags are hidden
from `--help`; nothing reads environment variables.

## Gotchas

- A backwards clock jump makes cargo skip rebuilds (sources older than
  artifacts): `cargo clean -p falcon-ocr` before an A/B.
- Run `cargo test` alone; a second concurrent cargo command can leave it
  printing nothing.
- The packed round-trip gate writes 1.5 GB of temporary files.
- Timing A/Bs are meaningless while other jobs run: interleave fresh
  processes on a quiet host and compare arms within one run.
- Keep the mapped checkpoint or packed file unchanged while a `Model` is
  alive.

## Repository layout

`src/` (library and the two binaries), `tests/` (Rust gates and fixtures),
`examples/ocr_bench.rs`, `tools/` (product-path helpers), `docs/` (this
folder: ARCHITECTURE, MODES, PERFORMANCE, PORTABILITY, DEVELOPMENT),
`scripts/` (the frozen GPU-reference closure the receipts hash by path; do
not edit), `reference/` and `requirements/` (frozen receipts and pins),
`research/` (indexed history: the phase-4 hill climb, the GPU reference
protocol, vLLM serving, corpus qualification, benchmarks, the retired BF16
graph, AOCL and quantization feasibility, and the reviews of 2026-09-20).
