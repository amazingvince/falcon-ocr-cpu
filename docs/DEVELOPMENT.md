# Development

Status: current as of 2026-10-02.

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
  domain, the split cache against the compact kernel over its decoded values
  at one position chunk (rotated caches with the rotation as the only pre-
  and post-transform).
  A fast path without such a test does not ship. The prefill stages are also
  compared bitwise across pool sizes (`model::pool_tests`, `kernels::tests`),
  which the page pipeline's second pool relies on.
- **Integration tests that need no weights** run in CI
  (`cargo test --release --locked --tests`): negative inputs, preprocessing
  fixtures, tokenizer prompts, and `run`'s flag, input, list and output-file
  checks (`tests/run_cli.rs`). The router's parity tests (library) compare
  its 26 statistics, both tree scores and the route bit for bit with the
  Python specification on synthetic pages and the decode fixtures.
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
  yields a prefix; the three trace hashes are pinned. `tests/batch_parity.rs`
  compares fixed cohorts, continuous batching (`run --batch-size N`) and the
  page pipeline (`run --pipeline`) with sequential pages: tokens, stops,
  input sizes, crops and routes, the safety net included, in the reference
  configuration and with the packed near-exact and fast files under the
  automatic one. `tests/batch_trace.rs` checks that traced cohorts and
  streamed runs name each page's tensors by its input index (`request.{i}`
  for a prefill or a page that runs alone, `batch.{i}` for a cohort's joint
  decode), singleton cohorts and pages dropped from a cohort included.
  `cargo test --release --locked --test negative_inputs --test margin_crop
  --test run_cli -- --ignored --test-threads=1` checks the entry points.
  With `artifacts/model`: every public entry point rejects invalid requests
  before model work, including tensor traces and teacher scoring with the
  margin crop, and a page with margins reports its crop and fewer input
  tokens on every entry point. With the near-exact file in
  `artifacts/packed`: `run` resumes an interrupted output, under
  `--keep-going` records a failing page and runs the rest, and goes on when
  stderr is closed.
- **Draft head** (needs an exported head file and `artifacts/model`):
  `FALCON_OCR_DRAFT_HEAD=<head>.safetensors cargo test --release --lib
  draft_head -- --ignored --nocapture` checks the Rust chain against the
  PyTorch fixture written by `export_head.py` (equal tokens up to the first
  near tie) with FP32 and INT8 drafter caches, and prints the draft-step
  timing probe (warm and cold).
- **Token agreement** against the FP32 anchor
  ([below](#token-agreement)): near-exact has 1 flip and fast mode about 63
  in 24,262 steps. The CI `weights` job runs the gates and an agreement
  queue, on the 12 GPTQ capture pages at 8192 steps, on a self-hosted runner
  labelled `falcon-weights`.
- **Python**: `python -m unittest discover -s <folder> -p "test_*.py"` for
  `research/corpus-qualification/tests` and
  `research/quantization-feasibility/tests` (CI runs these two; they need
  `pillow`, `rapidfuzz` and `numpy`); `ruff check tools`
  (`pyproject.toml`). The tools have their own tests (`tools/tests`:
  `pdf_to_pages.py` on PDFs they build, the W8 overlay tools' exception
  columns on synthetic Grams): `pip install -r requirements/tools.txt`, then
  `python -m unittest discover -s tools/tests` and `ruff check tools/tests`
  (the `tests` exclude in `pyproject.toml` skips that folder in `ruff check
  tools`).

Fixtures: `tests/fixtures/README.md` says how the preprocessing, decode and
router fixtures are regenerated from the pinned Pillow/PyTorch environment;
`tests/fixtures/modes/README.md` how the mode fixtures are recorded with the
CLI. Regenerate a fixture only when the numerics change on purpose.

## Token agreement

`falcon-ocr-eval agree` teacher-forces each page along the tokens of an FP32
reference, a `falcon-ocr-eval --profile reference bench` report over the
calibration pages, and counts flips and KL
([MODES.md](MODES.md#the-metric)). The anchor pages are the reference's
pages outside the GPTQ capture set (`tools/gptq-calibration-pages.txt`).
Upstream's `artifacts/phase4/checks/calibration-reference.json` gives the
55-page, 24,262-step anchor whose flips MODES.md and the `--help` text
quote. A host without that file makes its own reference over the 64 pages of
`reference/corpus-v3-calibration-lock.json` with up to 512 tokens each,
which gives the 52-page, 22,726-step anchor of MODES.md's Ryzen 7 7700X
results.

From the repository root (bash; `falcon-ocr-eval` from `cargo build
--release --locked`), writing to the git-ignored `artifacts/agree/`; to use
upstream's reference, skip the two commands that make one and set `REF` to
its path:

```sh
B=target/release
W8=artifacts/model/w8-gptq.safetensors
A=artifacts/agree
mkdir -p $A
# This host's FP32 reference: the calibration lock's 64 pages, 512 tokens each
python - > $A/calibration-pages.txt <<'EOF'
import json
for page in json.load(open("reference/corpus-v3-calibration-lock.json"))["pages"]:
    print(page["canonical_path"])
EOF
$B/falcon-ocr-eval --profile reference bench $(cat $A/calibration-pages.txt) \
  --max-new-tokens 512 --warmup 0 --samples 1 --report $A/calibration-reference.json
REF=$A/calibration-reference.json   # upstream's: artifacts/phase4/checks/calibration-reference.json
# The anchor pages: the reference's pages outside the GPTQ capture set.
python - $REF > $A/anchor-pages.txt <<'EOF'
import json, sys
from pathlib import PureWindowsPath as P  # reads / and \
capture = {P(path).parent.name for path in open("tools/gptq-calibration-pages.txt").read().split()}
for page in json.load(open(sys.argv[1]))["inputs"]:
    if (name := P(page["path"]).parent.name) not in capture:
        print(f"artifacts/corpus/v3/{name}/canonical-rgb.png")
EOF
# FP32 top-K log-probabilities along the reference tokens, to score KL against
$B/falcon-ocr-eval --profile reference agree $(cat $A/anchor-pages.txt) \
  --reference $REF --max-steps 512 --report $A/fp32.json --dump-topk $A/fp32-topk.json
# One agreement arm: flips and KL against FP32 (tools/agree_queue.sh runs several)
AGREE_BIN=$B/falcon-ocr-eval AGREE_REFERENCE=$REF AGREE_ARGS="--reference-topk $A/fp32-topk.json" \
  bash tools/agree_queue.sh $A/arms $A/anchor-pages.txt 512 fast=w8-body-kv-q8=$W8
```

`tools/agree_queue.sh <out> <pages> <steps> name=profile[=overlay] ...`
runs one `agree` per arm with `--exp fast --backend avx2`. It scores against
upstream's file unless `AGREE_REFERENCE` names another report, so leave that
variable out only when `REF` is upstream's; `AGREE_BIN` defaults to the
Windows `.exe` path. The FP32 top-K run has 0 flips by construction; an
arm's `kl_mean` is KL(FP32 ‖ arm) over the top 32 tokens per step.

## Adding an instruction set

Implement `simd::Simd` for it (every method, including `sum_tree` and the
pair operations), add the `#[target_feature]` entry wrappers next to the
existing ones (`kernels/attention/decode64.rs`, `panels/panel.rs`,
`model/fused.rs`, `quant/linear/dot.rs`), extend `kernels::Simd` and `Backend`,
and run the library tests on that machine: they compare the new
instantiation with `Portable` bit for bit. Then run `tools/m4_check.sh`-style
checks with weights: smoke trace, token agreement, a speed report.

## Knobs for experiments

`--exp exact|fast` selects the prefill exp; `--tune key=value` (repeatable)
takes `prefill-bf16=off|attention|all`, `split-chunks=1..4` (position chunks
of the split cache scan), `decode-exp=exact|fast` (the exp of decode
attention over split caches; pin it when comparing caches, see
[MODES.md](MODES.md#rotated-kv-cache-experimental)), `phases=1` (phase split
of forwards on stderr), `prefill-profile=1` (prefill attention stage cycles),
and the draft head's `draft-backoff=N`, `draft-window=N`, `draft-kv=f32|q8`
and `draft-gate=token|path`. `--kv-cache compact|f32-split|q16|q8|q8r|q4r`
(`falcon-ocr` run and doctor) replaces the KV half of the resolved profile;
the plan and every result name the profile that runs, a research one unless
it is a mode's own. These flags are hidden from `--help`; nothing reads
environment variables.

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
folder: ARCHITECTURE, MODES, PERFORMANCE, PORTABILITY, DEVELOPMENT, and
NEXT-STEPS for what remains open and the proposals),
`scripts/` (the frozen GPU-reference closure the receipts hash by path; do
not edit), `reference/` and `requirements/` (frozen receipts and pins),
`research/` (indexed history: the phase-4 hill climb, the GPU reference
protocol, vLLM serving, corpus qualification, benchmarks, the retired BF16
graph, AOCL and quantization feasibility, and the reviews of 2026-09-20).
