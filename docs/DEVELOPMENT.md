# Development

Status: current as of 2026-10-28.

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

Run the gate from the repository root:

```sh
bash tools/check.sh
```

`tools/check.ps1` is the Windows equivalent. The gate runs
`cargo fmt --all -- --check`,
`cargo clippy --release --all-targets --locked -- -D warnings`, and
`cargo test --release --lib --locked`. With `--smoke` it also rebuilds the
binary and checks the exact-mode smoke trace hash (`e2dad223…`), requiring
the pinned GPU smoke outputs. Lints: `Cargo.toml` sets `clippy::all` and
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
  head and the explicit AVX2 backend preserve the gate's fixture tokens;
  the repetition stop yields a prefix; the three trace hashes are pinned.

  `tests/batch_parity.rs` compares fixed cohorts, continuous batching
  (`run --batch-size N`) and the page pipeline (`run --pipeline`) with
  sequential pages: tokens, stops,
  input sizes, crops and routes, the safety net included, in the reference
  configuration and with the packed near-exact and fast files under the
  automatic one.

  `tests/batch_trace.rs` checks that traced cohorts and
  streamed runs name each page's tensors by its input index (`request.{i}`
  for a prefill or a page that runs alone, `batch.{i}` for a cohort's joint
  decode), singleton cohorts and pages dropped from a cohort included.

  `cargo test --release --locked --test negative_inputs --test margin_crop
  --test run_cli -- --ignored --test-threads=1` checks the entry points.
  With `artifacts/model`: every public entry point rejects invalid requests
  before model work, including tensor traces and teacher scoring with the
  margin crop, and a page with margins reports its crop and fewer input
  tokens on every entry point.

  With the near-exact file in `artifacts/packed`: `run` resumes an
  interrupted output, under
  `--keep-going` records a failing page and runs the rest, and goes on when
  stderr is closed.
- **Draft head** (needs an exported head file and `artifacts/model`):
  `FALCON_OCR_DRAFT_HEAD=<head>.safetensors cargo test --release --lib
  draft_head -- --ignored --nocapture` checks the Rust chain against the
  PyTorch fixture written by `export_head.py` (equal tokens up to the first
  near tie) with FP32 and INT8 drafter caches, and prints the draft-step
  timing probe (warm and cold).
- **Token agreement** against the FP32 anchor
  ([below](#token-agreement)) measures flips and KL;
  [mode comparisons](MODES.md#the-metric). The CI `weights` job runs the
  gates and an agreement queue on the 12 GPTQ capture pages at 8192
  steps, on a self-hosted runner
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

Fixture generation: [preprocessing, decode and router fixtures](../tests/fixtures/README.md)
and [mode fixtures](../tests/fixtures/modes/README.md). Regenerate a fixture
only when the numerics change on purpose.

The full weights gates also require the original overlay and GPU outputs,
not just the downloadable packed files. The frozen reference preflight
pins its recording GPU and environment; a cross-check on another setup is
not a replacement receipt. Run available gates and report missing inputs
separately rather than regenerating fixtures to make a check pass.

## Token agreement

`falcon-ocr-eval agree` takes an FP32 reference from a
`falcon-ocr-eval --profile reference bench` report. Use upstream's
`artifacts/phase4/checks/calibration-reference.json` when available;
otherwise build a reference from `reference/corpus-v3-calibration-lock.json`
with up to 512 tokens per page. For the metric and the two anchor sets, see
[mode comparisons](MODES.md#the-metric).

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

## Rebuilding packed files and experimental overlays

The standard packed files use the pinned FP32 checkpoint; fast mode also
needs a GPTQ overlay. Fetch the checkpoint, build both binaries, and pack:

```sh
python scripts/fetch_reference.py --output artifacts/model
cargo build --release --locked --bins
mkdir -p artifacts/repacked
target/release/falcon-ocr --mode near-exact pack \
  --output artifacts/repacked/falcon-ocr-v1.5-near-exact.safetensors
BIN=target/release/falcon-ocr-eval bash tools/make_gptq_overlay.sh
target/release/falcon-ocr --mode fast pack \
  --output artifacts/repacked/falcon-ocr-v1.5-fast.safetensors
```

Choose a new output directory for each build. The overlay helper captures
activation Gram matrices (`XᵀX`) on 12 calibration pages, then quantizes
88 body matrices with GPTQ act-order; it takes about 20 minutes on a
16-core desktop. `REUSE_GRAMS=1` reuses the existing Gram directory without
checking which pages produced it. On Windows, use the `.exe` binary paths;
the helper's default `BIN` is `target/release/falcon-ocr-eval.exe`.

To investigate exception columns with existing Grams:

```sh
python tools/w8_proxy.py --gram-dir artifacts/w8/gram \
  --include 'feed_forward\.w2' --jobs 16
EXCEPTIONS=4 REUSE_GRAMS=1 BIN=target/release/falcon-ocr-eval \
  bash tools/make_gptq_overlay.sh
target/release/falcon-ocr --mode fast \
  --w8-artifact artifacts/model/w8-gptq-exc4.safetensors run page.png
target/release/falcon-ocr --mode fast \
  --w8-artifact artifacts/model/w8-gptq-exc4.safetensors pack \
  --output artifacts/repacked/fast-exc4.safetensors
```

`page.png` is a page you supply. `tools/w8_proxy.py` reports each matrix's
top-k activation energy and activation-weighted output error,
`sqrt(tr(ΔW G ΔWᵀ) / tr(W G Wᵀ))`, for round-to-nearest and GPTQ with
k = 0, 1, 2, 4, 8, 16 exception columns. `--overlay` evaluates an existing
v2 overlay offline. This proxy can narrow the choice in minutes before a
model run; it is not a held-out quality gate.

`tools/w8_variants.py --exceptions N --exceptions-include REGEX` keeps
the N highest-energy input columns of each matching matrix;
`--exception-select error` weights selection by expected rounding error.
For column values, compensation, format and fidelity, see
[exception columns](MODES.md#exception-columns-experimental).

## Measuring performance and input changes

Compare arms only within one interleaved run on the same quiet host.
Background work changes timings; a running GPU job (training or vLLM)
slowed CPU decode by about 30% in the draft-head experiments. Record the
resolved plan, including the decode team, and pin it when comparing arms
whose automatic choices differ.
[Measurements and host-specific limits](PERFORMANCE.md).

For bandwidth, the decode floor, and a reference-profile timing report:

```sh
B=target/release
PAGE=artifacts/corpus/v3/3f294b5e60a0c2d4/canonical-rgb.png
OUT=artifacts/measure-journal
mkdir -p "$OUT"
$B/falcon-ocr --model artifacts/packed doctor --probe --text
$B/falcon-ocr-eval --profile reference bench "$PAGE" \
  --samples 3 --report "$OUT/reference.json"
```

The page needs the restored corpus. Choose a new `OUT` for another run;
`bench` refuses an existing report. `--tune phases=1` prints the forward
phase split; `--tune prefill-profile=1` prints prefill-attention stage
cycles. Use explicit profiles or `--model-file` and runner flags to
measure the configuration you intend.

Other measurement entry points:

| Tool | What it measures |
|---|---|
| `python research/phase4-hillclimb/attempt3/ab.py --help` | Interleaved fresh-process A/B of two binaries on the same pages |
| `python research/draft-head/cpu_ab.py --help` | Interleaved drafter settings, page totals and token identity against no speculation; an arm's first flag `@<exe>` selects its binary |
| `cargo test --release --locked --lib panels::panel::tests::prefill_throughput_probe -- --ignored --nocapture` | Panel GEMM versus the `gemm` crate at prefill shapes; the recorded 7950X result was 1.7–1.8 TFLOP/s |
| `cargo run --release --locked --example ocr_bench -- --help` | Warm canonical-RGB recognition, with input and model hashes in its reports; provide images, CPU/environment labels and an output path |

For margin cropping, run a few dozen of your own book pages in the same
mode with and without `--crop-margins`. Compare each page's token ids,
character edit distance divided by the uncropped text's length, stop
reason, `total_ms`, and CER against ground truth when available.
`research/resolution-router/analyze_agreement.py` uses that text-distance
measure. Read the changed pages: lost page numbers or marginal notes are
failures even if the run is faster.

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
- Keep the mapped checkpoint or packed file unchanged while a `Model` is
  alive.

## Repository layout

See the [repository layout](../README.md#repository-layout).
