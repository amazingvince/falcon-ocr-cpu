# Falcon-OCR CPU runner

A Rust library and CLI that runs the Falcon-OCR v1.5 vision-language model
(pinned revision `fe757d59…`) on CPUs: full-page plain OCR of PNG and JPEG
pages, no GPU, no Python at inference time. Near-exact mode is the default;
fast mode trades fidelity for speed with 8-bit weights. The runner reports
its model, kernels and thread plan with each result.

Download the kernel-ready weights and draft head from
[BEE-spoke-data/falcon-ocr-v1.5-cpu](https://huggingface.co/BEE-spoke-data/falcon-ocr-v1.5-cpu).
Filenames identify the body weight encoding: FP32, INT16 group-64 or GPTQ
INT8 group-64. The [model catalog](docs/MODES.md#packed-model-files) maps
files to modes.

## Quickstart

Install Rust through rustup (the repository selects Rust 1.94.0), a C
compiler, CMake and NASM. Linux also needs GNU make; Windows needs the MSVC
C++ build tools. [Building](#building) covers the platform helpers and the
build without libjpeg-turbo. Python runs the download tool and optional
helpers; inference runs entirely in Rust.

Clone the repository, then build and download the model files:

```sh
git clone https://github.com/amazingvince/falcon-ocr-cpu.git
cd falcon-ocr-cpu
python -m pip install huggingface_hub
cargo build --release --locked
hf download BEE-spoke-data/falcon-ocr-v1.5-cpu --local-dir models/falcon-ocr-cpu
target/release/falcon-ocr --model models/falcon-ocr-cpu doctor --text --load
target/release/falcon-ocr --model models/falcon-ocr-cpu run page.png --text
target/release/falcon-ocr --model models/falcon-ocr-cpu --mode fast run page.png --text
target/release/falcon-ocr --model models/falcon-ocr-cpu --mode exact run page.png --text
```

`doctor --load` verifies that the model loads and prints the chosen plan and
load time. Replace `page.png` with a PNG or JPEG page. `--text` prints its
recognized text; without it, `run` prints one JSON line per page with the
text, token ids, stop reason, timings and plan, in input order. For whole
books and PDFs, see [Books and PDFs](#books-and-pdfs).

## Modes

| Mode | Body weights | Teacher-forced flips (24,262 steps) | Journal page, 7950X | Use when |
|---|---|---:|---:|---|
| `exact` | FP32 | 0 | 38 s | you need FP32's tokens |
| `near-exact` (default) | 16-bit | 1 | 21.5 s | general use, handwriting and degraded scans |
| `fast` | 8-bit GPTQ | 63 | 12.6 s | printed pages where speed matters |

FP32 means 32-bit floating point. A flip is a different greedy token choice
when each mode receives the same reference-token prefix.

The journal page has 6,544 input tokens. These timings are on a Ryzen 9
7950X, Windows 11, without the draft head. Near-exact is about 1.8× faster
than exact and fast about 3×. Fast mode failed its held-out budgets on
handwriting and degraded scans; use near-exact for those pages.

Exact mode matched FP32's tokens on every page measured. Its default
kernels can change the last bits of logits; `RunnerConfig::reference()` is
the configuration used for CPU tensor traces. Fast-mode tokens can differ
with the CPU's BF16 support, and x86 and aarch64 can differ in every mode.
See [Modes](docs/MODES.md) for storage, numerical choices and measured
fidelity.

## What `auto` decides

The runner uses the mode's packed file in `--model`. Without it, exact reads
the FP32 checkpoint; near-exact and fast quantize at load, and fast needs the
GPTQ overlay. The recorded quantized-file load is about 10 ms. Prefill
processes the input page; decode generates its output tokens. The runner
picks kernels for the host, uses all logical CPUs for prefill, and tunes the
decode team to the smallest within 2% of the fastest.

By default it drafts up to four tokens per step and verifies each against
the model's greedy choice. The downloaded
`falcon-ocr-v1.5-draft-head.safetensors` is used automatically when it sits
next to the model files; without it the runner uses n-gram drafts. Threads,
drafting and the screened head do not change tokens.

The repetition stop ends cycles of at most 128 tokens after at least
`max(256, 4 × cycle)` repeated tokens. Without `--max-new-tokens`, the
8,192-token cap is lowered when the input leaves less of the 16,384-token
context (`budget_clamped` in the result). Both can end a page early without
changing preceding tokens; `--stop-repetition=false` lets loops run.

`doctor` prints the plan without reading tensors; `--load` times model
loading and `--probe` measures memory bandwidth and the decode floor.
For model and thread selection, see
[automatic configuration](docs/ARCHITECTURE.md#automatic-configuration).

Example load and probe output for near-exact on a Ryzen 9 7950X (16 cores,
32 threads), Windows 11:

```text
load: 7 ms, screened head 53 MB
probe: 512 MB x5 on 16 threads: 51.3 GB/s median; decode floor 7.83 ms/token
```

## Command line

| Flag | Meaning |
|---|---|
| `--model DIR` | Checkpoint and/or packed files (default `artifacts/model`) |
| `--model-file F` | A packed file; it decides the mode. `--verify-model-file` checks every tensor digest first |
| `--mode exact\|near-exact\|fast` | Default near-exact; `--w8-artifact F` picks the fast-mode overlay, `--allow-rtn` lets fast mode quantize without one |
| `--backend auto\|avx2\|scalar\|neon` | `avx2` forces 8-lane FP32 kernels; `scalar` is for debugging |
| `--threads N` | Prefill pool (default: all logical CPUs) |
| `--decode-threads auto\|pool\|N` | Decode team (default auto) |
| `--speculate N` | Drafts per step, 0 = off (default 4); `--speculate-min-match M` (default 2); `--document-drafts=false` stops cross-page drafts |
| `--draft-head F` | Select a trained draft head ([research/draft-head](research/draft-head/README.md)) |
| `--drafter ngram\|head\|both` | Draft source: default `head` when its file is present, otherwise `ngram` |
| `--draft-confidence P` | Minimum draft-head confidence (default 0.35) |
| `--stop-repetition=false` | Let loops run to the cap |
| `--head screened\|full` | Both select the same token |
| `--batch-size N` | Pages decoded jointly (1..=8); in `run`, a finished page's row goes straight to the next page |
| `run --min-dimension 64 --max-dimension 1536 --max-new-tokens 8192 --text` | Image size bounds, output cap, text only |
| `run --max-dimension auto` | The resolution router picks 768, 1024 or 1536 per page ([docs/MODES.md](docs/MODES.md#resolution-routing); changes the output) |
| `run --crop-margins[=PAD]` | Trim blank margins, keeping PAD pixels (default 24): fewer image tokens at the same text size; changes the output, so off by default ([docs/MODES.md](docs/MODES.md#margin-cropping)) |
| `run --list F --output F --resume --keep-going` | Long runs: inputs from a list, records appended to a file, resume, go on past a failing page ([Books and PDFs](#books-and-pdfs)) |
| `run --pipeline [--prefill-threads N]` | Prefill the next page while the current one decodes; tokens unchanged |
| `run --escalate` | Fast mode: reread a page that got stuck in a loop with the near-exact model |
| `pack --output F`, `inspect`, `trace`, `doctor [--text] [--load] [--probe]` | Write a packed file; verify the checkpoint; capture tensors; show the plan |

`falcon-ocr-eval` is the research binary: any weights × KV profile, timed
benchmarks with telemetry, token agreement against the FP32 anchor, tensor
traces of quantized profiles, and GPTQ Gram capture.

## Books and PDFs

`run` reads PNG and JPEG. `tools/pdf_to_pages.py` turns a PDF into one PNG
per page plus a `pages.txt` list: a page that is just a scan is extracted at
its native resolution, and any other page is rendered at 1536 pixels on its
long side (`--help` has the rules). The whole book then runs as one job:

```sh
pip install -r requirements/tools.txt
python tools/pdf_to_pages.py book.pdf --output-dir book-pages
target/release/falcon-ocr --model models/falcon-ocr-cpu run --list book-pages/pages.txt --output book.jsonl --keep-going
# stopped by a crash or Ctrl-C? the same command with --resume runs only the missing pages
target/release/falcon-ocr --model models/falcon-ocr-cpu run --list book-pages/pages.txt --output book.jsonl --keep-going --resume
```

- `--output` appends each page's record to the file as soon as the page is
  done; each record names its input `path` and its `page` index.
- `--resume` skips the pages that already have a successful record, and
  `--keep-going` records a page that fails and goes on to the rest.
- Inputs and the output file are checked before the model loads, so a
  wrong path fails before inference starts.
- `--pipeline` prefills the next page while the current one decodes; on a
  scanned book it gave 12-20% more pages per hour
  ([docs/PERFORMANCE.md](docs/PERFORMANCE.md#book-runs)). `--batch-size N`
  decodes several pages together. Neither changes the tokens.
- In fast mode, `--escalate` rereads a page that got stuck in a loop with
  the near-exact model.

Record format and resume matching:
[long runs](docs/ARCHITECTURE.md#long-runs).

## Library

```rust,no_run
use std::sync::Arc;
use falcon_ocr::{GenerationOptions, Mode, Model, Runner, RunnerConfig};

fn main() -> anyhow::Result<()> {
    let model = Arc::new(Model::load_mode("models/falcon-ocr-cpu", Mode::NearExact)?);
    let runner = Runner::new(model, "models/falcon-ocr-cpu", RunnerConfig::default())?;
    println!("{}", runner.resolved());
    let page = runner.recognize_file("page.png", &GenerationOptions::default())?;
    println!("{}", page.text);
    Ok(())
}
```

`Model::load_mode` finds the packed file or the checkpoint; `Model::load_packed`
and `Model::load` are the explicit loaders. `RunnerConfig::default()` selects
automatic tuning and n-gram drafts. To discover the published draft head as
the CLI does, pass the config and model directory to
[`auto::with_default_draft_head`](src/auto/mod.rs) with `explicit` set to
`false`. Alternatively, set `draft_head` and `drafter` explicitly.

`RunnerConfig::reference()` uses the platform exp, full head, no speculation
or repetition stop, and a pool-sized decode team.

`Runner` owns its thread pool; `recognize`, `recognize_file`,
`recognize_batch` and `recognize_files` return `OcrResult`s in input order,
each carrying its `plan`. `recognize_files_streaming` hands each page's
result (or error) to a callback as soon as it and every earlier page are
done, and `recognize_files_pipelined` does the same with the page pipeline.
See [multi-page execution](docs/ARCHITECTURE.md#multi-page-execution) for
streaming, fixed cohorts and pipeline behavior.

## Building

Install Rust through rustup (the toolchain file selects 1.94.0). Windows:
the MSVC C++ build tools; `tools/build_windows.ps1 build --release` fetches
checksum-verified CMake and NASM when they are not installed. Linux: a C
compiler, GNU make, CMake and NASM.
`bash tools/build_linux.sh build --release --locked` fetches local CMake and
NASM copies when needed and writes binaries to `target/linux/release` by
default; use that path in place of `target/release` after a helper build.
macOS: `brew install cmake`. `cargo build --no-default-features` drops
libjpeg-turbo (JPEG then decodes through the `image` crate, not
Pillow-exact).

## The checkpoint and the GPTQ overlay

Packing and rebuilding the GPTQ overlay need the original FP32 checkpoint:
`python scripts/fetch_reference.py --output artifacts/model` downloads and
verifies it (SHA-256 `3df91e40…`). `falcon-ocr --mode exact pack --output F`
writes the FP32 file; near-exact and fast pack their respective encodings.
Exact mode can load the published FP32 file directly.
`tools/make_gptq_overlay.sh` builds `artifacts/model/w8-gptq.safetensors`
([rebuilding instructions](docs/DEVELOPMENT.md#rebuilding-packed-files-and-experimental-overlays));
the published fast file already contains it.
Experimental overlays with FP32 exception columns are described in
[Modes](docs/MODES.md#exception-columns-experimental).

## Fidelity and validation

- `tools/check.sh` (or `.ps1`): format, clippy with warnings denied, the
  library's bitwise kernel oracles; `--smoke` adds the exact-mode trace hash.
- `cargo test --release --locked --test modes --test gpu_parity -- --ignored`
  with `artifacts/`: exact mode reproduces the GPU smoke tokens under the
  automatic configuration; near-exact and fast reproduce the recorded
  journal tokens; packed files round-trip; the trace hashes are pinned.
- `falcon-ocr-eval agree` measures teacher-forced flips and KL
  ([mode comparisons](docs/MODES.md#the-metric)).

Tests and fixture regeneration: [Development](docs/DEVELOPMENT.md).

## Performance notes

Decode streams the weights and KV cache from memory; prefill is
compute-bound. The draft head helps prose as well as repetitive pages: on
16 held-out pages it cut total time against n-gram drafts by 21% in fast
mode, 25% in near-exact and 26% in exact, with identical tokens.
[Performance](docs/PERFORMANCE.md): measurements, bandwidth floors, and
accepted and rejected changes.

## Platform support

| | Windows x86-64 | Linux x86-64 | macOS / Linux aarch64 |
|---|---|---|---|
| Builds and tests in CI | yes | yes (plus an aarch64 cross check) | macOS arm64: library tests |
| Runs the model | measured | measured (under WSL, and natively on a Ryzen 7 7700X) | not yet run |
| Kernels | AVX2, AVX-512 tiles, AVX512-BF16 (fast) | same | NEON (fast mode's BF16 kernels are x86-only) |

## Repository layout

`src/` library and binaries · `tests/` Rust gates and fixtures ·
`examples/ocr_bench.rs` · `tools/` build, model and PDF helpers · `docs/`
architecture, modes, performance, portability, development · `scripts/`
the frozen GPU-reference closure that the receipts hash (do not edit) ·
`reference/`, `requirements/` receipts and pins · `research/` the indexed
history of every experiment ([research/README.md](research/README.md)).

## License and sources

Apache-2.0 ([LICENSE](LICENSE)); the model architecture follows Technology
Innovation Institute's Falcon-OCR / Falcon-Perception (Apache-2.0,
[licenses/falcon-perception-LICENSE](licenses/falcon-perception-LICENSE)),
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) covers Pillow and
libjpeg-turbo.

- [Pinned model and model card](https://huggingface.co/tiiuae/Falcon-OCR/tree/fe757d59ecd79d4d68760162306a70a015761ad9)
- [Technical report](https://arxiv.org/abs/2603.27365)
- [Pinned official repository](https://github.com/tiiuae/Falcon-Perception/tree/c457916c9974efbacfa91f0f6ecc2c49c6543e56)
- [Artifact hashes](reference/manifest.json), [GPU package pins](requirements/reference.txt), `Cargo.lock`

The model revision is `fe757d59ecd79d4d68760162306a70a015761ad9`; its weight
SHA-256 is `3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16`.
Downloaded weights and generated traces are ignored by Git.
