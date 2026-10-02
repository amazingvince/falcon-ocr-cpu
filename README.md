# Falcon-OCR CPU runner

A Rust library and CLI that runs the Falcon-OCR v1.5 vision-language model
(pinned revision `fe757d59…`) on CPUs: full-page plain OCR of PNG and JPEG
pages, no GPU, no Python at inference time. Exact mode matched the FP32
reference token for token on every page measured; near-exact mode (the
default) changed 1 token in 24,262 against FP32 and runs 1.8× faster; fast
mode runs 3× faster with 8-bit weights. Everything the runner decides is
reported, and every fast path is tested bitwise against a portable one.
Kernel-ready model files are published at
[huggingface.co/amazingvince/falcon-ocr-v1.5-cpu](https://huggingface.co/amazingvince/falcon-ocr-v1.5-cpu).

## Quickstart

```sh
cargo build --release --locked            # Rust 1.94 (rust-toolchain.toml); CMake and NASM for libjpeg-turbo
huggingface-cli download amazingvince/falcon-ocr-v1.5-cpu --local-dir models/falcon-ocr-cpu
target/release/falcon-ocr --model models/falcon-ocr-cpu run page.png --text
target/release/falcon-ocr --model models/falcon-ocr-cpu --mode fast run page.png
target/release/falcon-ocr --model models/falcon-ocr-cpu doctor --text --probe
```

`run` prints one JSON line per page as soon as the page is done: the text,
token ids, stop reason, timings and the plan the runner chose. `--text`
prints the text only. With no flags the runner picks near-exact mode, loads
the packed file from the model directory (about 10 ms) and the draft head
next to it, and sets threads, kernels, the decode team, speculation and the
repetition stop from the host. For whole books and PDFs, see
[Books and PDFs](#books-and-pdfs).

## Modes

| Mode | Body weights | KV cache | Prefill attention | Changed tokens vs FP32 (of 24,262) | Journal page, 7950X | Use when |
|---|---|---|---|---:|---:|---|
| `exact` | FP32 | FP32 | FP32 | 0 | 38 s | you need FP32's tokens |
| `near-exact` (default) | 16-bit, scale per 64 | 16-bit, scale per 32 | FP32 | 1 | 21.5 s | everything else |
| `fast` | 8-bit GPTQ, scale per 64 | 8-bit, scale per 32 | BF16 on AVX512-BF16 CPUs | 63 | 12.6 s | printed pages where speed matters; **failed its held-out budget on handwriting (loops) and degraded scans** |

The journal page has 6,544 input tokens; timings are with default flags on a
Ryzen 9 7950X. Exact mode's 0 counts tokens: by default its logits can
differ from FP32's in the last bits, and only `RunnerConfig::reference()` is
bitwise. [docs/MODES.md](docs/MODES.md) has the metric, the storage choices
and the held-out result.

## What `auto` decides

`doctor` prints the plan without reading tensors (`--load` times the load,
`--probe` measures memory bandwidth and the decode floor):

```text
host: windows x86_64, 32 logical / 16 physical cores (SMT); avx2 fma f16c avx512f avx512bw avx512bf16
file: packed near-exact  artifacts/packed\falcon-ocr-v1.5-near-exact.safetensors (806 MB)
file: packed fast        artifacts/packed\falcon-ocr-v1.5-fast.safetensors (638 MB)
file: checkpoint         artifacts/packed\model.safetensors (missing)
file: overlay            artifacts/packed\w8-gptq.safetensors (missing)
file: tokenizer          artifacts/packed\tokenizer.json (5 MB)
file: draft head         artifacts/packed\falcon-ocr-v1.5-draft-head.safetensors (94 MB)
plan: mode near-exact (w16-body-kv-q16) | weights packed artifacts/packed\falcon-ocr-v1.5-near-exact.safetensors | prefill 32 threads, projections panel-avx2, attention avx512-wide | decode auto {12,16,24} threads, avx2 kernels, kv q16 | head Screened | speculation 4 drafts (min match 2), draft head artifacts/packed\falcon-ocr-v1.5-draft-head.safetensors | repetition stop on | exp Fast
load: 7 ms, screened head 53 MB
probe: 512 MB x5 on 16 threads: 51.3 GB/s median; decode floor 7.83 ms/token
```

Weights: the packed file for the mode in `--model`, else the FP32 checkpoint
quantized at load (fast needs the GPTQ overlay). Prefill runs on every
logical CPU with the fastest kernels the CPU has (AVX2, 16-lane AVX-512
tiles, BF16 attention for fast mode, NEON on aarch64) and a portable
polynomial exp in attention. Decode times a few team sizes on the first steps
and keeps the smallest within 2% of the fastest. Up to 4 tokens are drafted
from the output so far and verified in one step; every accepted token is the
model's own greedy choice. A page that repeats a cycle for 256 tokens stops
with `finish_reason: "repetition"`. Without `--max-new-tokens`, the
8,192-token output cap is lowered with a warning when a page's input leaves
less of the 16,384-token context (`budget_clamped` in the result).

The weights file, thread counts, decode team, drafts, head and AVX-512 tiles
never change tokens (gated by `tests/modes.rs` and the library's bitwise
tests). Three kernel choices change rounding, and so can change tokens: the
polynomial exp in prefill attention, and in fast mode the same exp in decode
attention and BF16 prefill attention on AVX512-BF16 CPUs
([docs/MODES.md](docs/MODES.md#the-three-modes)). Fast mode's tokens can
therefore differ between CPUs with and without AVX512-BF16, and x86 and
aarch64 can differ in every mode. The repetition stop and a lowered output
cap can end a page early without changing the tokens before the end;
`--stop-repetition=false` lets loops run.

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
| `--draft-head F` | A trained draft head ([research/draft-head](research/draft-head/README.md)); `falcon-ocr-v1.5-draft-head.safetensors` next to the model files is used without the flag; `--drafter ngram\|head\|both` (default `head` with a head, else `ngram`), `--draft-confidence P` (default 0.35) |
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
falcon-ocr --model models/falcon-ocr-cpu run --list book-pages/pages.txt --output book.jsonl --keep-going
# stopped by a crash or Ctrl-C? the same command with --resume runs only the missing pages
falcon-ocr --model models/falcon-ocr-cpu run --list book-pages/pages.txt --output book.jsonl --keep-going --resume
```

- `--output` appends each page's record to the file as soon as the page is
  done; each record names its input `path` and its `page` index.
- `--resume` skips the pages that already have a record, and `--keep-going`
  records a page that fails and goes on to the rest.
- Inputs and the output file are checked before the model loads, so a
  wrong path fails in seconds rather than hours into the run.
- `--pipeline` prefills the next page while the current one decodes; on a
  scanned book it gave 12–20% more pages per hour
  ([docs/PERFORMANCE.md](docs/PERFORMANCE.md#book-runs)). `--batch-size N`
  decodes several pages together. Neither changes the tokens.
- In fast mode, `--escalate` rereads a page that got stuck in a loop with
  the near-exact model.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#long-runs) has the record format
and how `--resume` matches and repairs the file.

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
and `Model::load` are the explicit loaders. `RunnerConfig::default()` is the
automatic configuration, `RunnerConfig::reference()` the bit-exact one (the
platform exp, full head, no speculation or repetition stop, a pool-sized
decode team). `Runner` owns its thread pool; `recognize`, `recognize_file`,
`recognize_batch` and `recognize_files` return `OcrResult`s in input order,
each carrying its `plan`. `recognize_files_streaming` hands each page's
result (or error) to a callback as soon as it and every earlier page are
done, and `recognize_files_pipelined` does the same with the page pipeline.

## Building

Install Rust through rustup (the toolchain file selects 1.94.0). Windows:
the MSVC C++ build tools; `tools/build_windows.ps1 build --release` fetches
checksum-verified CMake and NASM when they are not installed. Linux: a C
compiler, GNU make, CMake and NASM (`tools/build_linux.sh build --release`
fetches CMake). macOS: `brew install cmake`. `cargo build --no-default-features`
drops libjpeg-turbo (JPEG then decodes through the `image` crate, not
Pillow-exact). Supported: x86-64 with AVX2 (AVX-512 used when present) on
Windows and Linux. aarch64 compiles with NEON kernels tested bitwise in CI
but has not yet run the model.

## The checkpoint and the GPTQ overlay

Exact mode, packing and the overlay need the FP32 checkpoint:
`python scripts/fetch_reference.py --output artifacts/model` downloads and
verifies it (SHA-256 `3df91e40…`). `falcon-ocr --mode near-exact pack
--output F` and `--mode fast pack` write the packed files.
`tools/make_gptq_overlay.sh` builds `artifacts/model/w8-gptq.safetensors`
(about 20 minutes: Gram capture on 12 calibration pages, GPTQ act-order over
the 88 body matrices); the published fast file already contains it.
Experimental: `EXCEPTIONS=N` builds an overlay that keeps the N
highest-energy input columns of every W2 projection unquantized in FP32
(`tools/w8_proxy.py` estimates the gain from the Grams first; see
[docs/MODES.md](docs/MODES.md#exception-columns-experimental)).

## Fidelity and validation

- `tools/check.sh` (or `.ps1`): format, clippy with warnings denied, the
  library's bitwise kernel oracles; `--smoke` adds the exact-mode trace hash.
- `cargo test --release --locked --test modes --test gpu_parity ... -- --ignored`
  with `artifacts/`: exact mode reproduces the GPU smoke tokens under the
  automatic configuration; near-exact and fast reproduce the recorded
  journal tokens; packed files round-trip; the trace hashes are pinned.
- `falcon-ocr-eval agree` teacher-forces the 55 anchor pages along the FP32
  tokens and counts the steps where the mode's own choice differs: 1 for
  near-exact, about 63 for fast.
- Fast mode's one pre-registered run on 200 held-out pages passed overall
  and on 5 of 7 categories and failed on handwriting and degraded scans;
  it is documented, not hidden ([docs/MODES.md](docs/MODES.md)).

[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) lists every test and how the
fixtures are regenerated.

## Performance notes

Decode streams the weights, the head screen and the KV cache once per token:
233 MB per token in fast mode, 401 MB in near-exact, 876 MB in exact, plus
the cache (30–113 KB per position). At the 7950X's 52 GB/s that floor is 8.2,
15 and 31 ms per token; the runner measures 8.5, 15 and 29. Prefill is
compute-bound (about 6 TFLOP per full page): 2.9 s in fast mode with BF16
attention, 4.1 s in FP32. N-gram speculation pays on tables and loops (a
looping page 44 → 19 s), not on prose; the published draft head also drafts
prose: on 16 held-out pages it cuts total time against n-gram drafts by 21%
in fast mode, 25% in near-exact and 26% in exact (decode 1.5–1.9× against no
speculation), tokens unchanged. It is used automatically when
`falcon-ocr-v1.5-draft-head.safetensors` sits next to the model files, as it
does after the Hugging Face download. [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
lists every accepted and rejected change with its measurement.

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
