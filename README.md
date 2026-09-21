# Falcon-OCR CPU runner

A model-specific Rust library and CLI for the updated Falcon-OCR v1.5 weights.
The implementation currently runs FP32 full-page plain OCR. It is under active
validation: the completed 200-page Windows comparison matches all 275,903 GPU
token IDs, decoded texts and stopping reasons. Twelve texts use validated replay
through the corrected decoder; original inference records remain preserved.
The shared 24-page subset also matches Windows, Linux under WSL and GPU across
all 29,205 IDs. The exact 16,384-token context boundary also matches GPU,
including all 8,284 generated IDs after an 8,100-token input.
The fixed 200-page quality regression gate passes with zero difference overall
and in every category; audited document-component results also match. The full
numerical gate remains **open**, with ten intermediate-tensor checks failing.

The project targets native Windows and Linux, single-page latency and small-batch
throughput. It does not include PDF rendering, layout detection or HTTP serving.
See [the implementation plan](docs/PLAN.md) and [qualification status](docs/STATUS.md).
Measured workloads and their limits are recorded in [performance evidence](docs/PERFORMANCE.md).
Separate [quality accounting](docs/QUALITY.md) and
[cross-platform corpus comparisons](docs/CORPUS_SUBSET_COMPARISON.md) keep partial
results distinct from completed qualification.

## Build and run

Install Rust through rustup; `rust-toolchain.toml` selects Rust 1.94.0. Windows
requires the MSVC C++ build tools. CMake and NASM build the vendored SIMD
libjpeg-turbo decoder; the tokenizer also uses native Oniguruma.
Model downloads require Python 3, Git and curl; CPU inference does not
require Python, CUDA or a GPU.

```sh
python scripts/fetch_reference.py
cargo build --release --locked
cargo run --release --locked -- doctor
cargo run --release --locked -- inspect
cargo run --release --locked -- --threads 16 run page.png --max-new-tokens 8192
```

The build wrappers can fetch checksum-verified CMake/NASM prerequisites into
ignored local artifacts when absent, without changing system configuration:
`./scripts/build_windows.ps1` or `bash scripts/build_linux.sh`. Linux also needs
a C compiler and GNU make. When Windows and WSL share the checkout, the Linux
wrapper uses a separate target directory.

JSON output includes text, token IDs, stop reason, resized dimensions, token counts,
precision, selected vector backend and timings. `--text` prints just text.
The loader verifies the pinned config, tokenizer and FP32 weight hashes. Keep
the mapped checkpoint file unchanged while a `Model` is alive.

`--backend auto|scalar|avx2|avx512` selects vector kernels. Auto currently chooses
AVX2/FMA when available; AVX-512 requires an explicit choice until whole-model
measurements justify promotion. Large GEMMs independently dispatch inside `gemm`;
`scalar` also replaces those GEMMs for numerical debugging.

Defaults are minimum dimension 64, maximum dimension 1536, 8192 requested output
tokens and 16384 total context. Input tokens plus the requested output budget must
fit. A 1536-square image contains 9216 image patches and cannot fit the default
8192 output budget. The runner returns a budget error; choose dimensions or an
output limit explicitly. The output limit is exact, unlike upstream's loop over
a rounded cache capacity.

Exact preprocessing tests cover canonical RGB buffers and 75 PNG/JPEG file-mode
fixtures, including alpha/palette/grayscale and CMYK source-mode resizing. Tests
are against the pinned Pillow reference; see the status document for broader
qualification still required.

## Library

```rust,no_run
use std::sync::Arc;
use falcon_ocr::{GenerationOptions, Model, Runner, RunnerConfig};

fn main() -> anyhow::Result<()> {
    let model = Arc::new(Model::load("artifacts/model")?);
    let runner = Runner::new(model, "artifacts/model", RunnerConfig::default())?;
    let result = runner.recognize_file("page.png", &GenerationOptions::default())?;
    println!("{}", result.text);
    Ok(())
}
```

`Model` shares immutable mapped weights. `Runner` owns a bounded Rayon compute
pool. `recognize` accepts an `image::RgbImage`. `recognize_batch` and
`recognize_files` prefill requests independently and share decode projections
across active rows. Finished requests leave the batch; results retain input order.
Set `--batch-size` for the CLI or `RunnerConfig.batch_size` before creating a runner.

`--cache-layout compact` selects reduced KV storage: sixteen image-prefix key
heads, eight generated-text key heads, and eight value heads. The default remains
`expanded` until performance qualification. Both paths match bit for bit on the
single/mixed fixtures on Windows and Linux; reported fixture heap savings are in
the status document. A full-page Windows batch-one comparison measured 8.7–9.4%
lower latency with compact caches and about 398 MB lower peak resident memory;
larger/mixed workload qualification remains pending. Prefill scratch is reused
across layers and batch requests.

A separate [fixed-width attention experiment](experiments/attention64/RESULTS-V1.md)
uses an isolated copied build and measured 6.15–9.43% lower full-page Windows
latency with expanded caches. All nine measured outputs and the complete smoke
tensor trace match the control. A [combined compact-cache build](experiments/attention64_compact/RESULTS-V1.md)
then measured a further 6.15–6.40% latency reduction over fresh compact controls,
at 63.10 seconds per page with exact outputs. Separate smoke checks confirmed
zero warm-decode allocations.
The combined build also passes Linux-under-WSL operator, smoke-tensor and
allocation checks. These are isolated copied builds, not live runtime options;
broader workload qualification and bare-metal Linux measurements remain pending.

`--weight-layout phase-packed` adds a once-packed shared weight copy for AVX2
batch decode. Repeated small-fixture runs reduce batch2/4/8 latency by about
18%, 28–30% and 21–24%, at an additional 835.5 MiB. It remains opt-in pending
full-page measurements. The default is `unpacked`.

An experimental single-request BF16 path is available with `--precision bf16`.
It has explicit BF16 weights/activations/KV storage and an AVX-512BF16 backend,
but has not passed the frozen numerical gates. See [BF16 evidence](docs/BF16.md).
INT8/INT4 remain isolated experiments; [quantization research](docs/QUANTIZATION.md)
records CPU instruction support, candidate libraries, exact storage costs and gates.

## Validation

```sh
cargo test --locked
cargo test --release --locked --test gpu_parity -- --ignored
cargo test --release --locked --test decode_allocations -- --ignored
cargo test --release --locked --test batch_parity --test cache_layout -- --ignored
cargo test --release --locked --test negative_inputs -- --include-ignored --test-threads=1
```

The ignored integration test requires the pinned weights and actual GPU export.
It checks free-running token IDs, EOS and an exact short output limit. The ordinary
tests cover image resizing/patch packing, prompt construction and numerical kernels.
The allocation test measures all heap allocations during warmed-up token decoding.
The negative-input suite covers public API and CLI errors, including two explicitly
selected tests that require the pinned model assets and perform no successful
generation. It passes on native Windows and WSL; see
[the validation note](reference/negative-input-validation-v1.md).

Follow [reference/README.md](reference/README.md) to export the strict FP32 GPU
baseline in an isolated WSL environment. Then capture Rust activations:

```sh
cargo run --release --locked -- trace \
  --fixture artifacts/reference/smoke-fp32/trace.safetensors \
  --output artifacts/cpu/smoke-trace.safetensors --max-new-tokens 17
```

This command uses the fixture's teacher tokens to hold the prefix constant; its
`teacher_forced` flag distinguishes it from free-running validation. Trace timings
include copying and are not performance measurements. GPU tolerances are frozen
from independent strict GPU operator comparisons before measuring Rust.

Measure a warm RGB-buffer baseline independently from tensor capture:

```sh
cargo run --release --locked --example ocr_bench -- \
  artifacts/reference/smoke-fp32/canonical-rgb.png \
  --cpu-label "Ryzen 9 7950X" --environment-label "native Windows" \
  --output artifacts/benchmarks/ocr.json
```

The report records every sample, per-request stages, process memory, artifact and
source hashes, token IDs, and reused image indices. Model loading and file decoding
are separate from warm recognition. CPU/environment labels are caller-supplied;
label WSL explicitly. Run alternatives sequentially on an otherwise idle machine.

## Sources and reproducibility

See [build and replay provenance](docs/REPRODUCIBILITY.md) for preserving an
executable, its source archive and compiler identity, and for distinguishing a
saved-token decoder correction from fresh inference.

The [full-page benchmark protocol](docs/REALISTIC_BENCHMARKS.md),
[AOCL experiments](docs/AOCL.md), and
[quantized operator results](experiments/quantization/RESULTS-V1.md) separate
completed functional evidence from pending performance and quality qualification.

- [Pinned model and model card](https://huggingface.co/tiiuae/Falcon-OCR/tree/fe757d59ecd79d4d68760162306a70a015761ad9)
- [Technical report](https://arxiv.org/abs/2603.27365)
- [Pinned official repository](https://github.com/tiiuae/Falcon-Perception/tree/c457916c9974efbacfa91f0f6ecc2c49c6543e56)
- [Artifact hashes](reference/manifest.json), [GPU package pins](requirements/reference.txt), and `Cargo.lock`
- [Third-party notices](THIRD_PARTY_NOTICES.md)

The model revision is `fe757d59ecd79d4d68760162306a70a015761ad9` and its weight
SHA-256 is `3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16`.
The downloaded weights and generated large traces are ignored by Git.
