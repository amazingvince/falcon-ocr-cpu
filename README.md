# Falcon-OCR CPU runner

> **Pre-packed CPU model files:** [huggingface.co/amazingvince/falcon-ocr-v1.5-cpu](https://huggingface.co/amazingvince/falcon-ocr-v1.5-cpu)
> (near-exact 806 MB, fast 638 MB). They load in about 10 ms and need no FP32
> checkpoint:
>
> ```sh
> cargo build --release --locked
> huggingface-cli download amazingvince/falcon-ocr-v1.5-cpu --local-dir models/falcon-ocr-cpu
> target/release/falcon-ocr --model-file models/falcon-ocr-cpu/falcon-ocr-v1.5-near-exact.safetensors run page.png --text
> ```
>
> On the journal benchmark page (Ryzen 9 7950X): exact ≈ 38 s, **near-exact
> ≈ 21.5 s** (1 changed token in 24,262 against FP32), **fast ≈ 12.6 s**, down
> from 63.8 s before Phase 4. See [Modes](#modes), the
> [phase 5 log](research/phase4-hillclimb/attempt3/HILLCLIMB.md) and [results](research/phase4-hillclimb/attempt3/RESULTS-V3.md).

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
See [the implementation plan](research/reviews-2026-09-20/docs/PLAN.md) and [qualification status](research/benchmarks/docs/STATUS.md).
Measured workloads and their limits are recorded in [performance evidence](research/benchmarks/docs/PERFORMANCE.md).
Separate [quality accounting](research/corpus-qualification/docs/QUALITY.md) and
[cross-platform corpus comparisons](research/corpus-qualification/docs/CORPUS_SUBSET_COMPARISON.md) keep partial
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

### Modes

| | `--mode exact` (default) | `--mode near-exact` | `--mode fast` |
|---|---|---|---|
| Body weights | FP32 | 16-bit integers, scale per 64 weights (quantized at load, or from a packed file) | 8-bit GPTQ act-order W8G64 (`<model>/w8-gptq.safetensors`) |
| KV cache | FP32 (split layout, bitwise) | 16-bit integers, BF16 scale per 32 values | 8-bit, BF16 scale per 32 values |
| Prefill attention | FP32 | FP32 | BF16 products on AVX512-BF16 CPUs (FP32 elsewhere) |
| Teacher-forced vs FP32 (55 pages, 24,262 tokens) | bitwise | KL 9e-8, 1 changed token | KL 2.7e-4, 63 changed tokens (2.8e-4, 64 with FP32 attention) |
| Journal page (7950X, default flags) | ≈ 38 s | ≈ 21.5 s | ≈ 12.6 s (≈ 14 s without AVX512-BF16) |

- **Packed files.** `falcon-ocr --mode near-exact|fast pack --output F` writes a kernel-ready file, and `--model-file F` maps it and uses it in place. Loading takes about 10 ms instead of about 2 s, peak memory drops by about 1 GB, and tokens are identical. Published files: [huggingface.co/amazingvince/falcon-ocr-v1.5-cpu](https://huggingface.co/amazingvince/falcon-ocr-v1.5-cpu).
- **Speculative decoding.** `--speculate 4` is the default. Up to 4 tokens are drafted from the output so far and verified in one step. Every accepted token is the model's own greedy choice, so outputs are unchanged. The verify step decodes the KV cache and the weights once for all its rows (about 17–20 ms for 4 drafts against 9 ms for one token). Drafting switches itself off while it doesn't pay (most prose) and on for tables, lists and loops: a table page 45.3 → 30.5 s, a looping page 44.3 → 19.2 s. With several images per `run`, drafts also continue 4-token matches from the earlier pages (`--document-drafts`, default on; about 2% on an 8-page document).
- **BF16 prefill (fast mode).** On CPUs with AVX512-BF16 (Zen 4, Sapphire Rapids and later), fast mode's prefill attention multiplies in BF16 (`vdpbf16ps`, twice the FP32 rate) while scores, softmax and outputs stay FP32: prefill 4.0 → 2.9 s with no measurable fidelity change against FP32. `--tune prefill-bf16=all` also runs the projections in BF16 (2.3 s) at +17% KL; `--tune prefill-bf16=off` turns BF16 off.
- **Decode threads.** `--decode-threads auto` is the default: the first decode steps time a few team sizes and keep the fastest.
- **Image size.** `--max-dimension 1280` (default 1536) is about 20% faster. On 64 calibration pages it made no difference to accuracy on pages that end normally; this is not yet validated on held-out pages.
- **Held-out check of fast mode.** On the 200 held-out pages, fast mode failed its pre-registered budget against FP32 on handwriting (loops) and degraded scans. Production BF16 fails the same budget by more. A blinded LLM judge rated fast mode's content equal to FP32 and production except for loop pages. See `research/phase4-hillclimb/attempt3/RESULTS-V3.md` §6 and §8.

All modes use the exact screened head (`--head screened`, the default; it
selects the same tokens as `--head full`) and a portable vector exp in prefill
attention, which is token-identical to the platform exp on all calibration
pages. Pass `--exp exact` to keep the platform exp; `trace` always does.

`--mode fast` loads the GPTQ overlay `<model>/w8-gptq.safetensors`. Build it
once with `bash tools/make_gptq_overlay.sh` (about 20 min). The overlay
roughly triples closeness to FP32 against plain rounding: 2.7 against 8.6
flipped greedy choices per 1,000 teacher-forced steps. Without the overlay,
fast mode quantizes round-to-nearest at load, and `--w8-artifact` selects
another overlay.

`--threads` defaults to all logical CPUs, used by the compute-bound prefill.
`--decode-threads` defaults to `auto`: decode is memory-bound, so the runner
times a few team sizes on the first decode steps and keeps the smallest within
2% of the fastest (12 of 16 cores on a 7950X). A number fixes it.
- Fast mode can send a page that ended normally into a repetition loop: 1 of
  the 46 such held-back calibration pages.
- `--stop-repetition` ends a page once it repeats a cycle of at most 128 tokens
  for at least max(256, 4 × cycle) tokens, with `finish_reason: "repetition"`.
  Output up to that point is unchanged.
- On calibration it never fired on a page that ended normally, and cut decode
  work by about 26–29%.

The build wrappers can fetch checksum-verified CMake/NASM prerequisites into
ignored local artifacts when absent, without changing system configuration:
`./scripts/build_windows.ps1` or `bash tools/build_linux.sh`. Linux also needs
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

`--cache-layout` defaults to `compact` for FP32: sixteen image-prefix key
heads, eight generated-text key heads, and eight value heads. `expanded`
duplicates every KV head and remains selectable (the experimental BF16 graph
still requires it). Both layouts match bit for bit on the single/mixed fixtures
on Windows and Linux; reported fixture heap savings are in the status document.
The earlier full-page Windows batch-one comparison measured 8.7–9.4% lower
latency with compact caches and about 398 MB lower peak resident memory.
Prefill scratch is reused across layers and batch requests.

Single-query decode attention with 64-wide heads on AVX2 now uses the
fixed-width kernels in `src/kernels/attention64.rs`, promoted from the
[expanded-cache](research/benchmarks/experiments/attention64/RESULTS-V1.md) and
[compact-cache](research/benchmarks/experiments/attention64_compact/RESULTS-V1.md) experiments.
They are bit-identical to the generic loops (eleven operator tests, the
byte-exact 1,904-tensor smoke trace on both layouts, zero warm-decode
allocations). The control/candidate/control promotion bracket for the new
default is recorded in [performance evidence](research/benchmarks/docs/PERFORMANCE.md). A matched
head-to-head against an external Rust/GGML implementation is documented in
[research/benchmarks/docs/COMPARISON.md](research/benchmarks/docs/COMPARISON.md).

`--weight-layout phase-packed` adds a once-packed shared weight copy for AVX2
batch decode. Repeated small-fixture runs reduce batch2/4/8 latency by about
18%, 28–30% and 21–24%, at an additional 835.5 MiB. It remains opt-in pending
full-page measurements. The default is `unpacked`.

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

See [build and replay provenance](research/corpus-qualification/docs/REPRODUCIBILITY.md) for preserving an
executable, its source archive and compiler identity, and for distinguishing a
saved-token decoder correction from fresh inference.

The [full-page benchmark protocol](research/benchmarks/docs/REALISTIC_BENCHMARKS.md),
[AOCL experiments](research/aocl/docs/AOCL.md), and
[quantized operator results](research/quantization-feasibility/experiments/quantization/RESULTS-V1.md) separate
completed functional evidence from pending performance and quality qualification.

- [Pinned model and model card](https://huggingface.co/tiiuae/Falcon-OCR/tree/fe757d59ecd79d4d68760162306a70a015761ad9)
- [Technical report](https://arxiv.org/abs/2603.27365)
- [Pinned official repository](https://github.com/tiiuae/Falcon-Perception/tree/c457916c9974efbacfa91f0f6ecc2c49c6543e56)
- [Artifact hashes](reference/manifest.json), [GPU package pins](requirements/reference.txt), and `Cargo.lock`
- [Third-party notices](THIRD_PARTY_NOTICES.md)

The model revision is `fe757d59ecd79d4d68760162306a70a015761ad9` and its weight
SHA-256 is `3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16`.
The downloaded weights and generated large traces are ignored by Git.
