# Attempt 3 — integrated Falcon-OCR v1.5 CPU experiment

**Status: experimental source, NOT a compiled or performance-qualified release.**

This package extends the uploaded `falcon-ocr-source.zip`. It keeps the model-specific Rust executor, full-page bidirectional image prefill, original reference CLI, and archived experiment results. It adds opt-in weight/cache profiles and a controlled measurement route. It does not replace the model with cropped-region OCR or adopt a general serving framework.

**What was checked here:** the new Python artifact mathematics and harness tests, existing available Python tests, source integration, and archive/patch integrity. **What was not checked:** Rust compilation, native operator tests, checkpoint inference, OCR quality, Windows execution, or page latency. This environment has no Rust toolchain or checkpoint. See `BUILD_STATUS.json` for exact test outcomes. The ten intermediate-tensor failures described in the inherited numerical gate remain unresolved; old quality results do not qualify these new profiles.

**The first development target remains approximately 20 seconds on the previously measured journal page, not an achieved result or a guarantee.** Ten seconds remains a stretch hypothesis. None of this package's JSON results claim a measured speedup until you run it on your hardware.

## 1. What this attempt implements

| Piece | Implemented behavior |
|---|---|
| Separate native entry point | `falcon-ocr-attempt`; original `falcon-ocr` remains the default executable. |
| W8 body / optional head | Signed group-64 weight codes with FP32 scales; FP32 activations and accumulation. Projector, embeddings, normalization parameters, sinks and positional data remain FP32. |
| Phase-specific W8 execution | Up to eight decode rows share each output channel's unpacked weights. AVX2/FMA is selected when supported, otherwise scalar. Large-M expands the **same quantized weights** into one reusable operation-sized FP32 scratch matrix for GEMM. |
| Split-prefix storage | Eight temporal key halves, sixteen distinct spatial key halves, eight value heads. Sharing requires bit-identical paired temporal halves; otherwise the conversion refuses. |
| Prefix storage precision | Opt-in FP32, BF16, or group-32 signed INT8 with **FP32** scales. Generated K/V remain **FP32**. |
| Cache lifetime | Compress after complete prefill; append generated rows without reconstructing the prefix; release completed requests' cache allocations in experimental profiles. |
| Scratch lifetime | Release full-page single-request scratch before decoding; retain only small decode buffers. |
| Batch execution | Existing fixed cohorts, layer-major dense projections and request-specific attention; log actual active rows per forward. No continuous refill in this version. |
| Artifact conversion | Optional custom W8 overlay in a standard safetensors container. Exact source identity, schema and shape validation. No executable model code imported by the converter. |
| Comparison harness | Fresh-process reference-before / candidate / reference-after; original image-file-to-text timings, binary/input/model hashes, output checks, EOS/truncation checks and control drift. |

### Deliberately not implemented

There is no new GGML adapter, GGUF exporter, W4 path, integer-activation W8A8 path, AMX-specific kernel, calibrated AWQ/GPTQ/SmoothQuant quantizer, lower-precision generated tail, speculative decoding, continuous batch refill, concurrent pixel ingestion, cross-request attention task scheduler, or final-layer liveness pruning. Those remain subsequent experiments. This attempt combines the highest-value **independently testable** steps instead of introducing every proposed optimization at once.

The split cache currently uses a new decode consumer, including bounded on-stack reconstruction of one 64-element key/value at a time. It does **not** prove that format conversion costs are lower than the bytes saved. A storage optimization may regress speed. The separate `split-f32` profile isolates much of that consumer/layout change before adding lossy cache encoding.

## 2. Numerical profiles

All non-reference profiles enable memory hygiene. Names shown are the exact CLI values.

| Profile | Body linears | Vocabulary head | Immutable prefix KV | Generated KV |
|---|---|---|---|---|
| `reference` | FP32 | FP32 | Existing compact FP32 | FP32 |
| `hygiene` | FP32 | FP32 | Existing compact FP32 | FP32 |
| `split-f32` | FP32 | FP32 | Split/shared FP32 | FP32 |
| `kv-bf16` | FP32 | FP32 | Split/shared BF16 | FP32 |
| `kv-q8` | FP32 | FP32 | Split/shared INT8, G32 | FP32 |
| `w8-body` | W8G64 | FP32 | Existing compact FP32 | FP32 |
| `w8-all` | W8G64 | W8G64 | Existing compact FP32 | FP32 |
| **`w8-body-kv-bf16`** | **W8G64** | **FP32** | **Split/shared BF16** | **FP32** |
| `w8-all-kv-bf16` | W8G64 | W8G64 | Split/shared BF16 | FP32 |
| `w8-all-kv-q8` | W8G64 | W8G64 | Split/shared INT8, G32 | FP32 |

“W8-all” means all transformer linears **and the output head**, not every tensor. Input embeddings and the image projector remain protected.

**Recommended first combined candidate:** `w8-body-kv-bf16`. This is an experimental choice that protects the vocabulary head, not a claim of established accuracy. Test its constituent changes first. Compare it with `w8-all-kv-bf16` only after observing actual head sensitivity. Keep Q8-prefix profiles as a later, higher-risk storage experiment.

### Weight quantization contract

For each output row and contiguous group of up to 64 input channels:

1. Compute the maximum absolute original FP32 weight.
2. Compute `maximum / 127` in FP64, round the scale to FP32, and clamp a nonzero underflowed scale to the smallest positive FP32 subnormal.
3. Divide original values by that FP32 scale using FP64; round ties to even and clamp to `[-127,127]`.
4. Store signed I8 codes and FP32 scales. The all-zero group has zero scale and zero codes.
5. Reconstruct a weight by FP32 `code * scale`.

This follows the uploaded Q8 reference experiment. It is absmax round-to-nearest quantization, **not activation-aware calibration**. CPU floating-point control settings can affect subnormal handling; keep the default environment and run the edge-case tests on each target.

Prefill and decode use the same reconstructed numerical weights. Different reduction trees can still produce different FP32 rounding. There is no hidden switch from FP32 source weights in prefill to separately quantized weights in decode.

### Cache boundary

The exact execution order is:

```text
Original encoded image
  -> existing reference-compatible preprocessing and token/position construction
  -> full bidirectional image prefill in FP32 arithmetic, using SELECTED weights
  -> first-token logits and selection
  -> complete prefix KV is sealed into the selected split/storage representation
  -> text decode using sealed prefix plus append-only FP32 generated tail
  -> greedy selection, existing stopping and tokenizer
```

Sealing only occurs when another forward is needed. The first token comes from pre-sealing logits. Thus the cache-precision experiment is **deferred KV compression**, not a claim of fully quantized prefill attention. New text queries attend to both segments under one online-softmax normalization and one sink contribution. All original image keys exist before any prefix result is finalized. Nothing treats partial image strips as independent causal prompts.

Temporal sharing applies only after validating identical corresponding halves. Head-specific spatial keys are not collapsed into eight already-rotated full keys.

## 3. Build prerequisites and local verification

Use the existing Rust **1.94.0** toolchain selected by `rust-toolchain.toml`. Windows requires the existing MSVC C++ build tools, CMake and NASM for the repository's native image/tokenizer dependencies. Linux requires the corresponding C/C++ build tools. Existing `scripts/build_windows.ps1` and `scripts/build_linux.sh` document dependency setup; no new native dependency is introduced by this attempt.

The Python experiment tools require Python 3.11+ and the packages in `attempt3/requirements.txt`. Python is used for offline conversion/tests/bracketing, **not Rust inference**.

From the extracted repository root, in PowerShell:

```powershell
python -m pip install -r attempt3/requirements.txt
python -m unittest discover -s attempt3 -p 'test_*.py' -v
cargo +1.94.0 test --locked --lib
cargo +1.94.0 build --release --locked --bin falcon-ocr-attempt
.\target\release\falcon-ocr-attempt.exe doctor
```

The new Rust unit tests live in `src/attempt/`: quantization ties/endpoints/finite checks, tails and matrix widths, scalar/AVX comparisons to reconstructed-weight FP64 sums, split-FP32 versus existing compact attention, lower-precision synthetic bounds, and telemetry. **They are present but were not run here.** The full library unit command also exercises existing reference tests. Ignored checkpoint-dependent tests require the original assets; do not silently declare them passed.

Run `cargo +1.94.0 fmt --all` locally before committing; rustfmt was unavailable here. A native build error or failed Rust unit test is a blocker, not a reason to relax a threshold.

The archive has no checkpoint or page corpus. Reuse your verified `artifacts/model` folder and original page files. The existing `python scripts/fetch_reference.py --output artifacts/model` can fetch the pinned assets when deliberately invoked; it also uses Git/curl and fetches the companion repository. The experiment scripts do not fetch anything automatically.

## 4. Minimal first run

Choose the **same original image** used in the recorded journal benchmark. Do not use a re-encoded screen capture. Substitute its actual path below.

```powershell
python attempt3/make_manifest.py C:\ocr-pages\journal.png --output attempt3-cases.json
.\attempt3\run.ps1 -Manifest .\attempt3-cases.json -Profiles hygiene -Output .\artifacts\attempt3\hygiene-bracket
```

`run.ps1` checks Python tests, native library tests, builds the native binary, prints CPU capabilities and executes the bracket. It does not install tools, download weights or change PowerShell execution policy. To avoid a rebuild after a known successful build, `-SkipBuild` is available; it does not skip Python harness tests. Paths are resolved from the caller's current directory.

The harness defaults to one warmup and three measured repetitions **per process**. Each comparison has three fresh processes: reference-before, candidate, reference-after. Loading/importing is excluded from warm-page timing and recorded separately. Existing output directories are refused so earlier evidence cannot be overwritten.

For a single diagnostic run without the bracket:

```powershell
.\target\release\falcon-ocr-attempt.exe --model artifacts/model --threads 16 --backend auto --profile w8-body-kv-bf16 bench C:\ocr-pages\journal.png --max-new-tokens 4096 --max-dimension 1536 --warmup 1 --samples 3 --report artifacts/attempt3/combined-one-arm.json
```

This quantizes weights once at load when no overlay is supplied. It is not a comparative speed claim on its own. `max-new-tokens=4096` is a deliberate explicit measurement budget, not a change to the released context or a guarantee that every page fits. Prefix plus requested output capacity must fit 16,384. Truncation is reported and blocks a complete-page same-output speed claim. Keep the same budget and processed image size across all compared profiles.

## 5. Ablation sequence

Run separate fresh brackets in this order:

1. **`hygiene`** against `reference`: isolate buffer/cache reclamation. Require exact output and stopping agreement.
2. **`split-f32`**: isolate split storage and its new consumer without reducing precision. Require exact output in the same backend; the synthetic native test also enforces this.
3. **`w8-body`**: isolate transformer weight quantization, keeping head/cache FP32.
4. **`kv-bf16`**: isolate low-precision prefix cache with original FP32 weights.
5. **`w8-body-kv-bf16`**: measure the interaction. Do not multiply isolated speedup percentages.
6. **`w8-all` / `w8-all-kv-bf16`**: test the vocabulary head rather than assuming it can be compressed freely.
7. **`kv-q8` / `w8-all-kv-q8`**: investigate more aggressive prefix storage only after the preceding quality results.

The harness always brackets against `reference`; compare constituent profile reports as well to attribute hygiene/layout versus numeric effects. It does not automatically promote any candidate.

Example direct invocation for several profiles:

```powershell
python attempt3/bench.py --binary target/release/falcon-ocr-attempt.exe --model artifacts/model --manifest attempt3-cases.json --profiles split-f32 w8-body kv-bf16 w8-body-kv-bf16 --output artifacts/attempt3/precision-brackets --threads 16 --backend auto --warmup 1 --samples 3
```

For real batching, use distinct pages with long outputs:

```powershell
python attempt3/make_manifest.py C:\ocr-pages\journal.png C:\ocr-pages\long-table.png --include-batch --output attempt3-long-cases.json
```

This emits individual-page cases plus a joint case. The engine still runs fixed cohorts; a short page is retired but its slot is **not refilled**. The report records each step's active-row count. A configured batch of two with occupancy near one is not evidence of sustained two-row weight reuse.

## 6. Artifacts: retain the portable format distinction

The source checkpoint remains the original pinned safetensors file. Optional deployment experiments use an additional **safetensors overlay** with an explicit custom schema:

```text
metadata format = falcon-ocr-attempt3-w8g64-v1
<original tensor name>.__w8_codes   : I8 [output,input]
<original tensor name>.__w8_scales  : F32 [output,ceil(input/64)]
```

Metadata records model revision, source SHA-256, group size, scale/activation dtypes, rounding, and whether the vocabulary head is included. Loading verifies schema, source identity, tensor count, names, dtypes, dimensions, finite scales, valid codes, and zero-scale consistency. No code is loaded from the artifact. Reports record the overlay digest.

Create two separate overlays:

```powershell
python attempt3/convert_w8.py --model artifacts/model --output artifacts/attempt3-weights/body.w8.safetensors
python attempt3/convert_w8.py --model artifacts/model --include-head --output artifacts/attempt3-weights/body-head.w8.safetensors
```

Use them with direct `--w8-artifact ...` on the Rust binary, or `--w8-body-artifact ...` / `--w8-all-artifact ...` on `attempt3/bench.py`. Body-only and body+head profiles require matching metadata. The Python converter and runtime quantizer follow the same specified arithmetic; their equality on the complete checkpoint must still be verified on the target.

**This is not GGML Q8_0, Q8_K, or a GGUF artifact.** Do not rename the extension to suggest compatibility. Standard GGUF remains an appropriate future deployment format when using a corresponding implemented GGML encoding and arithmetic path. This attempt intentionally does not mix a new format migration with weight/cache numerical changes.

### Memory caveat

The original full-precision checkpoint **remains mapped** for protected tensors and reference access. It is also hashed at startup, which touches its pages. The package does not deliver a compact, standalone deployment loader. Quantized payload size, source mapping length, process RSS, private committed memory and actual DRAM traffic are different quantities.

Large-M W8 execution holds one operation-sized expanded matrix in reusable scratch, not a second full FP32 model. Single-request hygiene releases that large scratch before decoding. Prefix sealing happens layer-by-layer, temporarily retaining one old layer plus its new representation; persistent snapshots do not capture this transient peak.

For 6,544 prefix positions, excluding text tail and allocator overhead, the logical payloads are:

| Prefix representation, 22 layers | Bytes | Binary MiB |
|---|---:|---:|
| Existing compact FP32 | 884,539,392 | 843.56 |
| Split/shared FP32 | 737,116,160 | 702.97 |
| Split/shared BF16 | 368,558,080 | 351.48 |
| Split/shared INT8 with FP32 G32 scales | 207,313,920 | 197.71 |

The earlier theoretical INT8 table assumed FP16 scales; **this implementation uses FP32 scales**. Generated-token K/V remain 90,112 bytes per reserved token across 22 layers. Do not claim the earlier hypothetical FP16-tail footprint for this attempt.

## 7. What a valid result looks like

`summary.json` combines every native repetition; it does not select the fastest process. Each arm saves its exact command, stdout, stderr, input/model/binary identities, options, outputs and telemetry.

The comparison refuses mismatched identities, dimensions, budgets or inconsistent within-profile outputs. It requires both control arms to agree. The field `same_output_complete_page_comparison` is true only if:

- reference and candidate have identical token IDs, text, stopping, processed dimensions and prefix count;
- both finish with EOS, not the token cap;
- control median drift stays within the predeclared 5% diagnostic threshold.

This is a **conservative same-output speed screen**, not an assertion that every harmless quantized-token difference means bad OCR. Output drift is retained for substantive evaluation. Optional `ground_truth` paths, one per image, in the manifest add exact character edit-distance diagnostics where the computation budget allows. Null CER means the exact calculation was not completed; there is no substituted approximate score. Numeric-string sequence comparison is a diagnostic, not clinical/financial correctness certification or a complete table/formula metric.

Every report remains `quality_qualified: false`, including exact-match results. Qualification requires your held-out long-page corpus, ground-truth OCR metrics, representative scripts/formulas/tables/numerals, and explicitly chosen tolerances. Preserve failures and output lengths. Do not lower resolution, truncate text, relax the historical numerical gate, or cherry-pick samples to make a profile look fast.

### Clocks and counters

- `wall_ms`: warm-model execution from original image files through all returned text; includes decode, preprocessing, cache construction/sealing and scheduling; excludes loading/import and JSON serialization.
- `load_and_import_ms`: initialization, hashes, and quantization/overlay import; reported separately. This is not an instrumented OS cold-cache benchmark.
- Native `timings`: inherited per-request breakdowns. Prefill/first-token sub-timing is not yet standardized between single and cohort paths around sealing; use wall time and explicit sealing telemetry for primary comparisons.
- `active_rows_histogram`: count of forwards with each actual active-row count. Together with `decoded_request_rows / decode_forwards`, reveals whether a purported batch actually stays occupied.
- `decode_kernel_ms`: aggregate decode forward intervals, excluding token selection. Not a hardware-event counter and not the full decode stage.
- KV counters: capacity payload snapshots/retirement totals. Not process RSS, private commit, exact live-token bytes, or measured memory bandwidth.

`run.ps1` succeeds when execution/comparison completed; it does **not** assert an accepted speed/quality result. Inspect the comparison status and reasons. A process failure, missing report or invalid comparison exits nonzero. Output drift produces a valid, explicitly ineligible result rather than being discarded.

## 8. Decision after the first results

The working hypothesis is that W8 reduces recurring weight traffic and compact BF16 prefix storage reduces recurring private attention traffic enough to outweigh decoding/conversion overhead. Prefill still has the large quadratic image-attention cost. This build cannot eliminate that cost by changing a file format.

Use the results to decide the next step:

- If isolated W8 is slower, profile its actual small-M conversion/accumulation and output-channel scheduling before adopting it. The new AVX2 path is not yet a mature microkernel benchmark winner.
- If prefix BF16 is slower despite lower storage, inspect conversion/function-call/head-sharing overhead. Do not infer DRAM speedup from payload ratios.
- If either changes critical OCR content, retain the full-precision profile and investigate calibration, sensitive-layer exceptions, and precision restoration.
- If both are useful but batching underfills, next implement bounded refill and request/head-pair scheduling with byte-based admission.
- If prefill dominates after decode improves, run the established large-M backend and attention-tile experiments on actual page-sized shapes.
- Only then decide whether a GGML-compatible quantized backend, lower-precision generated cache, speculative verification, or more aggressive kernels are warranted.

## 9. Files changed and provenance

New source: `src/attempt/{mod,quant,prefix}.rs`, `src/bin/falcon-ocr-attempt.rs`.

Integration: `src/model.rs` (selected weights and cache state), `src/runner.rs` (sealing, scratch/cached-request lifetime, telemetry), `src/trace.rs` (default no-op counters), `src/kernels.rs` (internal dot/AXPY exposure and automatic non-x86 large-M dispatch correction), `src/lib.rs`, and the named/default binary entries in `Cargo.toml`.

Tooling: `attempt3/{convert_w8,bench,make_manifest,test_attempt3}.py`, `run.ps1`, requirements, this runbook and validation receipts.

The scalar-vs-automatic large-M selection is intentionally disentangled: explicit `scalar` remains the debug matrix path, while `auto` can use GEMM on a platform without the handwritten x86 kernels. This dispatch correction is not a claim that ARM performance or output parity has been tested.

The source archive is the user-uploaded snapshot based on the author's `95daabc...` tree with its uncommitted edits. The patch is based on a **local clean snapshot commit**, not an upstream public commit. It preserves original reports/licenses and does not rewrite any existing benchmark as an attempt-3 result. Apply to the exact supplied original archive, not blindly to a later repository state.

## 10. Primary sources revisited

Source/code inventory is grounded in the supplied repository. External documentation checked on 21 September 2026:

- [Falcon-OCR pinned configuration](https://huggingface.co/tiiuae/Falcon-OCR/blob/fe757d59ecd79d4d68760162306a70a015761ad9/config.json): dimensions and context contract.
- [Falcon-OCR attention helper](https://huggingface.co/tiiuae/Falcon-OCR/blob/fe757d59ecd79d4d68760162306a70a015761ad9/attention.py): same-image bidirectional mask; not ordinary causal strip prefill.
- [Safetensors 0.8.0 Rust API](https://docs.rs/safetensors/0.8.0/safetensors/tensor/struct.SafeTensors.html): typed tensor views and validated container parsing. Application-level quantization semantics are still required.
- [half BF16 API](https://docs.rs/half/latest/half/struct.bf16.html): BF16 conversion/storage facilities. Storage does not imply native BF16 arithmetic.
- [PyTorch numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html): different kernels, batch forms and operation order can change floating-point outputs.
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): portable tensor/metadata format, distinct from compute kernels and from this custom W8 encoding.

These references inform contracts and experiment design. They do not establish Falcon-OCR quantization quality or CPU speed for this new code.
