# Architecture

Status: current as of 2026-09-24.

## Model contract

The runner implements the pinned Falcon-OCR v1.5 checkpoint (revision
`fe757d59…`, weight SHA-256 `3df91e40…`, both in `reference/manifest.json`):
22 layers, hidden width 768, 16 query heads, 8 KV heads, head dimension 64,
FFN width 2304, vocabulary 65,536, context 16,384. The query width is 1,024,
not the hidden width. QKV weights are `[2048, 768]`; the FFN gate and up rows
are interleaved in `[4608, 768]` and combine as `ReLU(gate)² · up`. Learned
attention sinks enter only the softmax denominator. Input and Q/K
normalizations have implicit unit weights; the final normalization has its own
epsilon. Q and K are normalized before rotary positions, and K heads are
repeated before the spatial rotation because every paired head has its own
learned spatial frequencies (collapsing rotated prefix keys to eight heads is
wrong). Temporal positions do not advance over patches, registers or the
image-end marker; non-patch spatial positions are NaN. The attention mask is
causal except inside the bidirectional image prefix.

Preprocessing is two Pillow-exact bicubic uint8 stages with the processor's
truncation and ties-to-even dimension rules, then pixel/channel patch packing
with FP64-to-FP32 scaling. The prompt starts at the image class token (no
BOS) and plain extraction ends at `OCR_PLAIN`. JPEG goes through
libjpeg-turbo, matching the pinned Pillow decoder byte for byte; PNG through
the `image` crate with Pillow's 16-bit truncation.

## Execution order

```text
image file -> decode, resize, patches, tokens and positions   (preprocess.rs)
  -> image projection and full bidirectional prefill in FP32 arithmetic
     with the mode's weights                                    (model/)
  -> first token from the pre-sealing logits
  -> the prefix KV cache is sealed into the mode's storage      (quant/kv.rs)
  -> decode: one token per step against the sealed prefix plus the
     generated tail, greedy selection, stop on EOS, the repetition stop
     or the budget                                              (runner/)
```

Sealing happens only when another forward pass is needed, so KV compression
is deferred: prefill attention itself runs in FP32 (or BF16 in fast mode on
AVX512-BF16 CPUs). Every image key exists before any prefix result is
finalized; nothing treats partial image strips as causal prompts.

## Modules

| Module | Holds |
|---|---|
| `lib.rs`, `config.rs` | Public re-exports; `ModelConfig`, `GenerationOptions` (with `fit_budget`), `RunnerConfig` (`Default` = automatic, `reference()` = bit-exact), `Tuning`, `Backend`, `HeadMode`, `ExpMode`, `Speculation`, `Drafter`, `DraftKv`, `DecodeThreads` |
| `auto.rs` | `HostInfo::detect`, `Mode`, `resolve_weights` (packed file, then checkpoint), `Resolved` (the plan a runner executes), `doctor` |
| `cli.rs` | `RunnerArgs` shared by both binaries and `print_doctor` |
| `main.rs`, `bin/falcon-ocr-eval/` | The CLI; the research binary (profiles, bench, agree, trace, Gram capture) with its telemetry |
| `preprocess.rs`, `tokenizer.rs` | Pillow-exact image preparation; the pinned tokenizer and stop ids |
| `model/` | `Model` (mapped weights, `Weight`, `Layer`), `load.rs` (checkpoint and profile loaders, `MemoryReport`), `packed.rs` (kernel-ready files), `cache.rs` (`Session`, `LayerCache`, workspaces, `FeatureCapture`: the layer outputs a draft head reads), `fused.rs` (the fused prefill QKV row), `rope.rs` (rotary factors, `split_norm_rope`), `profile.rs` (phase clocks) |
| `quant/` | `Profile { weights, kv }`, `linear.rs` (`QuantLinear`: 8/16-bit codes, FP32 scales, panel prefill, multi-row decode dots), `kv.rs` (the split KV cache: F32, Q16 and Q8 records) |
| `kernels/` | `Simd` dispatch, `linear*`, `rms_norm*`, GLU, `prefill_plan`; `attention/` (`Geometry`, `CompactKv`, one online-softmax head, the tiled GEMM, fixed-width decode, prefill tiles and their BF16 form); `panels/` (the shared panel-GEMM scheduler and the FP32 and BF16 micro-kernels); `exp.rs` |
| `simd.rs` | The 8-lane `Simd` trait with `Portable`, `Avx2`, `Avx2Fast` and `Neon` implementations |
| `runner/` | `Runner` (prefill, sealing, batches), `generate.rs` (`Generation`: the one stop ladder; results), `speculate.rs` (draft verification) |
| `team.rs`, `tune.rs`, `cpu.rs` | The decode spin team; the automatic team-size tuner and its report; host topology, performance cores, the bandwidth probe |
| `head_screen.rs`, `draft.rs`, `repetition.rs` | The exact INT8 vocabulary screen; n-gram drafts, the adaptive draft policy and document history; the repetition stop |
| `draft_head.rs` | The trained draft head (EAGLE-3 style, `research/draft-head`): INT8 projections on the decode kernels, its own INT8 key/value cache of verified positions, confidence-gated draft chains, an optional low-rank vocabulary head |
| `trace.rs`, `packed_kernels.rs`, `buf.rs` | Tensor traces and callbacks; the opt-in phase-packed batch-decode weights; mapped or owned buffers |

## Kernels: one source per operation, instantiated per ISA

Hot kernels are `#[inline(always)]` generic functions over `simd::Simd` (an
8-lane logical `f32` vector with load, store, splat, FMA, the pair and tree
operations the rows need, exact integer widening loads and `exp`). Each ISA
gets a small entry wrapper carrying its `#[target_feature]`; NEON needs none.
Every implementation keeps the same lane layout and reduction trees, so an
AVX2 or NEON instantiation is bitwise the `Portable` one on the same inputs,
and the unit tests assert that on every machine. Dispatch is decided once per
process from runtime feature detection; one baseline-target binary runs
everywhere. Deviations from the portable arithmetic are explicit and
token-checked: the 16-lane AVX-512 prefill tiles (bitwise the 8-lane ones),
the `gemm` crate for FP32 prefill projections, and the BF16 prefill kernels
of fast mode. See [PORTABILITY.md](PORTABILITY.md).

`kernels::prefill_plan` is the one predicate that decides which prefill
projection and attention kernels a body and CPU get; `Model::forward_layers`
runs it and `auto::Resolved` reports it.

## KV cache formats

The compact cache (`LayerCache::Compact`) keeps prefix keys per query head
(their spatial rotations differ), generated keys and every value per KV head,
all FP32: the reference. For a single page, after the first token, the prefix
is sealed into the profile's split store (`quant/kv.rs`): one 160-element
record per KV group and position holding the temporal key half shared by the
pair, each head's spatial half and the value, so one decode task per group
streams its records contiguously. Records are FP32 (`split-f32`, bitwise the
compact kernel), 16-bit codes with a BF16 absmax scale per 32 values (`q16`),
or 8-bit codes (`q8`); generated positions go into 128-element tail records
in the same format as they are appended. Traces and batches keep the compact
cache, so their tensors stay comparable with the recorded references. The
`expanded` layout (every KV head duplicated) remains selectable.

## Decode machinery

Decode is memory-bound: it streams the body weights, the head and the KV
cache once per token. A spin team (`team.rs`) runs the roughly 110 short
parallel loops of a step without Rayon's wake-up latency; `tune.rs` times a
few team sizes on the first steps and keeps the smallest within 2% of the
fastest (tokens never depend on it). The screened head selects the argmax
through an INT8 copy with a proven error bound and recomputes only the rows
that can still win. Speculative decoding drafts up to four tokens and
verifies them in one multi-row step whose rows are bitwise the single-row
steps; `draft::DraftPolicy` turns drafting off while it does not pay. Drafts
come from n-gram matches in the output so far or, with `--draft-head`, from a
trained head that reads the outputs of target layers 2, 11 and 19 at every
verified position (captured during the forward) and drafts while its top
token's probability is at least 0.35. The repetition stop ends a page once it repeats a cycle of at most
128 tokens for at least `max(256, 4 × cycle)` tokens.

## Automatic configuration

`auto::resolve_weights` looks for `<model>/falcon-ocr-v1.5-<mode>.safetensors`
first (mapped in place, loads in about 10 ms), then the FP32 checkpoint,
quantized at load (fast mode needs the GPTQ overlay). `auto::Resolved` states
everything the runner will do: mode and profile, weights source, prefill and
decode kernels, thread plan, KV format, bytes per token, image decoder and
tuning. `falcon-ocr doctor` prints it without reading tensors; every result
carries it as `plan`.
