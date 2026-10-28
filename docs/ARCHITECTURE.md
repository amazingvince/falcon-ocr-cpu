# Architecture

Status: current as of 2026-10-28.

## Model contract

The runner implements the pinned Falcon-OCR v1.5 checkpoint (revision
`fe757d59…`, weight SHA-256 `3df91e40…`, both in `reference/manifest.json`):
22 layers, hidden width 768, 16 query heads, 8 KV heads, head dimension 64,
FFN width 2304, vocabulary 65,536, context 16,384. The query width is 1,024,
not the hidden width.

QKV weights are `[2048, 768]`; the FFN gate and up rows are interleaved in
`[4608, 768]` and combine as `ReLU(gate)² · up`. Learned attention sinks
enter only the softmax denominator. Input and Q/K
normalizations have implicit unit weights; the final normalization has its
own epsilon.

Q and K are normalized before rotary positions, and K heads are repeated
before the spatial rotation because every paired head has its own
learned spatial frequencies (collapsing rotated prefix keys to eight heads is
wrong). Temporal positions do not advance over patches, registers or the
image-end marker; non-patch spatial positions are NaN. The attention mask is
causal except inside the bidirectional image prefix.

Preprocessing follows the processor's two resize stages, truncation and
ties-to-even dimension rules, then pixel/channel patch packing with
FP64-to-FP32 scaling. RGB stages use Pillow-exact bicubic uint8 resampling.
Each stage resamples horizontally, then vertically, except that, as in the
pinned Pillow 12.3.0, a page more than 100 times taller than wide whose
height shrinks is resampled vertically first.

The first resize preserves the source mode: palette and 1-bit pages use
nearest-neighbour sampling, alpha pages use premultiplied resampling,
16-bit grayscale is resized before clipping to RGB, and CMYK JPEG channels
are resized before RGB conversion. Other 16-bit PNG channels use Pillow's
truncation to 8 bits. JPEG goes through libjpeg-turbo,
matching the pinned Pillow decoder byte for byte; PNG through the `image`
crate. The prompt starts at the image class token (no BOS) and plain
extraction ends at `OCR_PLAIN`.

## Execution order

```text
image file -> decode, resize, patches, tokens and positions   (preprocess/)
  (--max-dimension auto: the router picks the first resize's cap
   from the page at 1536 before the resize                   (router/))
  (--crop-margins: the first resize is cut to its content box
   before the second resize                              (preprocess/crop.rs))
  -> image projection and full bidirectional prefill
     with the mode's weights and kernels                        (model/)
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

### Resolution routing

`--max-dimension auto` (`router/`, about 12 ms per page) chooses a first
resize cap of 768, 1024 or 1536. It computes 26 statistics after the first
resize at 1536: size, ink, contrast, a projection-profile estimate of text
lines, column gaps, and ink changes after scaling to 1024 or 768 and back.
Two embedded gradient-boosted tree models (`src/router/trees.json`) score
the page. The router takes the smallest resolution whose model predicts
routability at p >= 0.5 and whose median text line remains at least 8 px tall.

A routed page has exactly the input that a fixed `--max-dimension 768` or
`1024` would produce. A repetition or length stop reruns the page at 1536;
`route.safety_net` records the first attempt. Routing changes the model
input and output, so exact and near-exact comparisons use a fixed
resolution. [The router study](../research/resolution-router/README.md)
reports Python parity, timing and quality results.

### Multi-page execution

`Runner::recognize_files_streaming` delivers each page's result or error to
a callback in input order, as soon as that page and every earlier one are
done. Returning `ControlFlow::Break` stops the run. Pages run sequentially
or share joint decode steps up to `--batch-size`; a finished row goes to the
next page (continuous batching, `runner/batch/`). Each page keeps the output
budget it would have alone.

A page failure, such as an unreadable file or a context-budget conflict,
does not stop other pages, but a failed joint decode step fails every page
in it. A buffered failure pauses admissions until earlier pages finish and
the callback decides whether to continue.

`recognize_files`, traced `recognize_files_streaming_with_trace` and
`recognize_batch` use fixed cohorts (`runner/cohort.rs`). Each cohort uses
the smallest budget fitted to any member; a failed prefill fails the whole
cohort. Traces use `request.{i}` for each page's tensors and `batch.{i}` for
joint decode, indexed by the first page that decodes (`batch.1` when page 0
failed to prepare). `recognize_files` stops at the first error.

The page pipeline (`recognize_files_pipelined`, `run --pipeline`,
`runner/pipeline.rs`) splits a sequential call at the prefill/decode
boundary into `Runner::prefill` and `Runner::decode_prefilled`. A prefetch
thread reads and prepares pages; a second pool prefills the next page while
the current page decodes.

The prefill pool takes the threads the decode team leaves. On three or more
threads, an automatic decode team takes at most half; below three, the pool
would have fewer than two threads, so prefill runs between decodes. When
every candidate is larger than half, the tuner measures a half-size team,
reported as `decode auto {6,8; 4 with --pipeline}`. Explicit
`--prefill-threads` sizes the pool and leaves the team unlimited, so that
reserve team goes unused. Prefill outputs are independent of the pool's
thread count, so the pipeline keeps the sequential run's tokens.

## Long runs

`falcon-ocr run` drives the multi-page runner through `book/` and `main.rs`.

**Inputs.** The positional images come first, then the paths of
`--list FILE`: UTF-8 with or without a byte order mark, one per line, blank
lines and lines starting with `#` skipped, relative paths resolved against
the current directory. Before the model loads, every input the run will
read must exist, be a regular file and start like a PNG or JPEG. A JPEG
above the pinned Pillow's limit (178,956,970 pixels) fails its page when it
is decoded.

**Records.** Each page gives one JSON line, in input order: `path` (the
input as given) and `page` (its 0-based index among the inputs), the
`OcrResult` fields, then the generation `options` the run asked for. A page
that fails under `--keep-going` gives `{"path", "page", "error"}`. With
`--text` and no `--output`, its error goes only to stderr. Without
`--keep-going` the run stops at the first failing page, after the records of
the pages before it, and pages after it that `--batch-size N` or `--pipeline` had
already finished are not written and count as not run.

With `--batch-size N` or `--pipeline`, an attempt's `total_ms` is the sum of
its own stages (`time_to_first_token_ms + decode_ms`), which overlap other
pages'. A router safety-net or escalation rerun adds the earlier attempt's
`total_ms`; that accumulated total is larger than the final attempt's stage
sum. The summary on stderr gives the pages done, failed, skipped and not
run, the elapsed time and pages per hour.

**The output file.** `--output FILE` appends the records to FILE, flushed
after every page. Checked with the inputs, before the model loads: FILE's
folder must exist; an existing FILE must hold only records (JSON objects
with a `path`; a byte order mark and blank lines are skipped), so an input
image or list given as FILE is refused; and every input path must be UTF-8,
since a record names its input by its path. A run without `--resume` over a
FILE that already holds successful records of some inputs warns that they
run again.

FILE is created or repaired only after the model has loaded. A device or
pipe (`/dev/stdout`) is written to as it is, without being read. With
`--text`, each page's text goes to stdout, and a closed stdout or stderr
does not stop the run.

**Resume.** `--resume` needs a regular `--output` file and skips the inputs
whose path, as given, already has a successful record there. Skipped inputs
are not read, so they need not exist any more, and no input may be listed
twice.

A last record cut short by a crash, or the NUL bytes a power loss can leave
in its place, is removed, so its page runs again. A record never holds a raw
NUL byte, so a line that does is damage from a power loss, which can
leave NULs in place of earlier records: it is skipped, unless a whole record
follows its leading NULs, and the pages of the records it replaced run
again (a file with damaged lines and no record is refused).

A record that ran with different settings is kept and skipped with a
warning naming the difference: the mode or precision, the GPTQ overlay or
round-to-nearest weights, `--max-dimension`, `--min-dimension`,
`--max-new-tokens` or crop padding, `--stop-repetition`, `--exp`, BF16, NEON
or scalar prefill attention, BF16 projections, a pinned `--tune decode-exp`,
the effective `--tune split-chunks`
or the build's image decoder; a record without `options` counts as one that
ran with different settings. Automatic chunking matches an explicit mode
default, and compact caches ignore the split-cache knob.

An escalated record compares its near-exact rerun's kernel settings,
allowing the expected switch from fast-mode BF16 prefill to FP32, and the
weights of the fast attempt it replaced, which `escalated_from` records;
an escalated record written before those were recorded has none to compare
and matches any weights. With
`--escalate`, a warning counts fast-mode records that the repetition stop
ended without a near-exact rerun, since their pages are skipped.

**Escalation.** `--escalate` (`runner/escalate.rs`) loads the near-exact
model when a fast-mode page first ends on the repetition stop and keeps it
beside the fast one. It selects the near-exact packed file next to
`--model-file`, otherwise the packed file or FP32 checkpoint in the model
directory, as `--mode near-exact` would. Pages are reread one at a time;
other batch rows wait with their caches. A routed page uses the size its
fast run ended at.

The near-exact result replaces the record, keeping the fast attempt in
`escalated_from` (`mode`, `finish_reason`, `output_tokens`, `total_ms`, like
`route.safety_net`); the page's `total_ms` includes that attempt. If loading
or the rerun fails, the fast result remains with `escalation_error` and a
warning on stderr. After a failed load, the run stops trying. The
[mode results](MODES.md#loops-and-the-repetition-stop) explain the
repetition-only policy and its measured effect.

## Modules

| Module | Holds |
|---|---|
| `lib.rs`, `config.rs` | Public re-exports; `ModelConfig`, `GenerationOptions` (with `fit_budget`, `route` and `crop_margins`), `MaxDimension`, `RunnerConfig` (`Default` = automatic, `reference()` = CPU traces and numerical gates), `Tuning`, `Backend`, `HeadMode`, `ExpMode`, `Speculation`, `Drafter`, `DraftKv`, `DecodeThreads` |
| `auto/` | `HostInfo::detect`, `Mode`, `resolve_weights` (packed file, then checkpoint), `Resolved` (the plan a runner executes); `doctor.rs` (`doctor`: host, files, plan, load and probe) |
| `cli.rs` | `RunnerArgs` shared by both binaries and `print_doctor` |
| `main.rs`, `bin/falcon-ocr-eval/` | The CLI; the research binary (profiles, bench, agree, trace, Gram capture) with its telemetry |
| `preprocess/`, `tokenizer.rs` | Pillow-exact image preparation (`SourceImage`: one decode, the first resize at any cap; `resample.rs`: the Pillow resampling port, with 8-bit bilinear and bicubic for the router; `crop.rs`: the opt-in margin crop); the pinned tokenizer and stop ids |
| `router/` | [Resolution routing](#resolution-routing): image statistics, embedded trees, resolution choice and safety net |
| `model/` | `Model` (mapped weights, `Weight`, `Layer`), `load.rs` (checkpoint and profile loaders, `MemoryReport`), `packed.rs` (kernel-ready files), `cache.rs` (`Session`, `LayerCache`, workspaces, `FeatureCapture`: the layer outputs a draft head reads), `fused.rs` (the fused prefill QKV row), `rope.rs` (rotary factors, `split_norm_rope`), `profile.rs` (phase clocks) |
| `quant/` | `Profile { weights, kv }`, `linear.rs` (`QuantLinear`: 8/16-bit codes, FP32 scales, optional FP32 exception columns, panel prefill; `linear/dot.rs` its single- and multi-row decode dots), `kv.rs` (the split KV cache: F32, Q16 and Q8 records, and the research Q8R and Q4R ones; `kv/codec.rs` its scalar codes and scales, `kv/decode.rs` its decode attention kernels), `rotation.rs` (the randomized Hadamard rotation of 32-value blocks behind Q8R and Q4R) |
| `kernels/` | `Simd` dispatch, `linear*`, `rms_norm*`, GLU, `prefill_plan`; `attention/` (`Geometry`, `CompactKv`, one online-softmax head, the tiled GEMM, fixed-width decode, prefill tiles and their BF16 form); `panels/` (the shared panel-GEMM scheduler and the FP32 and BF16 micro-kernels); `exp.rs` |
| `simd.rs` | The 8-lane `Simd` trait with `Portable`, `Avx2`, `Avx2Fast` and `Neon` implementations |
| `runner/` | `Runner` (prefill, sealing, decode), `cohort.rs` (fixed-cohort batches: one prefill per page, joint decode steps), `stream.rs` (results in input order as pages finish), `batch/` (continuous batching: the scheduler, the page feed, the rows), `pipeline.rs` (the page pipeline), `escalate.rs` (fast-mode loops reread in near-exact mode), `generate.rs` (`Generation`: the one stop ladder; results), `speculate.rs` (draft verification) |
| `book/` | Long runs of `falcon-ocr run`: the input list, the input check before the model loads, the per-page JSON record, the output file `--resume` continues (`resume.rs`), the summary |
| `team.rs`, `tune.rs`, `cpu.rs` | The decode spin team; the automatic team-size tuner and its report; host topology, performance cores, the bandwidth probe |
| `head_screen.rs`, `draft.rs`, `repetition.rs` | The exact INT8 vocabulary screen; n-gram drafts, the adaptive draft policy and document history; the repetition stop |
| `draft_head.rs` | The trained draft head (EAGLE-3 style, `research/draft-head`): INT8 projections on the decode kernels, its own INT8 key/value cache of verified positions, confidence-gated draft chains, an optional low-rank vocabulary head |
| `trace.rs`, `packed_kernels.rs`, `buf.rs` | Tensor traces and callbacks; the opt-in phase-packed batch-decode weights; mapped or owned buffers |

## Kernels: one source per operation, instantiated per ISA

Hot kernels are `#[inline(always)]` generic functions over `simd::Simd`, an
8-lane logical `f32` vector. It provides load, store, splat, add, sub, mul,
FMA, max, compare/select, exact `i8`/`i16` widening loads, and `exp`. The two
fixed reduction trees are `sum` for dots and `sum_tree` for RMS norm; RoPE
uses the pair operations `dup_even`, `dup_odd`, `swap_pairs` and `addsub`.

Each ISA gets a small entry wrapper carrying its `#[target_feature]`; NEON
needs none. Implementations share lane layouts and reduction trees, so with
the same exp choice their outputs are bitwise the `Portable` oracle on the
same inputs. Unit tests compare the fast paths against that oracle or the
scalar loop. Runtime feature detection decides dispatch once per process,
so one baseline-target binary runs everywhere.

The 16-lane AVX-512 prefill tiles are bitwise the 8-lane ones. The `gemm`
crate's FP32 prefill projections, fast mode's BF16 kernels, and the choice
between platform and polynomial exp can change rounding; their fidelity is
token-checked. See [PORTABILITY.md](PORTABILITY.md) for ISA-specific paths
and [MODES.md](MODES.md#the-three-modes) for the numerical choices.

`kernels::prefill_plan` is the one predicate that decides which prefill
projection and attention kernels a body and CPU get; `Model::forward_layers`
runs it and `auto::Resolved` reports it.

## KV cache formats

The compact cache (`LayerCache::Compact`) keeps prefix keys per query head
(their spatial rotations differ), generated keys and every value per KV head,
all FP32: the reference. After a page's first token (unless the page ends
there), its prefix is sealed into the profile's split store (`quant/kv.rs`):
one 160-element record per KV group and position holding the temporal key
half shared by the pair, each head's spatial half and the value, so one
decode task per group streams its records contiguously.

Records are FP32 (`split-f32`, bitwise the compact kernel), 16-bit codes
with a BF16 absmax scale per 32 values (`q16`), or 8-bit codes (`q8`);
generated positions go
into 128-element tail records in the same format as they are appended.

The research formats `q8r` and `q4r` (`falcon-ocr --kv-cache`,
`falcon-ocr-eval --profile`) rotate each 32-value block before quantizing:
the temporal and spatial key halves and both value halves, in prefix and
generated positions. The transform is `H D x`, where `H` is the 32 × 32
Sylvester Hadamard matrix and `D` a fixed ±1 diagonal per block kind
(`quant/rotation.rs`).
Rotation spreads an outlier channel across the block's other 31 values.
`q8r` uses `q8`'s codes and BF16 scale per 32 values; `q4r` uses 4-bit
codes (-7..=7) with the same scale.

Decode rotates queries, `(2^-5 H D q) · (H D k) = q · k`, and un-rotates
each head's output, `o = 2^-5 D H o'`, so both formats share `q8`'s kernels.
`q8r` streams 170 bytes per record. Its 32-point transforms run four times
per head, row and layer (two query halves and two output halves), four per
group for each appended position and five per record when sealing the
prefix. `q4r` takes 90 bytes per record and unpacks through the 8-bit loads;
there is no vector nibble unpack. The kernels are bitwise tested on every
instruction set.

With one position chunk (`--tune split-chunks=1`), output is bitwise the
compact kernel over the dequantized rotated records and rotated query,
followed by the inverse rotation. Coded caches scan four chunks by default;
merging their partial softmaxes changes rounding. See
[MODES.md](MODES.md#rotated-kv-cache-experimental) for agreement results.
The `expanded` layout (every KV head duplicated) remains selectable.

Every non-compact profile seals prefills for single pages, fixed cohorts
and continuous batching. The compact `reference` profile never seals.
`falcon-ocr` loads it for traces, which always run in exact mode, so their
tensors stay comparable with the recorded references, and for exact-mode
`run` at a batch size above 1. An exact single-page `run` takes `split-f32`;
near-exact and fast seal in every path.

## Decode machinery

Decode is memory-bound: it streams the body weights, the head and the KV
cache once per token. A spin team (`team.rs`) runs the roughly 110 short
parallel loops of a step without Rayon's wake-up latency; `tune.rs` times a
few team sizes on the first steps and keeps the smallest within 2% of the
fastest (tokens never depend on it).

The screened head selects the argmax through an INT8 copy with a proven
error bound and recomputes only the rows that can win.

By default speculative decoding drafts up to four tokens and verifies them
in one multi-row step whose rows are bitwise the single-row steps;
`draft::DraftPolicy` turns drafting off while it does not
pay. Drafts come from n-gram matches in the output so far or, with
`--draft-head`, from a trained head that reads the outputs of target layers
2, 11 and 19 at every verified position (captured during the forward) and
drafts while its top token's probability is at least 0.35.

The [repetition stop](MODES.md#loops-and-the-repetition-stop) can end a
page before EOS or the output budget.

## Automatic configuration

`auto::resolve_weights` looks for the mode's
[catalog file](MODES.md#packed-model-files) in `--model` first and maps it
in place. Otherwise exact reads the FP32 checkpoint; near-exact and fast
quantize it at load. Fast mode needs the GPTQ overlay; an explicit
`--w8-artifact` always reads the checkpoint. `auto::Resolved` states
everything the runner will do: mode and profile, weights source, prefill and
decode kernels, thread plan, KV format, bytes per token, image decoder and
tuning. `falcon-ocr doctor` prints it without reading tensors; every result
carries it as `plan`.
