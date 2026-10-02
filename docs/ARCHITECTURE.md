# Architecture

Status: current as of 2026-10-02.

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
truncation and ties-to-even dimension rules, then pixel/channel patch
packing with FP64-to-FP32 scaling. Each stage resamples horizontally, then
vertically, except that, as in the pinned Pillow 12.3.0, a page more than 100
times taller than wide whose height shrinks is resampled vertically first;
palette and 1-bit pages take Pillow's nearest-neighbour first resize. The
prompt starts at the image class token (no BOS) and plain extraction ends at
`OCR_PLAIN`. JPEG goes through libjpeg-turbo, matching the pinned Pillow
decoder byte for byte; PNG through the `image` crate with Pillow's 16-bit
truncation.

## Execution order

```text
image file -> decode, resize, patches, tokens and positions   (preprocess/)
  (--max-dimension auto: the router picks the first resize's cap
   from the page at 1536 before the resize                   (router/))
  (--crop-margins: the first resize is cut to its content box
   before the second resize                              (preprocess/crop.rs))
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

A run of many files (`Runner::recognize_files_streaming`) hands each page's
result or error to a callback as soon as the page and every earlier one are
done, in input order, and the callback stops the run by returning
`ControlFlow::Break`. A page that fails on its own (an unreadable file, a
context budget conflict) does not stop the others, but a joint decode step
that fails fails every page in it, and in a fixed cohort so does a failed
prefill. Pages run one after another, or `--batch-size` at a time in joint
decode steps, where a finished page's row goes to the next page
(continuous batching, `runner/batch/`), each keeping the output budget it
would have alone. A buffered failure pauses admissions until earlier pages
finish and the callback decides whether to continue. `recognize_files`, the
traced `recognize_files_streaming_with_trace` and `recognize_batch` keep fixed
cohorts of the batch size (`runner/cohort.rs`): a cohort decodes to the
smallest budget fitted to any page in it, and a trace names each page's
tensors by its input index (`request.{i}`; a cohort's joint decode is
`batch.{i}`, with the index of the cohort's first page that decodes, so
`batch.1` when page 0 failed to prepare). `recognize_files` stops at the
first error. The page pipeline (`recognize_files_pipelined`,
`run --pipeline`, `runner/pipeline.rs`) splits each page at the
prefill/decode boundary (`Runner::prefill`, `Runner::decode_prefilled`, the
two halves of the one sequential call): a prefetch thread reads and prepares
the next pages, and the next page's prefill runs on a second pool while the
current page decodes. That pool gets the threads the decode team leaves, and
on three or more threads an automatic team then takes at most half of them
(below three the default pool would have fewer than two threads, and pages
prefill between decodes); when every candidate is larger, it measures a
half-size team the plan lists as `decode auto {6,8; 4 with --pipeline}`.
An explicit `--prefill-threads`
sizes the pool and leaves the team unlimited, so the half-size team goes
unused. Every prefill stage computes each output independently of its pool's
thread count, so the pipeline's tokens are a sequential run's.

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
that fails under `--keep-going` gives `{"path", "page", "error"}`; without
it the run stops at the first failing page, after the records of the pages
before it, and pages after it that `--batch-size N` or `--pipeline` had
already finished are not written and count as not run. With
`--batch-size N` or `--pipeline`, a page's `total_ms` is the sum of its own
stages (`time_to_first_token_ms + decode_ms`), which overlap other pages'. A
summary on stderr gives the pages done, failed, skipped and not run, the
time and the pages per hour.

**The output file.** `--output FILE` appends the records to FILE, flushed
after every page. Checked with the inputs, before the model loads: FILE's
folder must exist; an existing FILE must hold only records (JSON objects
with a `path`; a byte order mark and blank lines are skipped), so an input
image or list given as FILE is refused; and every input path must be UTF-8,
since a record names its input by its path. A run without `--resume` over a
FILE that already holds successful records of some inputs warns that they
run again. FILE is created or repaired only after the model has loaded. A
device or pipe (`/dev/stdout`) is written to as it is, without being read.
With `--text`, each page's text still goes to stdout, and a closed stdout or
stderr does not stop the run.

**Resume.** `--resume` needs a regular `--output` file and skips the inputs
whose path, as given, already has a successful record there. Skipped inputs
are not read, so they need not exist any more, and no input may be listed
twice. A last record cut short by a crash, or the NUL bytes a power loss can
leave in its place, is removed, so its page runs again. A record never holds
a raw NUL byte, so a line that does is damage from a power loss, which can
leave NULs in place of earlier records: it is skipped, unless a whole record
follows its leading NULs, and the pages of the records it replaced run
again (a file with damaged lines and no record is refused). A record that
ran otherwise is kept and skipped with a warning naming the difference: the
mode or precision, the GPTQ overlay or round-to-nearest weights, the
`--max-dimension`, `--min-dimension`, `--max-new-tokens` or crop padding,
`--stop-repetition`, `--exp`, BF16, NEON or scalar prefill attention, BF16
projections, or a pinned `--tune decode-exp`; a record without `options`
counts as one that ran otherwise. An escalated record compares its
near-exact rerun's kernel settings, allowing the expected switch from
fast-mode BF16 prefill to FP32. With `--escalate`, a warning counts the
fast-mode records that the repetition stop ended without a near-exact rerun,
since their pages are skipped.

**Escalation.** `--escalate` (`runner/escalate.rs`) loads the near-exact
model when a fast-mode page first ends on the repetition stop and keeps it
beside the fast one; the other batch rows wait with their caches while the
page is reread. The record becomes the near-exact result with the fast
attempt in `escalated_from`, or keeps the fast result with an
`escalation_error` when the near-exact model cannot be loaded or the rerun
fails ([MODES.md](MODES.md#loops-and-the-repetition-stop)).

## Modules

| Module | Holds |
|---|---|
| `lib.rs`, `config.rs` | Public re-exports; `ModelConfig`, `GenerationOptions` (with `fit_budget`, `route` and `crop_margins`), `MaxDimension`, `RunnerConfig` (`Default` = automatic, `reference()` = bit-exact), `Tuning`, `Backend`, `HeadMode`, `ExpMode`, `Speculation`, `Drafter`, `DraftKv`, `DecodeThreads` |
| `auto/` | `HostInfo::detect`, `Mode`, `resolve_weights` (packed file, then checkpoint), `Resolved` (the plan a runner executes); `doctor.rs` (`doctor`: host, files, plan, load and probe) |
| `cli.rs` | `RunnerArgs` shared by both binaries and `print_doctor` |
| `main.rs`, `bin/falcon-ocr-eval/` | The CLI; the research binary (profiles, bench, agree, trace, Gram capture) with its telemetry |
| `preprocess/`, `tokenizer.rs` | Pillow-exact image preparation (`SourceImage`: one decode, the first resize at any cap; `resample.rs`: the Pillow resampling port, with 8-bit bilinear and bicubic for the router; `crop.rs`: the opt-in margin crop); the pinned tokenizer and stop ids |
| `router/` | The resolution router (`--max-dimension auto`, `research/resolution-router`): 26 image statistics of the page after the first resize at 1536, bit-exact with the Python specification; the two embedded gradient-boosted tree models (`trees.json`); the 768/1024/1536 choice with its 8-px text-line floor; the safety-net record |
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
all FP32: the reference. After a page's first token (unless the page ends
there), its prefix is sealed into the profile's split store (`quant/kv.rs`):
one 160-element record per KV group and position holding the temporal key
half shared by the pair, each head's spatial half and the value, so one
decode task per group streams its records contiguously. Records are FP32
(`split-f32`, bitwise the compact kernel), 16-bit codes with a BF16 absmax
scale per 32 values (`q16`), or 8-bit codes (`q8`); generated positions go
into 128-element tail records in the same format as they are appended. The
research formats `q8r` and `q4r` (8- and 4-bit codes; `falcon-ocr
--kv-cache`, `falcon-ocr-eval --profile`) store every 32-value block as
`H D x` (a Hadamard matrix times fixed signs, `quant/rotation.rs`) before
quantizing it; decode rotates the query and un-rotates each head's output
instead of the records, so they share `q8`'s kernels
([MODES.md](MODES.md#rotated-kv-cache-experimental)). The `expanded` layout
(every KV head duplicated) remains selectable.

Every path that prefills a page seals it: a page that runs alone, the pages
of a fixed cohort and the rows of continuous batching alike. Only a profile
whose cache is compact (`reference`) never seals. `falcon-ocr` loads it for
traces, which always run in exact mode, so their tensors stay comparable with
the recorded references, and for exact-mode `run` at a batch size above 1; an
exact single-page `run` takes `split-f32`. Near-exact and fast seal in every
path.

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
quantized at load (fast mode needs the GPTQ overlay; an explicit
`--w8-artifact` always reads the checkpoint). `auto::Resolved` states
everything the runner will do: mode and profile, weights source, prefill and
decode kernels, thread plan, KV format, bytes per token, image decoder and
tuning. `falcon-ocr doctor` prints it without reading tensors; every result
carries it as `plan`.
