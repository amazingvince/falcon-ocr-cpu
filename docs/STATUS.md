# Qualification status

Updated 2026-09-20. The persistent goal remains active.

The user explicitly requested a compiler update after the RTen 0.26 dependency
review identified a Rust 1.94 requirement. The project now pins Rust 1.94.0;
that toolchain is installed on native Windows and in the existing WSL distro.
The original 1.92 toolchains and evidence remain available. New matrix probes
compile control and candidate together under 1.94; old performance results are
not relabeled. Upgrade validation is complete: native Windows and WSL each
passed 50 enabled regression tests and the saved GPU-reference OCR smoke test.
The native 1.94 trace matches all 1,904 saved 1.92 CPU tensors bit for bit.
See `experiments/toolchain_194/RESULTS-V1.md`. The first RTen 0.26 probe builds
and passes its scoped arithmetic checks; timing and full-model qualification
remain pending.

The head-contiguous prefix experiment completed with all nine full-page outputs
exact, but its 0.61–1.40% gain missed the 5% target. It is not promoted. See
`experiments/head_contiguous_prefix/RESULTS-V1.md`. Matrix-kernel comparisons
are now the main performance priority, alongside the continuing numerical work.

The user approved continuing locally and leaving bare-metal Linux performance
measurements pending. WSL results remain explicitly labeled; this deferral does
not imply bare-metal Linux performance qualification.

The user subsequently prioritized single-page performance: capture and validate
a usable CPU profile, optimize its dominant decode costs in isolated variants,
measure against the completed unprofiled baseline, and check output parity after
each change. The ten intermediate numerical failures remain under investigation
alongside this work; they, deferred bare-metal Linux tests and the remaining
batch matrix do not block isolated performance experiments. The early mixed-B2
benchmark attempt was stopped and preserved as incomplete to make this shift.

A native full-page CPU capture now preserves three exact outputs and a usable
exploratory hotspot ranking: attention call paths account for 51.86% of sampled
CPU work, followed by linear projections at 26.85%. Stack attachment is 99.9469%.
One kernel-only sample 189.9 microseconds beyond process stop remains a strict
trace-audit failure; PerfView also returned an exception after writing its
parseable export. These limits are explicit in
`experiments/profiling/FULLPAGE-RESULTS-V1.md`. The first isolated candidate
specializes fixed-size AVX2 attention without changing its arithmetic order;
six operator checks and all 1,904 saved smoke tensors now match bit for bit.
Its completed full-page control/candidate/control bracket measured 68.444 s
against 75.570/72.931 s controls: 6.15–9.43% less whole-page latency, with all
nine measured outputs exact. Whole-page control drift passed at 3.49%; stage
timings varied more, so their precise gains remain less certain. See
`experiments/attention64/RESULTS-V1.md`. This isolated candidate is not promoted
and has not rerun the full corpus or other workloads. No approval is pending
for the completed capture or continued local work.

The combined performance experiment applies that specialization to the
compact-cache path. Its source retains the original prefix/generated indexing
and arithmetic. Eleven operator checks and the byte-identical 1,904-tensor
compact smoke passed; eight warmed decode intervals allocated zero heap memory.
The fresh compact control/candidate/control bracket completed at 63.102 s per
page versus 67.414/67.239 s compact controls: a further 6.15–6.40% reduction,
with all nine measured outputs exact and 0.259% control drift. See
`experiments/attention64_compact/RESULTS-V1.md`. This proves an added gain over
compact caches on this page; it does not establish additive historical gains
or default-promotion eligibility. Its unchanged source also passed Linux-under-WSL
qualification: eleven operators, all 1,904 same-platform smoke tensors, 17 teacher
IDs/text/stopping behavior and eight zero-allocation decode intervals. See
`reference/linux-attention64-compact-functional-v1.json`. A fresh three-process
compact mixed-B2 comparison completed for the frozen sparse/table pair:
128.912 s per pair versus 142.475/141.837 s controls, a 9.11–9.52% reduction
with 0.448% control drift. All 18 measured request outputs exactly match the
pinned GPU outputs (6 and 2,280 tokens per pair); the independent saved-result
review passed in `reference/benchmarks/windows-attention64-compact-mixed-b2-review-v1.json`.
The short shared decode interval and long singleton tail do not establish
sustained B2/B4/B8 throughput. The interrupted earlier matrix remains preserved.
In parallel, the bounded layer-7
numerical crossover completed with all ten original controls exact. At the
preselected hidden coordinate [112,249], the incoming-state term is +0.00233459
and the same-entry engine term is −0.00003815; they partially cancel. This
supports an earlier origin for that coordinate's difference, without identifying
a faulty operator or extending the attribution to every row. See
`reference/fp32-crossover-layer7-decomposition-v1.md`; the independent saved-result
review also passed. No numerical gate or tolerance has changed.

The completed single-page experiment in `experiments/gemv_pair` pairs
AVX2 output channels with unchanged per-output FMA/reduction order, limited to
the five actual single-row projection shapes. Seven operator checks are authored,
including full real-operand outputs; source review is complete, with the fresh
Cargo benchmark-artifact check added to the capture helper. After the mixed-B2
window closed, all four source guards and a fresh copied-project build passed.
All seven operators (including 73,728 real-input outputs), the byte-identical
1,904-tensor/17-token smoke and eight zero-allocation intervals passed. Compiled
code confirms shared input loads, independent accumulators and no hot-loop
calls or spills. The independently reviewed full-page bracket completed at
62.041 s versus 62.849/62.766 s controls, with all nine outputs exact and 0.132%
control drift. Its 1.16–1.29% latency reduction misses the frozen 5% target, so
the candidate remains unpromoted and further qualification is not queued.
See `experiments/gemv_pair/RESULTS-V1.md`.

The next isolated candidate, `experiments/attention64_temporal`, combines shared
prefix temporal-key storage with direct fixed64 AVX2 attention. It has an
explicit cache mode and retains the combined-attention compact build as control.
The fresh Windows build passed thirteen operator/cache tests, all 1,904 canonical
and 2,144 mixed tensor hashes, seventeen teacher decisions, independent free
outputs and two zero-allocation decode intervals. Its independently reviewed
single-page bracket measured 60.730 s versus 62.818/62.758 s controls, with all
nine outputs exact and 0.096% control drift. The 3.23–3.32% gain misses the 5%
target, so this candidate remains unpromoted. Its analytic 140.594 MiB reserved
KV saving is a separate benefit. A test-import compile failure and its source
snapshot were preserved before the corrected fresh build.
See `experiments/attention64_temporal/RESULTS-V1.md`.

The numerical follow-up returned to the original layer-9 V failure through a
controlled downstream continuation. That diagnostic
completed with all 17 fresh control stages exact and all 34 new payloads checked.
At V[112,14,2], the original +0.00662231 difference decomposes into downstream
engine +0.00003815, propagated layer-7 engine −0.00004578 and earlier-state
+0.00662994. The earlier-state term alone still exceeds the unchanged bound;
no numerical gate has closed. See
`reference/fp32-crossover-downstream-completion-v1.json`; the independent
saved-result audit in `reference/fp32-crossover-downstream-independent-review-v1.json`
also passed.

## Established

- Pinned v1.5 artifacts downloaded, hashes verified and dependencies recorded.
- Real strict FP32 HF GPU reference exported on an isolated RTX 4090 in WSL.
- Rust FP32 library/CLI run on native Windows and Linux under WSL.
- Free-running Windows Rust output matches the smoke fixture's 17 GPU token IDs,
  including end-of-query token 263: `Falcon OCR\n\nHello, world!\n\n12345`.
- Canonical RGB resizing/patch fixtures pass exact pixel comparisons. Floating
  spatial coordinates use separately measured PyTorch/NumPy bounds, maximum
  absolute difference 5.960464477539063e-8 and maximum 2 ULP on those fixtures.
- Windows and Linux tests cover 75 exact PNG/JPEG preprocessing cases (39 PNG, 36 JPEG)
  and 12 exact JPEG RGB decodes. The initial Rust JPEG decoder differed by up to
  three pixel levels; vendored libjpeg-turbo passes the pinned Pillow fixtures.
- Five focused negative-input tests pass on Windows and WSL, including explicitly
  selected model-dependent checks. Public RGB/file APIs and the CLI reject the
  tested malformed inputs and invalid options; valid empty library batches return
  no results. Mixed failures are covered within the first chunk only. These close
  the two bounded gaps in the first-release image/API audit without changing
  production code. See `reference/negative-input-validation-v1.md`.
- The strict FP32 GPU reference now completes all 200 corrected-corpus pages:
  275,903 generated IDs, 181 EOS stops and 19 explicit 4,096-token length stops.
  Windows also completes all 200 pages, matching every ID, corrected text and
  stop with valid provenance and no failed records. Twelve texts use separately
  bound saved-token decoding; the original inference is preserved. The fixed
  corpus quality gate passes with zero difference overall and in every category,
  and audited official component outputs match exactly. See
  `reference/windows-rust-corpus-v3-fp32-redecoded-200.json` and
  `reference/quality-regression-v3-fp32-complete-v1.json`.
- All 15 original supplemental pages match CPU/GPU token IDs, literal text,
  prefix lengths and stopping behavior: 1,056 IDs and 15 EOS stops. The CPU
  saved-token replay changes none of these texts. Intended-text diagnostics
  remain separate from parity: the two blank pages retain the upstream marker,
  and some rotated/multilingual content errors remain. See
  `reference/original-supplement-fp32-redecoded-gpu-comparison-v1.md`.
- The official repository's strict FP32 GPU engine matches the same 17 HF tokens.
- Reusable hidden/logit/rotary buffers eliminate measured heap allocations during
  warmed-up smoke decoding. The entire 17-step tensor trace remains byte-identical
  after this refactor; this is separate from the still-open GPU numerical gate.
- Mixed batches of 2/4/8 preserve sequential token IDs, order, EOS and exact caps
  on Windows and Linux integration fixtures. Joint decode projections are implemented;
  both single and four-request warm decode allocation tests pass with zero calls.
- All 24 full-size canonical smoke pages match GPU text, stopping behavior and
  every token exactly (28,734 tokens). Artifact provenance checks pass. These are
  qualitative v1 smoke results, not the corrected corpus quality gate; see
  `reference/windows-rust-corpus-smoke-fp32.json`.
- The corrected v3 smoke subset also completes on Linux under WSL: all 24 pages
  match GPU token IDs, literal text and stops exactly, totaling 29,205 IDs. This
  is fresh inference from the preserved Linux build, with no text replay. It
  closes this selected-subset output comparison, not full-corpus quality or
  bare-metal performance. A direct saved-output join also confirms all 24 pages
  agree with the completed Windows run, including CPU dimensions and GPU prefix
  counts. See `reference/linux-corpus-v3-smoke-fp32-summary.json` and
  `reference/windows-linux-corpus-v3-smoke-output-agreement-v1.json`.
- The actual pinned vLLM HTTP service matches the tiny fixture's 17 output IDs,
  text and 144-token prompt. Compiled Triton attention PTX confirms IEEE controls
  and no TF32 instructions. See `reference/vllm-smoke-fp32.json`.
- The official plain engine and digest-pinned direct vLLM service additionally
  match all three selected natural full pages: prose, a table and multiple
  columns. Each engine matches 4,881 HF token IDs, literal text, prefix lengths
  and EOS stops at maximum side 1536 and output cap 4096. vLLM returned the EOS
  IDs directly; no inferred stop token was needed. Its compiled attention PTX
  contains no TF32 instructions under the recorded IEEE controls. Saved response
  bytes, reference bindings and harness source archives were independently
  checked. These are single-request functional comparisons under concurrent
  load, not corpus-wide or performance qualification. The project-owned server
  is stopped. See `reference/official-fullpages-fp32.json` and
  `reference/vllm-fullpages-fp32.json`.
- A strict GPU numerical stress page executed an 8,100-token prefix and exactly
  8,192 generated tokens, stopping at the requested limit with a 16,384-token
  cache capacity. Windows Rust matches all 8,192 IDs, text and exact stopping.
  Its synthetic OCR quality is poor; this is boundary execution evidence only.
  See `reference/windows-rust-long-numeric-fp32.json`.
- Native Windows additionally completes the exact 16,384-token budget: an
  8,100-token prefix plus 8,284 emitted IDs, with every ID, text and length stop
  matching the GPU reference. The preserved v6 build and provenance checks pass.
  This is a synthetic functional boundary test; CPU final KV cursor was not
  recorded, and its timings are not a quiet performance measurement. See
  `reference/windows-rust-exact-context-boundary-fp32-v1.json`.
- The pinned official document component evaluator produces identical CPU/GPU
  results after auditing all 200 v3 pages. Coverage is 194 text/reading-order
  pages, 44 formula pages and 89 tables on 62 pages. Two dangling source merge
  links on two paper pages are explicitly removed in evaluator inputs; original
  annotations remain frozen. The negative upstream TEDS sample is preserved
  without clamping or exclusion. These are adapted component scores, without
  rendered CDM or an official Overall score. The earlier v1 smoke report remains
  historical. See `reference/official-components-v3-fp32.json` and `docs/EVALUATION.md`.
- Shared prefill scratch preserves all 2,256 mixed-batch tensor values bit for bit
  and saves 25.25 MiB of measured Rust heap on that fixture. Opt-in compact caches
  preserve 1,904 single-request and 2,256 mixed-batch tensors bit for bit, with
  another 25.4375 MiB saved on the mixed fixture. This excludes mapped weights and
  native allocations. Both layouts pass warmed single/batch zero-allocation tests
  on Windows and Linux. Expanded caches remain the default pending performance
  qualification. See `reference/shared-prefill-workspace.json` and
  `reference/compact-cache-parity.json` for the Linux measurements.
- An isolated prefix-key storage candidate shares identical temporal halves
  while preserving all spatial heads and existing attention arithmetic. Eight
  focused tests pass on Windows and WSL/Linux from identical candidate source,
  including exact comparisons against each platform's compact attention on
  scalar, AVX2 and AVX-512. For one layer with 144 prefix and 17 reserved generated
  tokens, measured Vec capacity payload falls from 954,368 to 806,912 bytes.
  This excludes allocator overhead and process memory. The isolated five-run
  model comparison passes separately on Windows and WSL/Linux: all 1,904
  canonical and 2,144 mixed-batch tensor records, actual decisions and free
  outputs match each platform's original expanded/compact controls. Each
  platform's ten warmed decode intervals report zero allocations. Windows/Linux
  free outputs and actual argmaxes also agree, but 1,846 canonical and 2,066
  mixed tensor hashes differ; hashes alone do not quantify numerical error.
  A separate reviewed Windows report corrects a teacher-forced stop-label
  assertion; the original failed report and rejected shared-build-target attempt
  remain preserved. These fixed smoke/mixed checks do not establish cross-OS
  tensor equivalence, GPU hidden-stage or full-corpus candidate quality, or
  performance. Builds are source-bound, not hermetic attestations; live source
  and defaults are unchanged. See `reference/prefix-temporal-model-windows-v1.md`,
  `reference/prefix-temporal-model-linux-v1.md`,
  `reference/prefix-temporal-candidate-windows-v1.json` and
  `reference/prefix-temporal-candidate-linux-v1.json`.
- Independent numerical kernel tests cover matrix shapes, masking, sinks, tails
  and bounded tiled attention. AVX2/FMA and explicit AVX-512 paths are present.
- Opt-in phase-packed FP32 weights preserve all tested single/mixed/batch2/4/8
  traces and outputs on Windows and WSL, including live-batch compaction and
  warmed zero-allocation decode. Packing is shared once per Model and adds
  835.5 MiB. Repeated native Windows small-fixture runs lower batch2/4/8 latency
  by 18%, 28–30% and 21–24%, with unchanged single-page latency within variation.
  Full-page qualification remains pending; Unpacked stays the default.
- The expanded corpus exposed punctuation-space mismatches despite identical
  generated IDs. The pinned Transformers 5.14.1 BPE backend ignores the model's
  WordPiece cleanup flag. Rust now preserves that spacing and matches Python's
  outer-strip character set. All 14 independently exported decoder cases pass on
  Windows and WSL. The running corpus binary remains unchanged; separately bound
  saved-token text replay preserves its original inference evidence. This is not
  a fresh inference run. See `reference/tokenizer-cleanup-v1.json`.
- A subsequent fresh inference regression covers the three known affected pages:
  all 1,054 token IDs, literal text and stops match their GPU references. This
  deliberately selected regression is separate from the held-out corpus quality
  result. See `reference/tokenizer-regression-fp32-v1.json`.
- A native Windows AOCL-DLP FP32 operator probe passed its checked FFI and
  known-answer cases. On 28 equal-input projections, its maximum error versus
  GPU improves/ties/worsens on 11/7/10 cases relative to Rust. Prefill W2 improves,
  but prefill W13 loses Rust's bit-exact match; no blanket replacement, model
  qualification or timing claim follows. The pinned build uses no OpenMP and
  remains separate from the runner. See `reference/aocl-windows-build-v1.json`
  and `reference/aocl-fp32-operator-probe-v1.json`.
- INT4 feasibility and an isolated scalar reference are documented in
  `docs/QUANTIZATION.md`. No quantized production backend or speed/quality result
  is claimed. Transformer-only group-64 W4A32 would use 476.75 MiB of weight
  payload with the other weights kept in FP32, excluding caches and overhead.

## Open numerical gate

The original Windows trace passed all 17 logit/argmax checks but failed 40 of 1904
intermediate tensor thresholds frozen from strict GPU versus dense GPU attention.
Linux failed 37. Reproducing upstream's separate attention-sink sigmoid operation
reduced the Windows failures to 29. Pairwise FP32 RMS reduction, independently
validated against GPU and F64 operator references, reduces this to 10. Investigation continues; the frozen policy
has not been relaxed. The reports live in `reference/`.

Some image-patch hidden states amplify small floating-point differences even when
the final prompt logits agree closely. Passing output tokens alone does not close
this gate. Teacher-forced traces are distinct from free-running output checks.

Equal-input diagnostics find exact gate/residual arithmetic, with small projection,
normalization and attention differences. Injecting exact GPU image embeddings
does not close the full trace gate. A test-only normalization tree based on the
pinned PyTorch CUDA source increases the Windows failures from 10 to 24; adding
a once-rounded reciprocal square root increases them to 60. All 17 argmax values
still match. These alternatives were not promoted, and the frozen policy is
unchanged. See `reference/projector-intervention-summary.json`,
`reference/equal-input-substages-windows-v3.json` and
`reference/rms-intervention-summary.json`.

A subsequent isolated-copy test retains the production width64 Q/K norm and
applies the CUDA-shaped, once-rounded reciprocal-square-root variant only at
width768. Its intermediate failures increase from 10 to 44; all 17 argmax values
still match. Production controls before/after are byte-identical. This candidate
is also rejected; see `reference/rms-width768-intervention-summary-v1.json`.
Separate Windows/WSL scale captures agree bit for bit, and an independent integer
rounding analysis finds one scale consistent with each of 1,444 saved GPU rows.
The subsequent actual fused CUDA export reproduces all 14 original output tensors
bit for bit and confirms every one of those 1,444 scale values. This observes
rstd, not the GPU variance or sum. See `reference/rms-observed-rstd-v1.json`.
The pinned CUDA source agrees with the diagnostic reduction tree, but its
`rsqrtf` operation is not guaranteed to match either CPU square-root variant.
Actual same-argument GPU replay now matches all 1,444 fused rstd values when
given the CUDA-shaped CPU variance arguments, versus 1,184 with production
variance arguments. This establishes compatible arguments and an intrinsic
difference without observing GPU variance. See
`artifacts/reference/rms-rsqrt-replay-fp32/report.json` and
`reference/rms-cuda-source-diagnosis-v1.md`. A diagnostic-only rsqrt table then
matches all 1,266,679,808 positive normal F32 arguments at or above EPS against
the observed GPU operation; the Rust lookup matches all 1,057 captured boundary
pairs. Its isolated width768 full-graph intervention nevertheless increases
failures from 10 to 18, with all 17 argmax decisions unchanged and byte-identical
production controls before/after. The candidate is rejected. No production
table design is adopted; see `reference/rms-gpu-rsqrt-width768-intervention-v1.json`.

A test-only AOCL intervention applies its FP32 kernel to every prefill W2 while
retaining the current Rust arithmetic elsewhere. Despite closer equal-input W2
results, full-trace failures increase from 10 to 47. All 17 argmax values still
match, and independent production controls before/after the intervention are
byte-identical. This candidate is not promoted; see
`reference/aocl-w2-intervention-summary-v1.json`. Isolated operator accuracy does
not predict the accumulated error of the full graph.

A bounded CUDA profiler capture now reproduces four saved projection outputs
exactly. Prefill W2 launches an SGEMM kernel followed by an explicitly named
split-K reduction kernel; W13 and the two decode projections have their own
recorded launch details. A subsequent supported logger replay preserves both
prefill outputs and launch sequences, and records actual `cublasSgemm_v2` calls:
W13 uses two splits with in-place reduction; W2 uses fourteen splits with
compute-type reduction. These are logged execution fields, not grid deductions.
These logs alone do not establish the K boundaries or accumulation order.
Profiled times are not benchmarks. See
`reference/linear-cuda-kernel-identities-fp32.json` and
`reference/linear-cublas-logged-algorithms-fp32.json`.

A fresh public cuBLAS handle with an owned 32 MiB workspace subsequently matches
the original W2 output bits, logged settings and kernel launches. Three coded
one-sparse probes test all 2,304 K positions and agree completely with one fixed
partial-matrix layout. Under that hypothesis, the observed K membership is
thirteen contiguous ranges of 165 and a final range of 159. The saved real
partial matrices can now be compared directly with CPU arithmetic. This remains
a conditional observation of one shape and execution configuration, not a public
workspace ABI or proof of within-partition or final accumulation order. See
`experiments/linear/OWNED-WORKSPACE.md` and
`artifacts/reference/linear-owned-workspace-fp32-v1/report.json`.

The fixed observed-partition Rust probe matches all 1,548,288 real partial values
and all 110,592 final values exactly on native Windows and Linux under WSL.
Its ascending FP32 fold is also byte-identical across platforms. The unsplit
control retains 98,306 differing values and maximum absolute error 0.00830078125,
matching the earlier diagnostic summary. All three focused tests pass on both
platforms. This isolates an equal-input projection discrepancy; it does not close
the full-graph gate or justify a performance claim. Production stays unchanged.
See `reference/w2-observed-partitions-rust-v1.json`.
The subsequent isolated full-graph intervention replaces exactly 22 prefill W2
calls and increases failures from 10 to 23: all ten original failures persist,
with thirteen additional failures. All 17 argmax decisions still match, and
production controls before/after remain byte-identical. The candidate is rejected
under the unchanged policy; see `reference/w2-observed-fullgraph-intervention-v1.json`.

## Preliminary measurements

The later [same-binary Windows comparison](PERFORMANCE.md) pauses other project
CPU/GPU jobs and brackets candidates with repeated controls. On the small fixture,
joint decoding improves group-8 throughput by 1.93–1.94x; single-page latency is
1.2–2.5% slower. Control drift is at most 1.28%, and every output ID matches.
Full-page/mixed-length performance qualification is still pending. Compact cache
batch speed gains are below 5% here, so Expanded remains the default.

A native Windows Ryzen 7950X run of the 256x128 three-line fixture (144 prompt
tokens, 17 emitted tokens) took about 513 ms after model loading at 16 threads:
130 ms prefill and 383 ms decode. Verified model load took about 770 ms. This was
one development smoke run, not a stabilized benchmark or representative OCR rate.

The later [seven-sample sequential baseline](../reference/benchmarks/ocr-windows-avx2-16t-sequential.json)
uses the corrected sink/RMS arithmetic and reusable decode buffers. At 16 threads,
medians are 562 ms for one page and 1.210/2.489/4.857 seconds for sequential groups
of 2/4/8 pages. Peak process resident memory was about 1.154 GB. A GPU corpus job
with CPU orchestration ran concurrently, so this is a development baseline;
quiet, matched runs are still required for performance promotion.

Operator sample reports are under ignored `artifacts/benchmarks/`, with source and
dependency hashes. Small-row custom GEMV removed substantial observed packing
overhead. These measurements do not qualify whole-model performance or promote
AVX-512 as the default.

## Required work still outstanding

- Resolve the ten frozen FP32 intermediate-tensor failures. Actual GPU
  normalization and projection diagnostics now provide operator evidence, but
  their isolated full-graph interventions were rejected; the gate remains open.
- Complete backend and full-context qualification. The bounded first-release
  image/API audit and its two negative-input follow-ups are complete; more
  successful codec fixtures are not currently required by that audit.
- The selected 200-page corpus and 24-page smoke output comparisons are complete;
  the fixed-corpus differential quality gate and component reporting now pass.
  Broader domain coverage and the numerical gate remain separate requirements.
  Visual inspection identified notebook pages sharing a document family in the
  initial manifest. That smoke run is diagnostic only. The corrected v2 manifest
  separates those notebook families. All 264 evaluation/calibration images are
  frozen in v2 locks; exact RGB/text checks found no cross-split duplicates, and
  four perceptual candidates were visually distinct. However, the completed
  visual review found mislabeled ordinary pages, five uncertain labels, and
  three additional book/newspaper family leaks. The v3 remedy is now frozen:
  all 264 selected sources are visually reviewed and hash-verified, known
  document families are split-separated, and materialization found no exact
  RGB/text duplicates or cross-split perceptual candidates. The full 200-page
  FP32 GPU and Windows runs are complete; all saved records pass validation.
  The immutable 200-page comparison matches all 275,903 token IDs,
  corrected text and stopping reasons, with valid provenance and no failed
  records: `reference/windows-rust-corpus-v3-fp32-redecoded-200.json`.
  Twelve texts were corrected by replaying the original IDs through the fixed
  decoder; these are derived text results, not fresh inference or new timings.
  The snapshot contains 181 EOS and 19 length stops, with no missing pages.
  The selected-corpus output gate passes; earlier partial snapshots remain preserved.
  The separate [quality accounting](QUALITY.md) revalidates original/replayed
  records and reports diagnostic CER/WER. The earlier 140-page negative check
  remains preserved; complete literal text identity proves zero differential
  without choosing a primary metric after observing results. The completed
  report attaches identical audited official component scores and records their
  coverage and annotation adaptations. Absolute OCR errors remain in both
  implementations; this is no claim of universal accuracy. The separate 15-page original supplement has complete
  exact CPU/GPU output parity and bounded intended-text diagnostics.
  Official-score preparation now applies the shared strict GPU-record checks;
  quality accounting binds attached scores to exact run, record and prediction
  hashes and rechecks evaluator artifacts. The corrected unclamped TEDS validator
  passes 19 workflow tests and 20 quality tests; independent review retains all
  previous failed evidence. Both fresh 200-page evaluator audits completed.
  See `reference/official-teds-reporting-independent-review-v1.json` and
  `reference/official-evaluator-git-interop-v2-completion.json`.
- The planned three-page official-engine and vLLM comparisons are complete;
  broader corpus-wide serving qualification has not been measured.
- Qualify matrix batching across platforms and workloads, packed weight/cache improvements,
  and end-to-end benchmark/memory reporting. A reproducible baseline harness and
  a narrow allocation-free decode test now exist. The first six-process full-page
  batch-one bracket now passes output identity and control stability checks.
  Compact caches take 67.829 s versus 74.310–74.876 s for expanded controls, an
  8.7–9.4% reduction, and lower process peak resident memory from 2.974 to 2.576 GB.
  Packed weights show no single-row gain; their intended batch compute path is
  inactive at batch one. Larger and mixed workloads remain required, and no
  default is promoted. See `reference/benchmarks/windows-fullpages-b1-v1.json`
  and [performance evidence](PERFORMANCE.md).
  The independently reviewed [full-page functional batch protocol](FUNCTIONAL_BATCH_REGRESSION.md)
  now has a completed frozen-CLI four-input run: a natural prose page,
  blank page, sparse scene and receipt, each at maximum side 1536 and output cap
  4096. Fresh same-binary sequential controls precede joint batch-four runs of
  all four expanded/compact and unpacked/phase-packed combinations. All four
  layouts match the controls' 1,140/2/6/94 tokens, text, stops and dimensions.
  The final execution receipt and source/artifact checks pass, covering 4,968
  candidate token IDs across 16 request outputs. This is concurrent functional
  evidence only, with no timing or corpus-quality qualification. A subsequent
  exact-contract join also matches all 20 saved request outputs (the sequential
  control plus four joint layouts) against GPU references at the same 4096 cap:
  6,210 compared IDs across repeated executions of four inputs. CPU dimensions
  match frozen expectations; GPU dimensions were not recorded. Batch2/8 and
  other orders remain separate work. See `reference/functional-batch-mixed-b4-v1.json`
  and `reference/functional-batch-v1-gpu-parity.md`.
- Broaden long-output/context boundary comparisons beyond the completed exact
  8,192-output / 16,292-total-token numerical stress case.
  `reference/exact-context-boundary-v1-lock.json` now freezes the same image
  with 8,284 requested outputs after its 8,100-token prefix: exactly 16,384 total.
  The GPU reference completed all 8,284 outputs with a length stop, and its first
  8,192 IDs match the previous reference. The actual final KV cursor is 16,383
  with capacity 16,384: the last emitted token is not fed back into KV. This
  synthetic boundary check is separate from OCR quality. Windows Rust now
  matches all 8,284 IDs, text and exact length stop at this cap; its final KV
  cursor is not separately instrumented. See
  `reference/gpu-exact-context-boundary-fp32-summary.json` and
  `reference/windows-rust-exact-context-boundary-fp32-v1.json`.
  A native Windows CLI call requesting 8,285 outputs correctly rejects the
  resulting 16,385-token total before prefill; see
  `reference/exact-context-budget-guard-windows-v1.json`.
- Broaden Linux and Windows backend agreement. Bare-metal Linux performance
  measurements are deferred by the user; WSL measurements stay labeled separately.
- Implement/qualify BF16; perform lower-priority INT8/INT4 experiments.

A bounded FP32 crossover now reproduces six saved controls on each platform
bit for bit. At the first failing prefill coordinate, layer-9 V row 112,
99.424% of the signed difference comes from passing the already-different
layer-7 states through the same GPU segment; 0.576% comes from the segment's
engine difference on the CPU entry state. The incoming-state term alone still
exceeds the frozen bound. This narrows the investigation upstream of this
segment but identifies no earlier root cause and closes no numerical gate.
All 17 stages and 144 rows are retained, with zero FP64 accounting residual.
See [the controlled crossover report](../reference/fp32-crossover-decomposition-v1.md).

The batch API now shares decode projections; independent prefills remain sequential.
Experimental scalar and AVX-512BF16 matrix operators exist, with explicit BF16
operands and FP32 accumulation. All 160 existing accumulation checks (40 cases
times two implementations and two layouts) pass the independently frozen local
contract. Nine normalization/gate checks also pass. Native BF16-output linear,
attention, graph/output and corpus checks remain open. The first full-trajectory
BF16 tolerance was rejected as too broad; it cannot qualify the backend. A
separate experimental BF16 runner now matches the free 17-token smoke output on
Windows and WSL. The first full same-prefix run passes 16/17 logit checks, with all
argmax decisions matching; sparse attention element failures remain open.
A CPU-only sink replay reproduces 832 selected saved CPU values exactly. Replacing
raw attention values resolves all 20 recorded scaled-output violations, while
replacing LSE alone resolves none. This narrows the investigation upstream of
sink scaling; it does not close any numerical gate. See
`reference/bf16-sink-replay-windows-v1.json`.
Two fused-observer captures preserve raw BF16 outputs but change the masked
denominator reduction and fail exact LSE checks. The single constrained
follow-up still fails the original output gates and a compiled-reduction check;
both captures are rejected as native-intermediate evidence. See
`reference/bf16-fused-observer-v5-summary.json`. A subsequent unmodified PTX
round-trip through the pinned assembler and explicit launch interface preserves
the complete cubin/SASS and all raw BF16, natural-LSE and log2-LSE outputs on
the three selected cases. This validates the toolchain control for a prospective
observer; no new observed intermediate or backend qualification follows. See
`reference/bf16-ptx-roundtrip-v1-summary.json`. The subsequent single stores-only
PTX observer was rejected before launch: assembling it added one FADD under the
unchanged machine-arithmetic inventory gate. No intermediate values were
captured or accepted; the restored original PTX and seven preserved diagnostic
artifacts were independently checked. See
`reference/bf16-ptx-observer-v1-rejection.json`. A second candidate omitting only
four implicated debug stores was also rejected before launch for one extra
FADD. No output or intermediate capture occurred; all numerical gates remain
unchanged. Its separate LF-only launcher recovery preserved the reviewed
sources. See `reference/bf16-ptx-observer-v2-rejection.json`. Standalone CUDA exp2 replay
does expose one BF16 rounding difference from Rust, but that argument belongs
to a mixed counterfactual; the corresponding actual saved CPU and oracle
arguments each agree with CUDA. It does not establish a fused-kernel cause.
See `reference/bf16-exp2-midpoint-origin-v1.json`.
Isolated W4A32/W8A32 scalar operators now have independently checked packing and
sampled arithmetic evidence; INT8/INT4 model backends remain unimplemented.
An isolated AVX2 W4A32 candidate also passes nine standalone tests and independent
checks of all 4,410 synthetic outputs per backend under the unchanged arithmetic
bound. This covers rows 1/2/4/8 and group sizes 64/128, with allocation-free calls;
the seven-channel synthetic operators do not establish model quality or speed.
See `experiments/quantization/AVX2-RESULTS-V1.md`.
The unchanged AVX2 kernel additionally passes all 1,261,568 full-channel outputs
per backend on the 28 saved real projection cases, across groups64/128 and
176 supported batch shapes. Activation rows are fixed subsets of the synthetic
diagnostic fixture; this is neither calibration nor model-quality evidence.
See `experiments/quantization/REAL-AVX2-RESULTS-V1.md`.
No broad GPU-equivalent or production performance claim is made.
