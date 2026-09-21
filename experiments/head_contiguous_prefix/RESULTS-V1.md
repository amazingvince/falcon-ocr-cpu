# Head-contiguous prefix cache: native Windows result

The candidate preserves the checked CPU tensors and outputs, but improves this
full page by only **0.61–1.40%** against the two controls. It misses the fixed
5% target. Retain the combined compact/fixed64 implementation; this experiment
is not promoted, and no follow-up layout or prefetch sweep was performed.

## Full-page measurement

Ryzen 7950X, native Windows, FP32/AVX2, 16 threads, unpacked weights, joint B1
(the singleton recognition path). The fixed page has 6,544 prefix tokens,
prepared dimensions 1088×1536, and 1,140 output tokens ending in EOS. Image
limit is 1536; output limit is 4096. Each fresh process ran two unmeasured
warmups and three measured recognitions. All nine measured token vectors,
literal texts, finish reasons, counts and dimensions match the frozen output.

| Process | Page median | Prefill median | Decode median |
|---|---:|---:|---:|
| Compact control before | 63.9422 s | 10.2406 s | 53.5523 s |
| Head-contiguous candidate | 63.0483 s | 10.2629 s | 52.6405 s |
| Compact control after | 63.4384 s | 10.2148 s | 53.1183 s |

Control drift is **−0.7879%**, within the prospective 5% limit. Candidate
page-latency reductions are 1.3980% and 0.6149%; decode reductions are
1.7026% and 0.8994%. Prefill increases by 0.2182% and 0.4716%.
Stage medians need not sum to the page median. This small observed improvement
does not establish a general speedup or its cause. There was one bracket on
one page, not a confidence-interval study.

No other task-owned model, build, profiler or bulk-analysis job ran during the
bracket. Ordinary OS/user activity was not excluded. Model loading and image
file decoding precede the warmed page interval; prefix packing, scratch
allocation, filling and release occur inside page execution.

| Process | Peak resident memory | Peak private commit |
|---|---:|---:|
| Compact before | 2456.633 MiB | 1701.613 MiB |
| Candidate | 2457.340 MiB | 1714.563 MiB |
| Compact after | 2457.031 MiB | 1702.059 MiB |

These are process-lifetime high-water marks, not per-phase peaks or KV payload.
Candidate peak private commit is 12.50–12.95 MiB higher. Its persistent cache
payload is unchanged: `22*4*(1536*P + 1024*Gmax)` bytes, or 1,253,638,144 bytes
at P=6544/Gmax=4096. The added transient prefill scratch has 12.78125 MiB of
logical FP32 payload. The process counters do not independently prove that
every byte of their difference belongs to that scratch.

## Change and qualification

The isolated copied project stores prefix K as `[16,P,64]` and prefix V as
`[8,P,64]`; generated K/V remain `[G,8,64]`. It retains every distinct spatially
rotated prefix-key head. The new explicit layout leaves existing Compact and
Expanded paths and the default unchanged. Four source files changed and two
were added; no dependency, precision or numerical-tolerance change was made.

Decode changes only prefix K/V addresses in the copied attention bodies.
Four-accumulator QK reduction, key-ordered PV FMAs, scalar exp/log, rescale,
sinks, masks and global 128-key tiles remain intact. The prefix boundary at
6544 stays inside the tile starting at 6528. Prefill uses the original compact
attention operator with one reusable local token-major V scratch.

- Seven source guards passed; all 12 selected Rust operator/storage/model tests
  passed, including supported SIMD backends, tile crossings and total context
  16384.
- Fresh Compact and candidate model processes reproduced all 1,904 canonical
  CPU tensor hashes and all 17 GPU teacher token decisions. Their complete
  mixed-request traces match across all 2,144 tensors.
- Free-running requests retained independent EOS counts 17, 2 and 6. Four
  warmed single/mixed decode intervals had zero allocations.
- The [compiled-code review](COMPILED-REVIEW-V1.md) confirms direct prefix
  addressing, retained generated addressing and the expected arithmetic.
  A separate [artifact verification](../../reference/head-contiguous-prefix-assembly-artifact-verification-v1.json)
  regenerated all four saved assembly artifacts exactly. Neither is a speed
  measurement or a full-binary proof.
- The controlled full-page bracket reproduced all nine measured outputs.

These checks do not resolve the ten existing CPU/GPU intermediate numerical
failures or transfer the baseline's 200-page GPU agreement to this candidate.
No candidate corpus, throughput or Linux qualification was run after the B1
experiment missed its target. Bare-metal Linux remains deferred.

## Decision and next direction

The profile correctly directed attention to substantial QK/PV work, but the
address-order hypothesis produced too little end-to-end improvement here.
Instruction samples and this timing result do not establish whether cache
misses, bandwidth, arithmetic, scheduling or another cost limits the decoder.

Keep the existing compact control. The next useful performance comparison is
a bounded projection-kernel experiment using actual decode/prefill shapes and
activation fixtures, with the existing implementation as control. The
[math and library review](../../docs/CPU_MATH_AND_KERNEL_REVIEW.md) identifies
candidate matrix backends and distinguishes GEMV from prefill GEMM. Evaluate
packing cost, thread ownership and numerical differences before any full-page
claim. Vector-exp changes, BF16 and quantization remain separate numerical
experiments. No new library was installed by this cache experiment.

## Reproducibility

The [prospective plan](../../artifacts/benchmarks/head-contiguous-prefix-fullpage-window-v1/plan.json)
binds the workload, binaries, qualification and thresholds. The
[comparison](../../artifacts/benchmarks/head-contiguous-prefix-fullpage-window-v1/comparison.json)
records every measured sample, output decision and stage median. Raw reports,
commands, logs and process start/exit records are alongside it. Functional
receipts are under `artifacts/diagnostics/head-contiguous-prefix-v1`.
An independent [saved-result review](../../reference/benchmarks/windows-head-prefix-fullpage-review-v1.json)
passed 838 checks across 45 unchanged files, reproducing output equality,
commands, process ordering, source/layout joins, timing arithmetic and memory
values. It performed no inference and explicitly inherited 16 model/image/tensor
hashes rather than rereading those large inputs.

| Artifact | SHA-256 |
|---|---|
| Preparation | `09263f7c068af2578131d4520c9bece7919cac970f77ecf0b4c66be63033def4` |
| Build | `bd5c36231e613950c01a64e36f20d851e3067a32b8e1786dd74318872702e621` |
| Operators | `6f1f9dfeb2a2ca6451cc17cc50729010168050b5f8e3f7846913e579716ffd5a` |
| Model comparison | `2b34ceac86e3cbac1774093d030cd11d0e2b4aa5e514a8a98c37231fa6f24a0f` |
| Benchmark executable | `5695e415c58432305905d4e7f46cdf958118249ad880816434dd6fc75e2afe52` |
| B1 plan | `164016585b6075199ee016b7016125fe704643d82a6f3f67c7d7ecfb166e4888` |
| B1 comparison | `b9fd89428424e9a093996378b0623d6d4755aad89d1865c70fd9f7878f4a11db` |
| Independent B1 review | `df3748a80ab130438b09f0823f073758756609a5d12423831d066658517869c5` |

Source archives, emitted binaries and prior experiment receipts remain intact.
This report is post-execution evidence, outside the frozen build-source set.
