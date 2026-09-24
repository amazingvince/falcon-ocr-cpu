# Direct attention over split prefix keys

Status: native correctness checks passed; the completed B1 bracket improved
latency by 3.23–3.32%, below the frozen 5% target. The candidate is unpromoted.
This isolated candidate starts from the accepted combined fixed64-attention
experiment and adds an explicit `TemporalCandidate` cache mode. It excludes
the paired-GEMV experiment, whose measured gain missed its 5% target.

Prefix keys store eight shared temporal halves and sixteen distinct spatial
halves. AVX2 attention reads these halves directly into the original four FMA
accumulators. Generated keys, values, reduction order, softmax, masks and sinks
retain the baseline arithmetic. Prefill borrows the original expanded workspace
keys. Compact and Expanded remain independent controls.

For the selected 6,544-token prefix, analytic reserved KV payload decreases
from 1,253,638,144 to 1,106,214,912 bytes at the 4,096-output capacity, saving
147,423,232 bytes (140.59375 MiB). This excludes allocation overhead and process
memory, and does not predict physical memory traffic or speed.

Six source guards and eleven benchmark-protocol tests passed. Independent
source and protocol reviews found no blocker. The first copied build compiled
the benchmark and CLI but failed to compile a new test import. No operator,
model or benchmark executed from that attempt. Its logs, generated source
archive and all nine preparation-bound experiment sources are preserved under
`artifacts/diagnostics/attention64-temporal-v1`; the latter archive has SHA256
`8898cd26de372788fbd30afb877389057f8c3f8970c7ced9ac5329ead17af9b1`.

The only correction is the test import `use crate::kernels::{self, Simd};`.
The fresh `attention64-temporal-v2` preparation is
`e4cd25e94123c60c2e08d62ec6ecc9fbb35338cd143a56ff90f80a973feb0d96`
and uses a separate D: build target. The runtime candidate is unchanged by
that correction. Its complete build is
`c7fc4453f462992868906db3fc4f33a31963045ae6b4626327684bea116d17b5`.
All thirteen selected operator/storage/model-dispatch tests actually passed;
operator receipt `06b6881544b822aa362b971d2de67e6504fd5c7de96f04ee738ad79355de6619`.

One fresh Windows qualification invocation matched all 1,904 canonical and
2,144 mixed tensor hashes against compatible, accepted historical Windows CPU
controls. All seventeen teacher decisions matched, with the intentional length
stop; independent single and mixed free outputs matched at 17, 2 and 6 tokens,
all EOS. Actual active rows shrank from three to two to one. Both warmed decode
intervals made zero allocations. The unchanged harness and surrounding source
compatibility are bound in execution receipt
`85e4b2679deaed18e52b6a26621dffd0f1e81af59b17af2981d969b08d180492`.
This compares complete raw-F32 tensor digests, not newly preserved tensor payload
files, and makes no fresh GPU or full-corpus claim.

The benchmark build receipt is
`737d8cf857d3db2d31b9b8960c99232c6a239fba8cfb569b9146a74afd7b8790`;
its executable is `9c655c627b219b56a4126d594cd3b1fb7764853fffee2cec0119bcd563f9b5ce`.
The matching PDB and exact 2,559-byte temporal-head function were inspected.
Manual review confirms separate temporal/spatial loads, each accumulator's
temporal-then-spatial FMA history, contiguous generated-key dot and eight ordered
AXPY FMAs. There are no reconstruction copies, calls or vector stack traffic in
those three inspected arithmetic blocks. The compiler commutes the prefix pair
sum operands while retaining the grouping; this is not a claim of identical
instruction order or NaN-payload arithmetic. Broader stack traffic includes the
logits array and Windows register preservation; full decoded-byte coverage is
not independently established. Review receipt:
`reference/attention64-temporal-compiled-dataflow-v1.json`, SHA256
`5f58fb12e12f5486f3d500301fece849f928ecdf3600c9ba7fa7974b814fb458`.

The saved-model result audit checked 83 file identities and explicitly inherited
bulk startup/end checks from the actual execution. See
`reference/attention64-temporal-model-windows-v1.json`, SHA256
`009b550d0980b9de417894bb70a7075829463df131d929ef402d02d0a9426312`.

The B1 protocol is frozen in
`research/benchmarks/experiments/profiling/TEMPORAL-B1-PLAN-V1.md`: two warmups and three measured
recognitions per fresh control/candidate/control process, FP32 AVX2, sixteen
threads, unpacked weights and the existing full-page prose fixture. It requires
all nine outputs to match, absolute control drift at most 5%, and a gain of at
least 5% against each control. No default promotion follows from this experiment.
The ten existing GPU intermediate numerical failures remain separate and open.

Completed timing plan:
`artifacts/benchmarks/attention64-temporal-fullpage-window-v1/plan.json`, SHA256
`efcfae3c67c3f3e9da9ff83a346538b7a4f9d5d4b6fe019561876f4fd55b715e`.
Execution ran from 18:35:50 to 18:51:24 UTC on 2026-09-20. All three processes
exited zero. Native/WSL selected-workload checks and 27.63 GiB available physical
memory were recorded before launch; no profiler, build, model diagnostic or
artifact scan ran alongside the bracket. These are point-in-time checks, not
continuous thermal/frequency/counter measurements.

| Process | Page median | Three page samples (seconds) | Prefill median | Decode median |
| --- | ---: | --- | ---: | ---: |
| Compact control before | 62.818 s | 62.933, 62.498, 62.818 | 12.259 s | 50.460 s |
| Temporal candidate | 60.730 s | 60.616, 60.730, 61.026 | 12.397 s | 48.233 s |
| Compact control after | 62.758 s | 62.273, 62.758, 63.004 | 12.300 s | 50.347 s |

Whole-page improvement is 3.3235% and 3.2307% against the respective controls.
Absolute control drift is 0.09594%, within 5%, but the latency target fails.
All nine measured outputs match 1,140 IDs, literal text, EOS, dimensions and
prefix/output counts. The compiled harness checks each measured token vector;
warmup outputs are not individually exported. Stage medians are descriptive
wall intervals and need not sum to the page median; they are not hardware
counter measurements or proof of an isolated kernel's speedup.

The reports also preserve whole-process high-water memory values, including
setup and warmups. Peak private commit was 1,701.855 / 1,560.766 / 1,701.551 MiB
for before / candidate / after; peak resident memory was 2,456.559 / 2,316.477 /
2,456.547 MiB. These observed reductions are consistent with the separate KV
payload calculation, but are not isolated per-phase or allocator measurements.
Memory remaining after recognition differs again because session buffers have
been released.

The independent saved-result review passed 742 checks over 27 bound files:
`reference/benchmarks/windows-temporal-candidate-fullpage-review-v1.json`, SHA256
`a1669618e094ae8ed6d03a81709ae910db3af9bb1d1ea3ddf344a2550ab1670d`.
The full comparison is preserved in the timing directory, SHA256
`6cac723632a2423a4518fbc4e570e4876f93451fe5cd4ffd17656f8e19cab670`.

The analytic 140.59375 MiB cache saving remains a separate storage benefit; it
does not turn this result into a passing speed experiment. No further variants,
Linux run, batch timing or promotion are queued for this candidate. The next
performance step is to refresh CPU hotspot evidence for the combined-attention
compact baseline before selecting another optimization. Existing GPU numerical
gates and the broader runner goal remain open.
