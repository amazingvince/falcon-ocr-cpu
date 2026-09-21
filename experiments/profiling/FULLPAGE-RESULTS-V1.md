# First full-page native CPU profile

The saved trace is useful for exploratory hotspot selection. Attention is the
first optimization target. Its strict trace audit retains one lifetime-boundary
failure; neither that exception nor the ten existing model numerical failures
is silently relabeled as a pass.

The native Windows Ryzen 7950X capture used the preserved full-page benchmark
executable and matching PDB, FP32 AVX2, 16 threads, expanded cache and unpacked
weights. Three recognitions of the 1088×1536 prepared journal page each produced
the same 1,140 IDs, literal text and EOS as the frozen baseline. Recording stopped
cleanly. The raw capture is at
`D:/falcon-ocr-rust-builds/profiles/fullpage-expanded-v2`.

## Observed CPU work

The PerfView XML export contains 3,118,232 samples from PID 121604. The table
partitions samples by resolved operator names anywhere in each call path; these
are whole-process CPU sampling percentages, not page latency or exact decode
phase durations. The executable has no ETW phase markers.

| Call-path category | Samples | Share |
|---|---:|---:|
| Non-GEMM attention | 1,617,133 | 51.861% |
| Linear projection | 837,371 | 26.854% |
| GEMM attention | 364,329 | 11.684% |
| Rayon without a resolved operator caller | 192,753 | 6.181% |
| Other GEMM | 60,565 | 1.942% |
| Gate | 32,107 | 1.030% |
| Normalization | 10,569 | 0.339% |
| Remaining categories | 3,405 | 0.109% |

Across all call paths, exclusive leaf samples are led by `dot_avx2` (46.297%)
and `axpy_avx2` (25.657%). Dot is shared by projections and attention, so these
leaf percentages must not be interpreted as separate operator percentages.
The independent cross-tab attributes 724,708 dot samples to linear projections
(23.241% of the whole export) and 718,938 to attention (23.056%). Attention also
contains 799,442 AXPY leaf samples (25.638%). This supports prioritizing its
fixed-width vector loop before broader scheduling changes.
Inclusive call-tree values overlap and must not be summed. Likewise, the large
inclusive Rayon wait frame encloses useful worker execution; it does not mean
the workers spent that percentage idle.

The first isolated experiment specializes the single-query, 64-element AVX2
attention operations. It keeps the existing tile size, key order, dot reduction
tree, FMA order, exponentials and sink calculation. Fixed-size direct calls may
remove loop and indirect-call overhead. That is a hypothesis requiring exact
CPU comparisons and an unprofiled control/candidate/control bracket; no speedup
is inferred from this trace. Production defaults remain unchanged.

The candidate passed six bit-exact operator tests and the complete 1,904-tensor
CPU smoke comparison, with all 17 teacher tokens and text unchanged. See
`artifacts/diagnostics/attention64-v2/smoke/report.json`. The unprofiled bracket
is frozen at `artifacts/benchmarks/attention64-fullpage-window-v1/plan.json`,
SHA256 `38d1fa812e0cf2d78a05355593f6d236fafdc018d5398754e8c9bb0674e63c28`.
It completed with nine exact measured outputs and a 6.15–9.43% whole-page latency
reduction against the two unchanged controls. See the
[candidate result](../attention64/RESULTS-V1.md) for measurements and limits.

## Evidence and limits

- [Capture review](../../reference/benchmarks/windows-fullpage-profile-capture-v2-review.json):
  all three output signatures exact, source/binary/PDB/plan joins, owned recorder
  cleanup and no capture cap/watchdog termination.
- [Exact-PID audit](../../reference/windows-profile-trace-audit-v4.json):
  unique process, exact command, actual start and stop, zero reported ETW event
  loss and no truncation. Of 3,118,232 in-bounds samples, 3,116,575 have attached
  stacks (99.9469%); 1,657 do not. Separate buffer-loss count is unavailable.
- [Boundary investigation](../../reference/windows-profile-boundary-review-v1.md):
  one additional kernel-only sample occurs 189.9 µs after process stop. Its
  contribution is about 0.0000321% of target samples. This is consistent with
  teardown attribution, but the cause is unproved and the strict failure stays.
  The exported view ends before that sample and its count equals the audit's
  in-bounds total.
- [Parsed stacks](../../reference/benchmarks/windows-fullpage-profile-stacks-v1.json):
  complete XML parsing, each sample bound to the exact `Process64` PID root,
  227 frames and 315,837 stack nodes. The matching project PDB was loaded.
  Windows system symbols remain unresolved; sample attachment alone does not
  establish complete unwind or complete symbols.

PerfView returned exit 1 with a `NullReferenceException` after writing the XML
ZIP. [Independent export review](../../reference/benchmarks/windows-fullpage-profile-export-independent-review-v1.json)
confirms CRC, all declared/actual frame, stack and
sample counts, valid references, acyclic stacks and every exact-PID root. Pinned
PerfView source writes the entire stack payload before GUI metadata; its
headless path leaves a null log value that the metadata serializer dereferences.
The actual XML ends after Notes, immediately before that log field. This strongly
supports a metadata-only failure, though the tool's rethrow lost the original
throw location. Do not report the exporter itself as successful.

Unresolved leaf symbols account for 266,973 samples (8.562%), chiefly Windows
modules; 33 have no identified module. None is explicitly identified as an
unresolved project leaf. Two out-of-range project symbols occur only as ancestors of 9 and 15
samples, with zero exclusive samples. Broken-stack ancestry appears in 2,557
samples (0.082%). These remain visible rather than being counted as fully
resolved stack evidence.
The export SHA256 is
`18facb2cda667464e12332009052b8bc9d124551dbd10d12a77bf1acff70d671`;
the parsed summary is
`5d8e4b7747b8a22d850b781145b7ed81465d609ae4ce0a81e23c20af6975bf6a`.

CPU sampling does not establish DRAM bandwidth, cache-miss rates or a memory
bottleneck. Profiled wall times are diagnostic only. The completed unprofiled
baseline, including the approximately 9% compact-cache gain, remains the timing
reference. This expanded-cache trace does not measure compact-cache hotspots,
other pages, batching, AVX-512 or bare-metal Linux.
