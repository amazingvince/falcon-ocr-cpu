# Current compact runner: profile findings

The captured compact runner reproduced three full-page outputs exactly. Its
qualified CPU evidence prioritizes attention's QK and PV loops, followed by
linear projections. The complete-profile verdict remains **failed**: an ETL
lifetime exception, a separate wrong-process XML record, and exporter exit 1
are preserved. These findings support isolated optimization experiments; they
are not a newly accepted benchmark, complete-profile certificate, or hardware
bandwidth measurement.

## Capture and retained exceptions

The frozen plan used the combined compact/fixed64 control on the Ryzen 7950X:
native Windows, FP32, AVX2, 16 threads, joint B1, unpacked weights, minimum image
dimension 64, maximum 1536, output cap 4096, no warmup and three repetitions.
The saved page has prefix length 6544 and produced 1140 tokens ending in EOS.
IDs, literal text, dimensions and counts matched the frozen full-page baseline
on all three repetitions. Profiling changes execution conditions, so these
elapsed times are not substituted into the unprofiled performance bracket.

The elevated launcher exited zero; recording stopped and cleanup was confirmed.
The ETL audit found 2,692,193 in-bounds target samples, of which 2,690,438 had
attached stacks (99.9348%). Reported raw and conversion lost-event counts were
zero; separate lost-buffer accounting remains unavailable.

The strict audit failed on one additional target-PID sample at 194483.1326 ms,
725.1 microseconds after the observed process stop. It has only kernel frames.
The [boundary review](COMPACT-BOUNDARY-REVIEW-V1.md) retains the event identity;
its exact origin is not established. No lifetime tolerance was widened.

PerfView exited 1 after writing its XML ZIP, with a logged NullReferenceException.
The original strict XML analyzer also failed its exact-process-root check.
A separate diagnostic then read the complete ZIP, validating CRC, declared
counts, contiguous IDs, all references and every stack chain, including unused
ones. It found 233 frames, 306,114 stacks and 2,692,194 samples:

- 2,692,193 have exactly the expected runner process root (PID 79568).
- One has an NgcIso process root (PID 11532), at 120274.327 ms, with kernel
  frames and a BROKEN marker. Its complete record is saved without truncation.
- No rootless or multiply rooted samples were found.

The wrong-process XML record is **different from** the late target-PID ETL
sample. Equal target-subset and ETL in-bounds totals do not prove an event-level
bijection. The new diagnostic exits 1 and cannot change the strict verdict.
The original failed artifacts and frozen analyzers remain unchanged.

## What the samples support

These mutually exclusive call-path categories use the positively attributed
XML subset as their denominator. Category names reflect resolved caller symbols,
not exact prefill/decode time intervals. Interrupt/nonprocess observations are
not silently filtered.

| Call-path category | Samples | Share of attributed subset |
| --- | ---: | ---: |
| Attention, excluding GEMM attention | 1,203,347 | 44.70% |
| Linear projection | 847,914 | 31.50% |
| GEMM attention | 340,632 | 12.65% |
| Rayon without resolved operator caller | 194,112 | 7.21% |
| Other GEMM path | 60,914 | 2.26% |
| Gate | 31,783 | 1.18% |
| Normalization | 10,557 | 0.39% |
| Other/unresolved, including three bare dot samples | 2,934 | 0.11% |
| **Total** | **2,692,193** | **100%** |

A second diagnostic read the already-retained ETLX directly, with no conversion,
symbol loading or model execution. It matched the exact process, command,
lifetime, loaded image base, module identity and PDB GUID/age, and reconciled
all original sample and DPC/ISR/nonprocess counts. The ETLX was read-locked and
its size/write time checked; its byte digest was inherited from the audit,
not recomputed by this scan.

There are 1,147,243 leaf instruction samples in the compact head. All 214
distinct RVAs map to exact starts in its pinned disassembly. QK dataflow regions
contain 643,657 samples, or 23.91% of all in-bounds samples; PV update blocks
contain 491,637, or 18.26%. The remaining head regions contain 11,949.
External-leaf samples with a head frame remain separately reported and are not
automatically attributed to expf. Detailed exhaustive ranges and instruction
joins are in [COMPACT-IP-RESULTS-V1.md](COMPACT-IP-RESULTS-V1.md).

These are sampled instruction locations. Skid, scheduling and dependency chains
prevent equating a sampled load/FMA with a cache miss or a measured instruction
cost. There are no exact ETW phase markers or hardware counters here.

## Next experiment

Test a head-contiguous prefix K/V layout on the unchanged combined compact
baseline. Retain sixteen distinct rotated prefix-key heads, eight value heads
and the existing generated-token cache initially. Preserve the exact QK
reduction, scalar exponential, ascending key/PV accumulation order and original
128-key tile boundaries, including the tile crossing prefix 6544. Include
packing and allocation in page latency, then run the established operator,
trace, allocation and full-page parity gates before the controlled timing
bracket. No speed gain is inferred from this profile.

The staged-probability/register-PV candidate remains rejected: it was
2.01–2.04% slower despite achieving the intended assembly and checked parity.
Neither this profile nor a plausible cache hypothesis overrides that result.
Linear-library comparisons remain useful subsequent work. Vector-exp changes
and quantization add numerical changes and remain separate experiments.

## Evidence locations and hashes

The profile directory is `D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1`.

| Artifact | SHA-256 |
| --- | --- |
| `plan.json` | `530cfc0c5808cc525722379c0e0e0a2518d0f3d7067e62e9819030407080332b` |
| `capture.json` | `4e80ebf0bee2ed06b3e8a4f7d3400c5ff5b5ef8d3d23299d292dafb37851780b` |
| `trace-audit-v4.json` | `37d46af7873aec974f663d6dc8187b3df1322511b58ef8e7b671f23b681f20b5` |
| `export-execution.json` | `f7032924df456cea3d140c6a7b3296cd8c5570fb01f6c9f9a4bf60c746b1084b` |
| `export-scope-diagnostic-v1.json` | `77739789ed8910d058922ae675873fa6f49e923d4487cc2a8ef8c61b30a5171e` |
| `export-scope-diagnostic-v1.rejected-samples.jsonl` | `dcddc588ab50bb0cc18b73df2c35ba7d1f3ab2519b28f173d5861a2d1f8a64a5` |
| `compact-ip-histogram-v1.json` | `267897446850d8065be8957940efd64dc565111ecea83830e30adb945149acb6` |

The scope diagnostic's six synthetic tests passed; its source was held unchanged.
The C# histogram helper compiled and passed fourteen host checks before scanning.
Those checks validate the tools' bounded functions, not the completeness of the
original capture. All interpretation remains limited to the evidence above.
