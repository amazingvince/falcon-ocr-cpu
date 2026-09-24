# Compact profile boundary review — 2026-09-20

The strict complete-profile audit remains **failed**. The bounded saved lookup
reconciles exactly one out-of-process-lifetime CPU sample, with an entirely
kernel-module stack. Neither the sample nor the gate is removed or reclassified.
This review reads only the existing audit and boundary JSON, plus the small
lookup source; it does not open ETL/ETLX, resolve symbols, record, or run a model.

## Exact reconciliation

The lookup examined 226 retained events across two six-millisecond windows and
saved 68 target records. Its one out-of-bounds **sample** is event **7169251**,
`PerfInfo/Sample`, PID **79568**, owner process index **557**, thread **136944**
(thread index **16070**), processor 4. The following StackWalk is also outside
the process lifetime but is not another sampled-profile event.

| Event | Index | Relative time (ms) | Original QPC |
| --- | ---: | ---: | ---: |
| Process/Terminate | 7169188 | 194482.1234 | 14133553536716 |
| Thread/Stop, thread 136944 | 7169189 | 194482.1251 | 14133553536733 |
| Last process-in-bounds sample | 7169190 | 194482.1356 | 14133553536838 |
| Project image unload | 7169194 | 194482.3453 | 14133553538935 |
| Process/Stop | 7169242 | 194482.4075 | 14133553539557 |
| Rejected sample | 7169251 | 194483.1326 | 14133553546808 |
| Following StackWalk | 7169252 | 194483.1377 | 14133553546859 |

The rejected sample is **725.1 microseconds after ProcessStop**, **1,007.5
microseconds after its thread stop**, and inside the recording session. QPC
differences are respectively 7,251 and 10,075 ticks. Direct event UTC is
`2026-09-20T20:18:13.7377202Z`; ProcessStop is
`2026-09-20T20:18:13.7369951Z`. These independently stored representations agree
on the interval. The following StackWalk is 51 QPC ticks / 5.1 microseconds later.
The audit reports no event-time inversion. Derived audit first/last UTC fields
are unnecessary for this determination.

The sample IP is `0xfffff800c863b6b1`, stack index **2362803**. All **11** recorded
frames belong to `c:\windows\system32\ntoskrnl.exe`; method names are empty,
and no project frame is present in that recorded stack. ExecutingDPC,
ExecutingISR and NonProcess are all false. The retained process ownership is
557 throughout these target events; this is not evidence of a different
same-numbered process. The audit independently reports one real start, one real
stop, a unique same-name process, and zero wrong/missing-owner samples.

The last process-in-bounds sample is itself 10.5 microseconds after the recorded
thread stop. That is an additional lifecycle observation, not a new acceptance
rule: the existing audit gates process/session bounds, not every thread bound.

## Interpretation and limits

The sequence is consistent with exit-time kernel work or ETW/TraceEvent lifetime
association timing. The unresolved kernel stack and flags do **not** identify
the exact kernel routine, prove teardown as the cause, or distinguish provider
timestamp semantics from attribution behavior. The prior expanded profile's
different outlier is not used to establish this one's cause. Original QPC and
direct event UTC rule out the audit's derived UTC rounding as an explanation
for the observed post-stop interval.

The audit counts **2,692,193 in-bounds samples**, with **2,690,438 attached
stacks** and **1,755 missing stacks** (99.9348115% attached). It excludes the
outlier before computing those counts, bins, and first/last sample times. The
target total including it is therefore **2,692,194**, and its count share is
approximately **0.0000371444%**. That small kernel-only contribution cannot
materially change broad project category count rankings, but does not establish
complete lifetime attribution or authorize a successful strict-profile label.
Any later exploratory stack analysis must retain this exception and separately
validate export identity, counts, unresolved frames, and missing stacks. Do not
silently filter the sample or treat sampled counts as elapsed wall time.

Available raw/converted event-loss counters are zero, no loss callback is
recorded, and ETLX truncation is false. The separate BuffersLost count remains
explicitly unknown. These facts do not override the lifetime failure.

The lookup used only the retained ETLX boundary windows under a read lock and
checked file length/write time. It did **not** rehash that 1,454,653,081-byte file;
its SHA-256 `93a683324c4dce110ec43a7af0a932a829cb1600de759d6ed1288edf83a894c9`
is carried from the original audit, not newly verified by this review.

## Small-file identity window

The three files below were hashed before examination and again at the end of
the examination; each end SHA-256 equals its before SHA-256. No input changed.
The new note does not mutate the audit, lookup, helper, or acceptance policy.

| File | Before and end SHA-256 |
| --- | --- |
| `D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1/trace-audit-v4.json` | `37d46af7873aec974f663d6dc8187b3df1322511b58ef8e7b671f23b681f20b5` |
| `D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1/boundary-inspection-v1.json` | `57258787e45878ffc58f637441b90a6db7ac7a31c2d446bcbcba974704562b39` |
| `research/benchmarks/experiments/profiling/inspect_trace_boundary.ps1` | `63ee138840e9cc5fd40eea7a67279102080e55b158c22cd1dc7076014f8838ba` |
