# Retained profile boundary review

The strict audit remains failed. Its single excluded CPU sample is event
**8127701**, at **233438.8458 ms** / **15:09:17.9624275 UTC**, exactly
**189.9 microseconds after ProcessStop** and **454.8 microseconds after its
thread's recorded stop**. It is inside the session bounds. The direct event UTC
and original QPC values agree, so the helper's derived UTC rounding does not
explain this failure.

The sample retains PID 121604 / process index 566, thread 47800 / index 20459,
and stack 2593922. Its ten frames are entirely `wtd.sys` and `ntoskrnl.exe`, with
no project frame. A StackWalk event follows 5.2 microseconds later. The retained
ordering is thread stop, image unloads, process stop, then the kernel sample;
the original trace audit reports no timestamp inversion.

This is consistent with teardown or ETW lifetime-to-sample association timing,
but the exact kernel or TraceEvent mechanism is not proved. It remains a real
boundary attribution exception, not a reason to change the failed gate.

The profile is usable for **exploratory hotspot ranking with that qualifier**.
One kernel-only sample among 3,118,233 contributes about 0.0000321% of counts;
it cannot materially change broad project hotspot rankings. Downstream
PerfView totals may exceed the in-bounds audit by one sample. Keep that join,
missing-stack coverage and unresolved symbols explicit. Preserve raw counts;
label any lifetime-filtered view as derived. This is no model qualification or
new performance claim.

The bounded lookup examined 230 retained-ETLX events across two six-millisecond
windows. It performed no ETL scan/conversion, symbol resolution, model or GPU
work. The existing audit hash and ETLX size/write-time remained unchanged under
a read-only lock; the 1.64 GB ETLX was not rehashed. Exact source, event, stack
and input identities are in `windows-profile-boundary-review-v1.json` and its
bound lookup receipt.
