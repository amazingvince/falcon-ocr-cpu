# Prospective compact profile acceptance v1

This carries forward PROFILE_ACCEPTANCE.md for the next combined fixed64,
compact B1 capture described in NEXT-PROFILE-V1.md. The new reviewed plan binds
this file before recording. The old expanded profile and its limitations remain
unchanged; neither its partial usability nor its exporter failure is transferred.

Require all three complete recognitions to match the frozen compact baseline:
all 1,140 IDs, literal text, EOS, 1088x1536 dimensions and 6,544 input tokens.
The exact executable, matching PDB, input, options and commands must retain
their plan hashes. Joint B1 follows the runner's singleton path.

Require actual exact-PID/image/command start and stop, unique name selection,
full process lifetime inside recording and all target samples inside the actual
process lifetime. A watchdog, collector cap, truncated capture, missing lifetime
or out-of-lifetime sample rejects complete-profile status. No boundary widening
or silent sample exclusion is permitted.

Require zero available raw ETL/ETLX event-loss counters and zero loss/truncation
callbacks. A separately unavailable BuffersLost counter is explicitly unknown,
not zero. Require at least 99% attached stacks, retaining denominator, missing
counts and their time distribution. Capture status logs are preliminary evidence;
the exact-PID TraceAudit result is a separate required gate.

Use the compact PDB GUID 1797F1F9-EC73-4291-98A8-5C9C8AAF2AE1, age 1. Quantify
unresolved project and system evidence. A PerfView nonzero exit remains failure;
any useful payload requires its own CRC/count/reference/cycle/PID checks and
separate explicit exploratory qualification. Export must not select a different
same-named process. Retain missing or rootless samples as rejected evidence.

Only resolved qualified compact_head and compact worker closure names add new
attention attribution. attention_gemm keeps precedence, then attention, linear,
generic GEMM and the existing remaining categories. Inlined dot/PV instructions
cannot be subdivided from a function-only stack export. No exact phase markers
exist: whole-process samples cover load, image processing, prefill and decode.
Inclusive percentages overlap and do not establish bandwidth, cache misses or
wall-time shares. Profiled latency is diagnostic, never a speed-target result.
