# Criteria for the first useful single-page CPU profile

These criteria were selected before the profile ran. They concern sampling
quality, not the model's frozen numerical tolerances or speedup qualification.
The user has prioritized isolated single-page performance experiments even while
the ten intermediate numerical differences and wider batch matrix remain open.

Use the frozen expanded-cache B1 capture plan v2, SHA256
`f44883243e8d681d737319b517f215b1c179db12cb8659f7cca4453f00ae6733`.
The exact executable, matching PDB, full-page input and options remain bound by
that plan. Preserve the trace, process identity and diagnostic output.

For a usable complete sampled profile:

- All three full-page recognitions must match the saved baseline's token IDs,
  literal text, stop, dimensions and counts. No model arithmetic changes.
- Match the recorded child PID, executable, command and lifetime to real ETW
  start/end events. The name-based PerfView export must select that unique process.
- The process must finish within the recording interval, without the capture
  size/time/watchdog limits ending collection early.
- Require zero reported ETW event loss and no analyzer truncation. Inspect
  recorder/ETL evidence for buffer-loss counters as well. If a separate counter
  is unavailable, record it as unknown; do not claim measured zero buffer loss.
- At least 99% of the target CPU samples should have attached stacks. Record
  the full denominator, missing counts and time distribution. Lesser coverage
  requires investigation before treating the profile as complete; preserve any
  useful partial evidence without silently discarding missing samples.
- Resolve the project binary with the matching PDB and report unresolved
  project samples, unresolved system frames and truncated/rootless stacks.
  Only attribute resolved evidence to specific kernels. Inlining and generic
  shared kernel symbols may limit attribution.

The capture plan's earlier descriptive phrase “zero event/sample/stack losses”
does not require an unattainable claim of perfect stack collection. The explicit
99% attached-stack criterion above is the prospective analysis rule. Missing
coverage stays visible and is never relabeled as model agreement.

The existing binary has no exact prefill/decode ETW markers. Use actual caller
paths and report ambiguity; aggregate stage timers do not establish exact trace
phase boundaries. Inclusive call-tree percentages overlap. Sampled CPU time is
not wall time, bandwidth, cache-miss rate or proof of a memory bottleneck.

After selecting a measured hotspot, isolate the optimization, compare output
parity, and run unprofiled before/after controls against the completed baseline.
Profiled duration alone is never a speedup measurement. No batch-matrix,
bare-metal Linux or intermediate numerical-gate completion is required to begin
that experiment cycle.
