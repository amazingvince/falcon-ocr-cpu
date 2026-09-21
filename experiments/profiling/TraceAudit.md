# Offline exact-PID trace audit

`TraceAudit.cs` reads a completed merged ETL through the matching PerfView 3.2.6
TraceEvent assembly. It cannot start a recorder or workload and does not resolve
or download symbols. The parent capture harness owns recording and PerfView
`SaveCPUStacks` / `SaveCPUStacksAsCsv` analysis.

```powershell
./experiments/profiling/build_trace_audit.ps1 -OutputDirectory artifacts/tools/trace-audit-build-v3
./experiments/profiling/run_trace_audit.ps1 -Executable artifacts/tools/trace-audit-build-v3/TraceAudit.exe -Etl <merged.etl> -TargetProcessId <pid> -Output <fresh-audit.json> -ExpectedImage <basename.exe>
```

`-ExpectedCommandLine` optionally checks an exact recorded string. `-BinMilliseconds`
defaults to 1000. The environment variable `TRACEAUDIT_DEPENDENCIES` supplies the
extracted dependency directory when calling the EXE directly. The run wrapper
sets and restores it. Do not use the PowerShell reserved `$PID` as a variable.

The output JSON, `<output>.etlx`, and `<output>.conversion.log` must all be fresh.
Conversion retains all events, skips no initial interval, does not continue on
errors, and explicitly reports ETLX truncation. Existing ETLX files are never
reused. The original ETL stays open without write/delete sharing and its SHA256
is checked before and after. Loaded extracted dependencies are similarly bound.

Exit 0 means the exact PID resolved to one process lifetime, some valid sampled
profile events existed, the image/command line were available and optional
identity checks passed, loss counts were zero and agreed, conversion was not
truncated, and sample ownership/time bounds passed. It also requires one real
start and stop, at least 99% of sampled events with attached stacks, and same-name process
uniqueness for the separate name-based PerfView export join. Exit 1 is a completed failed
audit; exit 2 is a malformed input or processing error. A processing error after
report creation still writes its rejected JSON. Argument/setup errors before
report creation fail without inventing an audit. Existing reports are preserved.

`pid_matches` identifies PID reuse explicitly. `same_name_processes` and
`same_name_unique` describe all same-name processes in the trace; they do not
replace exact PID selection. `lifetime_observation` distinguishes observed
ProcessStart/ProcessStop events from clipped TraceProcess times and rundown.
One-sided/missing boundary events are reported and fail complete-profile
acceptance. Exact PID uniqueness alone cannot prove both lifetime boundaries.

`samples` contains event counts, count-payload totals, attached-stack coverage,
thread counts, flags, and first/last times. `sample_bins` uses half-open intervals
relative to session start; omitted bins have zero samples. Stack attachment says
nothing about unwind completeness or symbol quality. Module/PDB identities are
recorded trace metadata, never identities inferred from current disk files.
Zero `EventsLost` is an observed source/header counter, not a blanket guarantee
that every provider emitted everything. A separate buffer-loss counter is not
exposed by the used public API and remains `null`/unknown. The recorder's retained
evidence must support any separate claim about buffer loss. Missing stack counts
remain quantitative even when the 99% threshold passes; symbol coverage is a
separate downstream assessment. These limits remain in every report.

API evidence is the extracted local `Microsoft.Diagnostics.Tracing.TraceEvent.xml`
and Microsoft's [TraceLog.cs at the assembly's source revision](https://github.com/microsoft/perfview/blob/9a99c8202310fc0ab2f84690881f6a4a4ada0c44/src/TraceEvent/TraceLog.cs).
The exact assembly is pinned in source: SHA256
`530946dc20e89754783f0ac76d86b7a4eedf95326f255db34c32fcbf5c3ce0ff`,
version `3.2.6.0`, informational version
`3.2.6+9a99c8202310fc0ab2f84690881f6a4a4ada0c44`.

Host checks (no ETW recording) cover argument and UTC semantics, rejection of an
invalid ETL with a failure receipt, and refusal to overwrite that receipt:

```powershell
./experiments/profiling/test_trace_audit.ps1 -Executable artifacts/tools/trace-audit-build-v3/TraceAudit.exe -OutputDirectory artifacts/tools/trace-audit-host-tests-v3
```

A real completed ETL remains necessary to validate actual sample/process/module
coverage. Host tests do not constitute that validation.
