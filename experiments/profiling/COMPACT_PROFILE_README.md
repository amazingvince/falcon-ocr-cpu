# Combined fixed64 compact B1 profile v1

This version uses new capture, launcher and analyzer files. The old expanded
profile files, WPRP, source archives and receipts are unchanged.

Prepare only, with the native project Python:

```powershell
& C:/Users/amazi/mambaforge/python.exe experiments/profiling/capture_compact_windows.py prepare --output D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1
```

Preparation validates the fixed compact build, matching PDB receipt, workload,
baseline, model and tools; it reads token privileges but does not record. It
copies/hashes the new driver, launcher, analyzer and acceptance/readme plus the
unchanged WPRP and prospective plan note. Keep them immutable afterward. The
runtime Python is bound too. Review the exact printed plan SHA before launch.

Only the root's explicitly authorized, hidden native UAC dispatch may invoke:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File experiments/profiling/launch_compact_elevated.ps1 -Plan D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1/plan.json -PlanSha256 REVIEWED_SHA256 -LogDirectory artifacts/diagnostics/fullpage-compact-profile-launch-v1
```

The launcher does not elevate itself. It accepts only the pinned driver/runtime,
queries the retained child handle for a concrete exit code, and preserves a null
unknown result as failure rather than coercing it to zero. Test this path without
elevation or any recorder/model using a fresh directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File experiments/profiling/launch_compact_elevated.ps1 -ValidateExitCodeReporting -LogDirectory artifacts/diagnostics/compact-launcher-exit-validation-v1
```

That validation executes only fixed Python `exit(0)` and `exit(7)` equivalents.
It is not capture validation. Native hidden process execution, stdout/stderr
draining and concrete exit-code retrieval are shared with the real launch path.

The model command is fixed: joint B1, compact, unpacked, FP32, AVX2, 16 threads,
min64/max1536/cap4096, no warmup and three complete recognitions. WPR uses the
unchanged 2048 MiB sequential collector, 600-second model limit, 4 GiB folder
watchdog, 8 GiB starting and 4 GiB running free-space floors. Only a random,
observed-unused project instance may be stopped/cancelled. The Python finally
path performs cleanup; external forced termination/power loss cannot run it.

Offline tool commands, exact-PID TraceAudit v4, PerfView pins and local symbol
path are in NEXT-PROFILE-V1.md. Apply COMPACT_PROFILE_ACCEPTANCE.md unchanged.
After capture/audit/export, use the new analyzer, never the old category mapping:

```powershell
& C:/Users/amazi/mambaforge/python.exe experiments/profiling/analyze_compact_stacks.py --input FRESH_STACK_XML_ZIP --pid CAPTURED_PID --output FRESH_SUMMARY_JSON
```

The analyzer checks every exported sample's exact PID root and source/input
window. It does not accept an ETL capture by itself or infer missing phase
timestamps. Independent export-integrity and TraceAudit gates remain necessary.
No capture or timing claim follows from preparation or the launcher self-check.
