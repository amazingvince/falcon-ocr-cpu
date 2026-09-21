# Next profile: combined fixed64 attention, compact B1

Prospective, source-only assessment. No capture, model, build, tool installation,
elevation, tests or large artifact reads were performed for this plan. Reprofile
the accepted combined-attention compact binary before selecting another kernel:
the paired GEMV and temporal-storage B1 experiments missed their frozen 5% target.
The old expanded-cache profile does not measure this baseline's hotspots.

## Frozen subject and workload

Use `artifacts/diagnostics/attention64-compact-v1/benchmark-build/`:

| Artifact | SHA256 |
|---|---|
| `build.json` | `68b1667cb7fed0d5470ec8b9153e87dee7198787f70667e71f807be9f8cae7f0` |
| `ocr_bench.exe` | `8de750766ea4026303f7d3b8f0c1ce91da67b12b4af293245cf24faa98d3a3ac` |
| `source.zip` | `f39d1f4641237aa82dc318d9bc14d5a28c71c292c27fdd497d39253c257ac4d3` |
| Preserved `ocr_bench.pdb` | `1e7e0f0d0640a94daec5c9b990fd37d4dfdaa6c7e868360fd61f1bb137abcba4` |

These are recorded identities, to be rechecked during future preparation and
after capture. No rebuild is needed. Its existing
`compiled-compact-head/symbol-receipt.json` binds PE/PDB GUID
`1797F1F9-EC73-4291-98A8-5C9C8AAF2AE1`, age 1. Do not use the old expanded
binary's same-named PDB. Exclude paired GEMV and temporal storage from this run.

Keep the journal page `3f294b5e60a0c2d4`, canonical input
`artifacts/corpus/v3/3f294b5e60a0c2d4/canonical-rgb.png`, prepared 1088x1536,
6,544 prefix tokens. Bind the existing workload lock and literal output baseline
`artifacts/benchmarks/attention64-compact-fullpage-window-v1/candidate.json`
(SHA256 `de250d2ad37db564503762952fa8557c05dde24c48d2f4d659af136a7752fa91`).
Each recognition must reproduce all 1,140 IDs, literal text, EOS, dimensions and
counts. The proposed invocation is the existing executable with:

```text
--model <workspace>/artifacts/model --threads 16 --backend avx2
--execution joint --cache-layout compact --weight-layout unpacked --batches 1
--warmup 0 --repetitions 3 --min-dimension 64 --max-dimension 1536
--max-new-tokens 4096 --cpu-label "AMD Ryzen 9 7950X"
--environment-label "Native Windows; WPR profiled diagnostic"
--output <fresh-profile-directory>/benchmark.json <canonical-rgb.png>
```

Joint B1 matches the accepted baseline's execution option. This is one whole
process with three recognitions and no warmups, not a decode-only capture or an
unprofiled timing comparison. Existing source has no exact ETW phase markers.

## Minimal reuse and changes before launch

1. Make a **new versioned copy** of `capture_windows.py`. Change the frozen
   build/PDB/baseline identities, compact/joint command, new plan/receipt labels,
   and source closure. `benchmark_check` must expect
   `independent_prefill_joint_decode`. Adapt its old symbols-receipt schema to
   the actual compact receipt's `pe_codeview` and `pdb_identity_matches_pe`;
   validate the matching PE/PDB identity rather than retaining the old GUID.
   Preserve all existing model/input/tool checks, same-buffer result validation,
   before/after closure, fresh-output guards, watchdogs and owned-session cleanup.
   The old driver cannot be used unchanged: its pins and command are expanded.
2. Reuse `FalconOcrCpu.wprp` unchanged, including the 2,048 MiB sequential cap,
   64 x 1,024 KiB buffers and `FalconOcrCpu.Verbose` file-mode profile. Prepare a
   fresh directory such as
   `D:/falcon-ocr-rust-builds/profiles/fullpage-compact-fixed64-v1`, then review
   the printed plan hash before any recording. Keep 600-second model timeout,
   4 GiB folder watchdog, 8 GiB starting and 4 GiB running free-space floors.
3. Make a new launcher from `launch_elevated.ps1`: its current driver path is
   hardcoded. Bind the new driver/plan/Python and preserve hidden, explicitly
   authorized UAC execution, unique recorder instance and fresh launcher logs.
   The prior launch receipt's null child exit code must remain a limitation;
   require a concrete child exit result or an explicit failed/unknown outcome,
   never coerce null to success. Validate this reporting path before recording.
4. Pause other project CPU/GPU work for the capture. Recheck actual profiler
   privilege and disk space at launch. Allow additional space for merged ETL,
   fresh ETLX and exports: the old capture produced roughly 0.8 GB ETL and
   1.6 GB ETLX. These are past sizes, not a bound for the new run. No new tool or
   Rust build is required; current tool/artifact presence is not a fresh digest
   or privilege attestation.

Preserve old scripts, plans, receipts and capture directories. New capture
source/launcher/analyzer and acceptance criteria need their own prospective
review and hash binding; no inference or recorder launch is authorized by this
assessment itself.

## Existing offline tool entrypoints and unchanged acceptance

Use installed `C:/Windows/System32/wpr.exe` through the reviewed driver. Its
scoped start/stop/status/cancel commands remain unchanged. Offline exact-PID
audit can reuse `run_trace_audit.ps1` with:

```text
-Executable artifacts/tools/trace-audit-build-v4/TraceAudit.exe
-Etl <fresh>/cpu.etl -TargetProcessId <captured PID>
-Output <fresh>/trace-audit-v4.json -ExpectedImage ocr_bench.exe
-ExpectedCommandLine <exact captured Windows command line>
-DependencyDirectory C:/Users/amazi/AppData/Roaming/PerfView/VER.2026-09-20.08.05.04.237
```

Audit binary SHA256:
`930cace564616e131a94f1f9b871c73a2e2536e96692aa16149a0d8aa18a5a1e`.
Use fresh ETLX/output paths. Carry forward `PROFILE_ACCEPTANCE.md` criteria in a
new plan-bound note: one actual start/stop, exact PID/image/command, unique
matching process name, sample lifetime/event bounds, zero available raw/ETLX
event-loss and loss/truncation callbacks, and at least 99% attached stacks.
Separate buffer loss remains unavailable/unknown, not measured zero.

Set process-local `_NT_SYMBOL_PATH` to the preserved compact benchmark-build
directory and restore it afterward. Existing full-stack export entrypoint:

```text
artifacts/tools/perfview-3.2.6/PerfView.exe /AcceptEULA /NoGui
  /LogFile:<fresh>/perfview-stacks.log UserCommand SaveCPUStacks
  <fresh>/cpu.etl ocr_bench
```

PerfView SHA256:
`84b8523f7fb4783fd0baae6b080adb1b5aac388192145ad713e504c69954556d`.
CSV export is optional; one full-stack export is enough initially. Name-based
selection requires the audit's unique process and every exported sample's exact
PID root. Record the actual exporter exit code and log. The old exit-1 metadata
exception is **not** blanket permission to accept a new failed export: any such
payload needs fresh CRC, complete declared/actual inventory, reference, cycle and
PID checks, with the export failure retained separately.

The prior one kernel-only sample 189.9 microseconds after process stop still
failed the strict lifetime gate; its teardown explanation was unproved. If that
recurs, preserve the rejected complete-profile status and investigate separately
with `inspect_trace_boundary.ps1`. Do not discard it or widen the lifetime gate.
`FULLPAGE-RESULTS-V1.md` and its linked audit/export receipts document these limits.

## Attribution needed for this binary

Create a versioned adaptation of `analyze_stacks.py`, preserving exact-PID and
source-closure checks. Its current `category()` recognizes the old attention
dispatchers but not `falcon_ocr::kernels::attention64_candidate::compact_head`
or the corresponding resolved compact worker closure. Add those exact qualified
call paths under attention, keeping `attention_gemm` precedence and existing
linear attribution. Do not label all Rayon frames or generic `head` names as
attention. Retain missing, unresolved and rootless evidence explicitly.

Dot64 and AXPY64 are inlined in `compact_head`; the old `dot_avx2`/`axpy_avx2`
leaf percentages are therefore not directly comparable. Initially report the
resolved compact-head total and operator caller categories. Its preserved PDB
extent is preferred VA `0x1400b13f0..0x1400b1d36`. Further dot/PV subdivision
would require actual sample instruction addresses, loaded-module base/RVA and
the matching disassembly; function-level XML and static instruction counts do
not establish those shares. Defer that extra offline export unless needed.

Report whole-process sampled CPU work, unresolved project/system frames and
stack coverage. Inclusive percentages overlap; Rayon waits can enclose worker
execution. No sampled percentage proves bandwidth, cache misses, exact decode
wall time or a memory bottleneck. Choose one next experiment only after these
new observations; retain separate unprofiled before/after timing requirements.
