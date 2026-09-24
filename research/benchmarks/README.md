# benchmarks

Benchmark harnesses and the performance evidence before the phase-4 work.

| File | What it did or produced | Status |
|---|---|---|
| [docs/COMPARISON.md](docs/COMPARISON.md) | Head-to-head: this runner versus `focr` (pszemraj/falcon-ocr.rs) | history |
| [docs/CPU_PORTABILITY.md](docs/CPU_PORTABILITY.md) | Running fast on a broad base of CPUs | superseded by docs/PORTABILITY.md (2026-09-24) |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Performance evidence | superseded by attempt3/RESULTS-V3.md and docs/CPU_PORTABILITY.md |
| [docs/REALISTIC_BENCHMARKS.md](docs/REALISTIC_BENCHMARKS.md) | Frozen full-page FP32 benchmark protocol | history |
| [docs/STATUS.md](docs/STATUS.md) | Qualification status | superseded by attempt3/RESULTS-V3.md and docs/CPU_PORTABILITY.md |
| [examples/kernel_bench.rs](examples/kernel_bench.rs) | archived source (Stage A) | history |
| [experiments/attention64/README.md](experiments/attention64/README.md) | Fixed-width AVX2 attention experiment | history |
| [experiments/attention64/RESULTS-V1.md](experiments/attention64/RESULTS-V1.md) | Fixed-width attention: first full-page result | history |
| [experiments/attention64/candidate.rs](experiments/attention64/candidate.rs) | Copied-project experiment only. The head body is extracted from the frozen | history |
| [experiments/attention64/capture.py](experiments/attention64/capture.py) | Explicit copy/build/operator-test phases; never invokes a model or benchmark | history |
| [experiments/attention64/inspect_compiled.py](experiments/attention64/inspect_compiled.py) | One-function, offline PDB/PE inspection; no compilation or model execution | history |
| [experiments/attention64/patch.py](experiments/attention64/patch.py) | Exact-anchor copy patch. No live file is written by this module | history |
| [experiments/attention64/run_smoke.py](experiments/attention64/run_smoke.py) | One authorized copied-CLI smoke trace, compared with the saved CPU control | history |
| [experiments/attention64/test_source.py](experiments/attention64/test_source.py) | Source-hash tests for the attention64 candidate patch | history |
| [experiments/attention64/tests.rs](experiments/attention64/tests.rs) | Operator tests injected into the copied attention module | history |
| [experiments/attention64_compact/MIXED-B2-PLAN-V1.md](experiments/attention64_compact/MIXED-B2-PLAN-V1.md) | Mixed-B2 comparison protocol | history |
| [experiments/attention64_compact/README.md](experiments/attention64_compact/README.md) | Combined expanded/compact fixed64 attention candidate | history |
| [experiments/attention64_compact/RESULTS-V1.md](experiments/attention64_compact/RESULTS-V1.md) | Compact caches with fixed-width attention | history |
| [experiments/attention64_compact/candidate.rs](experiments/attention64_compact/candidate.rs) | SAFETY: The parent validated features, slices and shape relationships | history |
| [experiments/attention64_compact/capture.py](experiments/attention64_compact/capture.py) | Explicit copy/build/operator-test phases; never invokes a model or benchmark | history |
| [experiments/attention64_compact/capture_linux.py](experiments/attention64_compact/capture_linux.py) | Bounded WSL CPU qualification of the unchanged combined Windows source ZIP | history |
| [experiments/attention64_compact/inspect_compiled.py](experiments/attention64_compact/inspect_compiled.py) | One-function, offline PDB/PE inspection; no compilation or model execution | history |
| [experiments/attention64_compact/patch.py](experiments/attention64_compact/patch.py) | Extend the frozen reviewed expanded patch in a copied project only | history |
| [experiments/attention64_compact/run_allocations.py](experiments/attention64_compact/run_allocations.py) | Run the unchanged, captured warmed-decode allocation integration test once | history |
| [experiments/attention64_compact/run_smoke.py](experiments/attention64_compact/run_smoke.py) | One authorized copied-CLI smoke trace, compared with the saved CPU control | history |
| [experiments/attention64_compact/test_source.py](experiments/attention64_compact/test_source.py) | Source-hash tests for the combined candidate patch | history |
| [experiments/attention64_compact/tests.rs](experiments/attention64_compact/tests.rs) | Operator tests injected into the copied attention module | history |
| [experiments/attention64_staged/README.md](experiments/attention64_staged/README.md) | Staged-probability compact attention experiment | history |
| [experiments/attention64_staged/RESULTS-ASSEMBLY-REVIEW-V1.md](experiments/attention64_staged/RESULTS-ASSEMBLY-REVIEW-V1.md) | Independent staged-PV assembly review | history |
| [experiments/attention64_staged/RESULTS-V1.md](experiments/attention64_staged/RESULTS-V1.md) | Staged-probability attention: native Windows result | history |
| [experiments/attention64_staged/candidate.rs](experiments/attention64_staged/candidate.rs) | Isolated compact AVX2 attention experiment. Wrapper, head and dot64 are | history |
| [experiments/attention64_staged/capture.py](experiments/attention64_staged/capture.py) | Copy/build/operator phases; no full-model inference or benchmark execution | history |
| [experiments/attention64_staged/inspect_compiled.py](experiments/attention64_staged/inspect_compiled.py) | Inspect one staged compact attention head; no builds/model/timing execution | history |
| [experiments/attention64_staged/patch.py](experiments/attention64_staged/patch.py) | One compact AVX2 PV scheduling change in the frozen combined-attention copy | history |
| [experiments/attention64_staged/review_fullpage.py](experiments/attention64_staged/review_fullpage.py) | Independent saved-result arithmetic/identity review; no model or build execution | history |
| [experiments/attention64_staged/run_allocations.py](experiments/attention64_staged/run_allocations.py) | One unchanged eight-interval allocation test after staged operators and smoke | history |
| [experiments/attention64_staged/run_smoke.py](experiments/attention64_staged/run_smoke.py) | One copied-CLI staged-attention smoke after exact operators; no benchmark | history |
| [experiments/attention64_staged/test_source.py](experiments/attention64_staged/test_source.py) | Source-hash tests for the staged candidate patch | history |
| [experiments/attention64_staged/tests.rs](experiments/attention64_staged/tests.rs) | Operator tests injected into the copied attention module | history |
| [experiments/attention64_temporal/README.md](experiments/attention64_temporal/README.md) | Split temporal-prefix keys with fixed64 attention | history |
| [experiments/attention64_temporal/RESULTS-V1.md](experiments/attention64_temporal/RESULTS-V1.md) | Direct attention over split prefix keys | history |
| [experiments/attention64_temporal/candidate.rs](experiments/attention64_temporal/candidate.rs) | Isolated copied-project experiment. No live runner or default changes | history |
| [experiments/attention64_temporal/capture.py](experiments/attention64_temporal/capture.py) | Copy/build/operator phases; no full-model inference or benchmark execution | history |
| [experiments/attention64_temporal/inspect_compiled.py](experiments/attention64_temporal/inspect_compiled.py) | Inspect one frozen temporal attention head; no builds/model/timing execution | history |
| [experiments/attention64_temporal/patch.py](experiments/attention64_temporal/patch.py) | Exact-anchor copied-project patch; no import-time I/O or live source edits | history |
| [experiments/attention64_temporal/run_model.py](experiments/attention64_temporal/run_model.py) | One explicit temporal-candidate model test, after the frozen operator gates | history |
| [experiments/attention64_temporal/temporal_candidate.rs](experiments/attention64_temporal/temporal_candidate.rs) | Diagnostic-only copied-project cache. Never compiled into the live runner | history |
| [experiments/attention64_temporal/temporal_model_tests.rs](experiments/attention64_temporal/temporal_model_tests.rs) | Included only in the copied model module; no weights or inference required | history |
| [experiments/attention64_temporal/test_source.py](experiments/attention64_temporal/test_source.py) | Source-hash tests for the temporal candidate patch | history |
| [experiments/attention64_temporal/tests.rs](experiments/attention64_temporal/tests.rs) | Operator tests injected into the copied attention module | history |
| [experiments/cache_layout/TEMPORAL-CANDIDATE.md](experiments/cache_layout/TEMPORAL-CANDIDATE.md) | Copied-project temporal-prefix storage candidate | history |
| [experiments/cache_layout/TEMPORAL-MODEL-QUALIFICATION-V2.md](experiments/cache_layout/TEMPORAL-MODEL-QUALIFICATION-V2.md) | Temporal-prefix model qualification: isolated build targets | history |
| [experiments/cache_layout/TEMPORAL-MODEL-QUALIFICATION.md](experiments/cache_layout/TEMPORAL-MODEL-QUALIFICATION.md) | Isolated temporal-prefix model qualification | superseded by TEMPORAL-MODEL-QUALIFICATION-V2.md |
| [experiments/cache_layout/audit_temporal_sharing.py](experiments/cache_layout/audit_temporal_sharing.py) | Read-only bitwise audit of a possible split prefix-K representation | history |
| [experiments/cache_layout/capture_temporal_candidate.py](experiments/cache_layout/capture_temporal_candidate.py) | Copy-only prefix temporal-K storage diagnostic: focused tests, no model inference/timing | history |
| [experiments/cache_layout/capture_temporal_linux.py](experiments/cache_layout/capture_temporal_linux.py) | Build/test the exact frozen native candidate archive under Linux, CPU-only | history |
| [experiments/cache_layout/capture_temporal_model.py](experiments/cache_layout/capture_temporal_model.py) | Frozen-copy temporal-cache model qualification. Preparation never runs inference | superseded by capture_temporal_model_v2.py |
| [experiments/cache_layout/capture_temporal_model_linux_v1.py](experiments/cache_layout/capture_temporal_model_linux_v1.py) | Prepare/build/run unchanged temporal model qualification sources on Linux, in separate stages | history |
| [experiments/cache_layout/capture_temporal_model_v2.py](experiments/cache_layout/capture_temporal_model_v2.py) | Frozen-copy temporal-cache model qualification. Preparation never runs inference | history |
| [experiments/cache_layout/compare_temporal_model_saved_v1.py](experiments/cache_layout/compare_temporal_model_saved_v1.py) | Read-only correction of teacher-forced stop reporting; no inference or rebuild | history |
| [experiments/cache_layout/temporal_candidate.rs](experiments/cache_layout/temporal_candidate.rs) | Diagnostic-only copied-project cache. Never compiled into the live runner | history |
| [experiments/cache_layout/temporal_model_qualification.rs](experiments/cache_layout/temporal_model_qualification.rs) | Copied integration test only. Uses public Runner APIs; no production hooks | history |
| [experiments/cache_layout/temporal_model_tests.rs](experiments/cache_layout/temporal_model_tests.rs) | Included only in the copied model module; no weights or inference required | history |
| [experiments/cache_layout/test_temporal_candidate.py](experiments/cache_layout/test_temporal_candidate.py) | Source-only patch tests; no Rust build, inference, model loading or timing | history |
| [experiments/cache_layout/test_temporal_linux.py](experiments/cache_layout/test_temporal_linux.py) | Host-only extraction/provenance guards; no build or runtime execution | history |
| [experiments/cache_layout/test_temporal_model_capture.py](experiments/cache_layout/test_temporal_model_capture.py) | Bounded host-only preparation/comparison guards; no build or model execution | superseded by test_temporal_model_capture_v2.py |
| [experiments/cache_layout/test_temporal_model_capture_v2.py](experiments/cache_layout/test_temporal_model_capture_v2.py) | Bounded host-only preparation/comparison guards; no build or model execution | history |
| [experiments/cache_layout/test_temporal_model_linux_v1.py](experiments/cache_layout/test_temporal_model_linux_v1.py) | Host-only exact-source and platform-observation checks; no build/inference | history |
| [experiments/cache_layout/test_temporal_model_saved_v1.py](experiments/cache_layout/test_temporal_model_saved_v1.py) | Host-only correction tests using existing synthetic records; no inference | history |
| [experiments/gemv_pair/B1-PLAN-V1.md](experiments/gemv_pair/B1-PLAN-V1.md) | Paired-output GEMV single-page experiment | history |
| [experiments/gemv_pair/README.md](experiments/gemv_pair/README.md) | Paired-output AVX2 GEMV: source-only candidate | history |
| [experiments/gemv_pair/RESULTS-V1.md](experiments/gemv_pair/RESULTS-V1.md) | Paired-output GEMV candidate | history |
| [experiments/gemv_pair/candidate.rs](experiments/gemv_pair/candidate.rs) | Five actual Falcon row-one shapes, expressed as (input, output) | history |
| [experiments/gemv_pair/capture.py](experiments/gemv_pair/capture.py) | Copy/build/operator phases; no full-model inference or benchmark execution | history |
| [experiments/gemv_pair/inspect_compiled.py](experiments/gemv_pair/inspect_compiled.py) | Offline inspection of one PDB-resolved GEMV block | history |
| [experiments/gemv_pair/patch.py](experiments/gemv_pair/patch.py) | One exact-anchor patch of the frozen combined-attention copy; no I/O on import | history |
| [experiments/gemv_pair/run_allocations.py](experiments/gemv_pair/run_allocations.py) | One unchanged warmed-decode allocation test after operators and exact smoke | history |
| [experiments/gemv_pair/run_smoke.py](experiments/gemv_pair/run_smoke.py) | One copied-CLI smoke run after the exact seven-operator gate; no benchmark | history |
| [experiments/gemv_pair/test_source.py](experiments/gemv_pair/test_source.py) | Source-hash tests for the paired GEMV candidate patch | history |
| [experiments/gemv_pair/tests.rs](experiments/gemv_pair/tests.rs) | Operator tests injected into the copied kernels module | history |
| [experiments/head_contiguous_prefix/COMPILED-REVIEW-V1.json](experiments/head_contiguous_prefix/COMPILED-REVIEW-V1.json) | Machine-readable compiled head-contiguous prefix review | history |
| [experiments/head_contiguous_prefix/COMPILED-REVIEW-V1.md](experiments/head_contiguous_prefix/COMPILED-REVIEW-V1.md) | Compiled head-contiguous prefix review | history |
| [experiments/head_contiguous_prefix/README.md](experiments/head_contiguous_prefix/README.md) | Head-contiguous prefix K/V candidate | history |
| [experiments/head_contiguous_prefix/RESULTS-V1.md](experiments/head_contiguous_prefix/RESULTS-V1.md) | Head-contiguous prefix cache: native Windows result | history |
| [experiments/head_contiguous_prefix/candidate.rs](experiments/head_contiguous_prefix/candidate.rs) | Exact frozen fixed64 arithmetic with prefix K/V addressing changes only | history |
| [experiments/head_contiguous_prefix/capture.py](experiments/head_contiguous_prefix/capture.py) | Copy/build/operator phases; no full-model inference or benchmark execution | history |
| [experiments/head_contiguous_prefix/head_contiguous_prefix.rs](experiments/head_contiguous_prefix/head_contiguous_prefix.rs) | Copied-project experiment only: reorder retained prefix K/V, preserve bits | history |
| [experiments/head_contiguous_prefix/head_prefix_model_qualification.rs](experiments/head_contiguous_prefix/head_prefix_model_qualification.rs) | Copied integration test only. Uses public Runner APIs; no production hooks | history |
| [experiments/head_contiguous_prefix/inspect_compiled.py](experiments/head_contiguous_prefix/inspect_compiled.py) | Inspect one frozen head-contiguous prefix attention head; no builds/model/timing | history |
| [experiments/head_contiguous_prefix/model_tests.rs](experiments/head_contiguous_prefix/model_tests.rs) | Private copied-model cache dispatch checks; no weights or model generation | history |
| [experiments/head_contiguous_prefix/patch.py](experiments/head_contiguous_prefix/patch.py) | Pinned copied-project patch. Importing does not read/write source or build | history |
| [experiments/head_contiguous_prefix/run_model.py](experiments/head_contiguous_prefix/run_model.py) | Fresh Compact/candidate functional comparison; never a timing measurement | history |
| [experiments/head_contiguous_prefix/storage_tests.rs](experiments/head_contiguous_prefix/storage_tests.rs) | Independent focused tests injected as the storage module's test child | history |
| [experiments/head_contiguous_prefix/test_model_protocol.py](experiments/head_contiguous_prefix/test_model_protocol.py) | Synthetic functional-protocol tests; no model files, tensors, builds or inference | history |
| [experiments/head_contiguous_prefix/test_source.py](experiments/head_contiguous_prefix/test_source.py) | Source-hash tests for the head-contiguous candidate patch | history |
| [experiments/head_contiguous_prefix/verify_saved_assembly_v1.py](experiments/head_contiguous_prefix/verify_saved_assembly_v1.py) | One supplementary rerender; never rewrite original inspection artifacts | history |
| [experiments/matrix_backend_compare/Cargo.lock](experiments/matrix_backend_compare/Cargo.lock) | Lockfile of the RTen matrix backend probe crate | history |
| [experiments/matrix_backend_compare/Cargo.toml](experiments/matrix_backend_compare/Cargo.toml) | Manifest of the RTen matrix backend probe crate | history |
| [experiments/matrix_backend_compare/RESULTS-V1.md](experiments/matrix_backend_compare/RESULTS-V1.md) | RTen 0.26.0 FP32 operator screening, native Windows | history |
| [experiments/matrix_backend_compare/candidate.rs](experiments/matrix_backend_compare/candidate.rs) | Isolated RTen 0.26.0 FP32 adapter. No production integration | history |
| [experiments/matrix_backend_compare/capture.py](experiments/matrix_backend_compare/capture.py) | Prepare and build one isolated RTen/control operator probe; never run it | history |
| [experiments/matrix_backend_compare/main.rs](experiments/matrix_backend_compare/main.rs) | Isolated actual-operands matrix comparison. No Model/Runner or GPU work | history |
| [experiments/matrix_backend_compare/run_timing.py](experiments/matrix_backend_compare/run_timing.py) | Run the preserved RTen operator probe once without rebuilding | history |
| [experiments/matrix_backend_faer/Cargo.lock](experiments/matrix_backend_faer/Cargo.lock) | Lockfile of the faer matrix backend probe crate | history |
| [experiments/matrix_backend_faer/Cargo.toml](experiments/matrix_backend_faer/Cargo.toml) | Manifest of the faer matrix backend probe crate | history |
| [experiments/matrix_backend_faer/README.md](experiments/matrix_backend_faer/README.md) | Isolated faer FP32 matrix adapter | history |
| [experiments/matrix_backend_faer/candidate.rs](experiments/matrix_backend_faer/candidate.rs) | Isolated faer 0.24.4 FP32 adapter. No production integration | history |
| [experiments/matrix_backend_faer/capture.py](experiments/matrix_backend_faer/capture.py) | Prepare and build one isolated faer/control operator probe; never run it | history |
| [experiments/matrix_backend_faer/main.rs](experiments/matrix_backend_faer/main.rs) | Isolated actual-operands matrix comparison. No Model/Runner or GPU work | history |
| [experiments/profiling/COMPACT-BOUNDARY-REVIEW-V1.md](experiments/profiling/COMPACT-BOUNDARY-REVIEW-V1.md) | Compact profile boundary review — 2026-09-20 | history |
| [experiments/profiling/COMPACT-IP-HISTOGRAM-V1.md](experiments/profiling/COMPACT-IP-HISTOGRAM-V1.md) | Retained compact-head instruction-address diagnostic | history |
| [experiments/profiling/COMPACT-IP-RESULTS-V1.md](experiments/profiling/COMPACT-IP-RESULTS-V1.md) | Compact-head instruction samples — 2026-09-20 | history |
| [experiments/profiling/COMPACT-PROFILE-RESULTS-V1.md](experiments/profiling/COMPACT-PROFILE-RESULTS-V1.md) | Current compact runner: profile findings | history |
| [experiments/profiling/COMPACT_PROFILE_ACCEPTANCE.md](experiments/profiling/COMPACT_PROFILE_ACCEPTANCE.md) | Prospective compact profile acceptance v1 | history |
| [experiments/profiling/COMPACT_PROFILE_README.md](experiments/profiling/COMPACT_PROFILE_README.md) | Combined fixed64 compact B1 profile v1 | history |
| [experiments/profiling/CompactIpHistogram.cs](experiments/profiling/CompactIpHistogram.cs) | Offline retained-ETLX diagnostic. No recording, conversion, symbols or model | history |
| [experiments/profiling/FULLPAGE-RESULTS-V1.md](experiments/profiling/FULLPAGE-RESULTS-V1.md) | First full-page native CPU profile | history |
| [experiments/profiling/FalconOcrCpu.wprp](experiments/profiling/FalconOcrCpu.wprp) | WPR recording profile for the native CPU capture | history |
| [experiments/profiling/NEXT-HEAD-CONTIGUOUS-V1.md](experiments/profiling/NEXT-HEAD-CONTIGUOUS-V1.md) | One prefix-only head-contiguous cache experiment | history |
| [experiments/profiling/NEXT-KERNEL-V1.md](experiments/profiling/NEXT-KERNEL-V1.md) | Next single-page experiment: paired AVX2 GEMV | history |
| [experiments/profiling/NEXT-PROFILE-V1.md](experiments/profiling/NEXT-PROFILE-V1.md) | Next profile: combined fixed64 attention, compact B1 | history |
| [experiments/profiling/NEXT-TEMPORAL-V1.md](experiments/profiling/NEXT-TEMPORAL-V1.md) | Next bounded candidate: split-prefix storage with direct fixed64 attention | history |
| [experiments/profiling/PROFILE_ACCEPTANCE.md](experiments/profiling/PROFILE_ACCEPTANCE.md) | Criteria for the first useful single-page CPU profile | history |
| [experiments/profiling/README.md](experiments/profiling/README.md) | Native single-page CPU profiling | history |
| [experiments/profiling/TEMPORAL-B1-PLAN-V1.md](experiments/profiling/TEMPORAL-B1-PLAN-V1.md) | Explicit temporal-cache B1 experiment | history |
| [experiments/profiling/TraceAudit.cs](experiments/profiling/TraceAudit.cs) | Offline only. No recorder, symbol download, process launch, or workload execution | history |
| [experiments/profiling/TraceAudit.md](experiments/profiling/TraceAudit.md) | Offline exact-PID trace audit | history |
| [experiments/profiling/analyze_compact_stacks.py](experiments/profiling/analyze_compact_stacks.py) | Summarize a PerfView CPU-stack XML export | history |
| [experiments/profiling/analyze_stacks.py](experiments/profiling/analyze_stacks.py) | Summarize a PerfView CPU-stack XML export | history |
| [experiments/profiling/benchmark_candidate.py](experiments/profiling/benchmark_candidate.py) | One isolated expanded-cache candidate between two full-page CPU controls | history |
| [experiments/profiling/benchmark_compact_candidate.py](experiments/profiling/benchmark_compact_candidate.py) | One combined fixed-width/compact-cache candidate between compact CPU controls | history |
| [experiments/profiling/benchmark_head_prefix_candidate.py](experiments/profiling/benchmark_head_prefix_candidate.py) | One explicit head-contiguous prefix B1 candidate between pinned compact CPU controls | history |
| [experiments/profiling/benchmark_mixed_compact_candidate.py](experiments/profiling/benchmark_mixed_compact_candidate.py) | One fixed mixed-B2 compact candidate between unchanged compact CPU controls | history |
| [experiments/profiling/benchmark_temporal_candidate.py](experiments/profiling/benchmark_temporal_candidate.py) | One explicit temporal-prefix B1 candidate between pinned compact CPU controls | history |
| [experiments/profiling/build_compact_ip_histogram.ps1](experiments/profiling/build_compact_ip_histogram.ps1) | Build the CompactIpHistogram TraceEvent tool | history |
| [experiments/profiling/build_trace_audit.ps1](experiments/profiling/build_trace_audit.ps1) | Build the TraceAudit TraceEvent tool | history |
| [experiments/profiling/capture_compact_windows.py](experiments/profiling/capture_compact_windows.py) | Source-bound native CPU diagnostic. Preparation is inert; run requires an exact plan | history |
| [experiments/profiling/capture_windows.py](experiments/profiling/capture_windows.py) | Source-bound native CPU diagnostic. Preparation is inert; run requires an exact plan | history |
| [experiments/profiling/check_compact_export_integrity.py](experiments/profiling/check_compact_export_integrity.py) | Independently check saved compact PerfView XML; never open ETL or run tools | history |
| [experiments/profiling/decode-source-map.md](experiments/profiling/decode-source-map.md) | Warm single-page FP32 decode: source map | history |
| [experiments/profiling/inspect_compact_export_scope.py](experiments/profiling/inspect_compact_export_scope.py) | Diagnose a rejected saved XML export without weakening the frozen analyzer | history |
| [experiments/profiling/inspect_trace_boundary.ps1](experiments/profiling/inspect_trace_boundary.ps1) | Inspect an ETL trace boundary with TraceEvent | history |
| [experiments/profiling/launch_compact_elevated.ps1](experiments/profiling/launch_compact_elevated.ps1) | Elevated launcher for the compact profile capture | history |
| [experiments/profiling/launch_elevated.ps1](experiments/profiling/launch_elevated.ps1) | Elevated launcher for the expanded profile capture | history |
| [experiments/profiling/review_candidate_results.py](experiments/profiling/review_candidate_results.py) | Independent saved-result arithmetic/identity review; no model or build execution | history |
| [experiments/profiling/review_compact_candidate_results.py](experiments/profiling/review_compact_candidate_results.py) | Independent saved-result arithmetic/identity review; no model or build execution | history |
| [experiments/profiling/review_gemv_pair_candidate_results.py](experiments/profiling/review_gemv_pair_candidate_results.py) | Paired-GEMV B1 saved-result review, released only after the timed bracket | history |
| [experiments/profiling/review_head_prefix_candidate_results.py](experiments/profiling/review_head_prefix_candidate_results.py) | Explicit head-contiguous-prefix B1 saved-result review after parent timing release | history |
| [experiments/profiling/review_mixed_compact_candidate_results.py](experiments/profiling/review_mixed_compact_candidate_results.py) | Independent saved-result review. Run only after the quiet bracket has ended | history |
| [experiments/profiling/review_temporal_candidate_results.py](experiments/profiling/review_temporal_candidate_results.py) | Explicit temporal-prefix B1 saved-result review after parent timing release | history |
| [experiments/profiling/run_compact_offline.py](experiments/profiling/run_compact_offline.py) | Run one existing offline tool on the completed compact capture; no recording | history |
| [experiments/profiling/run_trace_audit.ps1](experiments/profiling/run_trace_audit.ps1) | Run TraceAudit on a completed ETL | history |
| [experiments/profiling/test_analyze_stacks.py](experiments/profiling/test_analyze_stacks.py) | Offline attribution checks; no recorder, model or real trace is executed | history |
| [experiments/profiling/test_benchmark_candidate.py](experiments/profiling/test_benchmark_candidate.py) | Offline protocol regression tests: no subprocesses, model loads, or asset hashes | history |
| [experiments/profiling/test_benchmark_compact_candidate.py](experiments/profiling/test_benchmark_compact_candidate.py) | Offline protocol regression tests: no subprocesses, model loads, or asset hashes | history |
| [experiments/profiling/test_benchmark_head_prefix_candidate.py](experiments/profiling/test_benchmark_head_prefix_candidate.py) | Offline head-prefix protocol tests. All process and model-input work is mocked | history |
| [experiments/profiling/test_benchmark_mixed_compact_candidate.py](experiments/profiling/test_benchmark_mixed_compact_candidate.py) | Offline protocol regression tests: no subprocesses, model loads, or asset hashes | history |
| [experiments/profiling/test_benchmark_temporal_candidate.py](experiments/profiling/test_benchmark_temporal_candidate.py) | Offline temporal protocol tests. All process and model-input work is mocked | history |
| [experiments/profiling/test_compact_export_integrity.py](experiments/profiling/test_compact_export_integrity.py) | Tiny synthetic XML only; no existing trace, ETL, model or profiler access | history |
| [experiments/profiling/test_compact_export_scope.py](experiments/profiling/test_compact_export_scope.py) | Tiny synthetic fixtures only; no actual export, ETL or model is accessed | history |
| [experiments/profiling/test_compact_ip_histogram_source.py](experiments/profiling/test_compact_ip_histogram_source.py) | Small prospective contract checks only; never load TraceEvent or ETLX | history |
| [experiments/profiling/test_compact_profile.py](experiments/profiling/test_compact_profile.py) | Bounded offline checks: no preparation, recorder, model, elevation or build | history |
| [experiments/profiling/test_trace_audit.ps1](experiments/profiling/test_trace_audit.ps1) | Tests for the TraceAudit tool | history |
| [experiments/toolchain_194/RESULTS-V1.md](experiments/toolchain_194/RESULTS-V1.md) | Rust 1.94 upgrade | history |
| [experiments/toolchain_194/validate_linux.sh](experiments/toolchain_194/validate_linux.sh) | Validate the Rust 1.94 toolchain build on Linux | history |
| [experiments/toolchain_194/validate_windows.py](experiments/toolchain_194/validate_windows.py) | Compare a fresh Rust 1.94 trace against the saved 1.92 trace | history |
| [scripts/benchmark_packed_windows.ps1](scripts/benchmark_packed_windows.ps1) | Repeat the packed/unpacked native benchmark binary on Windows | history |
| [scripts/capture_rust_build.py](scripts/capture_rust_build.py) | Build and preserve a Rust example with a checked source archive and | history |
| [scripts/compare_benchmark_modes.py](scripts/compare_benchmark_modes.py) | Compare matched same-binary CPU runs without treating one fixture as a release | history |
| [scripts/compare_focr.py](scripts/compare_focr.py) | Fair head-to-head driver: our `falcon-ocr` runner versus pszemraj's `focr` | history |
| [scripts/compare_packed_benchmarks.py](scripts/compare_packed_benchmarks.py) | Compare repeated packed/unpacked runs from the same native benchmark binary | history |
| [scripts/compare_realistic_benchmarks.py](scripts/compare_realistic_benchmarks.py) | Validate frozen full-page benchmark evidence and compare bracketed controls | history |
| [scripts/promotion_bracket.py](scripts/promotion_bracket.py) | Control / candidate / control latency bracket for a default-runtime promotion | history |
| [scripts/realistic_benchmark.py](scripts/realistic_benchmark.py) | Freeze/validate a benchmark plan; inference requires --run and a quiet-host attestation | history |
| [scripts/summarize_operator_benchmarks.py](scripts/summarize_operator_benchmarks.py) | Summarize the two native Windows operator-only measurement rounds | history |
| [scripts/test_compare_focr.py](scripts/test_compare_focr.py) | Unit tests for scripts/compare_focr.py (synthetic inputs only, no binaries) | history |
| [scripts/test_realistic_benchmark.py](scripts/test_realistic_benchmark.py) | Protocol regression tests only: synthetic JSON, no inference or measured timings | history |
