# Full-page functional batch regression

This protocol compares fresh free-running CLI outputs for one frozen four-request stress case. It does not measure speed or establish corpus quality. The existing production runner, generic build capture, and benchmark protocol are unchanged.

The frozen order in `reference/functional-batch-v1-lock.json` is the original content-selected prose page (`3f294b5e60a0c2d4`), blank-white, sparse-room, and receipt-cafe. Its raw SHA-256 is `c9ba1bcbadb7ffe8865aa8a149a75fb239053fb79955d86083c7a28b4de3fb72`; the harness independently pins the complete canonical JSON digest. Every source page must be an exact member of its preserved source lock, with matching source image, canonical PNG, RGB pixels, and expected prepared dimensions. All four model assets are checked against fixed revision/digest pins.

The runtime is FP32, AVX2, four threads, minimum dimension 64, maximum dimension 1536, and output cap 4096. A fresh same-binary sequential expanded/unpacked control processes all four inputs first. Four subsequent processes use joint batch size four with expanded/unpacked, compact/unpacked, expanded/phase-packed, and compact/phase-packed layouts, in that order. Existing cap512 supplement outputs and earlier CPU corpus outputs cannot replace these controls.

On native Windows, prepare a fresh captured CLI and plan without inference:

```powershell
& C:/Users/amazi/mambaforge/python.exe research/corpus-qualification/scripts/capture_functional_cli.py --output artifacts/builds/functional-cli-v1-windows --jobs 2
& C:/Users/amazi/mambaforge/python.exe research/corpus-qualification/scripts/functional_batch_regression.py --build artifacts/builds/functional-cli-v1-windows/build.json --output artifacts/functional-batch/mixed-b4-v1
```

The dedicated capture preserves the Cargo-emitted executable, complete Rust/Cargo/build-wrapper source inventory, Python protocol helpers, explicit build flags/target directory, compiler identity, and a checked source ZIP. It rejects source changes during the build. It is not a hermetic toolchain image, and Cargo may reuse dependency objects.

Planning validates these identities and freezes exact commands in `plan.json`. It neither launches the CLI nor reserves memory. After coordinating the workload with the other ongoing jobs, explicitly launch it:

```powershell
& C:/Users/amazi/mambaforge/python.exe research/corpus-qualification/scripts/functional_batch_regression.py --plan artifacts/functional-batch/mixed-b4-v1/plan.json --run
```

The execution host must match the planning platform. Before each process, the harness verifies the plan/binary identity and requires at least 12 GiB available physical memory. Native Windows uses `GlobalMemoryStatusEx`; Linux uses `MemAvailable` and explicitly describes a WSL guest budget. The frozen analytic batch cache reservations are 6,049,759,232 bytes expanded or 3,799,121,920 bytes compact, plus a 381,960,192-byte shared largest prefill workspace. These are capacity estimates with the exclusions listed in the input lock; the memory check is not a measured peak-memory guarantee.

Each request must produce valid nonempty IDs, literal text, stop reason, counts, prepared dimensions, and prefix length. Every field must equal the corresponding sequential control. The declared backend/layout and packing metadata must match the invoked mode. Teacher forcing, malformed results, replay markers, extra/missing requests, and nonfinite timing metadata are rejected. The pass also requires an observed prefix of at least 4096 tokens, at least two requests remaining after prefill, and one EOS completion before the longest request finishes. Those length-derived checks establish eligibility for mixed completion and multirow decode; they do not claim internal attention or allocation tracing.

Raw JSONL, stderr, invocation receipts, and comparisons are preserved. Their hashes and the source/build/input closure are checked again before the final report. `execution.json` records start identity; `execution-final.json` records end identity and success/failure. A complete accepted result requires both the final execution receipt and `report.json` to pass. An interrupted process may leave only the start receipt and partial files, which never count as completion. An exception, CLI error, or OOM produces a failed final receipt when the harness can handle it. There is no automatic retry, reduced image size, lower token cap, overwrite, or resume; preserve the failed directory and create a fresh plan.

Per-request timings remain raw diagnostic fields. Joint batch stage attribution is shared, and this once-per-layout concurrent functional workload must not be quoted as a benchmark. Batch sizes two/eight and additional orders require a separately frozen workload and protocol revision; they are not covered by this v1 result.

The synthetic test suite runs without inference:

```powershell
& C:/Users/amazi/mambaforge/python.exe -m unittest discover -s tests -p test_functional_batch_regression.py -v
```
