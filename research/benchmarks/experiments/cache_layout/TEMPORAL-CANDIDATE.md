# Copied-project temporal-prefix storage candidate

`capture_temporal_candidate.py` creates a new project copy, archives its source, and adds one explicit `TemporalCandidate` cache layout there. The live project, defaults, Cargo files, reference policies and frozen executable artifacts are not edited. The experiment follows `reference/prefix-temporal-dedup-design-v1.json` and the separately preserved prefix sharing byte audit.

Each prefix token stores eight 32-channel temporal keys, sixteen 32-channel spatial keys and eight full value heads. Generated keys and all values keep the current compact eight-head representation. Cache insertion validates every discarded duplicate with `to_bits()` before writing anything. Only the fixed 16Q/8KV/64-channel architecture, one complete prefix insertion and contiguous one-token continuation are accepted. Capacity overflow, partial prefixes, multiquery continuation and mismatched duplicate bits return errors.

Prefill borrows the existing expanded `work.k` and calls the unchanged compact attention implementation. Decode copies a prefix key's two halves into a stack array and passes it to the existing selected 64-channel dot function. The diagnostic adapter is generated from the pinned compact function: only prefix loading is changed, and the unavailable multiquery branch is removed under a one-query assertion. Dot, scale, key order, 128-key tiles, max/normalizer, value accumulation and sink operations remain text-identical. A source-only test checks that property. The current compact implementation remains the comparison oracle inside the copy.

The eight focused Rust tests cover exact reconstruction (including signed zero and NaN payload bits), insertion rejection before mutation, capacity/shape/continuation rejection, unchanged full-prefix prefill, decode with mask/tile/buffer boundaries and a 1,025-token prefix, extreme sinks, copied model cache dispatch and unchanged default selection. Scalar, Auto, AVX2/FMA and AVX-512 are tested when available; backends are recorded in the execution's memory report. No weights are loaded and no full-model generation is executed.

The memory test compares requested arithmetic with **Vec-reported buffer capacity payload**, and verifies that pointers/capacities remain unchanged while filling the reservation. This does not measure allocator overhead, actual process heap, resident memory or peak scratch. The candidate owns four buffers instead of compact's three. For the frozen mixed-b4 workload, the analytic payload changes from 3,799,121,920 to 3,412,000,768 bytes, saving 387,121,152 bytes (369.1875 MiB). That is 16⅔% of prefix payload and about 10.19% of this total reservation. Copy costs, model outputs, allocation behavior of the full runner, and throughput remain unqualified.

Native Windows commands, each using a new output directory:

```powershell
python -m unittest discover -s experiments/cache_layout -p test_temporal_candidate.py -v
python research/benchmarks/experiments/cache_layout/capture_temporal_candidate.py --output artifacts/diagnostics/prefix-temporal-candidate-review-v1 --prepare-only
python research/benchmarks/experiments/cache_layout/capture_temporal_candidate.py --output artifacts/diagnostics/prefix-temporal-candidate-windows-v1
```

The last command builds only the copied library test executable (offline, locked, two compiler jobs) and runs only the eight selected tests, with four-thread attention pools. It records original and copied source hashes, the patch and source archive, compiler/Cargo versions, selected build environment, emitted executable identity, test log, and buffer capacity report. It rejects source/archive/binary drift and refuses output-directory reuse. `reference/prefix-temporal-candidate-independent-review-v1.json` records the separate source-only review; an execution receipt is separate evidence.

Further work needs separately authorized same-prefix traces and single/mixed free-output comparisons, warmed full-runner allocation tests, and quiet performance measurement before any promotion. This candidate does not address the open GPU numerical gates and does not alter their tolerances.
