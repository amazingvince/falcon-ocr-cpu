# Full-page benchmark scheduling audit

The existing 72-process plan is source-compatible at this read-only check, but its runner cannot resume across quiet windows. Use new immutable single-profile/single-batch plans, keeping each complete six-process bracket together. The existing 72-process plan and all prior evidence remain unchanged. This audit ran no inference or benchmark; its hash-bound details are in [the receipt](realistic-benchmark-scheduling-review-v1.json).

After that receipt was written, the parent froze the first concrete group at `artifacts/benchmarks/fullpages-b1-window-v1/plan.json`, SHA256 `760250cf8bc62187d7e58d427c71a59bebd80ec4dcf7a0c465148866c87828f3`. Its saved fields independently confirm six processes, fullpages b1, three measured repetitions, `selected_groups_only`, and the same frozen v2 binary. The parent reports its full planning validation passed; no inference ran. Use this existing plan for the first execution, rather than create the illustrative duplicate below.

The frozen plan uses two warmups and three measured repetitions, making 360 batch passes and 1,350 page recognitions. Three is the existing plan's setting; the CLI default is seven, so pass `--repetitions 3` explicitly when preserving it. The six fresh processes remain sequential A, expanded A, compact, phase-packed, expanded B, sequential B. Controls must not straddle separate quiet windows.

Existing concurrent functional observations give a **mixed-source serial-equivalent workload proxy of about 62.6 hours**, not a runtime forecast, confidence bound or speed comparison. The proxy applies each observed sequential per-page total to every mode/pass. Table and columns observations use 16 threads; prose and three originals use four threads and cap4096; two other originals use four threads and cap512. Builds and concurrent load also differ. No thread scaling or batch speedup is assumed, and these timings cannot replace exact-contract benchmark samples.

| Six-process group | Proxy hours | Expanded joint reserved KV payload, GiB |
| --- | ---: | ---: |
| fullpages b1 | 2.98 | 1.79 |
| fullpages b2 | 4.83 | 3.67 |
| fullpages b4 | 8.62 | 7.34 |
| fullpages b8 | 16.11 | 14.77 |
| mixed-sparse-first b1 | 0.14 | 1.21 |
| mixed-page-first b1 | 1.84 | 1.88 |
| either mixed order b2 | 1.98 each | 3.09 |
| either mixed order b4 | 5.14 each | 5.97 |
| either mixed order b8 | 6.93 each | 12.02 |

KV figures count reserved FP32 elements using `sum((prefix+4096)*22*1024*4*2)` for simultaneous expanded sessions. They exclude weights, workspace, packing, allocators and the OS, and are not observed resident memory. The current benchmark runner has no physical-memory preflight. Verify headroom before long b8 windows; the functional harness's separate 12 GiB floor is not a sufficient total-memory budget for expanded fullpages b8.

Start with **fullpages b1 after corpus and boundary jobs finish**, then mixed b2/b4 and fullpages b2. Reserve the long b8 groups for sufficiently long quiet windows. The actual first quiet bracket will provide better scheduling evidence; it does not authorize changing the frozen control order, repetitions or input selection.

These existing CLI commands create and validate a new six-process plan without running a model. The proposed output directory must not already exist:

```powershell
python scripts/realistic_benchmark.py --build artifacts/builds/ocr-bench-fullpages-v2-windows/build.json --output artifacts/benchmarks/quiet-fullpages-b1-v1 --cpu-label "AMD Ryzen 9 7950X" --environment-label "Native Windows" --profiles fullpages --batches 1 --repetitions 3
python scripts/realistic_benchmark.py --plan artifacts/benchmarks/quiet-fullpages-b1-v1/plan.json
```

Only after the operator verifies the quiet window, the existing execution and comparison commands are:

```powershell
python scripts/realistic_benchmark.py --plan artifacts/benchmarks/fullpages-b1-window-v1/plan.json --run --quiet-attestation "Operator verified other project CPU/GPU work is paused for the complete six-process bracket"
python scripts/compare_realistic_benchmarks.py --plan artifacts/benchmarks/fullpages-b1-window-v1/plan.json --output artifacts/benchmarks/fullpages-b1-window-v1/comparison.json
```

For each later group, make a new output directory and select exactly one frozen profile (`fullpages`, `mixed-sparse-first`, or `mixed-page-first`) and one batch (`1`, `2`, `4`, or `8`). Each plan correctly records `selected_groups_only`. An interrupted bracket remains a partial attempt; preserve it and create a fresh plan for that complete group. The current comparator has no verified cross-plan complete-matrix aggregation. Keep a metadata inventory of completed group reports without changing their coverage labels or pairing controls across windows.

The recorded current Rust source map and both protocol script hashes match the existing build/plan. This scheduling audit did not repeat the full model/image/executable/archive validation; the planning/validation commands perform those checks before execution. Current frozen source provides literal text and projection/transformer-prefill timing fields required by the schema. Historical functional timing records are not substitute benchmark reports. Profiling readiness is documented separately by the parent; no profiler recording occurred here.
