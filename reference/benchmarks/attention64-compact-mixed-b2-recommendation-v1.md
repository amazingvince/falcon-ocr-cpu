# Selected mixed-batch regression check

Use the existing **mixed-sparse-first B2** order, `[sparse-room, table]`, with the combined attention64 candidate and the frozen control. Both binaries should run `joint`, `compact`, `unpacked`, FP32/AVX2, 16 threads, minimum dimension 64, maximum dimension 1536, output cap 4096. Keep two warmups and three measured repetitions in each fresh process, ordered **control-before, candidate, control-after**. This is three processes, 15 batch passes and 30 recognitions, rather than the interrupted six-mode comparison or the full matrix.

The order comes from the frozen workload, not measured speed or output agreement. B2 enters the actual batch path and exercises unequal prefixes and removal of a finished request. The sparse page produces six IDs and the table 2,280, both EOS; only about five decode forwards have two live rows before the long one-row tail. Attention still receives one query per live request inside the joint decoder. This checks the selected mixed workload; it does not establish sustained multirow throughput or B4/B8 behavior.

| Input, fixed order | Prepared size | Prefix | Expected output | Requested compact KV payload |
| --- | ---: | ---: | ---: | ---: |
| sparse-room | 1024 x 768 | 3,088 | 6 IDs, EOS | 786,497,536 bytes |
| table `2c243b31d36eb729` | 1184 x 1536 | 7,120 | 2,280 IDs, EOS | 1,331,494,912 bytes |

Total requested KV capacity is **2,117,992,448 bytes (1.973 GiB)**, using `22 * 4 * (1024*P + 512*4096 + 512*(P+4096))` per request. The weight file is another 1,079,789,464 bytes; their combined payload is about 2.978 GiB. These are capacity/payload calculations, not peak resident or committed memory. Prefill/decode scratch, image buffers, allocators and the OS require additional headroom. There is no phase-packed weight copy. Check available physical memory before each process and record the actual process peak counters; do not infer a peak-RSS bound from reserved element counts.

For scheduling only, the saved concurrent CPU observations are 16.358 seconds for sparse-room (four threads, cap4096) and 221.198 seconds for table (16 threads, cap4096). Applying their sum serially to all 15 passes gives **59.39 minutes**. This mixed-source workload proxy is neither a forecast nor a speed comparison: cache mode, concurrency, build and thread count differ. A prospective **1,800-second per-process timeout** is reasonable; the old 900-second limit could interrupt five table passes. Reserve roughly an hour with room up to the 90-minute process-time budget, or defer the whole bracket while continuing single-page development. Do not shorten warmups/repetitions after seeing outcomes.

Use these existing strict FP32 GPU outputs solely as output references:

- `artifacts/reference/functional-originals-fp32-4096/sparse-room.json`, SHA256 `a9a77300c5f25f03d420603b2172eeeb4f35c09b215bf0d793aadca44148dd0c`.
- `artifacts/reference/corpus-v3-fp32-4096/2c243b31d36eb729.json`, SHA256 `5434fc7bd87510672eb81c7227412cb43f370977c99d8e4d3575612304e361d3`.

The sparse pin is in `reference/gpu-functional-originals-fp32-4096-validation.json`; the table pin is in `reference/windows-rust-corpus-v3-fp32-redecoded-200.json`. Preserve their own manifest/configuration identities. Historical table startup fields remain missing, and neither GPU record contains observed prepared dimensions. Compare GPU literal text/IDs/stop/prefix, CPU dimensions against the frozen workload, and actual canonical input hashes/options. Reject changed goldens before adopting signatures; reject explicit error, replay or teacher-forcing metadata while allowing documented historical absence. This is a saved-output join, not new GPU inference or an accuracy evaluation.

All **18 measured request outputs** must match the two fixed signatures. The benchmark stores two token vectors per process and its archived compiled harness checks each measured repetition against them; literal text, stops, counts and dimensions are separately exported for every request/repetition. Bind the plan, protocol source/archive, build manifests, binaries, source archives and raw result/start/exit files. Only `src/kernels.rs` may differ between the matched builds. Preserve model/input validation and the final source/artifact checks outside the measured bracket. Keep the interrupted `mixed-sparse-first-b2-window-v1` directory untouched; it contains no completed output and is not a control or resumable bracket.

Prospectively require absolute control median drift <=5%. Report candidate change against **both** controls. The selected-pair no-regression screen passes only if the candidate is no more than 5% slower than either control; an unstable bracket is inconclusive. A >=5% gain against both is useful secondary evidence, not required to pass the no-regression screen. Stop on parity/identity/process/timeout failure and preserve partial artifacts. Complete the after-control before drawing timing conclusions. Per-request stage medians include batch waiting and must not be summed into batch wall time. No automatic default promotion, additive combination with historical expanded-cache gains, or wider matrix completion claim follows.

The root-owned wrapper is `experiments/profiling/benchmark_mixed_compact_candidate.py`. These proposed commands use a fresh output directory; only the final command executes a model, after plan/source review and a quiet-window check:

```powershell
python experiments/profiling/benchmark_mixed_compact_candidate.py --control artifacts/builds/ocr-bench-fullpages-v2-windows/build.json --candidate artifacts/diagnostics/attention64-compact-v1/benchmark-build/build.json --output artifacts/benchmarks/attention64-compact-mixed-b2-window-v1
python experiments/profiling/benchmark_mixed_compact_candidate.py --plan artifacts/benchmarks/attention64-compact-mixed-b2-window-v1/plan.json --plan-sha256 <frozen-plan-sha256>
python experiments/profiling/benchmark_mixed_compact_candidate.py --plan artifacts/benchmarks/attention64-compact-mixed-b2-window-v1/plan.json --plan-sha256 <frozen-plan-sha256> --run --quiet-attestation "Operator verified other project CPU/GPU work is paused for this complete three-process bracket"
```

This recommendation ran no models, builds, tests or benchmarks. The completed single-page gain remains useful independently; this bounded follow-up need not reopen the full batch matrix or block further isolated single-page experiments.
