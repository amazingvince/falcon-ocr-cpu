# Frozen full-page FP32 benchmark protocol

`reference/benchmarks/realistic-fp32-v1-workloads.json` freezes eight existing RGB PNG inputs. It contains no measured performance. The three natural pages retain the earlier serving selection and visual review: journal opening prose, a numerical table, and two-column game-manual prose. Five original supplement pages add a sparse room label, two narrow receipts, a blank page, and an orthogonal rotation. Selection uses source dimensions and visible content; generated lengths, agreement, and speed do not select or remove inputs. Blank/sparse content is intended to exercise short completions, but an early EOS is not assumed. Actual lengths and length-cap stops remain in the report.

| Input | Prepared dimensions | Input tokens |
| --- | ---: | ---: |
| Journal prose | 1088 × 1536 | 6544 |
| Numerical table | 1184 × 1536 | 7120 |
| Two-column manual | 1184 × 1536 | 7120 |
| Sparse room label | 1024 × 768 | 3088 |
| Cafe receipt | 640 × 960 | 2416 |
| Blank white | 1024 × 1280 | 5136 |
| Market receipt | 640 × 1024 | 2576 |
| Rotated notice | 960 × 1280 | 4816 |

These dimension expectations follow the pinned bounded resize and 16-pixel alignment; every measured report must agree. The processor pixel ceiling remains 10,035,200. At a 1536 longest side it does not bind these inputs. All runs use FP32 AVX2, 16 Rayon threads, minimum dimension 64, maximum dimension 1536, an output cap of 4096, two warmups, and at least three measured repetitions (seven recommended). Prefix plus maximum generation remains below 16,384. Phase packing is explicit and remains experimental.

`fullpages` cycles prose, table, columns. Its batch-one case is an actual full journal page. `mixed-sparse-first` and `mixed-page-first` contain the same eight inputs in two fixed orders, with sparse and dense pages alternating near the front. Each batch uses the first N entries, cycling only when N exceeds the profile length; the manifest is authoritative. Batch 1/2/4/8 profiles therefore differ explicitly. No sorting by observed output length is allowed. Only inputs used by the current batch are passed to the harness, avoiding unused decoded-image memory.

Each profile/batch group runs six fresh processes in this order: sequential A, joint expanded A, joint compact, joint phase-packed with expanded cache, joint expanded B, sequential B. Each process measures one batch size, preventing earlier batch cases from inflating its process peak. Expanded candidates are compared against both sequential controls; compact and phase-packed candidates against both expanded controls. Both control pairs must drift by at most 5% in absolute latency. A candidate needs at least 5% lower latency against both relevant controls for a positive group-specific result, while regression reporting uses the same 5% boundary. No group result promotes a default, and selected groups do not imply complete matrix coverage. The complete matrix is 72 fresh processes and may be expensive; schedule complete six-process groups in separate quiet windows without reordering or dropping cases.

## Build, validate, then execute explicitly

First preserve a fresh executable and its complete recorded source inventory. Existing historical benchmark binaries lack literal text evidence and are rejected. Build capture uses Cargo's emitted executable and checks sources before/after compilation. It records tool versions and build flags; this is not a hermetic compiler/dependency image.

```powershell
python research/benchmarks/scripts/capture_rust_build.py --example ocr_bench --output artifacts/builds/ocr-bench-fullpages-v1-windows --jobs 2
python research/benchmarks/scripts/realistic_benchmark.py --validate-inputs-only
python research/benchmarks/scripts/realistic_benchmark.py --build artifacts/builds/ocr-bench-fullpages-v1-windows/build.json --output artifacts/benchmarks/fullpages-plan-v1 --cpu-label "AMD Ryzen 9 7950X" --environment-label "Native Windows" --profiles fullpages,mixed-sparse-first,mixed-page-first --batches 1,2,4,8 --repetitions 7
python research/benchmarks/scripts/realistic_benchmark.py --plan artifacts/benchmarks/fullpages-plan-v1/plan.json
```

The two planning commands execute no model. They stream SHA256 checks of model/config/tokenizer files; verify source and canonical image bytes, decoded RGB hashes and dimensions; check the preserved executable/source ZIP; and require the current source inventory to match the build. Plans pin both orchestration scripts and preserve them, the workload, and source locks in `protocol-source.zip`. Editing protocol scripts invalidates an existing plan; use its preserved protocol or make a new plan. Editing captured Rust source requires a fresh build. Input-only validation is labeled explicitly and does not qualify a binary or run plan. Pillow is used only to verify lossless RGB bytes, never to generate benchmark preprocessing pixels.

Only when other project CPU/GPU workloads are paused, execute a frozen plan:

```powershell
python research/benchmarks/scripts/realistic_benchmark.py --plan artifacts/benchmarks/fullpages-plan-v1/plan.json --run --quiet-attestation "Operator verified other project CPU/GPU work is paused"
python research/benchmarks/scripts/compare_realistic_benchmarks.py --plan artifacts/benchmarks/fullpages-plan-v1/plan.json --output artifacts/benchmarks/fullpages-plan-v1/comparison.json
```

The quiet statement is a recorded operator assertion, not an automated idle detector. `--run` without an existing plan and the assertion is rejected. The runner writes command logs, execution start/finish records, and hashes every result. It refuses existing outputs and partial brackets; preserve an interrupted directory and create a new plan for that complete group. Comparison independently checks report identities, requested options, image order, all counts, every sample's actual text and stop, token IDs, derived medians and throughput. Missing text is an error; text is neither replayed nor inferred from equal IDs. Tokens following EOS, early length stops, missing requests, teacher forcing, mode substitutions and wrong build identities are rejected. All 4096-token truncations stay labeled `length` and never count as EOS. A comparison failure must be investigated; do not loosen numeric parity gates or drop the page.

## What these timings and memory counters mean

Warm batch wall time includes resizing/patch preparation, image projection and prefill, generation, and final text decoding. File reading/RGB decoding, verified model loading, and one-time packed-weight preparation are reported separately. The primary comparable latency is the wall time of the complete requested batch; throughput is batch pages divided by that wall time. This is no-arrival-queue inference, with independent prefills and joint decode, rather than a streaming document pipeline.

Per-request preprocessing and prefill are useful stage observations. New FP32 results also record `image_projection_ms` for the projector linear call, excluding feature allocation and embedding copies, and `transformer_prefill_ms` for the full transformer forward including final vocabulary logits. Both are contained within `prefill_ms`; do not add them again to total time. Historical results and the experimental BF16 runner omit these fields when unmeasured. The current benchmark validator requires them; the initial archived planning receipt predates this instrumentation and remains historical.

Joint `decode_ms` is elapsed shared decode time until that request completes, not exclusive work attributable to that page; summing it double-counts work. Joint TTFT and total time start at the chunk, include earlier requests' prefills/shared work, and have different origins from sequential per-request timing. First-token completion does not imply streaming delivery. These values cannot reconstruct independently timed per-page service latency or per-token speed. The harness does not isolate per-layer work, bandwidth, cache misses, thread affinity, or energy.

Memory snapshots and peak RSS/private commit cover the whole fresh process, including loading and warmup. They are not per-request KV measurements or isolated decode peaks. Packed payload bytes are exact tensor storage; RSS deltas also include allocators, scratch, image buffers, and page residency. Large batch/cap combinations may allocate substantial KV storage and take long enough for thermal drift; a failed allocation or unstable control stays visible, with no automatic smaller-shape retry. Native Windows, WSL compatibility results, and future bare-metal Linux performance are reported separately.

The numerical/quality scope is exact outputs across these CPU modes for the frozen inputs. It does not resolve outstanding GPU hidden-stage FP32 gates, qualify BF16, establish representative OCR quality, or predict corpus throughput. Old tiny-fixture reports remain unchanged.

Protocol regression checks, with synthetic JSON and no model execution:

```powershell
python -m unittest discover -s scripts -p test_realistic_benchmark.py -v
```
