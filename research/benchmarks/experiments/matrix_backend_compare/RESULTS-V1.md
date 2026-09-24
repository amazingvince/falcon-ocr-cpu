# RTen 0.26.0 FP32 operator screening, native Windows

**Do not integrate this RTen M1 mode into the model.** The fixed decode suite was 13.17–14.33% slower than the same-compiler control. The M144 prefill suite was 15.93–16.65% faster, which justifies considering an actual M6544 operator check; it does not establish a full-page improvement. No backend or default is promoted.

These are saved results from one native Windows process on the Ryzen 9 7950X, using Rust 1.94.0, LLVM 21.1.8, and one 16-thread Rayon pool. RTen selected `x86_64-f32-avx512`. The M1 control used the frozen AVX2 dot kernels; its M144 path used the separately dispatched `gemm 0.19` implementation. Both were compiled in the same executable. The comparison therefore changes library and selected SIMD implementation together, not only instruction width.

The candidate used FP32 inputs/output/accumulation, alpha=1, beta=0, no bias or quantization, and unpacked inputs. Its retained row-major weights were viewed as a transposed RHS without a per-call transpose copy. M1 consequently used RTen's dedicated GEMV path; M144 used its tiled GEMM with internal packing. Prepacked-B performance was not tested.

## Timing

Each arm performed two untimed warm suites and seven measured samples. Each sample contained five complete, ordered suites; the recorded sample is milliseconds per suite. Arms ran in the fixed order control-before, candidate, control-after. Constructor copies, file IO, FP64 reference computation, allocation counting, and output validation were outside timing; per-call library packing/scratch and scheduling were inside it.

| Fixed suite | Control before, ms | RTen, ms | Control after, ms | Gain vs before | Gain vs after | Control drift |
|---|---:|---:|---:|---:|---:|---:|
| M1 decode, 17 projections | 6.73306 | 7.69778 | 6.80214 | −14.328106% | −13.167033% | +1.025982% |
| M144 prefill, four projections | 3.19072 | 2.65958 | 3.16338 | +16.646400% | +15.926003% | −0.856860% |

Medians were independently recomputed from all seven saved samples. Gain is `100*(1 − candidate/control)`; drift is `100*(control_after/control_before − 1)`. Negative gain means slower. Both control drifts are below 5%, but these short operator measurements are one bracket, not a page-level acceptance test or a confidence interval.

The M1 suite is QKV/WO/W13/W2 from layers 9, 12, 19, and 21 at the fixed saved decode step, followed by the vocabulary projection: 17 distinct weight matrices totaling **324,009,984 bytes (309 MiB)**. It rotates more weights than one cached matrix; it is still not a transformer token latency measurement. The M144 suite is the four saved layer-0 prefill operators, totaling **30,670,848 weight bytes (29.25 MiB)**. No actual full-page M6544 operator or model run occurred.

The timing invocation records an owned-workload quiet window and an immediate process scan. It does not establish OS-wide isolation, affinity, cache flushing, measured bandwidth, or a causal stall diagnosis. The process started at `2026-09-20T21:58:18.146653+00:00`, ended at `21:58:22.081549+00:00`, and exited 0.

## Numerical and allocation scope

The preceding untimed accuracy run and timed run have identical numerical summaries and identical declared control/candidate output hashes for both suites. Every warmup and every sample's **final** complete suite was checked against that backend's own full saved FP32 output bits. Intermediate overwritten suite outputs within each five-iteration sample were not individually retained or checked. This is repeatability within each backend, not candidate/control equality.

| Saved comparison | M1 decode | M144 prefill |
|---|---:|---:|
| Candidate/control differing FP32 values | 76,222 / 98,304 | 1,088,504 / 1,179,648 |
| Candidate/GPU differing FP32 values | 25,779 / 32,768 | 1,086,826 / 1,179,648 |
| Maximum candidate/control absolute difference | 0.0001220703125 | 0.00030517578125 |
| Maximum candidate/GPU absolute difference | 0.0000762939453125 | 0.0006103515625 |
| Values checked against sequential FP64 dots | 98,304, all output channels | 24,576, all channels in rows 0/112/143 |
| Candidate values outside the existing `gamma_K` plus FP64-summation uncertainty envelope | 0 | 0 |

The vocabulary case has no paired GPU output in this fixture; its control exactly reproduced all 65,536 historical CPU logits after the pinned CPU final-norm reconstruction. Thus the decode GPU count covers only the other 16 projections. The prefill FP64 check samples three rows, not all 144; the full candidate/GPU and candidate/control comparisons cover all rows. The arithmetic envelope is a diagnostic bound and does not replace the frozen full-model numerical policy. These results establish neither exact GPU parity nor OCR quality. FP64 RMS error was lower for the candidate in 9/17 decode cases and 4/4 selected-row prefill cases; that does not qualify the full model.

Warm allocation counters were collected separately from timing:

| One warm suite | Control allocation calls / requested bytes | RTen allocation calls / requested bytes |
|---|---:|---:|
| Decode, accuracy and timing processes | 0 / 0 | 0 / 0 |
| M144, accuracy process | 170 / 222,536 | 1 / 131,072 |
| M144, timing process | 170 / 222,536 | 3 / 884,736 |

There were no reallocations in these intervals, and each reported allocation was accompanied by a same-sized deallocation. The varying prefill counts demonstrate why this single interval is not a fixed allocation guarantee. Counters cover Rust global-allocator requests across pool threads; they do not report TLS retained capacity, native allocations, allocator usable size, peak RSS, or allocation timing.

The candidate additionally owned weight copies of 309 MiB for decode and 29.25 MiB for prefill. Summed constructor times in the timing process were 33.265 ms and 3.3208 ms respectively, excluded from the suite timing. These setup costs and retained payloads must be accounted for in any future integration; the current experiment does not amortize them into a model result.

## Evidence and independent review

All paths below are under `artifacts/diagnostics/matrix-backends-rust194-v1/`:

| Artifact | SHA-256 |
|---|---|
| `preparation.json` | `3de4a504493198275c76bdb13a9aa9ce37a0d4dedf06aa5294349bfed9074629` |
| `source.zip` | `83b00aa3ddae15d6927bf1fa6e31a8781f0c82531fee878829995c130e402f0b` |
| `build.json` | `9f3b8f9660ca2173e1feba87f132a46fc3cb180913e84eb84c75e78e52a932bb` |
| `accuracy-v1/report.json` | `504547b2bec9e4f48abc1354a72808eb73861d713879c54a7496ba9c4d82a73a` |
| `timing-invocation-v1.json` | `9d2648098d142e735f9959e7be508e59f0e50977a7207bb60672a2761b61b91c` |
| `timing-execution-v1.json` | `5eb2e04e8a2a350c24bcebd5ae4c1a0d57217c94d66fe5bb1ddcfffde73e9c1e` |
| `timing-v1/report.json` | `0d274c500d0e428f90b04b2945a2c8ec742f77e8eaf53ba8dcfcb9f4593eda38` |

The shared executable digest recorded by the build and both runs is `c02e5d74a7de0be4306d55f1e2c2562c18540197ac3d8e40fea7b93747315e81`. Copied candidate source is `55a3f58e270e76a1fe03c60329ccace945de1d92cd116105fb5a534611c09fa4`; copied baseline kernels are `ad812805e03b536de0408d6f0dd21d3e944d6d752bc510331c59da44fe80fc5c`.

This independent review recomputed medians/percentages and numerical counts from the small reports, checked source/manifest/lock hashes, source ZIP CRC/hash, preparation/build/run joins, recorded platform/options, all JSON output-artifact hashes, and the timing log/exit receipt. It inspected the frozen repeat/allocation/error-bound code. Large model/fixture files and saved FP32 output payloads were not rehashed or numerically recomputed here; their recorded run-time validation and hashes are inherited evidence. No model, build, benchmark, or test was rerun, and no frozen evidence was changed.
