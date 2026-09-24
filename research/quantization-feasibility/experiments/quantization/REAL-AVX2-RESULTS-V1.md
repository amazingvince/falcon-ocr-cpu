# Real-projection AVX2 W4A32 operator evidence

The unchanged isolated AVX2/FMA kernel now passes arithmetic checks on the actual checkpoint matrices and saved FP32 reference inputs for all 28 projection cases. Both scalar and AVX2 outputs pass the existing `gamma_K * sum(abs(x * reconstructed_weight))` bound for **1,261,568 outputs per backend**, across **176 matrix/group/batch cases**. This is operator evidence only: no model execution, timing measurement, quality assessment, or production integration occurred.

The fixture contains five prefill layers (9, 12, 13, 17, 18) with 144 rows and two one-row decode layers (step 2/layer 21, step 6/layer 19). Each contributes QKV, WO, W13, and W2:

| Projection | Output channels N | Reduction K |
| --- | ---: | ---: |
| QKV | 2,048 | 768 |
| WO | 768 | 1,024 |
| W13 | 4,608 | 768 |
| W2 | 768 | 2,304 |

Both group 64 and group 128 are checked. Prefill batches use actual source indices: B1 `[0]`; B2 `[0,143]`; B4 `[0,47,95,143]`; B8 `[0,20,40,61,81,102,122,143]`. All selected batches contain that many distinct activation rows by exact bytes. The one-row decode fixtures support only B1; their B2/B4/B8 omissions are explicit in the plan. No row is replicated within a batch to manufacture diversity. Combining selected prefill rows checks the matrix kernel; it does not establish joint-request model behavior.

Every output channel is computed, preserved as FP32 little-endian bytes, and checked. Activation rows are deliberately selected subsets: their union covers 10 of 144 positions per prefill operator, with overlaps across the four batch sizes. The output count includes those repeated positions across separately executed batch shapes. It does not describe full-prefill coverage or independent quality samples.

The checker independently reconstructs the complete packed-code arrays, FP32 scale arrays and dequantized FP32 weights using NumPy. All 56 matrix/group combinations match the earlier pinned scalar experiment's exact code, scale, and reconstruction hashes. The original checkpoint, fixture, fixture-sidecar, source-trace, per-matrix weight, complete source-activation, and saved GPU-output hashes are preserved and checked. No format or quantizer arithmetic changed.

For every captured output, single-threaded NumPy FP64 dot products provide the dequantized-weight reference and absolute-product sum. The check conservatively includes the FP64 summation uncertainty: it requires `abs(candidate-reference64) + gamma64*A64/(1-gamma64) <= gamma32*A64/(1+gamma64)`. This is a sufficient condition for the unchanged FP32 bound and tightens the test rather than enlarging its tolerance. A separately selected 9,240-output `math.fsum` cross-check verifies the FP64 oracle within its own FP64 error bound. These oracle cross-checks are sampled; the candidate-output arithmetic gate covers all 1,261,568 outputs per backend.

All bound-violation counts are zero. The largest conservative fraction of the existing bound is 0.01121642 for scalar and 0.00332414 for AVX2. These fractions describe accumulation arithmetic on already quantized weights. Quantization error versus original weights and diagnostic error versus saved GPU outputs remain separately reported; they establish no OCR accuracy budget, clipping rule, group-size choice, or layer policy. The source activations come from the existing synthetic diagnostic fixture, not calibration or held-out quality data.

The capture also reruns all nine existing Rust kernel tests with one test thread; four new small checker tests pass for row selection, corrupted/nonfinite outputs, changed binary/plan/archive/output files, missing provenance, and mismatched embedded identities. The settled kernel, Q4 reference, existing probe, production Rust, Cargo, build wrappers, and shared capture helpers remain unchanged.

Evidence is preserved under `artifacts/quantization/q4-real-avx2-operator-v1/`:

- `plan.json`: exact row selections, all operand hashes, earlier real-weight build proof, and explicit omissions.
- `operands/`: the actual matrix and selected activation bytes passed to the standalone Rust adapter.
- `outputs/`: complete codes, scales, and scalar/AVX2 output arrays for all covered shapes.
- `source.zip`, executable, `build.json`, build/probe/test logs: source-before/after checks and exact executable identity.
- `independent-check.json`: all per-case arithmetic/diagnostic results and output digests.

Key SHA-256 identities:

- Check report: `4be655fe1cb3ae10092c5cac9794ce669341e81e9b21b221b1c383958feea55f`
- Build receipt: `1f5b91bed30b1397b97e2275d464af5591b2065056f6d64b0a279834283ba694`
- Executable: `e3272fd2945fa9e8267673d97114ef4fb6452f2396812c59dc7d216fece69087`
- Source ZIP: `68caa1286802740337889b4196a970a25a67e30c9d9136ef77f85458975f494b`
- Plan: `c8c750770387ef4e5aaf7dc3e8bcafeb33fbd4fefb1fc318dc939bb4cde80c24`

Capture into a fresh directory, using one operator thread and no benchmark collection:

```powershell
& C:/Users/amazi/mambaforge/python.exe research/quantization-feasibility/experiments/quantization/capture_real_avx2_probe.py --output artifacts/quantization/q4-real-avx2-operator-NEW
```

Recheck preserved evidence without changing it:

```powershell
& C:/Users/amazi/mambaforge/python.exe research/quantization-feasibility/experiments/quantization/capture_real_avx2_probe.py --check artifacts/quantization/q4-real-avx2-operator-v1 --report artifacts/quantization/q4-real-avx2-operator-v1/recheck-NEW.json
```

The scripts refuse report/capture overwrites. Preserved compiler/platform details are not a hermetic toolchain or cross-platform qualification. The test harness's ordinary elapsed line is retained as a test log, not an operator timing result. Later model integration, calibration, held-out quality evaluation, and quiet performance work remain separate decisions.
