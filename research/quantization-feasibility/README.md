# quantization-feasibility

Early quantized-operator feasibility experiments (source only, not compiled).

| File | What it did or produced | Status |
|---|---|---|
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | INT8 / INT4 feasibility, 2026-09-20 | superseded by attempt3/RESULTS-V3.md (GPTQ W8G64 fast mode) |
| [examples/quant_probe.rs](examples/quant_probe.rs) | archived source (Stage A) | history |
| [experiments/quantization/AVX2-RESULTS-V1.md](experiments/quantization/AVX2-RESULTS-V1.md) | AVX2 W4A32 operator evidence v1 | history |
| [experiments/quantization/README.md](experiments/quantization/README.md) | Isolated W4A32 / W8A32 arithmetic experiment | history |
| [experiments/quantization/REAL-AVX2-RESULTS-V1.md](experiments/quantization/REAL-AVX2-RESULTS-V1.md) | Real-projection AVX2 W4A32 operator evidence | history |
| [experiments/quantization/RESULTS-V1.md](experiments/quantization/RESULTS-V1.md) | Bounded operator evidence, 2026-09-20 | history |
| [experiments/quantization/audit_bf16_storage.py](experiments/quantization/audit_bf16_storage.py) | Check exact BF16 storage representability of the pinned FP32 checkpoint | history |
| [experiments/quantization/capture_avx2_probe.py](experiments/quantization/capture_avx2_probe.py) | Capture/check the std-only AVX2 experiment; no inference or speed measurements | history |
| [experiments/quantization/capture_probe.py](experiments/quantization/capture_probe.py) | Preserve a quant_probe build and optionally run its bounded scalar diagnostic | history |
| [experiments/quantization/capture_real_avx2_probe.py](experiments/quantization/capture_real_avx2_probe.py) | Full-channel, single-thread Q4 arithmetic checks on pinned real projection operands | history |
| [experiments/quantization/check_probe.py](experiments/quantization/check_probe.py) | Independent NumPy reconstruction and math.fsum oracles; no model/timing execution | history |
| [experiments/quantization/describe_layout.py](experiments/quantization/describe_layout.py) | Read tensor headers and calculate payloads; no tensor quantization or inference | history |
| [experiments/quantization/isa_probe.rs](experiments/quantization/isa_probe.rs) | Feature availability only; no arithmetic benchmark or production dispatch | history |
| [experiments/quantization/operator-probe-v1-summary.json](experiments/quantization/operator-probe-v1-summary.json) | Summary of the scalar W4/W8 operator probe v1 | history |
| [experiments/quantization/q4_avx2.rs](experiments/quantization/q4_avx2.rs) | Isolated W4A32 decode experiment; never part of the production runner | history |
| [experiments/quantization/q4_avx2_probe.rs](experiments/quantization/q4_avx2_probe.rs) | Standalone diagnostic: rustc --edition 2024 [--test] q4_avx2_probe.rs | history |
| [experiments/quantization/q4_real_avx2_probe.rs](experiments/quantization/q4_real_avx2_probe.rs) | Std-only transport adapter for the unchanged isolated Q4 quantizer/kernels | history |
| [experiments/quantization/q4_reference.rs](experiments/quantization/q4_reference.rs) | Standalone research oracle; not compiled into the production runner | history |
| [experiments/quantization/q8_reference.rs](experiments/quantization/q8_reference.rs) | Isolated W8A32 control: signed [-127,127] row-major bytes, FP32 | history |
| [experiments/quantization/real-avx2-operator-v1-summary.json](experiments/quantization/real-avx2-operator-v1-summary.json) | Summary of the real-projection AVX2 Q4 probe | history |
| [experiments/quantization/test_real_avx2_probe.py](experiments/quantization/test_real_avx2_probe.py) | Bounded tests of real-Q4 capture/checker guards; no model or timing run | history |
| [tests/test_quantization_provenance.py](tests/test_quantization_provenance.py) | Negative provenance checks independent of quantized model arithmetic | history |
