# bf16-graph

The retired experimental BF16 execution graph (source only, not compiled).

| File | What it did or produced | Status |
|---|---|---|
| [docs/BF16.md](docs/BF16.md) | BF16 implementation notes | history |
| [examples/bf16_attention_probe.rs](examples/bf16_attention_probe.rs) | archived source (Stage A) | history |
| [examples/bf16_ops_probe.rs](examples/bf16_ops_probe.rs) | archived source (Stage A) | history |
| [examples/bf16_probe.rs](examples/bf16_probe.rs) | archived source (Stage A) | history |
| [examples/bf16_qk_reduction_probe.rs](examples/bf16_qk_reduction_probe.rs) | archived source (Stage A) | history |
| [examples/support/bf16_qk_attention.rs](examples/support/bf16_qk_attention.rs) | archived source (Stage A) | history |
| [examples/support/bf16_qk_candidates.rs](examples/support/bf16_qk_candidates.rs) | archived source (Stage A) | history |
| [experiments/bf16_attention/analyze_crossings.py](experiments/bf16_attention/analyze_crossings.py) | Inspect saved tile evidence and CPU-only same-argument exp2 counterfactuals | history |
| [experiments/bf16_attention/analyze_sinks.py](experiments/bf16_attention/analyze_sinks.py) | Replay frozen BF16 sink failures with independent raw/LSE interventions | history |
| [experiments/bf16_attention/exp2_probe.rs](experiments/bf16_attention/exp2_probe.rs) | CPU-only diagnostic: exact F32 argument bits -> Rust exp2 and BF16 RN-even | history |
| [experiments/bf16_attention/map_exp2_midpoint.py](experiments/bf16_attention/map_exp2_midpoint.py) | CPU-only provenance join for one saved standalone exp2 BF16 midpoint | history |
| [experiments/bf16_attention/sink_probe.rs](experiments/bf16_attention/sink_probe.rs) | CPU-only replay of the production BF16 attention sink arithmetic | history |
| [scripts/analyze_bf16_trajectory.py](scripts/analyze_bf16_trajectory.py) | Record where independently rounded BF16 GPU trajectories first diverge | history |
| [scripts/assess_bf16_local_reports.py](scripts/assess_bf16_local_reports.py) | Assess recorded Rust operators against frozen GPU-only bounds, without edits | history |
| [scripts/bf16_ptx_observer_support_v2.py](scripts/bf16_ptx_observer_support_v2.py) | Host decoding, write-coverage checks and unchanged-ABI launch for PTX observer | history |
| [scripts/compare_bf16_attention_tiles.py](scripts/compare_bf16_attention_tiles.py) | Locate equal-input BF16 tile differences without changing acceptance gates | history |
| [scripts/compare_bf16_local_tensors.py](scripts/compare_bf16_local_tensors.py) | Apply frozen per-element BF16 local bounds to actual Rust operator outputs | history |
| [scripts/compare_bf16_outputs.py](scripts/compare_bf16_outputs.py) | Apply frozen same-prefix BF16 output gates without accepting hidden drift | history |
| [scripts/explain_bf16_local_failures.py](scripts/explain_bf16_local_failures.py) | Record concrete immutable-bound failures for kernel investigation | history |
| [scripts/export_bf16_ptx_observer_v2.py](scripts/export_bf16_ptx_observer_v2.py) | One reduced v2 PTX observer; all original runtime gates remain unchanged | history |
| [scripts/inspect_bf16_mma_ptx.py](scripts/inspect_bf16_mma_ptx.py) | Read existing Triton cache artifacts; never initialize CUDA or execute code | history |
| [scripts/inspect_flex_kernels.py](scripts/inspect_flex_kernels.py) | Read, never execute, generated kernel source to record selected tile metadata | history |
| [scripts/instrument_bf16_ptx_v2.py](scripts/instrument_bf16_ptx_v2.py) | Prepare the one reduced v2 observer for byte-pinned emitted PTX | history |
| [scripts/prepare_bf16_exp2_arguments.py](scripts/prepare_bf16_exp2_arguments.py) | Freeze exact saved GPU/CPU/counterfactual exp2 arguments for later GPU replay | history |
| [scripts/review_bf16_calibration.py](scripts/review_bf16_calibration.py) | Reject signal-scale BF16 calibration without modifying its immutable evidence | history |
| [scripts/run_bf16_graph_reference.sh](scripts/run_bf16_graph_reference.sh) | Independent GPU-only references. Dense BF16 is diagnostic, never a parity gate | history |
| [scripts/run_bf16_ptx_observer_v2.sh](scripts/run_bf16_ptx_observer_v2.sh) | Driver for the v2 PTX observer in the reference environment | history |
| [scripts/run_bf16_ptx_observer_v2_lf.sh](scripts/run_bf16_ptx_observer_v2_lf.sh) | LF line-ending copy of run_bf16_ptx_observer_v2.sh | one-off LF line-ending copy of run_bf16_ptx_observer_v2.sh |
| [scripts/test_bf16_ptx_observer.py](scripts/test_bf16_ptx_observer.py) | Unit tests for the v1 PTX observer support | superseded by scripts/test_bf16_ptx_observer_v2.py |
| [scripts/test_bf16_ptx_observer_v2.py](scripts/test_bf16_ptx_observer_v2.py) | Unit tests for the v2 PTX observer support | history |
| [scripts/test_cuda_ptx_reference.py](scripts/test_cuda_ptx_reference.py) | Unit tests for the CUDA driver ABI helper | history |
| [scripts/test_verify_bf16_reduction_lowering.py](scripts/test_verify_bf16_reduction_lowering.py) | Unit tests for verify_bf16_reduction_lowering.py | history |
| [scripts/verify_bf16_qk_candidates.py](scripts/verify_bf16_qk_candidates.py) | Cross-check the Rust diagnostic with the unchanged GPU-team comparator | history |
| [src/bf16_attention.rs](src/bf16_attention.rs) | archived source (Stage A) | history |
| [src/bf16_attention_diagnostics.rs](src/bf16_attention_diagnostics.rs) | archived source (Stage A) | history |
| [src/bf16_kernels.rs](src/bf16_kernels.rs) | archived source (Stage A) | history |
| [src/bf16_model.rs](src/bf16_model.rs) | archived source (Stage A) | history |
| [src/bf16_ops.rs](src/bf16_ops.rs) | archived source (Stage A) | history |
| [src/bf16_runner.rs](src/bf16_runner.rs) | archived source (Stage A) | history |
| [tests/bf16_runner.rs](tests/bf16_runner.rs) | archived source (Stage A) | history |
