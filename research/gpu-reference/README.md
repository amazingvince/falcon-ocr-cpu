# gpu-reference

The strict FP32 GPU export, reference capture and comparison protocol whose receipts live in `reference/`.

| File | What it did or produced | Status |
|---|---|---|
| [examples/attention_probe.rs](examples/attention_probe.rs) | archived source (Stage A) | history |
| [examples/linear_probe.rs](examples/linear_probe.rs) | archived source (Stage A) | history |
| [examples/rms_probe.rs](examples/rms_probe.rs) | archived source (Stage A) | history |
| [experiments/fp32_crossover/README.md](experiments/fp32_crossover/README.md) | One FP32 frozen-state crossover | history |
| [experiments/fp32_crossover/SOURCE_REVIEW_NOTES.md](experiments/fp32_crossover/SOURCE_REVIEW_NOTES.md) | Independent source-only review | history |
| [experiments/fp32_crossover/adapter.py](experiments/fp32_crossover/adapter.py) | Generate a test-only segment by retaining original forward operation text | history |
| [experiments/fp32_crossover/capture.py](experiments/fp32_crossover/capture.py) | Prospective copy/build/run capture. Each phase requires explicit quiet release | history |
| [experiments/fp32_crossover/compare.py](experiments/fp32_crossover/compare.py) | One offline three-branch decomposition; no model invocation or policy changes | history |
| [experiments/fp32_crossover/contract.py](experiments/fp32_crossover/contract.py) | Fixed crossover contract. Importing this module performs no I/O or execution | history |
| [experiments/fp32_crossover/export_gpu.py](experiments/fp32_crossover/export_gpu.py) | Prospective native GPU arm. Execution requires a separately reviewed, pinned plan | history |
| [experiments/fp32_crossover/rust_arm.rs](experiments/fp32_crossover/rust_arm.rs) | Included only inside model::fp32_crossover in an isolated test-only copy | history |
| [experiments/fp32_crossover/test_source.py](experiments/fp32_crossover/test_source.py) | Small source/contract tests. Run only after the quiet-window release | history |
| [experiments/fp32_crossover_block0/README.md](experiments/fp32_crossover_block0/README.md) | Block0 crossover on complete saved embeddings | history |
| [experiments/fp32_crossover_block0/SOURCE_REVIEW_NOTES.md](experiments/fp32_crossover_block0/SOURCE_REVIEW_NOTES.md) | Prospective block0 observation contract | history |
| [experiments/fp32_crossover_block0/adapter.py](experiments/fp32_crossover_block0/adapter.py) | Generate a test-only segment by retaining original forward operation text | history |
| [experiments/fp32_crossover_block0/capture.py](experiments/fp32_crossover_block0/capture.py) | Prospective copy/build/run capture. Each phase requires explicit quiet release | history |
| [experiments/fp32_crossover_block0/compare.py](experiments/fp32_crossover_block0/compare.py) | One offline three-branch decomposition; no model invocation or policy changes | history |
| [experiments/fp32_crossover_block0/contract.py](experiments/fp32_crossover_block0/contract.py) | Fixed crossover contract. Importing this module performs no I/O or execution | history |
| [experiments/fp32_crossover_block0/export_gpu.py](experiments/fp32_crossover_block0/export_gpu.py) | Prospective native GPU arm. Execution requires a separately reviewed, pinned plan | history |
| [experiments/fp32_crossover_block0/rust_arm.rs](experiments/fp32_crossover_block0/rust_arm.rs) | Included only inside model::fp32_crossover_block0 in an isolated test-only copy | history |
| [experiments/fp32_crossover_block0/source_guard.py](experiments/fp32_crossover_block0/source_guard.py) | Text-only guard against the preserved layer7 observer | history |
| [experiments/fp32_crossover_block0/test_source.py](experiments/fp32_crossover_block0/test_source.py) | Source/contract tests for the block0 crossover | history |
| [experiments/fp32_crossover_downstream/README.md](experiments/fp32_crossover_downstream/README.md) | Held downstream FP32 crossover | history |
| [experiments/fp32_crossover_downstream/compare.py](experiments/fp32_crossover_downstream/compare.py) | Offline four-endpoint telescoping only; never invokes a model or changes policy | history |
| [experiments/fp32_crossover_downstream/contract.py](experiments/fp32_crossover_downstream/contract.py) | Prospective fixed contract. Importing performs no I/O or numerical work | history |
| [experiments/fp32_crossover_downstream/evidence.py](experiments/fp32_crossover_downstream/evidence.py) | Read and validate saved evidence only when explicitly called after release | history |
| [experiments/fp32_crossover_downstream/export_gpu.py](experiments/fp32_crossover_downstream/export_gpu.py) | Held GPU arm: exact historical F(C7) control, then one native F(S) | history |
| [experiments/fp32_crossover_downstream/native_segment.py](experiments/fp32_crossover_downstream/native_segment.py) | Native segment copied from the pinned exporter; checked before future use | history |
| [experiments/fp32_crossover_downstream/prepare.py](experiments/fp32_crossover_downstream/prepare.py) | Prepare a fresh source-bound plan only after the quiet-window release | history |
| [experiments/fp32_crossover_downstream/source_guard.py](experiments/fp32_crossover_downstream/source_guard.py) | Compare native source fragments only when explicitly called; no imports run it | history |
| [experiments/fp32_crossover_downstream/test_source.py](experiments/fp32_crossover_downstream/test_source.py) | Source/contract tests for the downstream crossover | history |
| [experiments/fp32_crossover_layer7/README.md](experiments/fp32_crossover_layer7/README.md) | One native layer-7 FP32 state crossover | history |
| [experiments/fp32_crossover_layer7/SOURCE_REVIEW_NOTES.md](experiments/fp32_crossover_layer7/SOURCE_REVIEW_NOTES.md) | Prospective layer-7 observation contract | history |
| [experiments/fp32_crossover_layer7/adapter.py](experiments/fp32_crossover_layer7/adapter.py) | Generate a test-only segment by retaining original forward operation text | history |
| [experiments/fp32_crossover_layer7/capture.py](experiments/fp32_crossover_layer7/capture.py) | Prospective copy/build/run capture. Each phase requires explicit quiet release | history |
| [experiments/fp32_crossover_layer7/compare.py](experiments/fp32_crossover_layer7/compare.py) | One offline three-branch decomposition; no model invocation or policy changes | history |
| [experiments/fp32_crossover_layer7/contract.py](experiments/fp32_crossover_layer7/contract.py) | Fixed crossover contract. Importing this module performs no I/O or execution | history |
| [experiments/fp32_crossover_layer7/export_gpu.py](experiments/fp32_crossover_layer7/export_gpu.py) | Prospective native GPU arm. Execution requires a separately reviewed, pinned plan | history |
| [experiments/fp32_crossover_layer7/rust_arm.rs](experiments/fp32_crossover_layer7/rust_arm.rs) | Included only inside model::fp32_crossover_layer7 in an isolated test-only copy | history |
| [experiments/fp32_crossover_layer7/test_source.py](experiments/fp32_crossover_layer7/test_source.py) | Source/contract tests for the layer-7 crossover | history |
| [experiments/fp32_prefix_suffix_telescope/README.md](experiments/fp32_prefix_suffix_telescope/README.md) | Fixed CPU-prefix / GPU-suffix telescope | history |
| [experiments/fp32_prefix_suffix_telescope/RESULTS-V1.md](experiments/fp32_prefix_suffix_telescope/RESULTS-V1.md) | Fixed prefix/suffix telescope: accepted diagnostic | history |
| [experiments/fp32_prefix_suffix_telescope/compare.py](experiments/fp32_prefix_suffix_telescope/compare.py) | Offline fixed-boundary telescope; no model invocation or policy changes | history |
| [experiments/fp32_prefix_suffix_telescope/contract.py](experiments/fp32_prefix_suffix_telescope/contract.py) | Fixed eleven-branch diagnostic; imports do no I/O or numerical work | history |
| [experiments/fp32_prefix_suffix_telescope/evidence.py](experiments/fp32_prefix_suffix_telescope/evidence.py) | Extend the preserved historical loader; payload work occurs only on call | history |
| [experiments/fp32_prefix_suffix_telescope/export_gpu.py](experiments/fp32_prefix_suffix_telescope/export_gpu.py) | Fixed eleven-branch GPU diagnostic, only after reviewed-plan release | history |
| [experiments/fp32_prefix_suffix_telescope/legacy_contract.py](experiments/fp32_prefix_suffix_telescope/legacy_contract.py) | Prospective fixed contract. Importing performs no I/O or numerical work | superseded by contract.py in the same directory |
| [experiments/fp32_prefix_suffix_telescope/legacy_evidence.py](experiments/fp32_prefix_suffix_telescope/legacy_evidence.py) | Read and validate saved evidence only when explicitly called after release | superseded by evidence.py in the same directory |
| [experiments/fp32_prefix_suffix_telescope/native_segment.py](experiments/fp32_prefix_suffix_telescope/native_segment.py) | One fixed native prefix/suffix run; no arithmetic alternatives or retries | history |
| [experiments/fp32_prefix_suffix_telescope/prepare.py](experiments/fp32_prefix_suffix_telescope/prepare.py) | Prepare a fresh source-bound plan only after the quiet-window release | history |
| [experiments/fp32_prefix_suffix_telescope/review_saved_v1.py](experiments/fp32_prefix_suffix_telescope/review_saved_v1.py) | Independent post-capture recomputation; imports no experiment implementation | history |
| [experiments/fp32_prefix_suffix_telescope/source_guard.py](experiments/fp32_prefix_suffix_telescope/source_guard.py) | Bounded text/AST guards, not a machine-code or universal numerical proof | history |
| [experiments/fp32_prefix_suffix_telescope/test_source.py](experiments/fp32_prefix_suffix_telescope/test_source.py) | Source/contract tests for the telescope | history |
| [experiments/launch_prefix_suffix_telescope.py](experiments/launch_prefix_suffix_telescope.py) | Launch exactly the reviewed telescope once and retain native process logs | history |
| [experiments/linear/OWNED-WORKSPACE.md](experiments/linear/OWNED-WORKSPACE.md) | This is a prospective observation of one pinned W2 projection through the | history |
| [experiments/linear/capture_observed_partition_probe.py](experiments/linear/capture_observed_partition_probe.py) | Build/test and preserve only the fixed observed-W2 Rust operator diagnostic | history |
| [experiments/linear/capture_observed_w2_intervention.py](experiments/linear/capture_observed_w2_intervention.py) | One isolated W2 full-graph diagnostic; never edits the live production tree | history |
| [experiments/linear/export_owned_workspace.py](experiments/linear/export_owned_workspace.py) | Prospective direct-cuBLAS diagnostic; GPU execution belongs to the reference owner | named by the frozen driver; moved, path documented |
| [experiments/linear/observed_partition_probe/Cargo.lock](experiments/linear/observed_partition_probe/Cargo.lock) | Lockfile of the observed-W2 partitions probe crate | history |
| [experiments/linear/observed_partition_probe/Cargo.toml](experiments/linear/observed_partition_probe/Cargo.toml) | Manifest of the observed-W2 partitions probe crate | history |
| [experiments/linear/observed_partition_probe/src/main.rs](experiments/linear/observed_partition_probe/src/main.rs) | One diagnostic: production-style GEMM on the observed W2 K ranges | history |
| [experiments/linear/replay_observed_partials.py](experiments/linear/replay_observed_partials.py) | CPU-only validation and one fixed ascending-slot FP32 fold of observed scratch | history |
| [experiments/linear/test_observed_w2_intervention.py](experiments/linear/test_observed_w2_intervention.py) | Source/proof checks only; no model, build, GPU or subprocess execution | history |
| [experiments/linear/test_workspace_observation.py](experiments/linear/test_workspace_observation.py) | Host-only diagnostic tests; never import Torch or initialize CUDA | history |
| [experiments/linear/workspace_observation.py](experiments/linear/workspace_observation.py) | Host-only checks for the prospective, caller-owned W2 scratch observation | history |
| [experiments/rms_norm/RSQRT-TABLE.md](experiments/rms_norm/RSQRT-TABLE.md) | Observed CUDA rsqrt table diagnostic and its validation | history |
| [experiments/rms_norm/capture_cuda_source_identity.py](experiments/rms_norm/capture_cuda_source_identity.py) | Read installed Torch identity/headers and exact-revision RMSNorm sources | history |
| [experiments/rms_norm/capture_gpu_rsqrt_intervention.py](experiments/rms_norm/capture_gpu_rsqrt_intervention.py) | Isolated width768 RMS intervention using an exhaustively checked GPU rsqrt table | history |
| [experiments/rms_norm/capture_scales.py](experiments/rms_norm/capture_scales.py) | Capture CPU RMS scales and constrain them using saved GPU normalized outputs | history |
| [experiments/rms_norm/capture_width768_intervention.py](experiments/rms_norm/capture_width768_intervention.py) | Build/run one general width-specific RMS diagnostic in an isolated source copy | history |
| [experiments/rms_norm/export_rsqrt_table.py](experiments/rms_norm/export_rsqrt_table.py) | Observed CUDA rsqrt table and exhaustive normal-domain rescaling diagnostic | named by the frozen driver; moved, path documented |
| [experiments/rms_norm/review_saved_scales.py](experiments/rms_norm/review_saved_scales.py) | Read-only independent audit of the two preserved v1 RMS scale captures | history |
| [experiments/rms_norm/rsqrt_lookup.rs](experiments/rms_norm/rsqrt_lookup.rs) | Diagnostic only: a hash-verified observed CUDA rsqrt table, never a backend | history |
| [experiments/rms_norm/rsqrt_lookup_probe.rs](experiments/rms_norm/rsqrt_lookup_probe.rs) | Standalone observed-boundary probe: rustc -O rsqrt_lookup_probe.rs -o probe | history |
| [experiments/rms_norm/rstd_probe.rs](experiments/rms_norm/rstd_probe.rs) | Standalone CPU scale capture. No model, GPU call, or inference backend | history |
| [experiments/rms_norm/test_capture_scales.py](experiments/rms_norm/test_capture_scales.py) | Independent, CPU-only checks of the bounded RMS output-consistency search | history |
| [experiments/rms_norm/test_export_rsqrt_table.py](experiments/rms_norm/test_export_rsqrt_table.py) | Host-only mapping and exhaustive-range construction tests; no Torch execution | history |
| [scripts/compare_traces.py](scripts/compare_traces.py) | Calibrate before Rust comparison, then enforce a frozen numerical policy | history |
| [scripts/finalize_linear_logger.py](scripts/finalize_linear_logger.py) | Bind completed BLAS logs to the exact-output, unchanged-kernel replay | history |
| [scripts/inspect_divergence.py](scripts/inspect_divergence.py) | Locate peak errors and distinguish image/register tokens from output positions | history |
| [scripts/prepare_long_reference.py](scripts/prepare_long_reference.py) | Wrap the original numerical stress page in the shared corpus runner contract | history |
| [scripts/run_linear_logging_reference.sh](scripts/run_linear_logging_reference.sh) | Driver for the cuBLAS-logged linear replay reference | history |
| [scripts/summarize_corpus_reference.py](scripts/summarize_corpus_reference.py) | Freeze a compact, auditable summary from completed full-page GPU records | history |
| [scripts/summarize_exact_context_reference.py](scripts/summarize_exact_context_reference.py) | Freeze actual output count and cache cursor at the requested context boundary | history |
| [scripts/summarize_long_reference.py](scripts/summarize_long_reference.py) | Freeze observed output/context boundaries of the long stress reference | history |
| [scripts/test_compare_traces.py](scripts/test_compare_traces.py) | Regression checks for numerical comparison decisions, without a GPU | history |
| [scripts/test_reference_corpus_contract.py](scripts/test_reference_corpus_contract.py) | Unit tests for reference_corpus_contract.py | history |
| [scripts/test_reference_run_identity.py](scripts/test_reference_run_identity.py) | Exercise source-resume rejection without any GPU work or historical edits | history |
| [scripts/test_validate_gpu_reference_record.py](scripts/test_validate_gpu_reference_record.py) | Unit tests for validate_gpu_reference_record.py | history |
| [scripts/trace_error_rows.py](scripts/trace_error_rows.py) | Track failed tensor locations back through identical token positions | history |
| [scripts/validate_completed_gpu_run.py](scripts/validate_completed_gpu_run.py) | Validate and freeze a receipt for a completed new-style GPU corpus run | history |
