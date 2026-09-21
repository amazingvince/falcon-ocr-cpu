# Temporal-prefix model qualification: isolated build targets

The first model-qualification build is rejected before inference. Its two Cargo invocations shared a target directory: after compiling the original control, Cargo reported the candidate library and test as fresh, and both preserved executables had the same SHA256. The source copies were different, so those artifacts did not establish that the candidate had compiled. The original source, plan, logs and binaries remain unchanged; the rejection is recorded in `reference/prefix-temporal-model-build-rejection-v1.json`.

The v2 driver and its companion host tests are new files. The Rust integration harness, candidate source and genuine original controls are unchanged. A new plan binds distinct, initially absent native build targets at `D:/falcon-ocr-rust-builds/temporal-model-v2/control` and `D:/falcon-ocr-rust-builds/temporal-model-v2/candidate`. Every project artifact must be newly compiled, use the requested package manifest and exact source-copy paths, and reside inside its requested target. The preserved control and candidate executable hashes must differ before any model process can start. Build and execution records retain those target identities. Failed model processes retain their actual exit code.

The existing fixed five-process order and qualification scope remain unchanged: expanded and compact controls before the candidate, then compact and expanded controls after it; four runtime threads; all canonical and mixed trace identities; independently emitted single/mixed outputs; and warmed single/mixed decode allocation intervals. This provides bounded functional evidence under concurrent load, with no performance or production-promotion claim.

```powershell
python -m unittest discover -s experiments/cache_layout -p test_temporal_model_capture_v2.py -v
python experiments/cache_layout/capture_temporal_model_v2.py prepare --output artifacts/diagnostics/prefix-temporal-model-review-v2
python experiments/cache_layout/capture_temporal_model_v2.py build --plan artifacts/diagnostics/prefix-temporal-model-review-v2/plan.json
python experiments/cache_layout/capture_temporal_model_v2.py validate --plan artifacts/diagnostics/prefix-temporal-model-review-v2/plan.json
```

The build command requires write access to the two recorded D: build directories. Preparation and compilation do not execute model inference. Only after independent source/build review, the separately authorized execution command is:

```powershell
python experiments/cache_layout/capture_temporal_model_v2.py run --plan artifacts/diagnostics/prefix-temporal-model-review-v2/plan.json
```

Existing output directories and build targets must not be reused for retries. No historical source or result is overwritten.
