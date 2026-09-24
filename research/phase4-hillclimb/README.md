# phase4-hillclimb

The phase-4/5 hill climb of 2026-09-23: GPTQ overlay, near-exact and fast modes, speculation, packed files; every accepted and rejected attempt with its receipt.

| File | What it did or produced | Status |
|---|---|---|
| [attempt3/BUILD_STATUS.json](attempt3/BUILD_STATUS.json) | Validation receipt: attempt3 source uncompiled on 2026-09-21 | history (2026-09-21 uncompiled-source validation receipt) |
| [attempt3/HILLCLIMB.md](attempt3/HILLCLIMB.md) | Overnight hill-climb log of every accepted and rejected attempt | history |
| [attempt3/README.md](attempt3/README.md) | Attempt 3 package overview: integrated Falcon-OCR v1.5 CPU experiment | history |
| [attempt3/RESULTS-V1.md](attempt3/RESULTS-V1.md) | Attempt 3: first measured screen (Stage 1) | superseded by attempt3/RESULTS-V3.md |
| [attempt3/RESULTS-V2.md](attempt3/RESULTS-V2.md) | Phase 4: calibration verdict, generated-tail compression, loop stop (Stage 5) | superseded by attempt3/RESULTS-V3.md |
| [attempt3/RESULTS-V3.md](attempt3/RESULTS-V3.md) | A better quantized Falcon-OCR v1.5, and faster exact and fast modes (2026-09-23) | history |
| [attempt3/ab.py](attempt3/ab.py) | Interleaved A/B/... runs of attempt binaries on pages, with per-arm env | history |
| [attempt3/bench.py](attempt3/bench.py) | Bracketed full-page experiment runner. Each arm is a fresh native process | history |
| [attempt3/check_heldout.py](attempt3/check_heldout.py) | Check a held-out run against the pre-registered quality budget | history |
| [attempt3/compare_profiles.py](attempt3/compare_profiles.py) | Token agreement and speed of candidate profiles against an FP32 run | history |
| [attempt3/compare_runs.py](attempt3/compare_runs.py) | Compare several full-page runs of the same pages against one baseline run | history |
| [attempt3/draft_sim.py](attempt3/draft_sim.py) | Offline simulation of speculative drafting on reference token sequences | history |
| [attempt3/dtype_proxy.py](attempt3/dtype_proxy.py) | Activation-weighted output error of candidate storage types for every body matrix | history |
| [attempt3/gpu_harness.py](attempt3/gpu_harness.py) | GPU harness for quantization fidelity: reference generation and fake-quant scoring | history |
| [attempt3/judge_pages.py](attempt3/judge_pages.py) | Blinded LLM-as-judge packets for OCR outputs, and the unblinded report | history |
| [attempt3/kl_sweep.py](attempt3/kl_sweep.py) | Run teacher-forced KL arms against a stored FP32 top-K reference | history |
| [attempt3/loop_stop_sim.py](attempt3/loop_stop_sim.py) | Simulate a repetition stop on recorded outputs | history |
| [attempt3/make_manifest.py](attempt3/make_manifest.py) | Create explicit single-page cases and an optional joint batch from local images | history |
| [attempt3/requirements.txt](attempt3/requirements.txt) | Pinned numpy/safetensors versions for attempt3 offline tests | history |
| [attempt3/run.ps1](attempt3/run.ps1) | Run an attempt3 manifest of single-page cases from PowerShell | history |
| [attempt3/subset_overlay.py](attempt3/subset_overlay.py) | Drop matrices from a W8 overlay so they stay FP32 (mixed precision) | history |
| [attempt3/test_attempt3.py](attempt3/test_attempt3.py) | Offline checks for artifact math, comparison policy and source wiring | history |
| [attempt3/validation/existing-scripts.json](attempt3/validation/existing-scripts.json) | Receipt of the scripts/ unittest run on 2026-09-21 | history |
| [attempt3/validation/existing-scripts.log](attempt3/validation/existing-scripts.log) | Log of the scripts/ unittest run on 2026-09-21 | history |
| [attempt3/validation/existing-tests.json](attempt3/validation/existing-tests.json) | Receipt of the tests/ python run on 2026-09-21 | history |
| [attempt3/validation/existing-tests.log](attempt3/validation/existing-tests.log) | Log of the tests/ python run on 2026-09-21 | history |
| [attempt3/validation/new-python.json](attempt3/validation/new-python.json) | Receipt of the attempt3/test_attempt3.py run on 2026-09-21 | history |
| [attempt3/validation/new-python.log](attempt3/validation/new-python.log) | Log of the attempt3/test_attempt3.py run on 2026-09-21 | history |
| [attempt3/w8_output_error.py](attempt3/w8_output_error.py) | Activation-weighted output error of W8 overlays (offline, uses captured Grams) | history |
