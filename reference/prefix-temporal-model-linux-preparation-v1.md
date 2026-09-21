# Linux temporal-cache model qualification: prepared, not executed

The Linux plan is prepared at `artifacts/diagnostics/prefix-temporal-model-linux-plan-v1/plan.json`, SHA256 `87410dac06e056a58373ea37d5982355d7b09f2916ec0307389ebdc698cfa8ed`. Its source archive is `357905d7f412b0cdd48eeadd3f64704c4769111b7af3f0ebeba482367549dc76`. No Linux model-qualification build or inference has run.

The plan copies the exact sixty control/candidate project files from the reviewed Windows v2 archive. Both projects retain the identical Rust integration test. The accepted Windows evidence, failed historical execution and reporting-only comparator correction are preserved and hash-bound. Four new host tests passed for archive contents, changed/missing/added source rejection, Windows evidence-path mapping and explicit cross-platform difference reporting.

Build targets are separately reserved under the existing D-backed WSL filesystem:

- `/home/amazi/falcon-ocr-rust-reference/temporal-model-linux-v1-targets/control`
- `/home/amazi/falcon-ocr-rust-reference/temporal-model-linux-v1-targets/candidate`

The driver requires both target directories to be absent before compilation. It builds offline/locked with two build jobs and Rust 1.92.0, checks exact emitted manifests/source paths and fresh project artifacts, and preserves distinct executable hashes. Model execution uses the unchanged five-process control/candidate/control order, four runtime threads and `CUDA_VISIBLE_DEVICES=-1`. It first applies the reviewed within-Linux numerical/output/allocation checks; Windows/Linux equality is an additional explicit observation and is not assumed or used as the within-platform gate.

After independent review, run the stages separately from the shared repository in `Ubuntu-24.04-CUDA`:

```bash
/home/amazi/falcon-ocr-rust-reference/.venv/bin/python experiments/cache_layout/capture_temporal_model_linux_v1.py validate --plan artifacts/diagnostics/prefix-temporal-model-linux-plan-v1/plan.json
/home/amazi/falcon-ocr-rust-reference/.venv/bin/python experiments/cache_layout/capture_temporal_model_linux_v1.py build --plan artifacts/diagnostics/prefix-temporal-model-linux-plan-v1/plan.json
```

Only after build identity review and separate authorization:

```bash
/home/amazi/falcon-ocr-rust-reference/.venv/bin/python experiments/cache_layout/capture_temporal_model_linux_v1.py run --plan artifacts/diagnostics/prefix-temporal-model-linux-plan-v1/plan.json
```

There is no automatic retry/resume, performance measurement, GPU execution, production edit or default change. A future pass would concern only the bounded smoke and three-input mixed cases.
