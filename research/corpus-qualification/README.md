# corpus-qualification

Building and qualifying the evaluation corpus, the 200-page quality gate and its Python tests.

| File | What it did or produced | Status |
|---|---|---|
| [docs/CORPUS_SUBSET_COMPARISON.md](docs/CORPUS_SUBSET_COMPARISON.md) | Explicit corpus subset comparisons | history |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Document quality evaluation | history |
| [docs/FUNCTIONAL_BATCH_REGRESSION.md](docs/FUNCTIONAL_BATCH_REGRESSION.md) | Full-page functional batch regression | history |
| [docs/QUALITY.md](docs/QUALITY.md) | Quality regression accounting | history |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Build and replay provenance | history |
| [docs/corpus-v3-review.md](docs/corpus-v3-review.md) | Corpus v3 visual review and selection | history |
| [docs/original-supplement.md](docs/original-supplement.md) | Original supplemental OCR diagnostics | history |
| [examples/corpus_eval.rs](examples/corpus_eval.rs) | archived source (Stage A) | history |
| [examples/redecode_corpus.rs](examples/redecode_corpus.rs) | archived source (Stage A) | history |
| [examples/support/corpus_record.rs](examples/support/corpus_record.rs) | archived source (Stage A) | history |
| [scripts/audit_evaluation_annotations.py](scripts/audit_evaluation_annotations.py) | Run the pinned evaluator's annotation merger before model quality evaluation | history |
| [scripts/audit_evaluator_line_endings.py](scripts/audit_evaluator_line_endings.py) | Prove pinned evaluator working bytes differ only by CRLF before Git interop | history |
| [scripts/capture_functional_cli.py](scripts/capture_functional_cli.py) | Capture the existing CLI for functional batch checks; never run inference | history |
| [scripts/check_corpus_resume.py](scripts/check_corpus_resume.py) | Exercise build-bound corpus resume semantics with one capped CPU inference | history |
| [scripts/compare_corpus.py](scripts/compare_corpus.py) | Compare independently greedy CPU/GPU corpus outputs after checking provenance | history |
| [scripts/compare_cpu_corpora.py](scripts/compare_cpu_corpora.py) | Compare selected CPU outputs across platforms; validate text replay lineage | history |
| [scripts/compare_official_evaluation.py](scripts/compare_official_evaluation.py) | Validate and compare pinned component outputs on an explicit frozen manifest | history |
| [scripts/compare_original_supplement.py](scripts/compare_original_supplement.py) | Compare original-fixture CPU/GPU outputs without hiding partial/failed runs | history |
| [scripts/compare_tokenizer_regression.py](scripts/compare_tokenizer_regression.py) | Compare fresh targeted inference with preserved GPU and text-replay records | history |
| [scripts/corpus_comparison.py](scripts/corpus_comparison.py) | Shared read-only corpus comparison checks; never repair source run metadata | history |
| [scripts/create_long_output_fixture.py](scripts/create_long_output_fixture.py) | Create an original deterministic numerical document for long-output testing | history |
| [scripts/export_tokenizer_behavior.py](scripts/export_tokenizer_behavior.py) | Capture actual pinned Transformers BPE decoding, cleanup and Python strip behavior | history |
| [scripts/functional_batch_regression.py](scripts/functional_batch_regression.py) | Freeze and run exact-output batch/layout checks; planning never runs inference | history |
| [scripts/generate_original_supplement.py](scripts/generate_original_supplement.py) | Typeset original OCR diagnostics, separate from the frozen natural corpus | history |
| [scripts/inspect_corpus_sources.py](scripts/inspect_corpus_sources.py) | Inspect primary dataset revision, license statement and small metadata only | history |
| [scripts/join_cpu_platform_outputs.py](scripts/join_cpu_platform_outputs.py) | Join completed CPU/GPU comparisons and directly compare common CPU outputs | history |
| [scripts/prepare_corpus.py](scripts/prepare_corpus.py) | Materialize a frozen corpus selection and audit candidate duplicates | history |
| [scripts/prepare_corpus_smoke.py](scripts/prepare_corpus_smoke.py) | Download only the 24 selected smoke images; freeze bytes | history |
| [scripts/prepare_corpus_v2.py](scripts/prepare_corpus_v2.py) | Materialize a frozen corpus selection and audit candidate duplicates | superseded by scripts/prepare_corpus.py |
| [scripts/prepare_official_evaluation.py](scripts/prepare_official_evaluation.py) | Prepare pinned OmniDocBench inputs without changing model output text | history |
| [scripts/report_quality_regression.py](scripts/report_quality_regression.py) | Bind a corpus comparison to frozen inputs and report quality evidence only | history |
| [scripts/review_completed_corpus_v1.py](scripts/review_completed_corpus_v1.py) | Read-only independent audit of the frozen 200-page and platform output evidence | history |
| [scripts/run_evaluation_checked.py](scripts/run_evaluation_checked.py) | Run the pinned evaluator with per-page execution and output completeness audits | history |
| [scripts/run_official_evaluation.py](scripts/run_official_evaluation.py) | Preflight both complete runs, prepare fresh inputs, audit evaluation, compare | history |
| [scripts/select_corpus.py](scripts/select_corpus.py) | Build a frozen research evaluation split from pinned OmniDocBench annotations | superseded by scripts/select_corpus_v3.py |
| [scripts/select_corpus_v3.py](scripts/select_corpus_v3.py) | Freeze a visually reviewed v3 while preserving every v1/v2 artifact | history |
| [scripts/setup_evaluation.sh](scripts/setup_evaluation.sh) | Keep evaluator Python and packages isolated from the GPU reference runtime | history |
| [scripts/snapshot_cpu_corpus.py](scripts/snapshot_cpu_corpus.py) | Copy an exact completed manifest prefix without changing saved inference bytes | history |
| [scripts/test_corpus_comparison.py](scripts/test_corpus_comparison.py) | Mutation tests for explicit subset provenance and cross-platform outputs | history |
| [scripts/test_validate_text_replay.py](scripts/test_validate_text_replay.py) | Meaningful rejection checks and optional actual native replay integration | history |
| [scripts/validate_text_replay.py](scripts/validate_text_replay.py) | Shared, fail-closed provenance checks for text-only replay of saved Rust IDs | history |
| [tests/test_functional_batch_regression.py](tests/test_functional_batch_regression.py) | Functional harness tests use synthetic JSON only; no inference | history |
| [tests/test_official_evaluation_workflow.py](tests/test_official_evaluation_workflow.py) | Small synthetic reports exercise completeness and fail-closed evaluation | history |
| [tests/test_original_supplement_comparison.py](tests/test_original_supplement_comparison.py) | Regression checks for diagnostic scoring and incomplete-run gates | history |
| [tests/test_quality_regression.py](tests/test_quality_regression.py) | Bounded saved-record tests: no model execution or evaluator invocation | history |
| [tests/test_tokenizer_regression_comparison.py](tests/test_tokenizer_regression_comparison.py) | Targeted subset identity and comparison tests; no model inference | history |
