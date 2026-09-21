# Isolated temporal-prefix model qualification

This harness prepares two new projects: an unmodified production control verified against the original-source inventory from the frozen native candidate capture, and the exact candidate source ZIP that passed the native and Linux focused tests. Both receive the same new integration-test source. Live source, Cargo files, defaults, prior copies and reports remain unchanged.

The Windows-first invocation order is expanded-before, compact-before, candidate, compact-after, expanded-after. Controls use the original crate and its original cache variants; the original crate cannot deserialize the candidate layout. Every process loads the pinned model and corrected tokenizer, uses explicit FP32/AVX2 with four runtime threads and unpacked weights, and runs the same ignored integration test. The capture driver selects the actual Cargo-emitted executable, archives both source inventories and the harness, and checks source, model, input, executable and result identities through completion. Failed or partial attempts are preserved; there is no automatic retry or resume.

Each process performs these bounded checks:

- Canonical smoke tracing uses the existing teacher-forced `trace_reference` path for seventeen decisions. The report hashes all 1,904 tensors with names, F32 little-endian dtype, shapes and element counts. It rejects duplicate names and nonfinite data. Seventeen argmaxes are calculated from the actual captured logits and checked independently of the returned teacher-token sequence.
- Free inference runs separately for the smoke image, the existing 128x64 white blank, and the existing 256x48 first-line crop. A batch-capacity-four runner then executes those three inputs in the same order. All literal text, IDs, stops, dimensions, counts and free-running flags must agree. The fixed outputs have 17/2/6 tokens and the joint decode reaches active row counts three, two and one. Captured logits plus active request indices must reconstruct the emitted sequences. Cross-invocation mixed trace equality is assessed only after these independently emitted IDs establish common prefixes.
- Every discarded K/V duplicate is checked by bits during tracing: prefix temporal halves, generated full key heads and full value heads. All tensor inventories/hashes and the derived decisions must match the genuine original controls.
- The same single and mixed Runner instances are warmed before a disabled-trace allocation probe. Each measured call must produce the expected outputs and exactly one decode-start/decode-end interval. A process-global System-delegating allocator counts allocations from all worker threads; counters accumulate across callbacks and require zero calls and requested bytes. The interval excludes model load, preprocessing, prefill, final detokenization and result assembly. It is not a capacity, RSS, timing or throughput measurement.

Preparation and build do not execute the model:

```powershell
python -m unittest discover -s experiments/cache_layout -p test_temporal_model_capture.py -v
python experiments/cache_layout/capture_temporal_model.py prepare --output artifacts/diagnostics/prefix-temporal-model-review-v1
python experiments/cache_layout/capture_temporal_model.py build --plan artifacts/diagnostics/prefix-temporal-model-review-v1/plan.json
python experiments/cache_layout/capture_temporal_model.py validate --plan artifacts/diagnostics/prefix-temporal-model-review-v1/plan.json
```

Use a fresh preparation directory when those artifacts already exist. The existing Linux eight-test success is a mandatory pinned prerequisite. Before any full-model execution, the parent and an independent reviewer must review the frozen harness/plan/build. The separate explicit command after that review is:

```powershell
python experiments/cache_layout/capture_temporal_model.py run --plan artifacts/diagnostics/prefix-temporal-model-review-v1/plan.json
```

No full-model run has been performed merely by preparing this harness. A passing future result would qualify CPU layout equivalence only for the fixed smoke and three-input mixed fixtures. It would not close outstanding CPU/GPU hidden-stage numerical gates, establish natural-corpus OCR quality, measure performance, qualify arbitrary multiquery continuation, or promote a production default. Per-tensor SHA256 equality is the preserved trace comparison; full duplicate tensor payloads are not saved.
