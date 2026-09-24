# Explicit corpus subset comparisons

The completed v3 comparison matches all 24 shared pages across Windows, Linux
under WSL and GPU: 29,205 token IDs, literal texts and stopping reasons. The
[direct CPU-platform join](../../../reference/windows-linux-corpus-v3-smoke-output-agreement-v1.json)
also checks actual CPU dimensions and prefix/output counts. Linux records are
fresh inference; Windows texts come from the validated saved-token replay of
the completed 200-page run. Historical startup-evidence gaps remain explicit.

`--manifest` selects the pages and comparison order. Each original run still
belongs to its own unchanged source manifest. Without an explicit source
manifest argument, the comparator requires the original run's manifest hash to
equal the selected manifest hash, preserving the previous strict behavior.

When a source manifest is supplied, its actual SHA-256 must equal the run's
recorded hash. Selected pages must be **exact full-object members** of that
manifest, including image keys, original/canonical/RGB/ground-truth hashes,
annotation identity and selection metadata. Duplicate IDs or image keys in
either manifest fail. Dataset, revision and common preparation-policy identities
must agree. No run configuration or page record is rewritten.

For the v3 Linux smoke versus the GPU 200-page source:

```sh
python research/corpus-qualification/scripts/compare_corpus.py \
  --manifest reference/corpus-v3-smoke-lock.json \
  --gpu artifacts/reference/corpus-v3-fp32-4096 \
  --gpu-manifest reference/corpus-v3-evaluation-lock.json \
  --cpu artifacts/cpu/linux-corpus-v3-smoke-fp32-4096 \
  --cpu-build artifacts/builds/corpus-eval-v7-linux/build.json \
  --output artifacts/cpu/linux-v3-gpu-comparison-NEW.json
```

Use the pinned WSL reference Python environment for this command; its diagnostic
CER calculation requires `rapidfuzz`. The CPU-to-CPU comparator needs only the
existing Pillow dependency:

```sh
python research/corpus-qualification/scripts/compare_cpu_corpora.py \
  --manifest reference/corpus-v3-smoke-lock.json \
  --left artifacts/cpu/linux-corpus-v3-smoke-fp32-4096 \
  --right artifacts/cpu/corpus-v3-fp32-4096-redecoded-v2-snapshot-200 \
  --right-manifest reference/corpus-v3-evaluation-lock.json \
  --left-build artifacts/builds/corpus-eval-v7-linux/build.json \
  --output artifacts/cpu/linux-v3-platform-comparison-NEW.json
```

Both commands refuse an existing output report. Always use a new output
path for snapshots, preserving earlier reports. No inference
or performance measurement is performed. A partial snapshot exits successfully
when its available records are valid; automation must inspect
`completed_output_parity_passed`, not just the exit code. Missing pages and failed
records are explicit, and neither can qualify a complete selected-page comparison.

Both tools compare actual generated IDs and literal text, EOS/length termination,
prefix/output counts and precision. CPU-to-CPU also compares recorded processed
width/height and requires matching options/backend. OS, threads, source/build
versions and measured timing values may differ. Additional optional timing fields
are allowed and validated if present. Old GPU page records contain prefix counts
but no processed width/height: the report states that dimension comparison is
unavailable rather than inventing GPU dimensions from CPU output.

Replay validation uses the shared strict `validate_text_replay` helper. It checks
the original run and each original record, source and tokenizer files, and exact
preservation of every inference-result field except text. Thus Windows replay
text is explicitly derived from saved original inference tokens; it is never
presented as another model execution. Original timing values cannot be changed
within replay lineage even though timings may differ between the two platforms.

Reports bind both source manifests, selected membership, run snapshots and each
compared record. Actual config/tokenizer files are rehashed against fixed model
pins. An optional CPU `build.json` independently binds the preserved executable,
archive and embedded source hashes to its recorded startup contract. Model
revision, recorded weight digest, precision, recorded prompt/argmax policy and
generation options are checked. The checkpoint is not unnecessarily rehashed by
this read-only output comparison; its recorded digest and loader checks remain
the inference evidence.

Historical Windows/GPU records omit some prompt, argmax, asset or build fields.
`contract_evidence` lists those omissions; `startup_semantic_fields_complete`
stays false when fields are absent. Missing fields are never synthesized or
silently treated as independently startup-attested. The historical audit and
after-launch source preservation remain described in
`reference/provenance-audit.md` and
`reference/gpu-corpus-source-capture-after-launch.json`. A successful output
comparison means exact saved outputs on the selected pages with all available
checks passing. It is not a new launch attestation, completion of a larger parent
corpus, hidden-tensor parity, OCR quality qualification or bare-metal performance.

Validation:

```sh
python research/corpus-qualification/scripts/test_corpus_comparison.py
python research/corpus-qualification/scripts/test_validate_text_replay.py --replay artifacts/cpu/corpus-v3-fp32-4096-redecoded-v2
python research/gpu-reference/scripts/test_validate_gpu_reference_record.py
```

The comparison tests cover explicit versus implicit parents, wrong hashes and
nonmembers, modified annotations/hashes/metadata, duplicate IDs/image keys, empty
selection, missing or malformed records, output-field mismatches and replay
misuse. Three GPU CLI tests require `rapidfuzz` and are skipped on an environment
without it; run them in the pinned reference environment before accepting changes.
