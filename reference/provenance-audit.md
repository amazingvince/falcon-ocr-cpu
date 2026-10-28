# Corpus and serving provenance audit

Audit date: 2026-09-20. Scope: the active 200-page HF GPU corpus driver, Rust
`corpus_eval`, saved direct-vLLM smoke evidence, and the prepared full-page serving
launch. No inference, performance measurement, or modification of historical run
records was performed. Audit-time hashes and inspected source versions are in
[`provenance-audit-evidence.json`](provenance-audit-evidence.json).

The model identifiers are not merely labels: the HF preflight checks every file
in the model artifact manifest and separately requires the pinned weight digest.
Rust verifies the actual checkpoint, config, tokenizer, and tokenizer config
against compiled digests before inference. All 30 saved artifact hash checks in
this audit passed, including the 1,079,789,464-byte checkpoint, all 11 other model
assets, corpus/dependency manifests, and available serving request, response,
source-copy, config/tokenizer, precision-record and log files.

## Prioritized remaining work

1. **Bind GPU corpus source and verified assets to the next run/resume contract.**
   `scripts/run_corpus_reference.py:44` records model revision, weight digest,
   corpus manifest, dimensions, budget and precision. It omits the driver,
   `export_reference.import_model`, preflight, fetch helper, artifact-manifest
   digest and verified model/config/tokenizer source digest set. The dependency
   lock is recorded in `environment`, but is outside the configuration checked on
   resume. Consequently, a later driver, cast, tie-breaking, tokenizer, or
   dependency change can resume an existing output directory with the same
   configuration. The current tie selection is `torch.argmax` for both precisions
   (`:114`), but its explicit policy string is only recorded for BF16 (`:49`).
   Hash and archive the actual loaded sources/assets at the next launch; include
   their identity, cast/tie policy, prompt and preprocessing contract in resume
   equality. Keep a separate environment record for each invocation. Do not
   retrofit missing startup identity into the current run.

2. **Preserve complete CPU build identity alongside embedded source identity.**
   `examples/corpus_eval.rs:75` embeds hashes for 12 Rust source files plus
   `Cargo.lock` and correctly rejects a changed resume contract. It does not
   record the executing binary digest, compiler/version, target flags/profile,
   `Cargo.toml`, or the complete compiled source closure. The saved
   `artifacts/cpu/corpus-eval-v3.exe` hashes to
   `befd29f70f27ced1ff5289b34b28112c36f712e213ac774f3587c690fe0c08d5`
   **at audit time**; that digest is absent from its run contract. The current
   workspace differs from its embedded hashes for `kernels`, `bf16_attention`
   and `bf16_runner`, as expected during continued development. This does not
   invalidate the old binary; it means rebuilding the current tree is not a
   reproduction of that binary. For new runs, retain the executable, complete
   source archive, lockfile and build invocation, and bind them before launch.
   The existing original-supplement binary-snapshot workflow is a useful local
   precedent, not evidence that the v3 run already has that binding.

3. **Make reference preflight check the selected model path.**
   `scripts/export_reference.py:54` accepts `--model`, but calls
   `preflight(root)` at `:64`, which always verifies `artifacts/model`; it later
   imports and loads `args.model`. A nondefault path is therefore not covered by
   that launch gate. The corpus driver uses the same fixed directory that its
   preflight checks, so this omission does not apply to the active corpus run.
   Pass the resolved selected model directory into preflight and record it for
   future generic exports.

4. **Keep the nonweight artifact manifest independently pinned.**
   `scripts/fetch_reference.py:25` retains any existing file and writes its
   current digest into a new manifest; only the weight file is compared with an
   independently compiled digest. The immutable download URL is sound, and the
   present files pass their manifest, but rerunning fetch can bless changed
   nonweight files under the same revision label. Retain the original manifest
   digest or immutable expected digest table, and require an explicit new
   artifact identity when a model/tokenizer source file changes.

5. **Tighten run bookkeeping without changing old records.**
   The GPU driver permits replacement of same-configuration outputs without
   `--resume` and rewrites the run environment on resume (`:52–55`). Its page and
   summary qualification strings still describe a 24-page smoke (`:141,:150`).
   Rust records no planned/selected page count or final run status. The comparator
   correctly rechecks canonical PNG/RGB and ground-truth bytes, page identities,
   contracts and prefix counts, and keeps missing pages explicit; its
   `provenance_passed` means these available fields passed, not that missing
   source/build attestations were verified. New schemas should state that scope,
   preserve invocation history, require resume for an existing output directory,
   and derive descriptions/counts from the actual manifest and budget.

## Serving findings and fixes prepared during the audit

The saved direct-vLLM smoke is a real 144-prompt-token, 17-output-token FP32
request with exact IDs/text, an immutable container image digest, actual weight
verification, recorded entrypoint/config/tokenizer/template hashes, copied
installed model and attention source, and a successful compiled-PTX precision
inspection. Its preserved artifact hashes passed. It remains one small smoke,
not a full-page preprocessing or corpus qualification.

The image defaults are **1024 pixels / 1,003,520 pixels total**, whereas the HF
and Rust corpus target is **1536 / 10,035,200**. The GPU agent added a separate
`--full-pages` output directory and explicit `--hf-overrides` image configuration;
`request_vllm_fullpages.py` checks that exact override and compares prompt counts,
IDs, normalized outer whitespace and stop reasons. The queued full-page run must
still establish observed results. The previous smoke report is unchanged.

Three bounded fixes were implemented by the GPU agent before the next serving
launch and verified by source inspection during this audit:

- The entrypoint now checks every artifact-manifest file hash and size, including
  config, tokenizer and tokenizer config. The old entrypoint independently checked
  only the weights and merely recorded config/tokenizer digests.
- The precision audit now copies each inspected PTX file to `/out/ptx`. The old
  smoke preserves four PTX digests/instruction summaries but no PTX files in its
  output bundle, so those original bytes cannot be rechecked from that bundle.
- The smoke finalizer now derives attempt status and token/prompt counts from the
  actual report. Previously it unconditionally wrote successful 17/144 values.

The strengthened entrypoint has a different hash from the historical smoke's
recorded entrypoint; this is an expected future-run change, not a historical hash
failure. Future serving bundles should also preserve the launch/request/audit/
finalizer sources and chat-template bytes, bind a unique run identifier to the
environment/request records, and avoid reusing an existing output folder. The
entrypoint's hardcoded `precision="fp32"` currently matches the launch's explicit
`--dtype float32`; it must not be reused as a BF16 provenance label without
deriving and checking the selected dtype.

## Preservation of the active run

At 07:29:36 UTC, the current GPU driver, import helper, preflight, fetch helper,
dependency lock, model artifact manifest and all nonweight model assets were
captured in the ignored archive
`artifacts/provenance/gpu-corpus-source-after-launch-20260920.zip`.
Its SHA-256 is
`c8bc4465f2d5a9b219ef597de649c045976983779db93603ea1fd7ae4436f89a`.
The durable [capture sidecar](gpu-corpus-source-capture-after-launch.json) gives
every file hash and explicitly labels the snapshot **after launch, not
startup-attested**. The GPU agent stated that the four driver/helper files were
unchanged during this run; that statement is retained separately from measured
hash evidence. The active run/page records were not edited. Future source-binding
changes were coordinated with that agent for a new run, after the active corpus
work completes.
