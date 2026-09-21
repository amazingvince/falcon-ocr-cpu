# Independent replay/build-provenance review, 2026-09-20

Scope: `examples/redecode_corpus.rs`, `examples/corpus_eval.rs`,
`scripts/capture_rust_build.py`, and the shared replay/record validators while
their fixes were being integrated. This was a bounded correctness review, with
no model inference, production edits, or changes to old run records.

Four actionable issues were reported to the parent task. Fixes are now visible
in source; final rebuilt-binary and Windows-to-WSL integration checks remain
owned by the parent/GPU tasks.

| Finding | Concrete consequence | Resolution observed |
|---|---|---|
| Build capture guessed `target/release/examples/<name>` despite `CARGO_BUILD_TARGET` | A successful targeted build could archive an older host executable and mark the build complete | Capture now selects the executable reported by Cargo's JSON compiler-artifact event |
| Resume accepted any result object without an error | `result: {}` was skipped and counted as completed without IDs, stops or other required inference output | Shared Rust `corpus_record::validate_result` is called by resume and replay |
| Windows paths were interpreted directly by WSL `pathlib.Path` | The preserved native Windows decoder binary and joined source paths could not be found during WSL validation | Shared `resolve_recorded_path` explicitly maps Windows and WSL drive paths |
| Default `serde_json::Value` parsing changed saved floating numbers | A valid text-only replay could change original timing values by one ULP, violating exact inherited-result/run equality | `serde_json` now enables `float_roundtrip`; a new preserved replay binary is required |

The new record validator initially also accepted an interior stop token, for
example IDs `[263,561]` with a two-token `length` cap. This was reported as a
follow-up to the incomplete-result finding; it now rejects 11/263 before the
final position. Stop handling must remain identical to free inference.

## Reproduction and checks

- A mocked Cargo artifact-layout test executed the original capture helper with
  `CARGO_BUILD_TARGET=x86_64-pc-windows-msvc`, an older host executable, and a
  newly emitted target-directory executable. It reported `status=complete` and
  copied `OLD_HOST_BINARY`, proving the path-selection error without compiling
  or running a model. The replacement selector passed checks for the explicit
  target path and refusal of missing/wrong-named artifacts.
- A small Rust probe linked against the existing release `serde_json` dependency
  parsed and serialized 24 actual v3 CPU page records. **17 numbers changed**,
  including `13894.139000000001` to `13894.139` and
  `21.121299999999998` to `21.1213`. The ignored evidence is
  `artifacts/provenance/serde-roundtrip-review.json`. These differences are JSON
  round-trip behavior, not new model arithmetic. The shared validator correctly
  detects them; its exact inheritance gate must not be relaxed.
- Independently verified the preserved `corpus-eval-v4-windows` and initial
  `redecode-corpus-v1-windows` bundles: every one of their 25 archived source
  entries matched the build manifest, as did each archive and executable digest.
  Neither build used `CARGO_BUILD_TARGET`. This verifies the existing bundles,
  not the later corrected replay build; the initial replay binary is superseded.
- Checked Windows-side WSL-path conversion. The GPU task owns validation of an
  actual native-Windows-produced replay from WSL, rather than relying solely on
  synthetic path cases.

The corrected design clearly separates original free-running inference from
text decoding: it preserves the original inference contract, source/page hashes,
IDs, counters, stop reasons, precision/backend and timings; it adds an explicit
replay contract and immutable decoder source with its own binary/tokenizer
identity. The shared Python helper compares all original result fields except
text, not merely the token sequence. Missing or failed originals stay explicit
and official preparation refuses incomplete corpora. A partial replay progress
snapshot is not a new inference result or a completed qualification.

No further material issues were found in this bounded pass. Final qualification
still depends on the corrected binary's actual replay, strict inherited-field
checks and downstream parity/evaluator validation.
