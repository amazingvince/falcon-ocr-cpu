# Build and replay provenance

Inference records and text-decoding corrections have different identities. The
active v3 corpus run continues with its preserved executable. Its historical
records are not overwritten when source changes. The audit and explicitly
after-launch source capture are recorded in
[`reference/provenance-audit.md`](../../../reference/provenance-audit.md).

## Preserve a build before launching a new run

From the repository root, with Rust available on PATH:

```sh
python research/benchmarks/scripts/capture_rust_build.py --example corpus_eval \
  --output artifacts/builds/corpus-eval-new --jobs 4
```

The script invokes the platform build wrapper with `--locked --release`, archives
the source, hashes it before and after compilation, and records Cargo/rustc
versions and selected build flags. It uses Cargo's emitted executable path,
including any configured target triple, and copies that executable into the new
directory. Existing output directories are rejected. Cached native dependencies
and system compiler/linker installation mean this is not a hermetic build image.
Use a separate target directory when building Windows and Linux from one checkout.

Run the preserved executable, supplying `--build-manifest .../build.json` to the
corpus example. Its contract binds the executable hash, compiled source hashes,
archive/build-manifest identity, model/config, prompt, precision and generation
settings. Without a build manifest, executable/source binding still applies, but
the record explicitly lacks compiler/archive provenance. A changed binary or
contract requires a new output directory. Resume preserves the original run
record, adds invocation records, and accepts only complete valid page results.

[`research/corpus-qualification/scripts/check_corpus_resume.py`](../scripts/check_corpus_resume.py) checks a
fresh capped inference, matching resume, changed-contract rejection, wrong-build
rejection and preservation of the original run record. The first Windows check
is in `reference/corpus-resume-windows-validation.json`; later source fixes need
their own versioned build/check rather than replacing that evidence.

## Corrected text from preserved token IDs

The corpus uncovered a decoder compatibility issue: Transformers 5.14.1 ignores
`clean_up_tokenization_spaces=true` for the pinned BPE tokenizer. Rust's earlier
WordPiece-style cleanup removed meaningful spaces before punctuation. The fixed
decoder preserves raw BPE spacing, removes only the two EOS spellings, and uses
Python 3.12's outer-strip character set, including ASCII information separators.
The 14-case pinned-runtime fixture covers both actual failing pages, punctuation,
contractions, markup, stop tokens and Unicode whitespace; it passes on both OSes.

`examples/redecode_corpus.rs` reads saved CPU IDs and writes a separate derived
directory. It performs no model forward. Its contract retains the original
inference contract and adds hashes for the original run, frozen decoder binary,
preserved decoder source, tokenizer assets and replay code. Each derived page
binds the original record and text. Every result field other than text—including
IDs, stopping, counts and timings—must remain identical. Timings describe the
original inference and are not measurements of the corrected decoder.

Use `capture_rust_build.py --example redecode_corpus` and invoke that preserved
binary with `--source`, `--output` and `--manifest`. `--resume` adds newly available
pages only under the same replay contract. Missing/failed inference pages remain
explicit; a partial replay cannot qualify a completed corpus.

The first preliminary replay exposed a JSON issue: default `serde_json` parsing
changed some original f64 timings by one ULP. The shared validator correctly
rejected those records. The project now enables `serde_json/float_roundtrip`;
exact inherited-field validation is retained. That failed preliminary directory
is preserved separately, never silently repaired or presented as fresh inference.

GPU comparisons and official evaluation use the shared replay validator before
accepting derived text. Native Windows paths are explicitly resolved when those
checks run in WSL. Numerical tensor parity, original free-running token parity,
corrected text parity and OCR quality remain separate reported claims.
