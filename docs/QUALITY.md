# Quality regression accounting

The [completed v3 report](../reference/quality-regression-v3-fp32-complete-v1.json)
passes the nonquantized differential gate: all 200 literal predictions match,
with zero difference overall and in each of the seven frozen categories. All
official component outputs also match, and fixed-corpus quality reporting is
complete. The report binds the corrected saved-token CPU text replay to its
original inference records and the exact predictions used by both evaluators.
This result does not close the ten FP32 intermediate-tensor failures.

The plan permits at most 0.25 percentage points of nonquantized quality regression
overall and 1 point per category. It did not name a primary overall metric or
aggregation. The existing assembled-text comparator reports diagnostic CER;
the audited official workflow reports separate text/formula/table/reading-order
components and table TEDS. Neither defines an official Overall score.

`scripts/report_quality_regression.py` adds a separate fail-closed accounting
report. It accepts only the pinned 200-page v3 manifest and its seven frozen
category counts. It consumes a preserved schema2 corpus comparison, then checks
the current run/page/source/truth hashes, model pins, inference records, and any
text-replay ancestry using shared validators. Script/helper and input hashes
are checked again at the end. It leaves historical startup-attestation gaps
explicit and stores no source/prediction text or token sequences in the report.

Complete literal prediction identity proves zero differential for every
deterministic fixed quality metric applied to identical ground truth and
evaluation settings. This satisfies both differential budgets without choosing
a primary metric after seeing results. It does not establish high absolute
accuracy, numerical tensor parity, complete domain coverage, or performance.
Different valid token sequences can still yield identical text: token parity
and this text-quality proof remain separate.

Partial runs cannot pass the full-corpus gate. A complete run with nonidentical
literal text yields an explicit unresolved metric decision, even if normalized
CER/WER match. A prior comparison's `status=failed` due only to token/text parity
does not make otherwise valid quality evidence malformed. Invalid hashes,
records, replay ancestry, missing-category accounting, or nonfinite metrics
fail validation. Failed inference pages cannot become zero-error samples.

## Descriptive metrics and remaining requirements

CER preserves the existing NFC plus Unicode-whitespace collapse normalization.
Case, punctuation and emitted markup remain intact. The report shows character
edit totals, micro CER, and unweighted page-mean CER overall and per category;
neither aggregate is silently selected for acceptance.

Descriptive WER was introduced for this report **after the 82-page snapshot**,
with `normalized(text).split(' ')` for nonempty normalized text and `[]` for
empty text. It is not a preregistered primary gate. Unicode whitespace tokens
are not linguistically segmented CJK words. The report shows word-edit totals,
micro WER and page-mean WER using the same untouched content and normalization.

For either metric, positive-denominator rate numerators, all-page edit totals,
and zero-truth edit totals/counts remain separate. Empty-truth rates are absent;
nonempty hallucinated output is not assigned a zero error or arbitrary rate.

The report names the remaining conditions separately:

- All 200 successful paired predictions and every frozen category must be present.
- Absolute text diagnostics and audited structure components must be reported.
  Structure metrics retain component-specific coverage and annotation adaptations.
- A primary metric/aggregation remains unfixed. It is needed to decide a
  nonidentical-output regression gate; complete literal identity needs no such
  choice to prove zero differential.
- Natural-corpus domain coverage remains bounded by the reviewed v3 selection.
  Original receipts, blank pages, rotations and additional script fixtures are
  separate supplemental evidence, not extra pages inserted into the frozen200.

There is no invented minimum absolute accuracy requirement. Rendered CDM and
official Overall remain unavailable. Quantized models require their separate
acceptance decisions and cannot use this nonquantized gate.

## Completed v3 diagnostics

Both runtimes have the same descriptive scores below. All 200 pages have
positive ground-truth denominators: 627,467 characters and 79,368 whitespace
tokens. Nineteen outputs stop at the explicit 4,096-token cap and remain included.

| Diagnostic | GPU and Rust |
|---|---:|
| Character edits | 215,422 |
| Micro CER | 34.3320% |
| Unweighted page-mean CER | 206.3649% |
| Word edits | 50,181 |
| Micro WER | 63.2257% |
| Unweighted page-mean WER | 435.9973% |

Insertion errors can exceed the reference length, so these rates are not capped
at 100%. Page means give short references the same weight as long ones. These
assembled-text diagnostics preserve markup and are distinct from the official
matched text/formula/table/reading-order metrics reported in
[EVALUATION.md](EVALUATION.md). The measured accuracy limitations belong to both
implementations; exact output parity is not a claim that the model reads every
page correctly. Category scores and counts remain in the machine-readable report.

## Execution

Use an environment with the existing comparison dependencies, including
RapidFuzz and Pillow. No inference or evaluator is run by this command:

```bash
python scripts/report_quality_regression.py \
  --comparison reference/windows-rust-corpus-v3-fp32-redecoded-200.json \
  --official-gpu artifacts/evaluation/v3-checked-fp32-v2/gpu \
  --official-cpu artifacts/evaluation/v3-checked-fp32-v2/cpu \
  --output reference/quality-regression-v3-fp32-recheck.json
```

The official directories above contain the completed v3 evaluations. Their prepared inputs,
execution audits and result aggregates are revalidated without rerunning the
evaluator. Metrics must belong to the exact same source runs and per-page
records, checked by path/hash and a one-to-one ID/category join, with identical
literal UTF-8 prediction hashes. Shared configurations alone do not establish
that identity. Audits, results, prepared ground truth/configuration, logs and
predictions enter the final source-file recheck alongside inference artifacts.
Missing components stay absent, and their coverage remains visible.

Outputs must use new paths. The command exits nonzero when the differential
gate is incomplete/unresolved; malformed evidence also raises an error.
The historical 140-page negative check is
[`quality-regression-v3-fp32-partial-140-v1.json`](../reference/quality-regression-v3-fp32-partial-140-v1.json):
140 compared pages have identical literal predictions and 60 pages remain
missing. Its differential and full-reporting gates are false; no official
components are attached. It exits with `incomplete`, so it cannot qualify the
full200. The [100-page v3 snapshot](../reference/quality-regression-v3-fp32-partial-100-v1.json),
[82-page v3 snapshot](../reference/quality-regression-v3-fp32-partial-082-v3.json)
and earlier v1/v2 development snapshots remain unchanged historical evidence.

Twenty bounded tests in `tests/test_quality_regression.py` cover complete200,
partial/missing categories, nonidentical output, token-versus-text distinctions,
normalization, empty truth, nonfinite metrics, malformed records, changed source
bytes, model pins, replay ancestry, stale summary labels, unrelated official
prediction sources, missing/ambiguous joins and late metric-file changes. They
require no model execution, external dataset, evaluator invocation or GPU.
