# Independent quality-report review

Reviewed `scripts/report_quality_regression.py` and its saved-record tests after
the author completed the bounded fixes. No remaining concrete correctness defect
was found in the reviewed decision and provenance paths. This was a read-only
implementation review plus synthetic tests, not inference, evaluator execution,
an additional full-corpus scan or a model-quality acceptance decision.

The review addressed these concrete cases:

- Missing inventories must be unique, contain only known page keys, match their
  recorded counts and explain exactly the omitted comparison rows. A complete
  row list with stale missing entries is rejected.
- A comparison's passed labels cannot substitute for valid inference records.
  CPU/GPU IDs, counts, stops, finite fields and context dimensions are checked
  again with the shared validators. CPU contract hashes and fixed model pins are
  checked independently; consistently relabelled runs fail.
- Text replay is validated against its original run and records. Original
  records, decoder bytes/source and tokenizer assets join the checked source
  set. Changed ancestry fails even if newer summary hashes are self-consistent.
- Prediction origin and startup-completeness labels come from freshly validated
  evidence, rather than possibly stale prior-report flags.

Complete literal prediction equality supports zero differential for a fixed
deterministic text-quality metric using identical truth and settings. It does
not establish high absolute accuracy. Different text remains unresolved without
a prospective primary metric; neither retrospective CER nor whitespace WER is
silently selected as that metric. Even normalized-equal but literal-different
text does not obtain the universal identity proof. Different valid token
sequences may still have identical literal text, so token parity and text-quality
identity remain separate decisions.

CER and WER report micro and page-mean summaries explicitly. Zero-denominator
pages retain absent rates and separate emitted-character/token totals. Whitespace
WER is labelled retrospective, with its lack of linguistic CJK segmentation
stated. Missing official components are absent, not zero; attaching official
results requires the existing audited preparation/execution validation.

An independent execution of `tests/test_quality_regression.py` in the pinned WSL
reference Python environment passed all **15 tests**. They cover complete and
partial/category-missing cases, different/normalized-only text, malformed actual
records, changed model/contract/source/replay evidence, nonfinite metrics, empty
truth and stale summary labels.

The inspected immutable report is
`reference/quality-regression-v3-fp32-partial-082-v3.json`: 82 literal-exact pages,
118 missing pages, 7,334 current inference/lineage checks and 1,063 bound source
files. It correctly reports `incomplete`, leaves the differential and overall
reporting gates false, identifies saved-token replay, and retains incomplete
historical startup semantics. Its script and all seven recorded helper hashes
match the reviewed files. Earlier v1/v2 evidence remains unchanged.

Exact reviewed hashes and the report digest are in
`reference/quality-regression-independent-review-v1.json`.
