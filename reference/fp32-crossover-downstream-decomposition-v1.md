# Downstream continuation to the original FP32 failure

The one released continuation completed. A fresh GPU suffix on the original
CPU layer-7 state reproduced **all 17 prior GPU stages bit for bit** before the
new substitution ran. The offline join checked all 34 new payloads, two new
complete input states, 93 historical payloads, 22 original controls and eight
historical state joins. Source/input closures passed; every FP64 telescoping
residual is zero. No layer-7, Rust or full-model rerun occurred.

The target remains `layer.9.v[112,14,2]`. A is GPU89(original GPU7), C is
GPU89(original CPU7), D is Rust89(original CPU7), and B is the new
GPU89(saved GPU7(original CPU6)). All inputs retain the full 144 rows and the
original mask/position state. The observed endpoints are:

| Endpoint | FP32 value |
|---|---:|
| A | -19.037837982177734 |
| B | -19.031208038330078 |
| C | -19.031253814697266 |
| D | -19.03121566772461 |

The signed identity `D-A = (D-C) + (C-B) + (B-A)` gives:

| Term | Difference | Signed fraction of original total |
|---|---:|---:|
| Original D-A | +0.006622314453125 | 100% |
| Accumulated downstream engine D-C | +0.00003814697265625 | +0.5760% |
| Layer-7 engine effect through GPU suffix C-B | -0.0000457763671875 | -0.6912% |
| Earlier-state difference through GPU7/suffix B-A | +0.00662994384765625 | +100.1152% |

The last term remains above the unchanged original bound,
`0.005096435546875`. Substituting the observed GPU layer-7 and suffix arithmetic
therefore does not remove this fixed-coordinate failure. The two engine terms
partially cancel. These are conditional finite differences along one
substitution order, not probabilities or independent additive kernel causes.

Across the complete layer-9 V tensor, RMS is `0.00010810784355556972` for the
original difference, `0.000005652412524300419` for D-C,
`0.00000636576446180227` for C-B, and `0.00010854459441299701` for B-A. Maxima/RMS
must not be added as a signed attribution. The full report retains every row
and stage, including cancellation and the original coordinate.

This result keeps the original numerical failure open. It does not identify
which earlier operator created the incoming difference, qualify full hidden
parity, or justify a production change. Historical startup-source gaps,
changed allocation/observation history and the limits of same-input controls
remain explicit. No numerical tolerances changed and no further intervention
was launched.

Evidence:

- `reference/fp32-crossover-downstream-decomposition-v1.json`, SHA256
  `32d15683c8d226f4d0ea9163147cb6cee1d19b823a8d3c89fc5ec1861e792806`.
- `reference/fp32-crossover-downstream-completion-v1.json` binds the plan,
  exact launcher/environment, stdout/stderr, source closure and output records.
- Plan SHA256 `89d79b4717157f9d66c43092c9c83acd80880e0f823550e2a34111dafc6e7c36`;
  GPU report SHA256 `9d14800ac47f217afeb0f390e9cf96271fbb8e8d4765cdfb424d75eea43fbd78`;
  GPU tensors SHA256 `37ef2f78a224032d7193de5504ae4087259827fd23d4dfabcd60ade8649467a3`.

The isolated 4090 returned to 0% utilization/42 MiB after all owned export and
comparison processes exited. The 5090 was not used. Independent saved-result
review is separate from this execution/comparison receipt.
