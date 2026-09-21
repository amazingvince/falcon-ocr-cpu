# Fixed prefix/suffix telescope: accepted diagnostic

The single fixed run completed all 11 branches, 54 native transformer blocks and 11 native layer9 pre-QKV calls. Its three control branches reproduced all 94 saved arrays exactly before the eight unseen endpoints ran. The independently implemented saved-array review verified all 110 payload hashes, 11 full entry states, every global and per-row statistic, and zero FP64 telescoping residual across all 147,456 V9 elements. No model was run by the review.

At the original failing coordinate `layer.9.v[112,14,2]`, the original CPU−GPU error remains **+0.006622314453125**, above the unchanged bound **0.005096435546875**. The largest signed terms are:

| Fixed-order conditional term | Propagated difference | Fraction of original signed error | Full V9 RMS |
|---|---:|---:|---:|
| Embedding state, `E0−A` | +0.0028476715087890625 | +43.0012% | 0.00006293505 |
| Block0 substitution, `E1−E0` | +0.0044460296630859375 | +67.1371% | 0.00009641928 |
| Block1 substitution, `E2−E1` | −0.0015811920166015625 | −23.8767% | 0.00003930488 |
| Block3 substitution, `E4−E3` | +0.00101470947265625 | +15.3226% | 0.00001631339 |
| Layer9 pre-QKV on common CPU8 state, `D−E9` | +0.00000762939453125 | +0.1152% | 0.00000224832 |

The original full-V9 RMS error is `0.00010810784355556972`. Embedding and block0 substitutions are the largest full-array RMS terms as well as the largest positive terms at this coordinate. Layer9's local RMS/QKV path is small in this accounting. Block1 and several later terms cancel part of the positive error: the sum of absolute coordinate terms divided by the absolute total is **1.5858294930875576**.

These observations localize the dominant conditional contribution to earlier boundaries. They do not establish a defective block0 operator, show that one optimization will fix the error, or allocate nonlinear interactions uniquely. Attention mixes all 144 rows; the final coordinate does not identify a source row. The individual terms being below the original absolute bound does not make the total pass. Maxima and RMS values are not additive contributions. All ten frozen intermediate failures remain open, with no tolerance change or production intervention.

The next bounded question is: **on the same complete CPU embedding C0, which native block0 substage first develops the CPU/GPU difference whose substitution produces the large `E1−E0` effect?** A prospective one-block experiment can reuse the established crossover structure: GPU0(G0) with five original Q/K/V/attention/hidden controls; Rust0(C0) with the five original CPU controls; then GPU0(C0). Capture the same complete 14 stages (entry, attention norm/QKV/Q/K/V/attention/projection/residual, FFN norm/W13/gate/W2/hidden), keeping all144 rows, cache256 on GPU, original positions/mask/strictFP32 arithmetic and existing Rust behavior. No suffix or layer7 rerun is needed to ask that local question; the accepted telescope already establishes its conditional downstream effect. A local stage difference would still need a later same-input operator test before attributing the endpoint error to a particular kernel. This is a proposal, not an executed follow-up.

Evidence:

- Plan: `artifacts/diagnostics/fp32-prefix-suffix-telescope-plan-v1/plan.json`, SHA256 `17afd8974541318b3380f4c75b20dfbb8434500db68e125d0d3bc490211bf561`.
- Capture: `artifacts/diagnostics/fp32-prefix-suffix-telescope-gpu-v1/report.json`, SHA256 `1c9882505c5c381129435d1ae97ece3886a7f17dcd91e2007274971da9799c5a`; tensors `0f0a8d9c4de1af9ca10bc710768751d9da053c98e18851f312eeac9f247dacd8`.
- Complete signed results: `reference/fp32-prefix-suffix-telescope-decomposition-v1.json`, SHA256 `8bbe942e152340e6a0361e7e40aa8384dbd07b19177791ad2dcfe48e0f2a4f71`.
- Saved-array review: `reference/fp32-prefix-suffix-telescope-independent-review-v1.json`, SHA256 `c11085dd7e6b85afd654a843247157a90230f9329d44ed830711b626a538d329`; recomputation source `review_saved_v1.py` imports no experiment implementation.

The review performed 10,284 checks and rehashed 94 files before and after use, including all six tensor archives it used. The model weights and two unused large historical tensor archives inherit the accepted exporter's startup/end closure and were not rehashed by this review. The recomputation was independently implemented by the experiment author; a different agent performed the earlier source review. Historical startup provenance gaps and changed observation/allocation history remain explicit. These are finite-input diagnostic controls, not full-model numerical qualification or performance measurements.
