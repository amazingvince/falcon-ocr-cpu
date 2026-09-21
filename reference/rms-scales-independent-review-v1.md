# RMS scale capture review

Both preserved captures (`artifacts/diagnostics/rms-scales-windows-v1` and `rms-scales-wsl-v1`) pass the independent review. The receipt is `reference/rms-scales-independent-review-v1.json`, SHA256 `8380c45053945d7ad5ecd9274b77792a7a08459ca7601345fcbb655faf2b849b`.

The input, expected-output and captured-scale binaries are identical across platforms, as are all 5,776 scale triples and every reported row. The 14 width-768 cases contain 1,444 rows. An independent integer-significand/RNE oracle searched the entire positive finite F32 scale space by monotonic binary search for each anchor, then checked every surviving candidate against all 768 output bits: 1,231,104 candidate-output comparisons. It confirms one output-consistent scale per row and complete coverage by the capture's widened interval. Seven bounded synthetic tests also pass, covering power-of-two transitions, ties, signed zero, subnormal products, the smallest positive scale, inconsistent rows, and unsupported anchors.

The source/function extraction, fixture and sidecar pins, stage mapping, epsilon, preserved source copies, executable/data hashes, and final source window were checked. No concrete arithmetic defect was found. Production, production with the F64 reciprocal-square-root candidate, CUDA-shaped reduction, and that reduction with the F64 candidate reproduce respectively 862, 994, 959 and 1,138 of the saved GPU output rows.

These are output-consistent multipliers, **not observed CUDA rstd values or inferred GPU variances**. Actual CUDA capture remains necessary. The F64 sqrt/reciprocal expression is a specified CPU candidate, not a proof of correctly rounded mathematical rsqrt. The original capture is non-hermetic and records the final executable hash rather than a pre-run executable attestation; this review does not retroactively strengthen that history. No inference, probe re-execution, production changes or timing measurements were performed.

Reproduce with `python experiments/rms_norm/test_capture_scales.py -v` and `python experiments/rms_norm/review_saved_scales.py --output <new-receipt.json>`.
