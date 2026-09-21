# Bounded operator evidence, 2026-09-20

The isolated scalar W4A32/W8A32 experiment completed. Independent Rust and Python
implementations produced identical code bytes, FP32 scale bits, and reconstructed
FP32 matrix bytes for **all 112 matrix/format combinations**. All 8,880 sampled
outputs passed the independent FP64 arithmetic checks and FP32 accumulation error
bound; eight Rust unit tests and the Python format self-tests also passed.

The seven fixed synthetic fixture cases cover QKV, WO, W13 and W2 in 28 checkpoint
matrices. Each format reconstructs all 53,673,984 weight elements. Operator output
coverage is deliberately bounded: 2,220 of 5,914,624 possible output elements per
format (0.0375%), selected by fixed boundary/quartile rows and output channels,
including adjacent interleaved W13 channels and Q/K/V boundaries. Every format
receives the same original GPU activations; errors do not propagate through a
quantized graph.

| Format | Payload for these 28 matrices | Full weight relative L2 error | Sampled output relative L2 change, quantization only |
| --- | ---: | ---: | ---: |
| W4A32, group 64 | 30,191,616 bytes | 10.8801% | 7.9694% |
| W4A32, group 128 | 28,514,304 bytes | 11.9482% | 11.3391% |
| W8A32, group 64 | 57,028,608 bytes | 0.5999% | 0.4735% |
| W8A32, group 128 | 55,351,296 bytes | 0.6588% | 0.4945% |

Unquantized payload for these matrices is 214,695,936 bytes. Payload figures count
codes plus FP32 scales only. They are not process memory measurements or a whole
model storage estimate. Relative L2 is `norm(error)/norm(reference)` across the
explicitly included elements. The sampled-output column compares FP64 dot sums
using reconstructed weights against FP64 sums using original weights, separating
weight compression from FP32 accumulator rounding. Aggregation weights by signal
energy, so these percentages are not average per-case errors.

Per-case variability is substantial: sampled W4 group-64 W13 relative L2 spans
5.76–28.05%, for example. This limited fixture cannot establish accuracy after
gating/residuals or across generated tokens. W8 serves as the requested control;
these results select **no group size, layer policy, clipping rule or quality
budget**. The fixture remains neither calibration data nor held-out quality
evaluation. No production model changes, speed measurements, corpus assessment,
GPU parity qualification or tolerance changes were made.

The durable [summary](operator-probe-v1-summary.json) records aggregate and
per-operator ranges, original checkpoint/fixture/trace hashes, source identities,
and exact evidence paths. Full outputs, explicit sample indices, matrix/input/GPU
tensor hashes, the independent checker report, build log, binary and source ZIP
are preserved under `artifacts/quantization/operator-probe-v1/`. The runtime binary
and all four embedded source fingerprints were matched to the captured build;
every archived source entry was independently rehashed.

- Binary SHA256: `23e25fc538f87044cb5c9ff53df8f0e3b01f9f8700da06c9564c055aa9e0dfe5`
- Source ZIP SHA256: `3e096e11c64a59f8a0b0d6af3816909edf23364b17a2fcf13d912a62fc90af97`
- Original checkpoint SHA256: `3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16`
- Equal-input fixture SHA256: `ce7345c219d8923182ff66e9aad2f4d6bad3c193a19c3445a850c2d3a90e5417`

The format contract, sampling procedure, independent checks and reproduction
commands are in [README.md](README.md). This was a single-thread arithmetic run
concurrent with other functional work; the reports intentionally contain no timing
fields beyond `null`.
