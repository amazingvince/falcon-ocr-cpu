# Layer-7 FP32 crossover

The single bounded diagnostic completed. Five original GPU and five original
Rust Q/K/V/attention/hidden controls match bit for bit. The offline join rechecks
three complete layer-6 entry states and 42 stage payloads. All source/input
closures pass; FP64 accounting residuals are zero. One Rust layer-7 call and two
GPU layer-7 calls ran, with no retry, arithmetic variant or later-layer execution.

At the preselected layer-7 hidden coordinate `[112,249]`:

| Quantity | Value |
|---|---:|
| Rust on CPU entry | -58.89048767089844 |
| GPU on CPU entry | -58.89044952392578 |
| GPU on GPU entry | -58.892784118652344 |
| Total Rust minus GPU difference | 0.00229644775390625 |
| Incoming-state difference through GPU | 0.0023345947265625 |
| Segment engine difference on CPU entry | -0.00003814697265625 |

The incoming term is 101.661% of the signed total and the engine term is
-1.661%: they partially cancel. This is a causal partition for these fixed
states, not a probability or a universal attribution. Row112 hidden RMS is
0.000672061 total, 0.000675912 incoming, and 0.0000108982 engine. Full hidden
RMS is respectively 0.00213288, 0.00182221, and 0.000411043; the selected-row
finding must not be generalized to every row.

Most of the selected coordinate's discrepancy was already present before
layer7. This result does not identify an earlier faulty operator or qualify
full hidden-state parity. Later-stage engine terms include divergent internal
inputs within this segment, and are not isolated operator error estimates.
The original startup/build provenance gaps and changed observation/allocation
history remain explicit. The frozen policy and all production sources remain
unchanged. No further intervention is launched.

Exact inputs, runtime flags, every row/stage metric, source review, preparation,
build and execution hashes are bound by `fp32-crossover-layer7-completion-v1.json`
and `fp32-crossover-layer7-decomposition-v1.json`. The GPU returned to 0%/42MiB;
all owned build/model/compare processes exited. The
[independent saved-result review](fp32-crossover-layer7-independent-review-v1.json)
also passed all ten controls, three entry states, 42 payloads and the exact
signed decomposition, including zero full-array accounting residuals.
