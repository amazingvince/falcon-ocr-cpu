# KV-cache rotation

Rotating each 32-value cache block gives little benefit on this model's
weights: median attention-output error falls by 2% with isotropic inputs
and 9% with one massive input channel, with no gain on peaked queries. The
rotated formats (`kv-q8r`, `kv-q4r` and their 16- and 8-bit-weight profiles)
remain research options. Real-page token agreement and its limits are in
[docs/MODES.md](../../docs/MODES.md#rotated-kv-cache-experimental).

This static study uses real weights and synthetic activations, with no
inference. It reads the pinned checkpoint's `wqkv` matrices and golden
spatial frequencies and, for layer 0, token embeddings. The implementations
are `src/quant/rotation.rs` and `src/quant/kv.rs`.

| File | What |
|---|---|
| `kv_rotation_study.py` | The study (numpy, safetensors; about 3 minutes on 2 cores) |
| `results.json` | Recorded default study, with every layer's numbers |

To repeat the study, install `numpy` and `safetensors`, then run from the
repository root. The recorded output is `research/kv-rotation/results.json`;
write a new run under the ignored `artifacts/` directory:

```sh
mkdir -p artifacts/kv-rotation
python3 research/kv-rotation/kv_rotation_study.py artifacts/model/model.safetensors \
  --output artifacts/kv-rotation/results.json
```

## Method

For every layer, keys and values are built from the real `wqkv` exactly as
the runner builds them: `V = W_v x`; `K = RoPE(RMS-norm(W_k x))` per head, with
the image tokens at temporal position 0 and their patch positions through each
query head's learned golden frequencies, and text tokens at temporal positions
1, 2, ... with the spatial rotation off.

The geometry is the journal page's (1088 × 1536 pixels): a 68 × 96 patch
grid (6,528 image tokens) and 256 text tokens; 32 decode queries per head
at the next text position.

The inputs `x` are synthetic and RMS-normalized, as the attention norm leaves
them: isotropic Gaussian rows, or rows with one massive residual channel (the
same channel in every token and layer, 50 times the RMS of the others before
the norm). Layer 0 is also run on the one real input that needs no inference:
normalized token embeddings of 4,096 random vocabulary entries, as text keys.

The study compares Q8 and Q4 with their rotated forms Q8R and Q4R, using
the [runner's representation](../../docs/ARCHITECTURE.md#kv-cache-formats)
and absmax codes against a BF16 scale rounded up.

Reported: the relative error of the blocks per kind, and the relative
error of every head's attention output (FP64 softmax over the dequantized
keys and values) for diffuse queries (the model's own queries of random
inputs) and peaked ones (twice one key, so one position dominates). The
attention sinks and FP32 rounding are left out; quantization error
dominates.

## Results

Means over the 22 layers, expressed as relative errors. The value halves
are the low and high 32-value blocks of each value vector.

| Isotropic inputs | Temporal key | Spatial key | Value low | Value high |
|---|---:|---:|---:|---:|
| Q8 | 5.86e-3 | 5.56e-3 | 5.46e-3 | 5.46e-3 |
| Q8R | 5.42e-3 | 5.45e-3 | 5.46e-3 | 5.46e-3 |
| Q4 | 1.04e-1 | 9.89e-2 | 9.73e-2 | 9.73e-2 |
| Q4R | 9.66e-2 | 9.71e-2 | 9.73e-2 | 9.73e-2 |
| max/RMS, plain → rotated | 2.53 → 2.35 | 2.40 → 2.36 | 2.37 → 2.36 | 2.37 → 2.37 |

| Massive-channel inputs | Temporal key | Spatial key | Value low | Value high |
|---|---:|---:|---:|---:|
| Q8 | 5.91e-3 | 5.59e-3 | 5.50e-3 | 5.46e-3 |
| Q8R | 5.43e-3 | 5.44e-3 | 5.46e-3 | 5.42e-3 |
| Q4 | 1.05e-1 | 9.94e-2 | 9.80e-2 | 9.73e-2 |
| Q4R | 9.67e-2 | 9.70e-2 | 9.74e-2 | 9.66e-2 |
| max/RMS, plain → rotated | 2.55 → 2.35 | 2.42 → 2.36 | 2.37 → 2.37 | 2.37 → 2.35 |

Max/RMS is the mean over blocks; Gaussian blocks of 32 have max/RMS 2.36.
The corresponding attention-output errors are:

| Format | Isotropic, diffuse | Isotropic, peaked | Massive, diffuse | Massive, peaked |
|---|---:|---:|---:|---:|
| Q8 | 4.07e-3 | 5.46e-3 | 2.43e-4 | 2.07e-3 |
| Q8R | 3.94e-3 | 5.49e-3 | 2.21e-4 | 2.03e-3 |
| Q4 | 7.19e-2 | 9.72e-2 | 2.69e-3 | 3.56e-2 |
| Q4R | 7.03e-2 | 9.77e-2 | 2.68e-3 | 3.52e-2 |

Ratio of the rotated to the plain output error per layer (min / median / max):

| | Isotropic, diffuse | Isotropic, peaked | Massive, diffuse | Massive, peaked |
|---|---|---|---|---|
| Q8R / Q8 | 0.90 / 0.98 / 1.00 | 0.98 / 1.00 / 1.03 | 0.71 / 0.91 / 1.25 | 0.91 / 0.98 / 1.02 |
| Q4R / Q4 | 0.91 / 0.99 / 1.01 | 0.98 / 1.01 / 1.03 | 0.91 / 1.01 / 1.05 | 0.93 / 0.99 / 1.02 |

Layer 0 on real token embeddings: every block kind has max/RMS 2.36–2.40;
Q8R/Q8 output error 1.00 (diffuse) and 1.01 (peaked), Q4R/Q4 0.92 and 1.00.

Where the plain blocks are not Gaussian: the temporal key halves of layers
13 and 16–21 (max/RMS 2.60–2.77 with isotropic inputs, up to 2.96 with the
massive channel), where rotation cuts the Q8 block error by 12–23% (layer 16:
6.39e-3 → 5.30e-3).

The weights show why (`weights` in `results.json`): within a 32-value
block, the rows of `W_k` differ in norm by up to 4.1× (layer 19; 2.9 and
3.3 in layers 18 and 20), so a few key channels carry several times the
others' variance. The rows of `W_v` differ by at most 1.6× and its columns
look Gaussian (max/RMS 2.36–2.41), so the values have no outlier channels
with these inputs; the value halves gain nothing.

## Interpretation

### Eight-bit rotation

On this model's weights the cache blocks are nearly Gaussian unless the
activations make them otherwise. The unit tests find equal error on
Gaussian blocks, a quarter of the Q8 error with outlier channels at
10–50× the RMS, and more error than plain Q8 on blocks flatter than
Gaussian: 5.7e-4 against 2.4e-4 worst output error in
`low_precision_stays_close_to_fp32`.

The only structural outliers are in late-layer keys. Fixing them reduces
median output error by 2% (isotropic) to 9% (massive channel) for diffuse
queries, with no gain on peaked queries. One layer's output error rises
25% with the massive channel.

### Real-page agreement

The static study cannot see whether real activations carry outlier
channels into the keys or values. See the
[real-page agreement results](../../docs/MODES.md#rotated-kv-cache-experimental)
and [reproduction commands](../../docs/DEVELOPMENT.md#token-agreement).
On the screening set, FP32 weights with the Q8 cache flipped 0.77 steps per
1,000; per-channel key scales were flip-neutral in the hill climb.

### Four-bit cache

Four bits cost about 18 times the error of eight (10% of a block, rotated
or not); the real-page comparison is linked above.
