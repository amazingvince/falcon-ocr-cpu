"""Static study of the rotated KV cache on the real Falcon-OCR v1.5 weights.

Real weights, synthetic activations, no inference. For every layer the script
reads only `layers.{i}.attention.wqkv.weight` (and `freqs_cis_golden`), feeds
synthetic attention inputs through it and builds the keys and values exactly as
the runner does: V = W_v x; K = RoPE(per-head RMS-norm(W_k x)) with the image
tokens at temporal position 0 and their patch-grid positions through the learned
golden spatial frequencies (each query head of a GQA pair rotates its own copy),
and the text tokens at temporal positions 1.. with the spatial rotation off.
Queries are decode-time text queries built the same way.

The inputs x are RMS-normalized (the attention norm has unit weights) and come
in two kinds: isotropic Gaussian rows, and rows with one massive residual
channel (a fixed channel at `--massive` times the RMS of the others in every
token, before the norm). Layer 0 is also studied on the one real input that
needs no inference: the (normalized) token embeddings of random vocabulary
entries, as text keys and queries.

Each key and value block of 32 values (the cache's quantization groups: the
temporal and spatial key halves and the two value halves) is stored as
Q8 / Q4 (absmax codes, BF16 scale rounded up, as `src/quant/kv.rs`) or rotated
first (Q8R / Q4R: H D x with the signs of `src/quant/rotation.rs`). The script
reports the relative quantization error of the blocks and the relative error of
the attention output of every head (FP64 softmax attention over the dequantized
keys and values) for diffuse queries and for peaked ones (a query aligned with
one key). The attention sinks and the chunked kernels' rounding are left out:
the quantization error is orders of magnitude larger. Per layer it also
records how unevenly the rows of W_k and W_v spread variance over the channels
of a block, and how Gaussian W_v's columns look (`weights` in the JSON).

Usage:
    python3 kv_rotation_study.py <model.safetensors> --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

HEADS, KV_HEADS, HEAD_DIM, DIM, LAYERS = 16, 8, 64, 768, 22
ROPE_THETA = 10000.0
BLOCK = 32
# The sign patterns D of src/quant/rotation.rs (bit i set: element i negated).
SIGNS = {
    "temporal": 0x04DCF6B7,
    "spatial": 0x871F5F8B,
    "value_low": 0x489CE13C,
    "value_high": 0x4A81CBCB,
}
FORMATS = {"q8": (127, False), "q8r": (127, True), "q4": (7, False), "q4r": (7, True)}


def sylvester_hadamard(n: int) -> np.ndarray:
    """The n x n Sylvester Hadamard matrix, H[i][j] = (-1)^popcount(i & j)."""
    index = np.arange(n)
    parity = np.vectorize(lambda v: bin(v).count("1") & 1)(index[:, None] & index[None, :])
    return np.where(parity == 0, 1.0, -1.0)


HADAMARD = sylvester_hadamard(BLOCK)


def sign_vector(mask: int) -> np.ndarray:
    return np.array([-1.0 if (mask >> i) & 1 else 1.0 for i in range(BLOCK)])


def bf16_up(scale: np.ndarray) -> np.ndarray:
    """The smallest BF16 value at or above each FP32 scale (as the runner stores it)."""
    bits = scale.astype(np.float32).view(np.uint32)
    high = (bits >> 16) + ((bits & 0xFFFF) != 0).astype(np.uint32)
    return (high << 16).astype(np.uint32).view(np.float32).astype(np.float64)


def dequantized(x: np.ndarray, levels: int, rotated: bool, kind: str) -> np.ndarray:
    """What a cache of this format gives back for the blocks `x` (last axis 32):
    absmax codes in -levels..=levels against a BF16 scale per block, of `x` or
    of `H D x`; rotated blocks are mapped back with `D H / 32` (in the runner the
    query is rotated instead, which is the same in exact arithmetic)."""
    d = sign_vector(SIGNS[kind])
    y = (x * d) @ HADAMARD if rotated else x
    y = y.astype(np.float32).astype(np.float64)
    top = np.abs(y).max(axis=-1, keepdims=True)
    scale = bf16_up(np.maximum(top / levels, np.finfo(np.float32).smallest_subnormal))
    codes = np.clip(np.round(y / scale), -levels, levels)
    q = codes * scale
    return ((q @ HADAMARD) * d) / BLOCK if rotated else q


def rms_norm(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.finfo(np.float32).eps)


def rope(x: np.ndarray, temporal: np.ndarray, spatial: np.ndarray | None, golden: np.ndarray) -> np.ndarray:
    """RoPE of head vectors x [tokens, heads, 64] (interleaved pairs): pairs 0..16
    by the temporal position, pairs 16..32 by each head's golden spatial angle
    (`spatial` [tokens, 2] positions, None for text tokens: no rotation)."""
    pairs = np.arange(16)
    freq = 1.0 / ROPE_THETA ** (2 * pairs / 32)
    angle = np.zeros(x.shape[:2] + (32,))
    angle[:, :, :16] = temporal[:, None, None] * freq[None, None, :]
    if spatial is not None:
        # golden [heads, 16, 2]: angle = h * g0 + w * g1 per head and pair.
        angle[:, :, 16:] = (
            spatial[:, None, None, 0] * golden[None, :, :, 0] + spatial[:, None, None, 1] * golden[None, :, :, 1]
        )
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out


def grid_positions(width: int, height: int) -> np.ndarray:
    """The processor's patch positions: linspace over +-sqrt(aspect), rows first."""
    xlim, ylim = np.sqrt(width / height), np.sqrt(height / width)
    ys, xs = np.linspace(-ylim, ylim, height), np.linspace(-xlim, xlim, width)
    return np.stack(np.meshgrid(ys, xs, indexing="ij"), axis=-1).reshape(-1, 2)


def attention_inputs(rng: np.random.Generator, tokens: int, massive: tuple[int, float] | None) -> np.ndarray:
    """RMS-normalized inputs: Gaussian rows, optionally with one massive channel
    (`(channel, times the RMS)`) in every row."""
    h = rng.standard_normal((tokens, DIM))
    if massive is not None:
        channel, scale = massive
        h[:, channel] = scale * (1.0 + 0.1 * rng.standard_normal(tokens))
    return rms_norm(h)


def keys_values_queries(wqkv, golden, grid, text, queries, x):
    """Keys [tokens, heads, 64] (image then text), values [tokens, groups, 64] and
    decode queries [queries, heads, 64] for the attention inputs `x` (image, text,
    then query rows)."""
    image = grid.shape[0]
    wq, wk, wv = wqkv[:1024], wqkv[1024:1536], wqkv[1536:]
    k = rms_norm((x[: image + text] @ wk.T).reshape(-1, KV_HEADS, HEAD_DIM))
    k = np.repeat(k, 2, axis=1)  # each query head rotates its own copy
    v = (x[: image + text] @ wv.T).reshape(-1, KV_HEADS, HEAD_DIM)
    k_image = rope(k[:image], np.zeros(image), grid, golden)
    k_text = rope(k[image:], np.arange(1, text + 1, dtype=np.float64), None, golden)
    q = rms_norm((x[image + text :] @ wq.T).reshape(-1, HEADS, HEAD_DIM))
    q = rope(q, np.full(queries, text + 1.0), None, golden)
    return np.concatenate([k_image, k_text]), v, q


def attend(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Softmax attention per head in FP64: q [n, heads, 64], k [t, heads, 64],
    v [t, groups, 64] (each group serves two query heads); returns [heads, n, 64]."""
    logits = q.transpose(1, 0, 2) @ k.transpose(1, 2, 0) / np.sqrt(HEAD_DIM)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ np.repeat(v, 2, axis=1).transpose(1, 0, 2)


def relative(error: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.sum(error**2) / np.sum(reference**2)))


def crest(blocks: np.ndarray) -> float:
    """Mean max/RMS of 32-value blocks (about 2.36 for Gaussian blocks)."""
    return float(np.mean(np.abs(blocks).max(-1) / np.sqrt(np.mean(blocks**2, -1))))


def row_spread(w: np.ndarray) -> float:
    """Largest max/mean row norm within any 32-row block of `w` (output channels):
    how much more variance a channel of that block gets from isotropic inputs."""
    norms = np.linalg.norm(w, axis=1).reshape(-1, BLOCK)
    return float((norms.max(axis=1) / norms.mean(axis=1)).max())


def study_layer(wqkv, golden, grid, text, queries, x, rng) -> dict:
    k, v, q = keys_values_queries(wqkv, golden, grid, text, queries, x)
    k_blocks = k.reshape(k.shape[0], HEADS, 2, BLOCK)
    v_blocks = v.reshape(v.shape[0], KV_HEADS, 2, BLOCK)
    kinds = {
        "temporal": k_blocks[:, :, 0],
        "spatial": k_blocks[:, :, 1],
        "value_low": v_blocks[:, :, 0],
        "value_high": v_blocks[:, :, 1],
    }
    wk, wv = wqkv[1024:1536], wqkv[1536:]
    result = {
        "weights": {
            "k_row_spread": row_spread(wk),
            "v_row_spread": row_spread(wv),
            "v_column_crest": crest(wv.T.reshape(DIM, -1, BLOCK)),
        },
        "crest": {kind: crest(blocks) for kind, blocks in kinds.items()},
    }
    rotated_crest = {}
    for kind, blocks in kinds.items():
        d = sign_vector(SIGNS[kind])
        rotated_crest[kind] = crest((blocks * d) @ HADAMARD)
    result["crest_rotated"] = rotated_crest
    # Peaked queries: aligned with one image or text key (logit 16 above an
    # orthogonal key), the regime where one position's error is not averaged.
    picks = rng.integers(k.shape[0], size=queries)
    peaked = 2.0 * k[picks]
    exact = {"diffuse": attend(q, k, v), "peaked": attend(peaked, k, v)}
    for name, (levels, rotated) in FORMATS.items():
        deq = {kind: dequantized(blocks, levels, rotated, kind) for kind, blocks in kinds.items()}
        block_error = {kind: relative(deq[kind] - kinds[kind], kinds[kind]) for kind in kinds}
        k_eff = np.stack([deq["temporal"], deq["spatial"]], axis=2).reshape(k.shape)
        v_eff = np.stack([deq["value_low"], deq["value_high"]], axis=2).reshape(v.shape)
        output_error = {}
        for regime, query in (("diffuse", q), ("peaked", peaked)):
            out = attend(query, k_eff, v_eff)
            output_error[regime] = relative(out - exact[regime], exact[regime])
        result[name] = {"block_error": block_error, "output_error": output_error}
    return result


def summarize(layers: list[dict]) -> dict:
    """Means over layers of every metric."""

    def mean(path):
        values = []
        for layer in layers:
            node = layer
            for key in path:
                node = node[key]
            values.append(node)
        return float(np.mean(values))

    summary = {
        "crest": {kind: mean(("crest", kind)) for kind in SIGNS},
        "crest_rotated": {kind: mean(("crest_rotated", kind)) for kind in SIGNS},
    }
    for name in FORMATS:
        summary[name] = {
            "block_error": {kind: mean((name, "block_error", kind)) for kind in SIGNS},
            "output_error": {regime: mean((name, "output_error", regime)) for regime in ("diffuse", "peaked")},
        }
    # Per-layer ratios of the rotated format's output error to the plain one's.
    summary["ratio_min_median_max"] = {}
    for plain, rotated in (("q8", "q8r"), ("q4", "q4r")):
        spans = {}
        for regime in ("diffuse", "peaked"):
            ratios = [layer[rotated]["output_error"][regime] / layer[plain]["output_error"][regime] for layer in layers]
            spans[regime] = [float(np.min(ratios)), float(np.median(ratios)), float(np.max(ratios))]
        summary["ratio_min_median_max"][f"{rotated}/{plain}"] = spans
    return summary


def table(summary: dict) -> str:
    rows = ["| Format | T | S | V lo | V hi | out (diffuse) | out (peaked) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in FORMATS:
        blocks = summary[name]["block_error"]
        out = summary[name]["output_error"]
        cells = [f"{blocks[kind]:.2e}" for kind in SIGNS] + [f"{out['diffuse']:.2e}", f"{out['peaked']:.2e}"]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    crest_row = " | ".join(f"{summary['crest'][k]:.2f} -> {summary['crest_rotated'][k]:.2f}" for k in SIGNS)
    rows.append(f"| max/RMS (plain -> rotated) | {crest_row} | | |")
    for pair, regimes in summary["ratio_min_median_max"].items():
        spans = ", ".join(f"{regime} {lo:.2f} / {mid:.2f} / {hi:.2f}" for regime, (lo, mid, hi) in regimes.items())
        rows.append(f"\n{pair} output error per layer (min / median / max): {spans}")
    return "\n".join(rows)


def rounded(value):
    """`value` with every float rounded to four significant digits (for the JSON)."""
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def compact_json(results: dict) -> str:
    """`results` as JSON with one line per scalar, summary entry and layer."""
    lines = ["{"]
    for i, (key, value) in enumerate(results.items()):
        end = "," if i < len(results) - 1 else ""
        if not (isinstance(value, dict) and "layers" in value):
            lines.append(f" {json.dumps(key)}: {json.dumps(value)}{end}")
            continue
        summary, layers = value["summary"], value["layers"]
        lines.append(f" {json.dumps(key)}: {{")
        lines.append('  "summary": {')
        lines += [
            f"   {json.dumps(k)}: {json.dumps(v)}" + ("," if j < len(summary) - 1 else "")
            for j, (k, v) in enumerate(summary.items())
        ]
        lines.append("  },")
        lines.append('  "layers": [')
        lines += ["   " + json.dumps(layer) + ("," if j < len(layers) - 1 else "") for j, layer in enumerate(layers)]
        lines.append("  ]")
        lines.append(f" }}{end}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="the FP32 model.safetensors")
    parser.add_argument("--output", type=Path, help="write every layer's numbers as JSON")
    parser.add_argument(
        "--grid", default="68x96", help="image patch grid WxH (default: the 1088x1536-pixel journal page)"
    )
    parser.add_argument("--text", type=int, default=256, help="text positions after the image")
    parser.add_argument("--queries", type=int, default=32, help="decode queries per head")
    parser.add_argument("--massive", type=float, default=50.0, help="massive channel, times the RMS")
    parser.add_argument("--vocabulary", type=int, default=4096, help="layer-0 embedding rows (0: skip)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=LAYERS)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    width, height = (int(v) for v in args.grid.split("x"))
    grid = grid_positions(width, height)
    results = {"grid": args.grid, "text": args.text, "queries": args.queries, "massive_scale": args.massive}
    with safe_open(args.checkpoint, framework="numpy") as f:
        golden = f.get_tensor("freqs_cis_golden").astype(np.float64)
        tokens = grid.shape[0] + args.text + args.queries
        for variant in ("isotropic", "massive"):
            rng = np.random.default_rng(args.seed)
            # One residual channel, the same in every layer, as a residual stream's.
            massive = (int(rng.integers(DIM)), args.massive) if variant == "massive" else None
            layers = []
            for layer in range(args.layers):
                wqkv = f.get_tensor(f"layers.{layer}.attention.wqkv.weight").astype(np.float64)
                x = attention_inputs(rng, tokens, massive)
                layers.append(study_layer(wqkv, golden, grid, args.text, args.queries, x, rng))
                print(f"{variant} layer {layer}: done", file=sys.stderr)
            summary = summarize(layers)
            results[variant] = {"summary": summary, "layers": layers}
            print(f"\n{variant} inputs, mean over {args.layers} layers (relative errors):\n{table(summary)}")
        if args.vocabulary:
            # Layer 0's attention input for a text token is its normalized embedding.
            rng = np.random.default_rng(args.seed)
            ids = rng.choice(f.get_slice("tok_embeddings.weight").get_shape()[0], args.vocabulary + args.queries)
            x = rms_norm(f.get_tensor("tok_embeddings.weight")[np.sort(ids)].astype(np.float64))
            x = x[rng.permutation(len(x))]
            wqkv = f.get_tensor("layers.0.attention.wqkv.weight").astype(np.float64)
            layer = study_layer(wqkv, golden, np.zeros((0, 2)), args.vocabulary, args.queries, x, rng)
            summary = summarize([layer])
            results["vocabulary"] = {"summary": summary, "layers": [layer]}
            print(f"\nlayer 0 on {args.vocabulary} embedding rows (relative errors):\n{table(summary)}")
    if args.output:
        args.output.write_text(compact_json(rounded(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
