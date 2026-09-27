#!/usr/bin/env python3
"""Phase 2: the shipped router. Computes the version-2 statistics
(router_features.py, the Rust specification) for the Phase 0 pages and the
development set, trains the two gradient-boosted models exactly as
dev_image_probs.py does (noise-aware labels, stable pages) but with
scikit-learn's early stopping (a 10% split of the training pages picks the
tree count: about 2.8x fewer nodes, the same development result), writes the
development probabilities for dev_policy.py and exports the trees for the
runner (src/router/trees.json), checking that a plain evaluation of the export
reproduces scikit-learn's raw scores bit for bit.

  python research/resolution-router/train_trees.py --router /mnt/d/falcon-draft/router \\
      --lock reference/router-dev-v1-evaluation-lock.json --dev /mnt/d/falcon-draft/router-dev \\
      --export src/router/trees.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import native_path  # noqa: E402
from router_features import CAP, FEATURES, capped_page, features  # noqa: E402

SIZES = ("768", "1024")


def page_features(path: str) -> list[float]:
    return features(capped_page(native_path(path)))


def cached(path: Path, pages: list[str], workers: int) -> np.ndarray:
    if path.exists():
        x = np.load(path)
        if len(x) == len(pages):
            return x
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        x = np.asarray(list(pool.map(page_features, pages, chunksize=8)), dtype=np.float64)
    np.save(path, x)
    return x


def labels(router: Path, outputs: Path, meta: list[dict], tolerance: float, margin: float):
    noise, stop_b = [], []
    for m in meta:
        name = m["id"].replace("/", "__") + ".json"
        first = json.loads((outputs / name).read_text(encoding="utf-8"))
        second = json.loads((router / "outputs-1536-retest" / name).read_text(encoding="utf-8"))
        noise.append(Levenshtein.distance(second["text"], first["text"]) / max(1, len(first["text"])))
        stop_b.append(second["finish_reason"])
    stable = np.array([nz <= tolerance and m["stop"]["1536"] == sb
                       for m, nz, sb in zip(meta, noise, stop_b, strict=True)])
    ok = {s: np.array([m["disagreement"][s] <= max(tolerance, nz + margin) and m["stop"][s] in (m["stop"]["1536"], sb)
                       for m, nz, sb in zip(meta, noise, stop_b, strict=True)]) for s in SIZES}
    return stable, ok


def export(model: HistGradientBoostingClassifier) -> dict:
    """Leaves [value]; splits [feature, threshold, left, right] (x <= threshold goes left)."""
    trees = []
    for (predictor,) in model._predictors:
        nodes = []
        for node in predictor.nodes:
            assert not node["is_categorical"]
            if node["is_leaf"]:
                nodes.append([float(node["value"])])
            else:
                nodes.append([int(node["feature_idx"]), float(node["num_threshold"]),
                              int(node["left"]), int(node["right"])])
        trees.append(nodes)
    return {"baseline": float(model._baseline_prediction.ravel()[0]), "trees": trees}


def raw_score(model: dict, x: list[float]) -> float:
    """The runner's evaluation order: the baseline, then each tree in turn."""
    raw = model["baseline"]
    for nodes in model["trees"]:
        node = nodes[0]
        while len(node) == 4:
            node = nodes[node[2] if x[node[0]] <= node[1] else node[3]]
        raw += node[0]
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, default=Path("/mnt/d/falcon-draft/outputs/stage3"))
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True,
                    help="writes phase2-dev-features.npy and image-probs-v2.json here")
    ap.add_argument("--export", type=Path)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args()
    meta = json.loads((a.router / "phase1-meta.json").read_text(encoding="utf-8"))["pages"]
    sample = {p["id"]: p for p in map(json.loads, (a.router / "sample.jsonl").read_text(encoding="utf-8").splitlines())}
    x = cached(a.router / "phase2-features.npy", [sample[m["id"]]["path"] for m in meta], a.workers)
    assert np.isfinite(x).all()
    stable, ok = labels(a.router, a.outputs, meta, a.tolerance, a.margin)
    models = {s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                                 random_state=0).fit(x[stable], ok[s][stable]) for s in SIZES}
    print(f"{len(meta)} pages, {stable.sum()} stable; routable 768 {ok['768'][stable].mean():.1%}, "
          f"1024 {ok['1024'][stable].mean():.1%}; trees {', '.join(f'{s}: {models[s].n_iter_}' for s in SIZES)}")

    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    dev = cached(a.dev / "phase2-dev-features.npy", [p["canonical_path"] for p in lock], a.workers)
    assert np.isfinite(dev).all()
    exported = {s: export(models[s]) for s in SIZES}
    out = {}
    for entry, f in zip(lock, dev, strict=True):
        pid = Path(entry["canonical_path"]).parent.name
        out[pid] = {s: float(models[s].predict_proba(f[None])[0, 1]) for s in SIZES}
        for s in SIZES:
            assert raw_score(exported[s], list(f)) == models[s]._raw_predict(f[None])[0, 0], (pid, s)
        out[pid]["width"], out[pid]["height"], out[pid]["line_height_px"] = float(f[0]), float(f[1]), float(f[15])
    (a.dev / "image-probs-v2.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} development pages; mean p768 {np.mean([v['768'] for v in out.values()]):.2f}, "
          f"p1024 {np.mean([v['1024'] for v in out.values()]):.2f}; export reproduces raw scores")
    if a.export:
        a.export.parent.mkdir(parents=True, exist_ok=True)
        a.export.write_text(json.dumps({"version": 2, "cap": CAP, "features": FEATURES, "models": exported},
                                       separators=(",", ":")) + "\n")
        print(f"exported {a.export} ({a.export.stat().st_size / 1024:.0f} KiB, "
              f"{sum(len(t) for m in exported.values() for t in m['trees'])} nodes)")


if __name__ == "__main__":
    main()
