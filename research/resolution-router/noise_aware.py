#!/usr/bin/env python3
"""Phase 1 with serving noise accounted for.

vLLM's batched BF16 outputs differ between two runs of the same page at the
same resolution (a quarter of pages by more than 2%), so "within 2% of one
1536 transcript" mislabels many pages. With a second 1536 run (`outputs-1536-
retest`):

- noise: distance between the two 1536 transcripts;
- noise-aware routable at r: the r transcript is no further from the first
  1536 run than the second run is, plus `--margin` (at least `--tolerance`),
  and its stop reason is one of the two 1536 runs';
- stable pages: the two 1536 runs agree within the tolerance and stop alike
  (clean labels for training).

Routers (gradient-boosted trees): image-only 768/1024/1536, and the cascade
(image prior, then a 64-token probe at 768). Trained on stable pages with the
noise-aware labels; thresholds tuned on validation for the most modelled time
saved with at most `--budget` of pages mis-routed (routed to r although not
noise-aware routable there); reported on all test pages.

  python research/resolution-router/noise_aware.py --router /mnt/d/falcon-draft/router
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SIZES = ("768", "1024")


def prefill_s(n):
    return (0.33 * n + 0.000049 * n * n) / 1000


def decode_s(n, t):
    return t * (4.3 + 0.00064 * n) / 1.47 / 1000


def split_of(group):
    v = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
    return "train" if v < 70 else "val" if v < 85 else "test"


def tune(evaluate, grids, ix, budget):
    best = (-1.0, None)
    for point in np.array(np.meshgrid(*grids)).T.reshape(-1, len(grids)):
        saved, bad = evaluate(ix, *point)[:2]
        if bad <= budget and saved > best[0]:
            best = (saved, tuple(point))
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, default=Path("/mnt/d/falcon-draft/outputs/stage3"))
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--probe", type=int, default=64)
    a = ap.parse_args()
    info = json.loads((a.router / "phase1-meta.json").read_text(encoding="utf-8"))
    meta, img = info["pages"], np.load(a.router / "phase1-data.npz")["features"]
    noise, stop_b, conf = [], [], []
    for m in meta:
        name = m["id"].replace("/", "__") + ".json"
        first = json.loads((a.outputs / name).read_text(encoding="utf-8"))
        second = json.loads((a.router / "outputs-1536-retest" / name).read_text(encoding="utf-8"))
        noise.append(Levenshtein.distance(second["text"], first["text"]) / max(1, len(first["text"])))
        stop_b.append(second["finish_reason"])
        lp = np.asarray(json.loads((a.router / "outputs-768" / name).read_text(encoding="utf-8")).get("top1_logprob") or [0.0])
        part = lp[:a.probe]
        conf.append([part.mean(), (np.exp(part) < 0.5).mean(), part.min(), (np.exp(part) < 0.9).mean()])
    noise, conf = np.array(noise), np.asarray(conf, dtype=np.float32)
    stable = np.array([nz <= a.tolerance and m["stop"]["1536"] == sb for m, nz, sb in zip(meta, noise, stop_b)])
    ok = {s: np.array([m["disagreement"][s] <= max(a.tolerance, nz + a.margin) and m["stop"][s] in (m["stop"]["1536"], sb)
                       for m, nz, sb in zip(meta, noise, stop_b)]) for s in SIZES}
    types = np.array([m["type"] for m in meta])
    print(f"stable pages {stable.mean():.1%}; noise-aware routable 768 {ok['768'].mean():.1%}, 1024 {ok['1024'].mean():.1%}")
    for t in ("prose", "formulas", "tables", "slides", "handwriting", "newspaper"):
        sel = types == t
        print(f"   {t:12} n={sel.sum():5}  stable {stable[sel].mean():5.0%}  routable 768 {ok['768'][sel].mean():5.0%}  "
              f"1024 {ok['1024'][sel].mean():5.0%}")

    splits = np.array([split_of(m["group"]) for m in meta])
    idx = {s: np.nonzero(splits == s)[0] for s in ("train", "val", "test")}
    train = idx["train"][stable[idx["train"]]]
    n = {s: np.array([m["image_tokens"][s] for m in meta], float) for s in ("768", "1024", "1536")}
    t = {s: np.array([m["tokens"][s] for m in meta], float) for s in ("768", "1024", "1536")}
    full = {s: prefill_s(n[s]) + decode_s(n[s], t[s]) for s in n}
    base = lambda ix: full["1536"][ix].sum()  # noqa: E731

    # Image-only, 768 / 1024 / 1536.
    p = {}
    for s in SIZES:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(img[train], ok[s][train])
        p[s] = model.predict_proba(img)[:, 1]

    def image_only(ix, t768, t1024):
        choice = np.where(p["768"][ix] >= t768, "768", np.where(p["1024"][ix] >= t1024, "1024", "1536"))
        cost = np.array([full[c][i] for c, i in zip(choice, ix)])
        bad = np.array([c != "1536" and not ok[c][i] for c, i in zip(choice, ix)])
        return 1 - cost.sum() / base(ix), bad.mean(), {c: (choice == c).mean() for c in ("768", "1024", "1536")}

    # Cascade: image prior, then the probe at 768.
    both = np.concatenate([img, conf], 1)
    pa = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(img[train], ok["768"][train]).predict_proba(img)[:, 1]
    pb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(both[train], ok["768"][train]).predict_proba(both)[:, 1]
    probe_cost = prefill_s(n["768"]) + decode_s(n["768"], np.minimum(a.probe, t["768"]))

    def cascade(ix, ta, tb):
        probe = pa[ix] >= ta
        stay = probe & (pb[ix] >= tb)
        cost = np.where(stay, full["768"][ix], np.where(probe, probe_cost[ix] + full["1536"][ix], full["1536"][ix]))
        return 1 - cost.sum() / base(ix), (stay & ~ok["768"][ix]).mean(), {"probed": probe.mean(), "768": stay.mean()}

    test = idx["test"]
    print(f"test AUC (noise-aware labels, all test pages): image-only 768 {roc_auc_score(ok['768'][test], p['768'][test]):.3f}, "
          f"1024 {roc_auc_score(ok['1024'][test], p['1024'][test]):.3f}; cascade 768 {roc_auc_score(ok['768'][test], pb[test]):.3f}")
    oracle = np.where(ok["768"][test], full["768"][test], np.where(ok["1024"][test], full["1024"][test], full["1536"][test]))
    print(f"oracle (noise-aware): test time saved {1 - oracle.sum() / base(test):.1%}")
    grid = np.append(np.linspace(0.0, 0.99, 34), 1.01)
    for budget in (0.02, 0.05, 0.10):
        th = tune(image_only, [grid, grid], idx["val"], budget)
        s, b, mix = image_only(test, *th)
        thc = tune(cascade, [grid, grid], idx["val"], budget)
        sc, bc, mixc = cascade(test, *thc)
        print(f"budget {budget:.0%}: image-only saved {s:.1%} (mis-routed {b:.1%}; routes "
              f"{ {k: f'{v:.0%}' for k, v in mix.items()} }) | cascade saved {sc:.1%} (mis-routed {bc:.1%}; "
              f"{ {k: f'{v:.0%}' for k, v in mixc.items()} })")


if __name__ == "__main__":
    main()
