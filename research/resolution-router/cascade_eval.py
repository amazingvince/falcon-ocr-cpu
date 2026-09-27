#!/usr/bin/env python3
"""Phase 1: a confidence-gated cascade against image-only routing.

Cascade per page: (A) an image model sends pages that are unlikely to survive
768 straight to 1536; (B) every other page is prefilled at 768 and decodes
`--probe` tokens; a second model on the image statistics plus the probe's
confidence (top-1 log-probabilities of those tokens) decides to continue at
768 (the probe tokens are kept) or to restart at 1536 (the probe is wasted).
Thresholds are tuned on validation for the most modelled time saved with at
most `--budget` of pages mis-routed (continued at 768 although their 768 text
is not within the tolerance of the 1536 transcript or stops differently).
Costs: the router plan's fast-mode fits (prefill 0.33 n + 0.000049 n^2 ms,
decode (4.3 + 0.00064 n) / 1.47 ms per token, n image tokens).

  python research/resolution-router/cascade_eval.py --router /mnt/d/falcon-draft/router
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def prefill_s(n):
    return (0.33 * n + 0.000049 * n * n) / 1000


def decode_s(n, t):
    return t * (4.3 + 0.00064 * n) / 1.47 / 1000


def split_of(group: str) -> str:
    v = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
    return "train" if v < 70 else "val" if v < 85 else "test"


def confidence(lp: np.ndarray, k: int) -> list[float]:
    part = lp[:k] if len(lp) else np.zeros(1)
    p = np.exp(part)
    return [part.mean(), (p < 0.5).mean(), part.min(), (p < 0.9).mean(), float(len(lp) < k)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--probe", type=int, default=64)
    a = ap.parse_args()
    info = json.loads((a.router / "phase1-meta.json").read_text(encoding="utf-8"))
    meta = info["pages"]
    img = np.load(a.router / "phase1-data.npz")["features"]
    conf = []
    for m in meta:
        o = json.loads((a.router / "outputs-768" / (m["id"].replace("/", "__") + ".json")).read_text(encoding="utf-8"))
        conf.append(confidence(np.asarray(o.get("top1_logprob") or [], dtype=np.float64), a.probe))
    conf = np.asarray(conf, dtype=np.float32)
    y768 = np.array([m["disagreement"]["768"] <= a.tolerance and m["stop"]["768"] == m["stop"]["1536"] for m in meta])
    splits = np.array([split_of(m["group"]) for m in meta])
    idx = {s: np.nonzero(splits == s)[0] for s in ("train", "val", "test")}
    both = np.concatenate([img, conf], 1)
    stage_a = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(img[idx["train"]], y768[idx["train"]])
    stage_b = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(both[idx["train"]], y768[idx["train"]])
    p_a = stage_a.predict_proba(img)[:, 1]
    p_b = stage_b.predict_proba(both)[:, 1]
    print(f"test AUC for 768: image only {roc_auc_score(y768[idx['test']], p_a[idx['test']]):.3f}, "
          f"image + {a.probe}-token probe {roc_auc_score(y768[idx['test']], p_b[idx['test']]):.3f}")

    n = {s: np.array([m["image_tokens"][s] for m in meta], dtype=np.float64) for s in ("768", "1536")}
    t = {s: np.array([m["tokens"][s] for m in meta], dtype=np.float64) for s in ("768", "1536")}
    full = {s: prefill_s(n[s]) + decode_s(n[s], t[s]) for s in ("768", "1536")}
    probe_cost = prefill_s(n["768"]) + decode_s(n["768"], np.minimum(a.probe, t["768"]))

    def run(ix, ta, tb):
        probe = p_a[ix] >= ta
        stay = probe & (p_b[ix] >= tb)
        cost = np.where(stay, full["768"][ix], np.where(probe, probe_cost[ix] + full["1536"][ix], full["1536"][ix]))
        saved = 1 - cost.sum() / full["1536"][ix].sum()
        bad = (stay & ~y768[ix]).mean()
        return saved, bad, probe.mean(), stay.mean()

    grid_a = np.append(np.linspace(0.0, 0.9, 19), 1.01)
    grid_b = np.append(np.linspace(0.3, 0.99, 24), 1.01)
    best = (-1, None)
    for ta in grid_a:
        for tb in grid_b:
            saved, bad, _, _ = run(idx["val"], ta, tb)
            if bad <= a.budget and saved > best[0]:
                best = (saved, (ta, tb))
    ta, tb = best[1]
    saved, bad, probed, stayed = run(idx["test"], ta, tb)
    print(f"cascade (768 or 1536): thresholds image {ta:.2f}, probe {tb:.2f}; test time saved {saved:.1%} with "
          f"{bad:.1%} mis-routed; probed {probed:.0%}, kept at 768 {stayed:.0%}")
    oracle = np.where(y768[idx["test"]], full["768"][idx["test"]], full["1536"][idx["test"]])
    print(f"oracle 768-or-1536: test time saved {1 - oracle.sum() / full['1536'][idx['test']].sum():.1%}")
    for budget in (0.02, 0.03, 0.05, 0.08, 0.10):
        best = (-1, None)
        for x in grid_a:
            for z in grid_b:
                s_, b_, _, _ = run(idx["val"], x, z)
                if b_ <= budget and s_ > best[0]:
                    best = (s_, (x, z))
        s_, b_, pr, st = run(idx["test"], *best[1])
        print(f"   budget {budget:.0%}: test saved {s_:.1%}, mis-routed {b_:.1%}, probed {pr:.0%}, kept {st:.0%}")


if __name__ == "__main__":
    main()
