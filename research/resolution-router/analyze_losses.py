#!/usr/bin/env python3
"""Where the router loses and wins on the development set: per-page change
in error characters against ground truth (routed run minus 1536 run), by
category and route, the largest losses with their scores and statistics, and
whether losses sit near the decision boundary. Diagnostic only: any change
it suggests must be validated on pages the router has not seen.

  python research/resolution-router/analyze_losses.py --runs D:/falcon-draft/router-dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_features import FEATURES  # noqa: E402


def outputs(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    return {Path(i["path"]).parent.name: o for i, o in zip(r["inputs"], r["samples"][0]["outputs"], strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--lock", type=Path, default=Path("reference/router-dev-v1-evaluation-lock.json"))
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    fixed = {s: outputs(a.runs / f"fast-{s}.json") for s in (768, 1024, 1536)}
    auto = outputs(a.runs / "fast-auto.json")
    feats = np.load(a.runs / "phase2-dev-features.npy")
    rows = []
    for entry, f in zip(lock, feats, strict=True):
        pid = Path(entry["canonical_path"]).parent.name
        truth = Path(entry["ground_truth_path"]).read_text(encoding="utf-8")
        route = auto[pid]["route"]
        final = 1536 if route.get("safety_net") else route["max_dimension"]
        err = {s: Levenshtein.distance(truth, fixed[s][pid]["text"]) for s in (768, 1024, 1536)}
        rows.append({"page": pid, "category": entry["category"], "source": entry.get("source_family", ""),
                     "attrs": entry.get("attributes", {}), "chars": len(truth), "route": route["max_dimension"],
                     "final": final, "err": err, "delta": err[final] - err[1536],
                     "s768": route["score_768"], "s1024": route["score_1024"], "line": route["line_height_px"],
                     "f": f, "stop": {s: fixed[s][pid]["finish_reason"] for s in (768, 1024, 1536)},
                     "len": {s: len(fixed[s][pid]["text"]) for s in (768, 1024, 1536)}})
    chars = sum(r["chars"] for r in rows)
    print(f"{len(rows)} pages, {chars} truth chars; net change {sum(r['delta'] for r in rows):+d} error chars "
          f"({100 * sum(r['delta'] for r in rows) / chars:+.2f} pt)")
    for key in ("category", "route"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        print(f"\nby {key}: pages, net error chars (pt of the group), losses > 2 pt, gains > 2 pt")
        for k, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
            c = sum(r["chars"] for r in g)
            d = sum(r["delta"] for r in g)
            lose = sum(r["delta"] / r["chars"] > 0.02 for r in g)
            gain = sum(r["delta"] / r["chars"] < -0.02 for r in g)
            print(f"  {str(k):14} {len(g):3}  {d:+7d} ({100 * d / c:+.2f} pt)  lose {lose:2}  gain {gain:2}")
    routed = [r for r in rows if r["route"] != 1536]
    total_loss = sum(max(0, r["delta"]) for r in routed)
    total_gain = -sum(min(0, r["delta"]) for r in routed)
    print(f"\nrouted pages {len(routed)}: gross loss {total_loss} chars, gross gain {total_gain} chars")
    losers = sorted(routed, key=lambda r: -r["delta"])[:a.top]
    print(f"\nlargest losses (share of gross loss: {sum(r['delta'] for r in losers) / max(1, total_loss):.0%})")
    for r in losers:
        margin = r["s768"] if r["route"] == 768 else r["s1024"]
        print(f"  {r['page'][:28]:28} {r['category']:12} route {r['route']:4} margin {margin:+.2f} "
              f"line {r['line']:5.1f}px delta {r['delta']:+5d} ({100 * r['delta'] / r['chars']:+.1f} pt)  "
              f"CER 1536 {r['err'][1536] / r['chars']:.1%} -> {r['err'][r['final']] / r['chars']:.1%}  "
              f"len {r['len'][1536]}->{r['len'][r['final']]}  "
              f"stop {r['stop'][r['final']]}  {r['attrs'].get('data_source', '')}")
    winners = sorted(routed, key=lambda r: r["delta"])[:8]
    print("\nlargest gains")
    for r in winners:
        print(f"  {r['page'][:28]:28} {r['category']:12} route {r['route']:4} delta {r['delta']:+5d} "
              f"({100 * r['delta'] / r['chars']:+.1f} pt)  CER 1536 {r['err'][1536] / r['chars']:.1%} -> "
              f"{r['err'][r['final']] / r['chars']:.1%}  len {r['len'][1536]}->{r['len'][r['final']]}  "
              f"{r['attrs'].get('data_source', '')}")
    # Are losses near the decision boundary?
    print("\nrouted pages by margin (score of the chosen model): pages, net chars, losses > 2 pt")
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]
    for lo, hi in bins:
        g = [r for r in routed if lo <= (r["s768"] if r["route"] == 768 else r["s1024"]) < hi]
        if g:
            print(f"  [{lo}, {hi}): {len(g):3}  {sum(r['delta'] for r in g):+6d}  "
                  f"{sum(r['delta'] / r['chars'] > 0.02 for r in g)}")
    # Which statistics separate losing routed pages from the rest of the routed pages?
    lose = np.array([r["delta"] / r["chars"] > 0.02 for r in routed])
    x = np.stack([r["f"] for r in routed])
    print(f"\nstatistics of routed losers ({lose.sum()}) vs other routed pages ({(~lose).sum()}): "
          "medians, rank-biserial")
    out = []
    for j, name in enumerate(FEATURES):
        a_, b_ = x[lose, j], x[~lose, j]
        # Rank-biserial correlation (Mann-Whitney U based effect size).
        ranks = np.argsort(np.argsort(np.concatenate([a_, b_]))) + 1
        u = ranks[:len(a_)].sum() - len(a_) * (len(a_) + 1) / 2
        effect = 1 - 2 * u / (len(a_) * len(b_))
        out.append((abs(effect), name, np.median(a_), np.median(b_), effect))
    for _, name, ma, mb, signed in sorted(out, reverse=True)[:8]:
        print(f"  {name:16} losers {ma:10.4f}  others {mb:10.4f}  effect {signed:+.2f}")


if __name__ == "__main__":
    main()
