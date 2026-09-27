#!/usr/bin/env python3
"""Phase 1 against ground truth: routing policies on the router development
set (reference/router-dev-v1-*), from CPU fast-mode runs at 1536, 1280, 1024
and 768 (`falcon-ocr-eval bench` reports).

Per page and resolution: CER against the page's ground truth (raw-text
Levenshtein over its length, as check_heldout.py), the stop reason and the
measured CPU time. Policies: flat resolutions, the router plan's category
table on the corpus categories (formulas 1280; tables and multi-column 1024;
everything else 768), an image-model threshold (probabilities from
`--image-probs`, trained on the GPU labels), and the per-page oracle (lowest
resolution whose CER is at most the 1536 CER plus `--delta`). Reported per
policy: micro CER against 1536 (the plan's Phase 1 gate: within 0.5 pt),
per-category change, pages over 2 pt worse, stop-reason changes and CPU time
saved.

  python research/resolution-router/dev_policy.py --runs D:/falcon-draft/router-dev
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

SIZES = (768, 1024, 1280, 1536)
TABLE = {"formulas": 1280, "tables": 1024, "multi_column": 1024}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--lock", type=Path, default=Path("reference/router-dev-v1-evaluation-lock.json"))
    ap.add_argument("--image-probs", type=Path, help="JSON {page: {\"768\": p, \"1024\": p}} from the image model")
    ap.add_argument("--delta", type=float, default=0.005)
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    runs = {}
    for size in SIZES:
        f = a.runs / f"fast-{size}.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            runs[size] = {Path(i["path"]).parent.name: o for i, o in zip(r["inputs"], r["samples"][0]["outputs"])}
    sizes = [s for s in SIZES if s in runs]
    pages = []
    for entry in lock:
        pid = Path(entry["canonical_path"]).parent.name
        if any(pid not in runs[s] for s in sizes):
            continue
        truth = Path(entry["ground_truth_path"]).read_text(encoding="utf-8")
        pages.append({
            "page": pid, "category": entry["category"], "chars": len(truth),
            "cer": {s: Levenshtein.distance(truth, runs[s][pid]["text"]) / max(1, len(truth)) for s in sizes},
            "stop": {s: runs[s][pid]["finish_reason"] for s in sizes},
            "seconds": {s: runs[s][pid]["timings"]["total_ms"] / 1000 for s in sizes},
        })
    print(f"{len(pages)} pages with runs at {sizes}")
    probs = json.loads(a.image_probs.read_text()) if a.image_probs else {}

    def summarize(name, choose):
        chars = sum(p["chars"] for p in pages)
        base = sum(p["cer"][1536] * p["chars"] for p in pages) / chars
        routed = sum(p["cer"][choose(p)] * p["chars"] for p in pages) / chars
        worse = sum(p["cer"][choose(p)] - p["cer"][1536] > 0.02 for p in pages)
        stops = sum(p["stop"][choose(p)] != p["stop"][1536] for p in pages)
        t0, t1 = sum(p["seconds"][1536] for p in pages), sum(p["seconds"][choose(p)] for p in pages)
        mix = defaultdict(int)
        for p in pages:
            mix[choose(p)] += 1
        cats = defaultdict(lambda: [0.0, 0.0, 0])
        for p in pages:
            c = cats[p["category"]]
            c[0] += p["cer"][1536] * p["chars"]
            c[1] += p["cer"][choose(p)] * p["chars"]
            c[2] += p["chars"]
        per_cat = ", ".join(f"{k} {100 * (v[1] - v[0]) / v[2]:+.2f}" for k, v in sorted(cats.items()))
        print(f"{name:26} CER {base:.2%} -> {routed:.2%} ({100 * (routed - base):+.2f} pt)  pages >2 pt worse {worse:3}  "
              f"stop changes {stops:3}  CPU time saved {1 - t1 / t0:6.1%}  routes {dict(sorted(mix.items()))}")
        print(f"{'':26} per category (pt): {per_cat}")

    if 1536 not in runs:
        return
    for s in sizes:
        summarize(f"flat {s}", lambda p, s=s: s)
    if {768, 1024, 1280} <= set(sizes):
        summarize("category table", lambda p: TABLE.get(p["category"], 768))
    summarize("oracle", lambda p: next(s for s in sizes if p["cer"][s] <= p["cer"][1536] + a.delta))
    if probs:
        def floor_ok(p, size, px):
            # Median text line at least `px` pixels at the target resolution.
            q = probs.get(p["page"], {})
            long_side = max(q.get("width", 1536), q.get("height", 1536))
            return q.get("line_height_px", 99) * min(1.0, size / long_side) >= px

        def routed(p, t, px):
            q = probs.get(p["page"], {})
            if 768 in sizes and q.get("768", 0) >= t and floor_ok(p, 768, px):
                return 768
            if 1024 in sizes and q.get("1024", 0) >= t and floor_ok(p, 1024, px):
                return 1024
            return 1536

        def with_net(name, choose):
            # Safety net: a routed page that loops or hits the cap reruns at 1536 (both runs paid).
            chars = sum(p["chars"] for p in pages)
            base = sum(p["cer"][1536] * p["chars"] for p in pages) / chars
            final = [(p, choose(p)) for p in pages]
            final = [(p, r, r != 1536 and p["stop"][r] in ("repetition", "length")) for p, r in final]
            cer = sum(p["cer"][1536 if net else r] * p["chars"] for p, r, net in final) / chars
            t0 = sum(p["seconds"][1536] for p in pages)
            t1 = sum(p["seconds"][r] + (p["seconds"][1536] if net else 0) for p, r, net in final)
            worse = sum(p["cer"][1536 if net else r] - p["cer"][1536] > 0.02 for p, r, net in final)
            reruns = sum(net for _, _, net in final)
            print(f"{name:26} CER {base:.2%} -> {cer:.2%} ({100 * (cer - base):+.2f} pt)  pages >2 pt worse {worse:3}  "
                  f"reruns {reruns:3}  CPU time saved {1 - t1 / t0:6.1%}")

        for px in (6, 8, 10):
            for t in (0.4, 0.5, 0.6):
                with_net(f"image p>={t} floor {px}px +net", lambda p, t=t, px=px: routed(p, t, px))
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            def choose(p, t=t):
                q = probs.get(p["page"], {})
                if 768 in sizes and q.get("768", 0) >= t:
                    return 768
                if 1024 in sizes and q.get("1024", 0) >= t:
                    return 1024
                return 1536
            summarize(f"image model, p >= {t}", choose)


if __name__ == "__main__":
    main()
