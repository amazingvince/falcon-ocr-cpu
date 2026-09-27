#!/usr/bin/env python3
"""Phase 0 of the resolution router: how many pages keep their 1536-px
transcript at lower resolution.

For every sampled page and each lower resolution: the character edit distance
between its transcript and the 1536-px one, divided by the 1536 transcript's
length ("disagreement"), whether the stop reason is the same, and the token
counts. A page is routable to a resolution when the disagreement is within
the tolerance and the stop reason is unchanged; its oracle label is the
lowest such resolution. Page types come from the 1536 transcript and the
source (LaTeX, table markup, line count, density).

Decision rule (fixed before the lower-resolution transcripts existed): go if
at least half of the ordinary documents are routable within 2% at 1024 or
768. Ordinary documents: pages from olmocr_documents, olmocr_books, pdfa and
idl whose 1536 transcript has no LaTeX and no table markup.

Modelled page time (fast mode with the draft head, the router plan's fits,
n image tokens, t output tokens): prefill 0.33 n + 0.000049 n^2 ms and
decode t * (4.3 + 0.00064 n) / 1.47 ms.

  python research/resolution-router/analyze_agreement.py --router D:/falcon-draft/router
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from PIL import Image
from rapidfuzz.distance import Levenshtein

ORDINARY_SOURCES = {"olmocr_documents", "olmocr_books", "pdfa", "idl"}
LATEX = re.compile(r"\$|\\\(|\\\[|\\begin\{|\\frac|\\sum|\\int")
TABLE = re.compile(r"<table|<tr>|^\s*\|.*\|\s*$", re.MULTILINE)


def image_tokens(path: Path) -> int:
    with Image.open(path) as img:
        w, h = img.size
    return round(h / 16) * round(w / 16) + 16


def page_seconds(n: int, t: int) -> float:
    return (0.33 * n + 0.000049 * n * n + t * (4.3 + 0.00064 * n) / 1.47) / 1000


def page_type(source: str, text: str) -> str:
    if source.startswith("notes_"):
        return "handwriting"
    if source == "zenodo_slides":
        return "slides"
    if source == "loc_newspapers":
        return "newspaper"
    if LATEX.search(text):
        return "formulas"
    if TABLE.search(text):
        return "tables"
    return "prose"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, default=Path("D:/falcon-draft/outputs/stage3"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 768])
    ap.add_argument("--tolerances", type=float, nargs="+", default=[0.02, 0.05])
    a = ap.parse_args()
    sample = [json.loads(l) for l in (a.router / "sample.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for p in sample:
        name = p["id"].replace("/", "__") + ".json"
        base = json.loads((a.outputs / name).read_text(encoding="utf-8"))
        row = {"id": p["id"], "source": p["source"], "type": page_type(p["source"], base["text"]),
               "chars": len(base["text"]), "stop": {1536: base["finish_reason"]},
               "tokens": {1536: len(base["token_ids"])}, "image_tokens": {1536: image_tokens(Path(p["path"]))},
               "disagreement": {}}
        complete = True
        for size in a.sizes:
            f = a.router / f"outputs-{size}" / name
            if not f.exists():
                complete = False
                break
            r = json.loads(f.read_text(encoding="utf-8"))
            row["stop"][size] = r["finish_reason"]
            row["tokens"][size] = len(r["token_ids"])
            row["image_tokens"][size] = image_tokens(a.router / f"pages-{size}" / name.replace(".json", ".png"))
            row["disagreement"][size] = Levenshtein.distance(r["text"], base["text"]) / max(1, len(base["text"]))
        if complete:
            rows.append(row)
    print(f"{len(rows)} of {len(sample)} pages have every resolution")

    def routable(row, size, tol):
        return row["disagreement"][size] <= tol and row["stop"][size] == row["stop"][1536]

    def label(row, tol):
        for size in sorted(a.sizes):
            if routable(row, size, tol):
                return size
        return 1536

    def report(title, groups):
        print(f"\n{title}")
        head = "".join(f"  <= {int(t * 100)}% @{s}" for t in a.tolerances for s in a.sizes)
        print(f"{'group':18} {'pages':>6}{head}   median disagreement {' / '.join(str(s) for s in a.sizes)}   "
              f"stop changes   modelled time saved (oracle 2%)")
        for name, rs in groups:
            if not rs:
                continue
            fr = "".join(f"  {sum(routable(r, s, t) for r in rs) / len(rs):9.1%}" for t in a.tolerances for s in a.sizes)
            med = " / ".join(f"{sorted(r['disagreement'][s] for r in rs)[len(rs) // 2]:.1%}" for s in a.sizes)
            stops = sum(any(r["stop"][s] != r["stop"][1536] for s in a.sizes) for r in rs)
            base_t = sum(page_seconds(r["image_tokens"][1536], r["tokens"][1536]) for r in rs)
            routed = sum(page_seconds(r["image_tokens"][label(r, 0.02)], r["tokens"][label(r, 0.02)]) for r in rs)
            print(f"{name:18} {len(rs):6}{fr}   {med:>24}   {stops:12}   {1 - routed / base_t:10.1%}")

    report("By source", sorted(((s, [r for r in rows if r["source"] == s]) for s in {r["source"] for r in rows})))
    report("By page type", sorted(((t, [r for r in rows if r["type"] == t]) for t in {r["type"] for r in rows})))
    ordinary = [r for r in rows if r["source"] in ORDINARY_SOURCES and r["type"] == "prose"]
    report("Ordinary documents (decision set)", [("ordinary", ordinary)])
    share = sum(any(routable(r, s, 0.02) for s in a.sizes) for r in ordinary) / max(1, len(ordinary))
    print(f"\nDECISION: {share:.1%} of {len(ordinary)} ordinary documents are routable within 2% at 1024 or 768 -> "
          f"{'GO' if share >= 0.5 else 'NO-GO (flat 1280 default)'}")
    labels = collections.Counter(label(r, 0.02) for r in rows)
    print(f"oracle labels at 2%, all pages: {dict(sorted(labels.items()))}")
    (a.router / "phase0-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
