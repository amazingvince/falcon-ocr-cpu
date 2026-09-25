#!/usr/bin/env python3
"""Flag generated transcripts that share text with OmniDocBench.

A page is flagged when at least `--min-hits` distinct 13-word sequences
(lower-cased alphanumeric words, at least 8 different words each, so runs of
dots, zeros or table rules do not count) also occur in OmniDocBench's
annotated text.
Flagged ids are merged into the exclusion list that eagle3.py reads
(D:/falcon-draft/decontam-13gram.json), next to the image-hash blocklist the
page builder applies.

  python research/draft-head/data/decontam_13gram.py --outputs D:/falcon-draft/outputs/stage3
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--omnidocbench", type=Path,
                    default=Path("D:/falcon-draft/raw/opendatalab__OmniDocBench/OmniDocBench.json"))
    ap.add_argument("--exclusions", type=Path, default=Path("D:/falcon-draft/decontam-13gram.json"))
    ap.add_argument("--min-hits", type=int, default=3)
    a = ap.parse_args()
    grams = {}
    for page in json.loads(a.omnidocbench.read_text(encoding="utf-8")):
        name = page.get("page_info", {}).get("image_path", "?")
        text = " ".join(str(d.get("text") or d.get("latex") or d.get("html") or "") for d in page.get("layout_dets", []))
        w = words(text)
        for i in range(len(w) - 12):
            g = tuple(w[i:i + 13])
            if len(set(g)) >= 8:
                grams.setdefault(g, name)
    flagged = []
    for f in sorted(a.outputs.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        w = words(r["text"])
        matched = {tuple(w[i:i + 13]) for i in range(len(w) - 12)} & grams.keys()
        if len(matched) >= a.min_hits:
            source = collections.Counter(grams[g] for g in matched).most_common(1)[0][0]
            flagged.append((len(matched), r["id"], source))
    flagged.sort(reverse=True)
    known = set(json.loads(a.exclusions.read_text())) if a.exclusions.exists() else set()
    new = [i for _, i, _ in flagged if i not in known]
    a.exclusions.write_text(json.dumps(sorted(known | {i for _, i, _ in flagged})))
    print(f"{len(grams)} diverse OmniDocBench 13-grams; {len(flagged)} flagged pages ({len(new)} new); "
          f"{a.exclusions} now lists {len(known) + len(new)} ids")
    for n, page, source in flagged[:15]:
        print(f"  {n:5d} {page:28} <- {source}")


if __name__ == "__main__":
    main()
