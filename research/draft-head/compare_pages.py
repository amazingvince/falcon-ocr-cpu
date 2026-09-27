#!/usr/bin/env python3
"""Compare page outputs of two cpu_ab.py arms (same pages, same order).

Per page: identical tokens or not, the first diverging token, output lengths,
the finish reason, and the character error rate of each arm's text against
the FP32 calibration reference when the page has one. Prints a summary and
every changed page.

  python research/draft-head/compare_pages.py D:/falcon-draft/cpu-ab/fastexp-calib/nospec-r0.jsonl \\
      D:/falcon-draft/cpu-ab/fastexp-calib/nospec-fast-r0.jsonl --pages D:/falcon-draft/calib-english-16.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REFERENCE = Path("artifacts/phase4/checks/calibration-reference.json")


def edit_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--pages", type=Path, required=True)
    a = ap.parse_args()
    pages = [p.strip() for p in a.pages.read_text().splitlines() if p.strip()]
    ra, rb = load(a.a), load(a.b)
    assert len(ra) == len(rb) == len(pages), (len(ra), len(rb), len(pages))
    reference = {}
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
        for inp, out in zip(ref["inputs"], ref["samples"][0]["outputs"]):
            reference[Path(inp["path"]).parent.name] = out["text"]
    changed, tokens, cer_a, cer_b, n_ref = [], 0, 0.0, 0.0, 0
    for page, x, y in zip(pages, ra, rb):
        pid = Path(page).parent.name
        tokens += len(x["token_ids"])
        ref = reference.get(pid)
        ca = cb = None
        if ref is not None:
            ca = edit_distance(x["text"], ref) / max(len(ref), 1)
            cb = edit_distance(y["text"], ref) / max(len(ref), 1)
            cer_a, cer_b, n_ref = cer_a + ca, cer_b + cb, n_ref + 1
        if x["token_ids"] != y["token_ids"]:
            first = next((i for i, (s, t) in enumerate(zip(x["token_ids"], y["token_ids"])) if s != t),
                         min(len(x["token_ids"]), len(y["token_ids"])))
            between = edit_distance(x["text"], y["text"]) / max(len(x["text"]), 1)
            changed.append((pid, first, len(x["token_ids"]), len(y["token_ids"]), x["finish_reason"],
                            y["finish_reason"], between, ca, cb))
    print(f"{len(pages)} pages, {tokens} tokens in A; {len(changed)} pages differ")
    if n_ref:
        print(f"mean CER vs FP32 reference over {n_ref} pages: A {100 * cer_a / n_ref:.3f}%  B {100 * cer_b / n_ref:.3f}%")
    for pid, first, la, lb, fa, fb, between, ca, cb in changed:
        ref = "" if ca is None else f"; CER vs FP32 {100 * ca:.2f}% -> {100 * cb:.2f}%"
        print(f"  {pid}: first differing token {first} of {la} -> {lb} tokens, finish {fa} -> {fb}, "
              f"A vs B text distance {100 * between:.2f}%{ref}")


if __name__ == "__main__":
    main()
