#!/usr/bin/env python3
"""Blinded LLM-as-judge packets for OCR outputs, and the unblinded report.

  python attempt3/judge_pages.py build --lock reference/corpus-v3-evaluation-lock.json \
      --category handwriting --out artifacts/phase4/judge/handwriting \
      --run prod-bf16=artifacts/reference/vllm-heldout-bf16-4096/tokens.json \
      --run fp32=artifacts/cpu/corpus-v3-fp32-4096-redecoded-v3 ...
  python attempt3/judge_pages.py report --out artifacts/phase4/judge/handwriting

`build` writes one packet per page (`packets/<page>.md`): the image path, the
ground truth and every distinct output under a shuffled letter, plus
`key.json` (letter -> runs; never shown to judges). Identical outputs share a
letter. `--repeat N` also writes packets for N pages with fresh letters
(`packets/<page>.repeat.md`) to measure judge consistency. Judges write
`results/<packet name>.json`; `report` unblinds and aggregates them.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import load  # noqa: E402

FINISH = {"eos": "ended normally", "length": "hit the 4,096-token limit",
          "repetition": "stopped early by a repetition detector"}
VERDICTS = ["equivalent", "minor", "moderate", "major", "failed"]


def packet(pid, entry, root, candidates, letters):
    image = (root / entry["canonical_path"]).resolve()
    truth = (root / entry["canonical_path"]).parent.joinpath("ground-truth.txt").read_text(encoding="utf-8").strip()
    lines = [f"# Page {pid}", "", f"Category: {entry['category']}; language: {entry['attributes'].get('language')}",
             "", f"Image (read it): `{image}`", "", "## Ground truth (human annotation)", "", "```text", truth, "```", ""]
    for letter in letters:
        text, finish = candidates[letter]
        cap = max(3000, 3 * len(truth))
        shown = text.strip()
        note = ""
        if len(shown) > cap:
            note = f"\n[... truncated here: {len(shown) - cap} further characters not shown]"
            shown = shown[:cap]
        lines += [f"## Output {letter} ({FINISH.get(finish, finish)}; {len(text.strip())} characters)", "",
                  "```text", shown + note, "```", ""]
    return "\n".join(lines)


def build(a):
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    runs = dict(load(s) for s in a.run)
    root = Path.cwd()
    (a.out / "packets").mkdir(parents=True, exist_ok=True)
    (a.out / "results").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    key = {}
    pages = [e for e in lock if e["category"] == a.category]
    for entry in pages:
        pid = Path(entry["canonical_path"]).parent.name
        distinct = defaultdict(list)
        for name, run in runs.items():
            o = run[pid]
            distinct[(o["text"].strip(), o["finish_reason"])].append(name)
        groups = list(distinct.items())
        rng.shuffle(groups)
        letters = [chr(ord("A") + i) for i in range(len(groups))]
        candidates = {l: g[0] for l, g in zip(letters, groups)}
        key[pid] = {l: g[1] for l, g in zip(letters, groups)}
        (a.out / "packets" / f"{pid}.md").write_text(packet(pid, entry, root, candidates, letters), encoding="utf-8")
    for entry in rng.sample(pages, a.repeat):
        pid = Path(entry["canonical_path"]).parent.name
        original = key[pid]
        # Same outputs, new letters.
        texts = {l: (runs[names[0]][pid]["text"].strip(), runs[names[0]][pid]["finish_reason"])
                 for l, names in original.items()}
        old = list(original)
        new = old[:]
        while len(new) > 1 and new == old:
            rng.shuffle(new)
        remap = dict(zip(old, new))
        candidates = {remap[l]: texts[l] for l in old}
        key[f"{pid}.repeat"] = {remap[l]: original[l] for l in old}
        (a.out / "packets" / f"{pid}.repeat.md").write_text(
            packet(pid, entry, root, candidates, sorted(candidates)), encoding="utf-8")
    (a.out / "key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    print(f"{len(pages)} pages (+{a.repeat} repeats) -> {a.out / 'packets'}")


def report(a):
    key = json.loads((a.out / "key.json").read_text(encoding="utf-8"))
    per_run = defaultdict(lambda: {"scores": [], "verdicts": Counter(), "content_errors": 0, "loops": 0,
                                   "formatting": Counter(), "best": 0})
    repeats, missing = [], []
    results = {}
    for name, letters in key.items():
        path = a.out / "results" / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        results[name] = json.loads(path.read_text(encoding="utf-8"))
    for name, r in results.items():
        if name.endswith(".repeat"):
            continue
        cands = r["candidates"]
        top = max(c["content_score"] for c in cands.values())
        for letter, runs in key[name].items():
            c = cands[letter]
            for run in runs:
                s = per_run[run]
                s["scores"].append(c["content_score"])
                s["verdicts"][c["verdict"]] += 1
                s["content_errors"] += int(c.get("content_errors", 0))
                s["loops"] += bool(c.get("repetition_loop"))
                s["formatting"].update(c.get("formatting_differences", []))
                s["best"] += c["content_score"] == top
    for name, r in results.items():
        if not name.endswith(".repeat") or name[:-7] not in results:
            continue
        first, second = results[name[:-7]], r
        for letter, runs in key[name[:-7]].items():
            other = next(l for l, rr in key[name].items() if rr == runs)
            repeats.append((first["candidates"][letter]["content_score"], second["candidates"][other]["content_score"],
                            first["candidates"][letter]["verdict"], second["candidates"][other]["verdict"]))
    judged = sum(1 for n in results if not n.endswith(".repeat"))
    print(f"judged pages: {judged}; missing: {missing}")
    print(f"{'run':>14} {'mean score':>10} {'median':>7} {'best':>5} {'errors':>7} {'loops':>6}  verdicts")
    summary = {}
    for run, s in sorted(per_run.items(), key=lambda kv: -statistics.mean(kv[1]["scores"])):
        v = " ".join(f"{k}:{s['verdicts'][k]}" for k in VERDICTS if s["verdicts"][k])
        print(f"{run:>14} {statistics.mean(s['scores']):>10.1f} {statistics.median(s['scores']):>7.1f} "
              f"{s['best']:>5d} {s['content_errors']:>7d} {s['loops']:>6d}  {v}")
        summary[run] = {"mean_score": statistics.mean(s["scores"]), "median_score": statistics.median(s["scores"]),
                        "best_or_tied": s["best"], "content_errors": s["content_errors"], "loops": s["loops"],
                        "verdicts": dict(s["verdicts"]), "formatting": dict(s["formatting"].most_common())}
    for run, s in summary.items():
        print(f"{run:>14} formatting: {s['formatting']}")
    if repeats:
        diffs = [abs(x - y) for x, y, _, _ in repeats]
        same = sum(v1 == v2 for _, _, v1, v2 in repeats)
        print(f"judge consistency on {len(repeats)} repeated candidates: mean |score diff| {statistics.mean(diffs):.1f}, "
              f"max {max(diffs)}, same verdict {same}/{len(repeats)}")
        summary["_consistency"] = {"pairs": repeats, "mean_abs_diff": statistics.mean(diffs), "same_verdict": same}
    (a.out / "report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lock", type=Path, required=True)
    b.add_argument("--category", required=True)
    b.add_argument("--run", action="append", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=20260923)
    b.add_argument("--repeat", type=int, default=5)
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"build": build, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
