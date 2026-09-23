#!/usr/bin/env python3
"""Compare several full-page runs of the same pages against one baseline run.

  python attempt3/compare_runs.py --lock reference/corpus-v3-evaluation-lock.json \
      --baseline prod-bf16=artifacts/reference/vllm-heldout-bf16-4096/tokens.json \
      --run fp32=artifacts/cpu/corpus-v3-fp32-4096-redecoded-v3 \
      --run fast=artifacts/phase4/checks/heldout200-fast-gptq.json [--output summary.json]

A run is a bench-style report (`inputs` + `samples[0].outputs`) or a directory
of per-page records (`<page>.json` with a `result`). Per category it prints
micro CER against `ground-truth.txt` (outer whitespace stripped, same for every
run) on the pages where the baseline ends at EOS and on the pages where every
run ends at EOS (loops excluded), pages token-identical to the baseline, token
edits, and pages that end differently from the baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import levenshtein  # noqa: E402


def load(spec: str) -> tuple[str, dict]:
    name, path = spec.split("=", 1)
    path = Path(path)
    if path.is_dir():
        pages = {}
        for f in path.glob("*.json"):
            record = json.loads(f.read_text(encoding="utf-8"))
            if "result" in record:
                pages[f.stem] = record["result"]
        return name, pages
    report = json.loads(path.read_text(encoding="utf-8"))
    return name, {Path(i["path"]).parent.name: o for i, o in zip(report["inputs"], report["samples"][0]["outputs"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    runs = dict(load(s) for s in [a.baseline, *a.run])
    base = a.baseline.split("=", 1)[0]
    truth, category = {}, {}
    for entry in lock:
        path = Path(entry["canonical_path"])
        truth[path.parent.name] = (path.parent / "ground-truth.txt").read_text(encoding="utf-8").strip()
        category[path.parent.name] = entry["category"]
    pages = list(truth)
    rows = {}
    for name, run in runs.items():
        for pid in pages:
            o, b = run[pid], runs[base][pid]
            rows[name, pid] = {
                "stop": o["finish_reason"], "tokens": len(o["token_ids"]),
                "edits": levenshtein(truth[pid], o["text"].strip()),
                "identical": o["token_ids"] == b["token_ids"],
                "token_edits": levenshtein(b["token_ids"], o["token_ids"]),
            }
    eos = [p for p in pages if runs[base][p]["finish_reason"] == "eos"]
    cats = sorted(set(category.values()))

    def cer(name, subset):
        chars = sum(len(truth[p]) for p in subset)
        return sum(rows[name, p]["edits"] for p in subset) / max(1, chars)

    summary = {"baseline": base, "pages": len(pages), "baseline_eos_pages": len(eos), "runs": {}}
    header = f"{'':>14}" + "".join(f"{n:>18}" for n in runs)
    print(f"{len(pages)} pages; baseline {base} ends at EOS on {len(eos)}")
    print(f"micro CER on baseline-EOS pages\n{header}")
    for c in ["ALL", *cats]:
        subset = eos if c == "ALL" else [p for p in eos if category[p] == c]
        print(f"{c + f' ({len(subset)})':>14}" + "".join(f"{cer(n, subset):>18.3%}" for n in runs))
    print(f"{'all pages':>14}" + "".join(f"{cer(n, pages):>18.3%}" for n in runs))
    common = [p for p in pages if all(rows[n, p]["stop"] == "eos" for n in runs)]
    print(f"\nmicro CER on pages where every run ends at EOS\n{header}")
    for c in ["ALL", *cats]:
        subset = common if c == "ALL" else [p for p in common if category[p] == c]
        print(f"{c + f' ({len(subset)})':>14}" + "".join(f"{cer(n, subset):>18.3%}" for n in runs))
    summary["common_eos_pages"] = len(common)
    print(f"\nagreement with {base}\n{header}")
    for c in ["ALL", *cats]:
        subset = pages if c == "ALL" else [p for p in pages if category[p] == c]
        print(f"{c + f' ({len(subset)})':>14}" + "".join(
            f"{sum(rows[n, p]['identical'] for p in subset):>18d}" for n in runs))
    print(f"{'token edits':>14}" + "".join(f"{sum(rows[n, p]['token_edits'] for p in pages):>18d}" for n in runs))
    print(f"{'tokens':>14}" + "".join(f"{sum(rows[n, p]['tokens'] for p in pages):>18d}" for n in runs))
    for stop in ["length", "repetition"]:
        print(f"{'stop ' + stop:>14}" + "".join(
            f"{sum(rows[n, p]['stop'] == stop for p in pages):>18d}" for n in runs))
    for name in runs:
        changed = [{"page": p, "category": category[p], "baseline": runs[base][p]["finish_reason"],
                    "run": rows[name, p]["stop"], "tokens": rows[name, p]["tokens"],
                    "cer_baseline": rows[base, p]["edits"] / max(1, len(truth[p])),
                    "cer_run": rows[name, p]["edits"] / max(1, len(truth[p]))}
                   for p in pages if rows[name, p]["stop"] != runs[base][p]["finish_reason"]]
        summary["runs"][name] = {
            "eos_micro_cer": cer(name, eos), "all_micro_cer": cer(name, pages),
            "common_eos_micro_cer": cer(name, common),
            "common_eos_category_cer": {c: cer(name, [p for p in common if category[p] == c]) for c in cats},
            "eos_category_cer": {c: cer(name, [p for p in eos if category[p] == c]) for c in cats},
            "identical_pages": sum(rows[name, p]["identical"] for p in pages),
            "identical_by_category": {c: sum(rows[name, p]["identical"] for p in pages if category[p] == c) for c in cats},
            "token_edits": sum(rows[name, p]["token_edits"] for p in pages),
            "stop_changes": changed,
        }
        if name != base and changed:
            print(f"\n{name}: ends differently from {base} on {len(changed)} pages")
            for s in changed:
                print(f"  {s['page']} {s['category']:<12} {s['baseline']:>6} -> {s['run']:<10} "
                      f"{s['tokens']:5d} tok  CER {s['cer_baseline']:.1%} -> {s['cer_run']:.1%}")
    if a.output:
        a.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
