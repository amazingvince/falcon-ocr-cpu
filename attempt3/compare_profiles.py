#!/usr/bin/env python3
"""Token agreement and speed of candidate profiles against an FP32 run.

Inputs are `falcon-ocr-attempt bench` reports that each ran the same page
list in one process (one sample). For every page this reports whether the
candidate selected exactly the FP32 tokens, the first divergent token, the
token edit distance, CER of the candidate text against the FP32 text, and
CER of both against the page's `ground-truth.txt` (raw text, exact
Levenshtein, same treatment for both runs), plus per-page speed.

  python attempt3/compare_profiles.py --reference fp32.json \
      --candidate w8q8.json --candidate w8bf16.json \
      [--lock reference/corpus-v3-calibration-lock.json] [--output summary.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import first_divergence, levenshtein  # noqa: E402


DIVERGENT_KEYS = ("page", "category", "first_divergence", "reference_tokens", "candidate_tokens",
                  "reference_stop", "candidate_stop", "token_edit_distance", "cer_vs_reference",
                  "cer_truth_reference", "cer_truth_candidate")


def outputs(report: dict) -> dict[str, dict]:
    sample = report["samples"][0]
    paths = [item["path"] for item in report["inputs"]]
    return dict(zip(paths, sample["outputs"]))


def speed(o: dict) -> dict:
    t = o["timings"]
    return {
        "prefill_s": t["prefill_ms"] / 1000,
        "decode_ms_per_token": t["decode_ms"] / max(1, o["output_tokens"] - 1),
        "total_s": t["total_ms"] / 1000,
        "tokens": o["output_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--lock", type=Path, help="corpus lock for page categories")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    categories: dict[str, str] = {}
    if args.lock:
        for page in json.loads(args.lock.read_text(encoding="utf-8"))["pages"]:
            categories[Path(page["canonical_path"]).parent.name] = page["category"]

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    ref = outputs(reference)
    summary = {"reference": str(args.reference), "profile": reference.get("profile"), "candidates": []}
    for path in args.candidate:
        report = json.loads(path.read_text(encoding="utf-8"))
        cand = outputs(report)
        pages = []
        for image, a in ref.items():
            b = cand.get(image)
            if b is None:
                continue
            page_dir = Path(image).parent
            page = {
                "page": page_dir.name,
                "category": categories.get(page_dir.name, "full-page"),
                "identical": a["token_ids"] == b["token_ids"],
                "first_divergence": first_divergence(a["token_ids"], b["token_ids"]),
                "reference_tokens": a["output_tokens"],
                "candidate_tokens": b["output_tokens"],
                "reference_stop": a["finish_reason"],
                "candidate_stop": b["finish_reason"],
                "token_edit_distance": levenshtein(a["token_ids"], b["token_ids"]),
                "cer_vs_reference": levenshtein(a["text"], b["text"]) / max(1, len(a["text"])),
                "reference_speed": speed(a),
                "candidate_speed": speed(b),
            }
            truth_path = page_dir / "ground-truth.txt"
            if truth_path.is_file():
                truth = truth_path.read_text(encoding="utf-8")
                page["cer_truth_reference"] = levenshtein(truth, a["text"]) / max(1, len(truth))
                page["cer_truth_candidate"] = levenshtein(truth, b["text"]) / max(1, len(truth))
                page["truth_chars"] = len(truth)
            pages.append(page)
        identical = sum(p["identical"] for p in pages)
        truth_pages = [p for p in pages if "cer_truth_reference" in p]
        chars = sum(p["truth_chars"] for p in truth_pages) or 1
        micro_ref = sum(p["cer_truth_reference"] * p["truth_chars"] for p in truth_pages) / chars
        micro_cand = sum(p["cer_truth_candidate"] * p["truth_chars"] for p in truth_pages) / chars
        by_stop = {}
        for stop in ("eos", "length"):
            ps = [p for p in pages if p["reference_stop"] == stop]
            tp = [p for p in ps if "cer_truth_reference" in p]
            n = sum(p["truth_chars"] for p in tp) or 1
            by_stop[stop] = {
                "pages": len(ps),
                "identical": sum(p["identical"] for p in ps),
                "token_edits": sum(p["token_edit_distance"] for p in ps),
                "reference_tokens": sum(p["reference_tokens"] for p in ps),
                "micro_cer_truth_reference": sum(p["cer_truth_reference"] * p["truth_chars"] for p in tp) / n,
                "micro_cer_truth_candidate": sum(p["cer_truth_candidate"] * p["truth_chars"] for p in tp) / n,
            }
        by_category = defaultdict(list)
        for p in pages:
            by_category[p["category"]].append(p)
        entry = {
            "candidate": str(path),
            "profile": report.get("profile"),
            "pages": len(pages),
            "identical_pages": identical,
            "token_edit_distance_total": sum(p["token_edit_distance"] for p in pages),
            "reference_tokens_total": sum(p["reference_tokens"] for p in pages),
            "mean_cer_vs_reference": statistics.mean(p["cer_vs_reference"] for p in pages),
            "max_cer_vs_reference": max(p["cer_vs_reference"] for p in pages),
            "micro_cer_truth_reference": micro_ref,
            "micro_cer_truth_candidate": micro_cand,
            "stop_changes": sum(p["reference_stop"] != p["candidate_stop"] for p in pages),
            "reference_total_s": sum(p["reference_speed"]["total_s"] for p in pages),
            "candidate_total_s": sum(p["candidate_speed"]["total_s"] for p in pages),
            "by_reference_stop": by_stop,
            "by_category": {
                c: {
                    "pages": len(ps),
                    "identical": sum(p["identical"] for p in ps),
                    "mean_cer_vs_reference": statistics.mean(p["cer_vs_reference"] for p in ps),
                }
                for c, ps in sorted(by_category.items())
            },
            "divergent_pages": [
                {k: p[k] for k in DIVERGENT_KEYS if k in p}
                for p in pages if not p["identical"]
            ],
            "per_page": pages,
        }
        summary["candidates"].append(entry)
        print(f"== {entry['profile']} vs {summary['profile']}: {identical}/{len(pages)} pages token-identical; "
              f"token edits {entry['token_edit_distance_total']}/{entry['reference_tokens_total']}; "
              f"mean CER vs FP32 {entry['mean_cer_vs_reference']:.4%} (max {entry['max_cer_vs_reference']:.4%}); "
              f"micro CER vs truth {micro_ref:.4%} -> {micro_cand:.4%}; stop changes {entry['stop_changes']}; "
              f"time {entry['reference_total_s']:.0f}s -> {entry['candidate_total_s']:.0f}s")
        for stop, v in by_stop.items():
            print(f"   reference stop {stop:>6}: {v['identical']}/{v['pages']} identical, token edits "
                  f"{v['token_edits']}/{v['reference_tokens']}, micro CER vs truth "
                  f"{v['micro_cer_truth_reference']:.4%} -> {v['micro_cer_truth_candidate']:.4%}")
        for c, v in entry["by_category"].items():
            print(f"   {c:>20}: {v['identical']}/{v['pages']} identical, mean CER vs FP32 {v['mean_cer_vs_reference']:.4%}")
        for d in entry["divergent_pages"]:
            print(f"   diverges: {d}")
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
