#!/usr/bin/env python3
"""Check a held-out run against the pre-registered quality budget.

  python research/phase4-hillclimb/attempt3/check_heldout.py --candidate artifacts/phase4/checks/heldout200-fast-gptq.json \
      --reference-dir artifacts/cpu/corpus-v3-fp32-4096-redecoded-v3 \
      --lock reference/corpus-v3-evaluation-lock.json \
      --budget reference/phase4-quality-budget.json [--output summary.json]

The reference is the per-page FP32 corpus record directory (`<page>.json`
with a `result`); the candidate is a `falcon-ocr-attempt bench` report of the
same pages. CER is raw-text Levenshtein distance to the page's
`ground-truth.txt` over its length, micro-averaged over characters, with the
same treatment for both runs. Gates apply to pages where FP32 ends at EOS.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import first_divergence, levenshtein  # noqa: E402


def micro(pages, key):
    chars = sum(p["truth_chars"] for p in pages)
    return sum(p[key] * p["truth_chars"] for p in pages) / chars if chars else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--budget", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    budget = json.loads(a.budget.read_text(encoding="utf-8"))["criteria"]
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    report = json.loads(a.candidate.read_text(encoding="utf-8"))
    cand = {Path(i["path"]).parent.name: o for i, o in zip(report["inputs"], report["samples"][0]["outputs"])}
    pages = []
    for entry in lock:
        path = Path(entry["canonical_path"])
        pid = path.parent.name
        ref_record = json.loads((a.reference_dir / f"{pid}.json").read_text(encoding="utf-8"))
        ref = ref_record["result"]
        c = cand[pid]
        truth = (path.parent / "ground-truth.txt").read_text(encoding="utf-8")
        pages.append({
            "page": pid,
            "category": entry["category"],
            "reference_stop": ref["finish_reason"],
            "candidate_stop": c["finish_reason"],
            "reference_tokens": ref["output_tokens"],
            "candidate_tokens": c["output_tokens"],
            "identical": ref["token_ids"] == c["token_ids"],
            "first_divergence": first_divergence(ref["token_ids"], c["token_ids"]),
            "token_edits": levenshtein(ref["token_ids"], c["token_ids"]),
            "truth_chars": len(truth),
            "cer_reference": levenshtein(truth, ref["text"]) / max(1, len(truth)),
            "cer_candidate": levenshtein(truth, c["text"]) / max(1, len(truth)),
            "reference_wall_s": ref_record.get("wall_ms", 0) / 1000,
            "candidate_total_s": c["timings"]["total_ms"] / 1000,
        })
    eos = [p for p in pages if p["reference_stop"] == "eos"]
    overall_ref, overall_cand = micro(eos, "cer_reference"), micro(eos, "cer_candidate")
    by_cat = defaultdict(list)
    for p in eos:
        by_cat[p["category"]].append(p)
    categories = {c: {"pages": len(ps), "fp32": micro(ps, "cer_reference"), "candidate": micro(ps, "cer_candidate")}
                  for c, ps in sorted(by_cat.items())}
    stop_on_eos = [p["page"] for p in eos if p["candidate_stop"] == "repetition"]
    gates = {
        "overall": (overall_cand - overall_ref) * 100 <= budget["overall_micro_cer_increase_max_points"],
        "per_category": all((v["candidate"] - v["fp32"]) * 100 <= budget["per_category_micro_cer_increase_max_points"]
                            for v in categories.values()),
        "repetition_stop_on_eos": len(stop_on_eos) <= budget["repetition_stop_fires_on_fp32_eos_pages"],
    }
    summary = {
        "candidate": str(a.candidate), "reference": str(a.reference_dir), "budget": str(a.budget),
        "pages": len(pages), "fp32_eos_pages": len(eos),
        "gates": gates, "pass": all(gates.values()),
        "eos_micro_cer": {"fp32": overall_ref, "candidate": overall_cand},
        "eos_categories": categories,
        "repetition_stop_on_fp32_eos_pages": stop_on_eos,
        "identical_pages": sum(p["identical"] for p in pages),
        "identical_eos_pages": sum(p["identical"] for p in eos),
        "eos_token_edits": sum(p["token_edits"] for p in eos),
        "eos_reference_tokens": sum(p["reference_tokens"] for p in eos),
        "stop_changes": [{k: p[k] for k in ("page", "category", "reference_stop", "candidate_stop",
                                            "reference_tokens", "candidate_tokens", "cer_reference", "cer_candidate")}
                         for p in pages if p["reference_stop"] != p["candidate_stop"]],
        "all_pages_micro_cer": {"fp32": micro(pages, "cer_reference"), "candidate": micro(pages, "cer_candidate")},
        "tokens": {"fp32": sum(p["reference_tokens"] for p in pages), "candidate": sum(p["candidate_tokens"] for p in pages)},
        "time_s": {"fp32_recorded_wall": sum(p["reference_wall_s"] for p in pages),
                   "candidate_total": sum(p["candidate_total_s"] for p in pages)},
        "per_page": pages,
    }
    print(f"pages {len(pages)}, FP32 ends at EOS on {len(eos)}")
    print(f"GATE overall (EOS pages): micro CER {overall_ref:.3%} -> {overall_cand:.3%} "
          f"({(overall_cand - overall_ref) * 100:+.3f} pt, limit +{budget['overall_micro_cer_increase_max_points']}) "
          f"{'PASS' if gates['overall'] else 'FAIL'}")
    for c, v in categories.items():
        d = (v["candidate"] - v["fp32"]) * 100
        print(f"GATE {c:>13} ({v['pages']:3d} pages): {v['fp32']:.3%} -> {v['candidate']:.3%} ({d:+.3f} pt) "
              f"{'PASS' if d <= budget['per_category_micro_cer_increase_max_points'] else 'FAIL'}")
    print(f"GATE repetition stop on FP32-EOS pages: {len(stop_on_eos)} {stop_on_eos} "
          f"{'PASS' if gates['repetition_stop_on_eos'] else 'FAIL'}")
    print(f"=> {'PASS' if summary['pass'] else 'FAIL'}")
    print(f"info: identical to FP32 {summary['identical_pages']}/{len(pages)} "
          f"(EOS pages {summary['identical_eos_pages']}/{len(eos)}); EOS token edits "
          f"{summary['eos_token_edits']}/{summary['eos_reference_tokens']}; all-pages micro CER "
          f"{summary['all_pages_micro_cer']['fp32']:.2%} -> {summary['all_pages_micro_cer']['candidate']:.2%}; "
          f"tokens {summary['tokens']['fp32']} -> {summary['tokens']['candidate']}; "
          f"time {summary['time_s']['fp32_recorded_wall']:.0f}s -> {summary['time_s']['candidate_total']:.0f}s")
    for s in summary["stop_changes"]:
        print(f"info: stop change {s}")
    if a.output:
        a.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
