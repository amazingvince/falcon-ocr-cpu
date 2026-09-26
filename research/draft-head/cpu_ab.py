#!/usr/bin/env python3
"""CPU A/B of speculative drafters with the main binary (fast mode, packed file).

Arms: no speculation, the n-gram drafter, the trained draft head, and both.
Each (arm, round) is one `falcon-ocr run` over all pages; rounds alternate
the arm order. Reports per arm the median over rounds of total and decode
time, decode ms per token, drafted/accepted counts, and whether every page's
tokens equal the no-speculation arm's (speculation must never change output).

  python research/draft-head/cpu_ab.py --pages D:/falcon-draft/heldout-english-16.txt \
      --head D:/falcon-draft/heads/text-twophase.safetensors --out D:/falcon-draft/cpu-ab/heldout16
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

BINARY = str(Path("target/release/falcon-ocr.exe").resolve())
MODEL = "artifacts/packed/falcon-ocr-v1.5-fast.safetensors"
SPEC = re.compile(r"speculation: (\d+) tokens; (\d+) single steps \(([\d.]+) ms\), (\d+) verify steps \(([\d.]+) ms\),\s+"
                  r"drafted (\d+), accepted (\d+)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=Path, required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--decode-threads", default="12")
    ap.add_argument("--confidence", default="0.45")
    ap.add_argument("--max-new-tokens", default="4096")
    ap.add_argument("--arm", action="append", default=[],
                    help="name=extra flags (replaces the default arms; `nospec` is always added); "
                         "a first flag `@<exe>` runs that binary instead of target/release")
    a = ap.parse_args()
    pages = [p.strip() for p in a.pages.read_text().splitlines() if p.strip()]
    arms = {
        "nospec": ["--speculate", "0"],
        "ngram": ["--speculate", "4", "--drafter", "ngram"],
        "head": ["--speculate", "4", "--drafter", "head", "--draft-head", a.head, "--draft-confidence", a.confidence],
        "both": ["--speculate", "4", "--drafter", "both", "--draft-head", a.head, "--draft-confidence", a.confidence],
    }
    if a.arm:
        custom = {"nospec": arms["nospec"]}
        for spec in a.arm:
            name, flags = spec.split("=", 1)
            custom[name] = flags.replace("{head}", a.head).split()
        arms = custom
    a.out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[dict]] = {name: [] for name in arms}
    for r in range(a.rounds):
        order = list(arms) if r % 2 == 0 else list(reversed(arms))
        for name in order:
            # An arm may set its own --decode-threads.
            threads = [] if "--decode-threads" in arms[name] else ["--decode-threads", a.decode_threads]
            flags = arms[name]
            binary = BINARY
            if flags and flags[0].startswith("@"):
                binary, flags = str(Path(flags[0][1:]).resolve()), flags[1:]
            base = [binary, "--mode", "fast", "--model-file", MODEL, *threads,
                    "--document-drafts=false", "--tune", "phases=1"]
            cmd = base + flags + ["run", *pages, "--max-new-tokens", a.max_new_tokens]
            out, err = a.out / f"{name}-r{r}.jsonl", a.out / f"{name}-r{r}.stderr"
            t0 = time.time()
            with out.open("w", encoding="utf-8") as fo, err.open("w", encoding="utf-8") as fe:
                code = subprocess.call(cmd, stdout=fo, stderr=fe)
            wall = time.time() - t0
            results = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            spec = [tuple(map(float, m.groups())) for m in SPEC.finditer(err.read_text(encoding="utf-8"))]
            runs[name].append({"code": code, "wall": wall, "results": results, "spec": spec})
            print(f"round {r} {name}: exit {code}, {wall:.1f} s, {len(results)} pages", flush=True)
    reference = runs["nospec"][0]["results"]
    summary = {}
    for name, rs in runs.items():
        ok = [x for x in rs if x["code"] == 0 and len(x["results"]) == len(pages)]
        if not ok:
            continue
        identical = all(res["token_ids"] == ref["token_ids"] for x in ok for res, ref in zip(x["results"], reference))
        decode = statistics.median(sum(p["timings"]["decode_ms"] for p in x["results"]) for x in ok) / 1e3
        total = statistics.median(sum(p["timings"]["total_ms"] for p in x["results"]) for x in ok) / 1e3
        tokens = sum(p["output_tokens"] for p in ok[0]["results"])
        drafted = sum(s[5] for s in ok[0]["spec"])
        accepted = sum(s[6] for s in ok[0]["spec"])
        verifies = sum(s[3] for s in ok[0]["spec"])
        summary[name] = {"total_s": round(total, 2), "decode_s": round(decode, 2), "tokens": tokens,
                         "decode_ms_per_token": round(1e3 * decode / max(tokens, 1), 3),
                         "verify_steps": verifies, "drafted": drafted, "accepted": accepted,
                         "tokens_identical_to_nospec": identical}
    base = summary.get("nospec", {})
    for name, s in summary.items():
        speed = base.get("decode_s", 0) / s["decode_s"] if s["decode_s"] else 0
        print(f"{name:7} total {s['total_s']:7.1f} s  decode {s['decode_s']:7.1f} s  {s['decode_ms_per_token']:.2f} ms/token "
              f"(decode {speed:.2f}x vs nospec)  verify steps {s['verify_steps']:.0f}  drafted {s['drafted']:.0f} "
              f"accepted {s['accepted']:.0f}  identical {s['tokens_identical_to_nospec']}")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
