#!/usr/bin/env python3
"""Interleaved A/B/... runs of attempt binaries on pages, with per-arm env.

  python attempt3/ab.py --out DIR --rounds 3 --reference-dir artifacts/phase4/control-default-4096 \
      --arm "base=path/to/a.exe" --arm "fast=path/to/b.exe;--exp fast;--threads 32" \
      --args "--profile w8-body-kv-q8 --head screened --w8-artifact X" -- page1.png page2.png

Each run is a fresh process (`bench --warmup 0 --samples 1`) with
`--tune phases=1`; stderr is kept. An arm is `name=binary` followed by
`;KEY=VALUE` environment entries and `;--option value` global options, which
replace `--args` options of the same name (for example `;--threads 32`). Prints per-arm median prefill, decode
ms/token and total, plus token agreement with the reference token IDs.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import first_divergence, levenshtein  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--threads", default="16")
    ap.add_argument("--reference-dir", type=Path)
    ap.add_argument("pages", nargs="+")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        parts = rest.split(";")
        env = dict(p.split("=", 1) for p in parts[1:] if not p.startswith("--"))
        options = [shlex.split(p) for p in parts[1:] if p.startswith("--")]
        arms.append((name, str(Path(parts[0]).resolve()), env, options))
    results: dict[str, list] = {name: [] for name, _, _, _ in arms}
    for r in range(a.rounds):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for name, binary, env, options in order:
            report = a.out / f"{name}-r{r}.json"
            if report.exists():
                report.unlink()
            base = ["--threads", a.threads, "--tune", "phases=1", *shlex.split(a.args)]
            for option in options:
                if option[0] in base:
                    at = base.index(option[0])
                    del base[at:at + len(option)]
                base += option
            cmd = [binary, *base, "bench", *a.pages,
                   "--warmup", "0", "--samples", "1", "--report", str(report)]
            e = dict(os.environ, **env)
            t0 = time.time()
            with open(a.out / f"{name}-r{r}.stderr", "w", encoding="utf-8") as err:
                code = subprocess.call(cmd, env=e, stdout=subprocess.DEVNULL, stderr=err)
            print(f"round {r} {name}: exit {code} in {time.time() - t0:.1f}s", flush=True)
            if code != 0:
                continue
            j = json.loads(report.read_text(encoding="utf-8"))
            paths = [i["path"] for i in j["inputs"]]
            results[name].append(dict(zip(paths, j["samples"][0]["outputs"])))
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        pages = {}
        for path in runs[0]:
            outs = [run[path] for run in runs]
            t = [o["timings"] for o in outs]
            n = outs[0]["output_tokens"]
            page = {
                "prefill_s": statistics.median(x["prefill_ms"] for x in t) / 1000,
                "decode_ms_tok": statistics.median(x["decode_ms"] for x in t) / max(1, n - 1),
                "total_s": statistics.median(x["total_ms"] for x in t) / 1000,
                "tokens": n,
                "stable": all(o["token_ids"] == outs[0]["token_ids"] for o in outs),
            }
            if a.reference_dir:
                ref_file = a.reference_dir / f"{Path(path).parent.name}.json"
                if ref_file.is_file():
                    ref = json.loads(ref_file.read_text(encoding="utf-8"))["token_ids"]
                    ids = outs[0]["token_ids"]
                    page["identical"] = ids == ref
                    page["first_divergence"] = first_divergence(ref, ids)
                    page["token_edits"] = levenshtein(ref, ids)
            pages[Path(path).parent.name] = page
        summary[name] = pages
        for page, v in pages.items():
            print(f"{name:>10} {page}: prefill {v['prefill_s']:.2f}s decode {v['decode_ms_tok']:.2f}ms/tok "
                  f"total {v['total_s']:.2f}s tokens {v['tokens']} stable {v['stable']} "
                  f"identical {v.get('identical')} edits {v.get('token_edits')} first {v.get('first_divergence')}")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
