#!/usr/bin/env python3
"""Run teacher-forced KL arms against a stored FP32 top-K reference, one process each.

  python attempt3/kl_sweep.py --out artifacts/phase4/agree/kl/sweep1 \
      --pages artifacts/phase4/agree/kl-dev-pages.txt \
      --reference-topk artifacts/phase4/agree/kl/fp32-top.json \
      --arm "gptq=--profile w8-body-kv-q8 --w8-artifact artifacts/model/w8-gptq.safetensors" \
      --arm "w2fp32=--profile w8-body-kv-q8 --w8-artifact ... --keep-fp32 *.w2.weight"

Each arm is `name=<global options>`. Arms whose report already exists are
skipped, so an interrupted sweep resumes. Prints mean KL (FP32 || arm) per
step and flips, and writes `summary.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=Path, required=True)
    ap.add_argument("--reference-topk", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=Path("artifacts/phase4/checks/calibration-reference.json"))
    ap.add_argument("--binary", default="artifacts/phase4/bin/falcon-ocr-attempt-kl2.exe")
    ap.add_argument("--max-steps", default="384")
    ap.add_argument("--threads", default="32")
    ap.add_argument("--decode-threads", default="16")
    ap.add_argument("--arm", action="append", required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pages = [l.strip() for l in a.pages.read_text(encoding="utf-8").splitlines() if l.strip()]
    binary = str(Path(a.binary).resolve())
    summary = {}
    for spec in a.arm:
        name, options = spec.split("=", 1)
        report = a.out / f"{name}.json"
        if not report.exists():
            cmd = [binary, "--threads", a.threads, "--decode-threads", a.decode_threads, *shlex.split(options),
                   "agree", *pages, "--reference", str(a.reference), "--max-steps", a.max_steps,
                   "--reference-topk", str(a.reference_topk), "--report", str(report)]
            t0 = time.time()
            with open(a.out / f"{name}.stderr", "w", encoding="utf-8") as err:
                code = subprocess.call(cmd, env=dict(os.environ, FALCON_OCR_EXP="fast"),
                                       stdout=subprocess.DEVNULL, stderr=err)
            if code != 0:
                print(f"{name}: exit {code}", flush=True)
                continue
            elapsed = time.time() - t0
        else:
            elapsed = 0.0
        j = json.loads(report.read_text(encoding="utf-8"))
        summary[name] = {"kl_mean": j["kl_mean"], "flips": j["flips"], "steps": j["steps"],
                         "keep_fp32": j.get("keep_fp32"), "options": options}
        print(f"{name:>16}: mean KL {j['kl_mean']:.4e}  flips {j['flips']:3d}/{j['steps']}  ({elapsed:.0f}s)", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
