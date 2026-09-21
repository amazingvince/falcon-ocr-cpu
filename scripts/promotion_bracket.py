#!/usr/bin/env python3
"""Control / candidate / control latency bracket for a default-runtime promotion.

Each case runs three fresh `examples/ocr_bench.rs` processes in the frozen
order control-before, candidate, control-after (all on an otherwise idle
machine), then checks:

- every measured request's token IDs, text, stop reason and dimensions are
  identical across the three processes (and, when a pinned GPU record is
  given, identical to that record);
- control drift `|before - after| / min` is within the limit;
- the candidate's median latency against each control, reporting whether it
  meets the target reduction and whether it regresses beyond the limit.

Only the standard library is used. The bracket refuses an existing output
directory so partial evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

KIND = "promotion-bracket-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_command(binary: Path, case: dict, extra: list[str], output: Path, args) -> list[str]:
    return [str(binary), "--model", str(args.model), "--threads", str(args.threads), "--backend", args.backend,
            "--execution", case["execution"], "--weight-layout", "unpacked", *extra,
            "--batches", case["batches"], "--warmup", str(case["warmup"]), "--repetitions", str(case["repetitions"]),
            "--min-dimension", "64", "--max-dimension", str(case["max_dimension"]),
            "--max-new-tokens", str(case["max_new_tokens"]), "--cpu-label", args.cpu_label,
            "--environment-label", args.environment_label, "--output", str(output), *case["images"]]


def signatures(report: dict) -> list[dict]:
    """One signature per (case, sample, request): the fields that must be exact."""
    out = []
    for case in report["cases"]:
        for sample in case["samples"]:
            for index, request in enumerate(sample["per_request"]):
                out.append({
                    "batch_size": case["batch_size"], "image_index": case["image_indices"][index],
                    "token_ids": case["token_ids"][index] if len(case["token_ids"]) > index else None,
                    "text": request["text"], "finish_reason": request["finish_reason"],
                    "width": request["width"], "height": request["height"],
                    "input_tokens": request["input_tokens"], "output_tokens": request["output_tokens"],
                })
    return out


def case_medians(report: dict) -> dict:
    """Median latency per batch case: `median_ms` as ocr_bench reports it."""
    return {str(case["batch_size"]): float(case["median_ms"]) for case in report["cases"]}


def stage_medians(report: dict) -> dict:
    out = {}
    for case in report["cases"]:
        if case["batch_size"] != 1:
            continue
        prefill = [s["per_request"][0]["timings"]["prefill_ms"] for s in case["samples"]]
        decode = [s["per_request"][0]["timings"]["decode_ms"] for s in case["samples"]]
        out["prefill_ms"] = statistics.median(prefill)
        out["decode_ms"] = statistics.median(decode)
    return out


def evaluate(before: dict, candidate: dict, after: dict, drift_limit: float, target: float,
             regression_limit: float, gpu_records: dict) -> dict:
    result = {"cases": {}, "outputs_exact": True, "gpu_exact": None, "problems": []}
    sig_before, sig_cand, sig_after = signatures(before), signatures(candidate), signatures(after)
    if not (sig_before == sig_cand == sig_after):
        result["outputs_exact"] = False
        result["problems"].append("measured outputs differ between control/candidate/control")
    if gpu_records:
        checks = []
        for sig in sig_cand:
            record = gpu_records.get(str(sig["image_index"]))
            if record is None:
                continue
            same = sig["token_ids"] == record["token_ids"] and sig["text"] == record["text"] and sig["finish_reason"] == record["finish_reason"]
            checks.append(same)
            if not same:
                result["problems"].append(f"image {sig['image_index']} differs from the pinned GPU record")
        result["gpu_exact"] = all(checks) if checks else None
    mb, mc, ma = case_medians(before), case_medians(candidate), case_medians(after)
    for batch in mb:
        drift = abs(mb[batch] - ma[batch]) / min(mb[batch], ma[batch])
        change_before = (mc[batch] - mb[batch]) / mb[batch]
        change_after = (mc[batch] - ma[batch]) / ma[batch]
        result["cases"][batch] = {
            "control_before_ms": mb[batch], "candidate_ms": mc[batch], "control_after_ms": ma[batch],
            "control_drift_pct": drift * 100.0, "drift_ok": drift <= drift_limit,
            "change_vs_before_pct": change_before * 100.0, "change_vs_after_pct": change_after * 100.0,
            "meets_target": max(change_before, change_after) <= -target,
            "regresses_beyond_limit": max(change_before, change_after) > regression_limit,
        }
    result["stages"] = {"control_before": stage_medians(before), "candidate": stage_medians(candidate),
                        "control_after": stage_medians(after)}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True, help="control ocr_bench.exe")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate ocr_bench.exe")
    parser.add_argument("--control-flags", default="--cache-layout expanded", help="extra flags for the control binary")
    parser.add_argument("--candidate-flags", default="--cache-layout compact", help="extra flags for the candidate binary")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--backend", default="avx2")
    parser.add_argument("--cpu-label", default="AMD Ryzen 9 7950X")
    parser.add_argument("--environment-label", default="Native Windows")
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of cases")
    parser.add_argument("--drift-limit", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--regression-limit", type=float, default=0.05)
    parser.add_argument("--process-timeout", type=float, default=3600.0)
    parser.add_argument("--attestation", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    bracket = {
        "kind": KIND, "label": args.label, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "attestation": args.attestation, "threads": args.threads, "backend": args.backend,
        "control": {"binary": str(args.control), "sha256": sha256_file(args.control), "flags": args.control_flags},
        "candidate": {"binary": str(args.candidate), "sha256": sha256_file(args.candidate), "flags": args.candidate_flags},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "limits": {"control_drift": args.drift_limit, "target_reduction": args.target, "regression": args.regression_limit},
        "cases": [],
    }
    order = [("control-before", args.control, args.control_flags), ("candidate", args.candidate, args.candidate_flags),
             ("control-after", args.control, args.control_flags)]
    try:
        for case in cases:
            entry = {"name": case["name"], "definition": case, "jobs": {}}
            reports = {}
            for job, binary, flags in order:
                output = args.output / f"{case['name']}-{job}.json"
                command = job_command(binary, case, flags.split(), output, args)
                (args.output / f"{case['name']}-{job}.command.json").write_text(
                    json.dumps({"command": command, "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}, indent=1), encoding="utf-8")
                print(f"[{case['name']}/{job}] {subprocess.list2cmdline(command)}", flush=True)
                start = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                           timeout=args.process_timeout)
                (args.output / f"{case['name']}-{job}.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
                entry["jobs"][job] = {"exit_code": completed.returncode, "wall_s": time.perf_counter() - start,
                                      "output": str(output), "sha256": sha256_file(output) if output.exists() else None}
                if completed.returncode != 0:
                    raise RuntimeError(f"{case['name']}/{job} exited with {completed.returncode}")
                reports[job] = json.loads(output.read_text(encoding="utf-8"))
            gpu_records = {}
            for index, path in (case.get("gpu_records") or {}).items():
                gpu_records[str(index)] = json.load(open(path, encoding="utf-8"))
            entry["evaluation"] = evaluate(reports["control-before"], reports["candidate"], reports["control-after"],
                                           args.drift_limit, args.target, args.regression_limit, gpu_records)
            bracket["cases"].append(entry)
    finally:
        bracket["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        (args.output / "bracket.json").write_text(json.dumps(bracket, indent=2), encoding="utf-8")
        lines = [f"# {KIND}: {args.label}", ""]
        lines.append("| case | batch | control before | candidate | control after | drift % | vs before % | vs after % | ≥ target | regresses |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for entry in bracket["cases"]:
            ev = entry.get("evaluation", {})
            for batch, row in ev.get("cases", {}).items():
                lines.append(f"| {entry['name']} | {batch} | {row['control_before_ms'] / 1000:.3f} s | {row['candidate_ms'] / 1000:.3f} s | "
                             f"{row['control_after_ms'] / 1000:.3f} s | {row['control_drift_pct']:.2f} | {row['change_vs_before_pct']:+.2f} | "
                             f"{row['change_vs_after_pct']:+.2f} | {row['meets_target']} | {row['regresses_beyond_limit']} |")
            lines.append("")
            lines.append(f"{entry['name']}: outputs exact across processes = {ev.get('outputs_exact')}, GPU record exact = {ev.get('gpu_exact')}, "
                         f"stages = {json.dumps(ev.get('stages'))}; problems = {ev.get('problems')}")
            lines.append("")
        (args.output / "bracket.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {args.output / 'bracket.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
