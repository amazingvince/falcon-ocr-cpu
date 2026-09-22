#!/usr/bin/env python3
"""Bracketed full-page experiment runner. Each arm is a fresh native process.
Quality is fail-closed: output drift is reported, never silently counted as speed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROFILES = {"hygiene", "split-f32", "kv-bf16", "kv-q8", "w8-body", "w8-all",
            "w8-body-kv-bf16", "w8-all-kv-bf16", "w8-all-kv-q8"}

def load_manifest(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("schema") != "falcon-ocr-attempt3-cases-v1":
        raise ValueError("unexpected manifest schema")
    cases, seen = document.get("cases", []), set()
    if not cases:
        raise ValueError("manifest has no cases")
    for case in cases:
        identity = case.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity) or identity in seen:
            raise ValueError("case IDs must be unique safe basenames")
        seen.add(identity)
        images = case.get("images", [])
        batch = case.get("batch_size", 1)
        maximum = case.get("max_new_tokens", 4096)
        dimension = case.get("max_dimension", 1536)
        if not images or not isinstance(batch, int) or not 1 <= batch <= 8:
            raise ValueError(f"invalid images/batch in {identity}")
        if not isinstance(maximum, int) or maximum <= 0 or not isinstance(dimension, int) or dimension < 64 or dimension % 16:
            raise ValueError(f"invalid generation bounds in {identity}")
        resolved = [(path.parent / item).resolve() for item in images]
        if not all(p.is_file() for p in resolved):
            raise ValueError(f"missing input image in {identity}: {resolved}")
        case["images"] = [str(p) for p in resolved]
        case["batch_size"], case["max_new_tokens"], case["max_dimension"] = batch, maximum, dimension
        truth = case.get("ground_truth")
        if truth is not None:
            if len(truth) != len(images):
                raise ValueError("ground_truth must have one path per image")
            case["ground_truth"] = [(path.parent / item).read_text(encoding="utf-8") for item in truth]
    return cases

def edit_distance(a: str, b: str, max_cells: int = 20_000_000) -> int | None:
    """Exact two-row Levenshtein, bounded; never substitute an approximate CER."""
    if a == b:
        return 0
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    a, b = a[start:], b[start:]
    end = 0
    while end < min(len(a), len(b)) and a[-end - 1] == b[-end - 1]:
        end += 1
    if end:
        a, b = a[:-end], b[:-end]
    if not a or not b:
        return len(a) + len(b)
    if len(a) * len(b) > max_cells:
        return None
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        row = [i]
        for j, right in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = row
    return previous[-1]

def tokens_signature(output: dict) -> tuple:
    return (output["width"], output["height"], output["input_tokens"], output["finish_reason"],
            tuple(output["token_ids"]), output["text"])

def validate_report(report: dict) -> None:
    if report.get("schema") != "falcon-ocr-attempt3-report-v1" or not report.get("samples"):
        raise ValueError("invalid or empty native report")
    count = len(report.get("inputs", []))
    if not count:
        raise ValueError("report lacks input identities")
    for sample in report["samples"]:
        wall = sample.get("wall_ms", 0)
        if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("invalid sample wall time")
        if len(sample["outputs"]) != count:
            raise ValueError("missing page output")
        for item in sample["outputs"]:
            if item["output_tokens"] != len(item["token_ids"]) or item["output_tokens"] <= 0:
                raise ValueError("token count mismatch")
        if any(tokens_signature(a) != tokens_signature(b) for a, b in zip(report["samples"][0]["outputs"], sample["outputs"])):
            raise ValueError("outputs changed across repetitions of the same profile")

def compare(before: dict, candidate: dict, after: dict, ground_truth: list[str] | None = None) -> dict:
    for report in (before, candidate, after):
        validate_report(report)
    keys = ("weights_sha256", "model_revision", "threads", "backend", "batch_size", "options", "inputs", "binary_sha256")
    if any(before[k] != other[k] for k in keys for other in (candidate, after)):
        raise ValueError("comparison has mismatched input/model/binary/execution settings")
    if before.get("profile") != "reference" or after.get("profile") != "reference":
        raise ValueError("both bracket controls must use reference")
    outputs = [r["samples"][0]["outputs"] for r in (before, candidate, after)]
    if [tokens_signature(x) for x in outputs[0]] != [tokens_signature(x) for x in outputs[2]]:
        raise ValueError("reference outputs changed across bracket")
    medians = [statistics.median(s["wall_ms"] for s in r["samples"]) for r in (before, candidate, after)]
    drift = 100 * (medians[2] / medians[0] - 1)
    details = []
    for i, (a, b) in enumerate(zip(outputs[0], outputs[1])):
        equal_shape = all(a[k] == b[k] for k in ("width", "height", "input_tokens"))
        if not equal_shape:
            raise ValueError("candidate changed processed resolution or input length")
        exact = tokens_signature(a) == tokens_signature(b)
        numbers_a = re.findall(r"[-+]?\d+(?:[.,]\d+)*", a["text"])
        numbers_b = re.findall(r"[-+]?\d+(?:[.,]\d+)*", b["text"])
        detail = {"index": i, "exact_reference_match": exact, "reference_tokens": a["output_tokens"],
                  "candidate_tokens": b["output_tokens"], "reference_stop": a["finish_reason"],
                  "candidate_stop": b["finish_reason"], "numeric_string_sequence_equal": numbers_a == numbers_b,
                  "reference_is_complete": a["finish_reason"] == "eos",
                  "candidate_is_complete": b["finish_reason"] == "eos"}
        if ground_truth is not None:
            truth = ground_truth[i]
            for label, result in (("reference", a), ("candidate", b)):
                distance = edit_distance(truth, result["text"])
                detail[label + "_cer"] = None if distance is None else distance / max(1, len(truth))
            detail["cer_note"] = "exact Levenshtein; null means computation budget exceeded; no acceptance threshold inferred"
        details.append(detail)
    unchanged = all(x["exact_reference_match"] for x in details)
    complete = all(x["reference_is_complete"] and x["candidate_is_complete"] for x in details)
    control_mean = (medians[0] + medians[2]) / 2
    eligible = unchanged and complete and abs(drift) <= 5
    return {"profile": candidate["profile"], "control_before_ms": medians[0], "candidate_ms": medians[1],
            "control_after_ms": medians[2], "control_drift_pct": drift,
            "raw_latency_reduction_pct": 100 * (1 - medians[1] / control_mean),
            "raw_speedup": control_mean / medians[1], "same_output_complete_page_comparison": eligible,
            "quality_qualified": False,
            "status": "same-output evidence; still needs broader quality qualification" if eligible else "not eligible for a same-quality speed claim",
            "reasons": {"exact_output_match": unchanged, "complete_eos_outputs": complete, "control_drift_within_5pct": abs(drift) <= 5},
            "pages": details, "candidate_active_rows_histograms": [s["telemetry"]["active_rows_histogram"] for s in candidate["samples"]],
            "candidate_memory_policy": candidate.get("memory_policy")}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=["hygiene"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w8-body-artifact", type=Path, help="optional body-only W8G64 safetensors overlay")
    parser.add_argument("--w8-all-artifact", type=Path, help="optional body+head W8G64 safetensors overlay")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--backend", choices=["auto", "scalar", "avx2", "avx512"], default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if not args.binary.is_file() or args.output.exists() or args.threads <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("check binary, fresh output directory, positive threads/samples and nonnegative warmup")
    for artifact in (args.w8_body_artifact, args.w8_all_artifact):
        if artifact is not None and not artifact.is_file():
            parser.error(f"missing W8 overlay: {artifact}")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    cases = load_manifest(args.manifest.resolve())
    args.output.mkdir(parents=True)
    summary: dict[str, Any] = {"schema": "falcon-ocr-attempt3-comparison-v1", "created_unix": time.time(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "comparisons": [], "quality_qualified": False}
    failures = 0
    for case in cases:
        for profile in args.profiles:
            arms, errors = [], []
            directory = args.output / case["id"] / profile
            directory.mkdir(parents=True)
            for arm, mode in [("before", "reference"), ("candidate", profile), ("after", "reference")]:
                report = (directory / f"{arm}.json").resolve()
                command = [str(args.binary.resolve()), "--model", str(args.model.resolve()), "--profile", mode,
                    "--threads", str(args.threads), "--backend", args.backend, "--batch-size", str(case["batch_size"]),
                    "bench", *case["images"], "--max-new-tokens", str(case["max_new_tokens"]),
                    "--max-dimension", str(case["max_dimension"]), "--warmup", str(args.warmup),
                    "--samples", str(args.samples), "--report", str(report)]
                artifact = args.w8_all_artifact if mode.startswith("w8-all") else args.w8_body_artifact if mode.startswith("w8-body") else None
                if artifact is not None:
                    at = command.index("bench")
                    command[at:at] = ["--w8-artifact", str(artifact.resolve())]
                (directory / f"{arm}.command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
                print(f"{case['id']} / {profile} / {arm}", flush=True)
                try:
                    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=args.timeout_seconds)
                    (directory / f"{arm}.stdout.txt").write_text(run.stdout, encoding="utf-8")
                    (directory / f"{arm}.stderr.txt").write_text(run.stderr, encoding="utf-8")
                    if run.returncode:
                        raise RuntimeError(f"{arm} exited {run.returncode}; see stderr")
                    arms.append(json.loads(report.read_text(encoding="utf-8")))
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    errors.append(str(exc))
            if errors:
                result = {"case": case["id"], "profile": profile, "status": "execution failed", "errors": errors}
                failures += 1
            else:
                try:
                    result = {"case": case["id"], **compare(*arms, ground_truth=case.get("ground_truth"))}
                except (KeyError, ValueError) as exc:
                    result = {"case": case["id"], "profile": profile, "status": "comparison failed", "error": str(exc)}
                    failures += 1
            summary["comparisons"].append(result)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output / "summary.json")
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
