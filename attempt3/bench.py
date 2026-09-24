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

PROFILES = {"reference", "hygiene", "split-f32", "kv-bf16", "kv-q8", "w8-body", "w8-all",
            "w8-body-kv-bf16", "w8-body-kv-q8", "w8-all-kv-bf16", "w8-all-kv-q8"}

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

def levenshtein(a, b) -> int:
    """Exact Levenshtein distance of two sequences of hashable items.

    Bit-parallel (Myers 1999 / Hyyro 2003) with Python integers as bit vectors:
    O(len(a) * len(b) / word) instead of the O(len(a) * len(b)) table, so
    whole-page texts and token-ID lists are exact and fast.
    """
    if len(a) < len(b):
        a, b = b, a
    m = len(a)
    if m == 0:
        return len(b)
    peq: dict = {}
    for i, item in enumerate(a):
        peq[item] = peq.get(item, 0) | (1 << i)
    mask = (1 << m) - 1
    last = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for item in b:
        eq = peq.get(item, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score

def first_divergence(a, b) -> int | None:
    """Index of the first differing item, None for identical sequences."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))

def edit_distance(a: str, b: str, max_cells: int | None = None) -> int | None:
    """Exact Levenshtein; optional cell bound; never substitute an approximate CER."""
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
    if max_cells is not None and len(a) * len(b) > max_cells:
        return None
    return levenshtein(a, b)

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

def arm_stats(report: dict) -> dict:
    """Median stage timings of one fresh-process arm, per page and overall."""
    samples = report["samples"]
    pages = []
    for i in range(len(samples[0]["outputs"])):
        rows = [s["outputs"][i] for s in samples]
        tokens = rows[0]["output_tokens"]
        prefill = statistics.median(r["timings"]["prefill_ms"] for r in rows)
        decode = statistics.median(r["timings"]["decode_ms"] for r in rows)
        pages.append({"prefill_ms": prefill, "decode_ms": decode, "output_tokens": tokens,
                      "decode_ms_per_token": decode / max(1, tokens - 1),
                      "total_ms": statistics.median(r["timings"]["total_ms"] for r in rows)})
    telemetry = [s.get("telemetry", {}) for s in samples]
    memory = report.get("process_memory") or {}
    return {"profile": report.get("profile"), "wall_ms": statistics.median(s["wall_ms"] for s in samples),
            "wall_ms_samples": [s["wall_ms"] for s in samples], "pages": pages,
            "prefix_seal_ms": statistics.median(t.get("prefix_seal_ms", 0.0) for t in telemetry),
            "kv_allocated_high_water": max(t.get("kv_allocated_high_water", 0) for t in telemetry),
            "load_and_import_ms": report.get("load_and_import_ms"),
            "peak_resident_bytes": memory.get("peak_resident_bytes"),
            "peak_private_commit_bytes": memory.get("peak_private_commit_bytes")}

def divergence(a: dict, b: dict) -> dict:
    """Token and text drift of candidate output b against reference output a."""
    ids_a, ids_b = a["token_ids"], b["token_ids"]
    text_ed = edit_distance(a["text"], b["text"])
    return {"first_divergence_token": first_divergence(ids_a, ids_b),
            "token_edit_distance": levenshtein(ids_a, ids_b),
            "text_edit_distance": text_ed,
            "text_cer_vs_reference": text_ed / max(1, len(a["text"]))}

def compare(before: dict, candidate: dict, after: dict, ground_truth: list[str] | None = None,
            candidate_binary_may_differ: bool = False) -> dict:
    for report in (before, candidate, after):
        validate_report(report)
    keys = ("weights_sha256", "model_revision", "threads", "backend", "batch_size", "options", "inputs")
    if any(before[k] != other[k] for k in keys for other in (candidate, after)):
        raise ValueError("comparison has mismatched input/model/binary/execution settings")
    if before["binary_sha256"] != after["binary_sha256"] or (
            not candidate_binary_may_differ and before["binary_sha256"] != candidate["binary_sha256"]):
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
        if not exact:
            detail.update(divergence(a, b))
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
            "pages": details, "arms": {"before": arm_stats(before), "candidate": arm_stats(candidate),
                                       "after": arm_stats(after)},
            "candidate_active_rows_histograms": [s["telemetry"]["active_rows_histogram"] for s in candidate["samples"]],
            "candidate_memory_policy": candidate.get("memory_policy")}

def schedule(profiles: list[str], kind: str, control_every: int) -> tuple[list[str], list[tuple[int, int, int]]]:
    """Arm sequence and (before, candidate, after) index triples.

    bracket: reference/candidate/reference per profile (6 processes for 2 profiles).
    interleaved: shared controls, reference first, after every `control_every`
    candidates and last; each candidate is judged against its nearest controls.
    """
    if kind == "bracket":
        arms, triples = [], []
        for profile in profiles:
            base = len(arms)
            arms += ["reference", profile, "reference"]
            triples.append((base, base + 1, base + 2))
        return arms, triples
    # Controls are tracked by position: a candidate may itself be "reference"
    # (for example a new build compared with an old one).
    arms, pending, triples, last_control = ["reference"], [], [], 0
    for i, profile in enumerate(profiles):
        pending.append(len(arms))
        arms.append(profile)
        if (i + 1) % control_every == 0 or i + 1 == len(profiles):
            arms.append("reference")
            triples += [(last_control, k, len(arms) - 1) for k in pending]
            last_control = len(arms) - 1
            pending = []
    return arms, triples

def run_arm(args, case: dict, mode: str, directory: Path, name: str, control: bool = False) -> dict:
    report = (directory / f"{name}.json").resolve()
    binary = args.control_binary if control and args.control_binary else args.binary
    command = [str(binary.resolve()), "--model", str(args.model.resolve()), "--profile", mode,
        "--threads", str(args.threads), "--backend", args.backend, "--batch-size", str(case["batch_size"]),
        "bench", *case["images"], "--max-new-tokens", str(case["max_new_tokens"]),
        "--max-dimension", str(case["max_dimension"]), "--warmup", str(args.warmup),
        "--samples", str(args.samples), "--report", str(report)]
    artifact = args.w8_all_artifact if mode.startswith("w8-all") else args.w8_body_artifact if mode.startswith("w8-body") else None
    if artifact is not None:
        at = command.index("bench")
        command[at:at] = ["--w8-artifact", str(artifact.resolve())]
    extra = args.control_args if control else args.candidate_args
    if extra:
        at = command.index("bench")
        command[at:at] = extra.split()
    (directory / f"{name}.command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    print(f"{case['id']} / {name}", flush=True)
    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=args.timeout_seconds)
    (directory / f"{name}.stdout.txt").write_text(run.stdout, encoding="utf-8")
    (directory / f"{name}.stderr.txt").write_text(run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(f"{name} exited {run.returncode}; see stderr")
    return json.loads(report.read_text(encoding="utf-8"))

def table(summary: dict) -> str:
    """Plain-text overview of every comparison; the JSON remains authoritative."""
    lines = ["case | profile | ctrl ms | cand ms | delta % | drift % | prefill s | decode ms/tok"
             " | exact | first div | tok ED | CER vs ref | CER gt ref/cand | peak RSS GB"]
    for c in summary["comparisons"]:
        if "arms" not in c:
            lines.append(f"{c.get('case')} | {c.get('profile')} | {c.get('status')}")
            continue
        cand = c["arms"]["candidate"]
        for i, page in enumerate(c["pages"]):
            cp = cand["pages"][i]
            rss = cand.get("peak_resident_bytes")
            gt = ""
            if page.get("reference_cer") is not None and page.get("candidate_cer") is not None:
                gt = f"{page['reference_cer']:.4f}/{page['candidate_cer']:.4f}"
            cer = page.get("text_cer_vs_reference")
            lines.append(" | ".join([str(c["case"]), str(c["profile"]),
                f"{(c['control_before_ms'] + c['control_after_ms']) / 2:.0f}", f"{c['candidate_ms']:.0f}",
                f"{-c['raw_latency_reduction_pct']:+.2f}", f"{c['control_drift_pct']:+.2f}",
                f"{cp['prefill_ms'] / 1000:.2f}", f"{cp['decode_ms_per_token']:.2f}",
                str(page["exact_reference_match"]), str(page.get("first_divergence_token", "")),
                str(page.get("token_edit_distance", "")), "" if cer is None else f"{cer:.4f}",
                gt, "" if rss is None else f"{rss / 2**30:.2f}"]))
    return "\n".join(lines) + "\n"

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
    parser.add_argument("--backend", choices=["auto", "scalar", "avx2"], default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--schedule", choices=["bracket", "interleaved"], default="bracket",
                        help="bracket: fresh reference before/after every profile; interleaved: shared controls")
    parser.add_argument("--control-every", type=int, default=2,
                        help="interleaved schedule: candidates between reference controls")
    parser.add_argument("--control-binary", type=Path,
                        help="run reference controls with this binary instead of --binary (A/B of builds)")
    parser.add_argument("--candidate-args", default="",
                        help="extra global options for candidate arms, e.g. '--head screened'")
    parser.add_argument("--control-args", default="", help="extra global options for control arms")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.control_binary is not None and not args.control_binary.is_file():
        parser.error("missing control binary")
    if not args.binary.is_file() or args.output.exists() or args.threads <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("check binary, fresh output directory, positive threads/samples and nonnegative warmup")
    if args.control_every <= 0:
        parser.error("control-every must be positive")
    for artifact in (args.w8_body_artifact, args.w8_all_artifact):
        if artifact is not None and not artifact.is_file():
            parser.error(f"missing W8 overlay: {artifact}")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    cases = load_manifest(args.manifest.resolve())
    args.output.mkdir(parents=True)
    summary: dict[str, Any] = {"schema": "falcon-ocr-attempt3-comparison-v1", "created_unix": time.time(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "schedule": args.schedule,
        "comparisons": [], "quality_qualified": False}
    failures = 0
    for case in cases:
        arms, triples = schedule(args.profiles, args.schedule, args.control_every)
        directory = args.output / case["id"]
        directory.mkdir(parents=True)
        reports: list[dict | None] = []
        errors: dict[int, str] = {}
        control_indices = {b for b, _, a in triples} | {a for _, _, a in triples}
        for index, mode in enumerate(arms):
            control = index in control_indices
            label = f"{index:02d}-{'control' if control else 'candidate'}-{mode}"
            try:
                reports.append(run_arm(args, case, mode, directory, label, control=control))
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                reports.append(None)
                errors[index] = str(exc)
        for before, candidate, after in triples:
            profile = arms[candidate]
            failed = [errors[i] for i in (before, candidate, after) if i in errors]
            if failed:
                result = {"case": case["id"], "profile": profile, "status": "execution failed", "errors": failed}
                failures += 1
            else:
                try:
                    result = {"case": case["id"], "arm_indices": [before, candidate, after],
                              **compare(reports[before], reports[candidate], reports[after],
                                        ground_truth=case.get("ground_truth"),
                                        candidate_binary_may_differ=args.control_binary is not None)}
                except (KeyError, ValueError) as exc:
                    result = {"case": case["id"], "profile": profile, "status": "comparison failed", "error": str(exc)}
                    failures += 1
            summary["comparisons"].append(result)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.txt").write_text(table(summary), encoding="utf-8")
    print(table(summary))
    print(args.output / "summary.json")
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
