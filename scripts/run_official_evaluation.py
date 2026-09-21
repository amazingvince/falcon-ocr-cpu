#!/usr/bin/env python3
"""Preflight both complete runs, prepare fresh inputs, audit evaluation, compare."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from compare_official_evaluation import compare, require
from prepare_official_evaluation import prepare, validate_source_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=Path("artifacts/OmniDocBench-eval"))
    parser.add_argument("--evaluation-python", type=Path, default=Path("/home/amazi/falcon-ocr-evaluation/.venv/bin/python"))
    parser.add_argument("--drop-dangling-truncated-relations", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    checked, errors = {}, []
    for runtime in ("cpu", "gpu"):
        try:
            checked[runtime] = validate_source_run(args.manifest, getattr(args, runtime), runtime)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"runtime": runtime, "error": str(error)})
    if errors:
        print(json.dumps({"status": "inputs_incomplete_or_invalid", "evaluation_started": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    cpu, gpu = checked["cpu"][2], checked["gpu"][2]
    for key in ("model_revision", "weights_sha256", "precision"):
        require(cpu[key] == gpu[key], f"CPU/GPU contract differs: {key}")
    for key in ("min_dimension", "max_dimension", "max_new_tokens"):
        require(cpu["options"][key] == gpu[key], f"CPU/GPU options differ: {key}")
    if args.preflight_only:
        print(json.dumps({"status": "inputs_complete", "pages": len(checked["cpu"][0]["pages"]), "evaluation_started": False}, indent=2))
        return
    require(not args.report.exists(), "Report already exists; refusing to replace durable scores")
    require(not args.output_root.exists() or not any(args.output_root.iterdir()), "Evaluation output root must be new or empty")
    require(args.evaluation_python.is_file(), "Pinned evaluation Python is missing; run setup_evaluation.sh")
    project = Path(__file__).resolve().parents[1]
    prepared = {}
    for runtime in ("gpu", "cpu"):
        destination = args.output_root / runtime
        prepared[runtime] = destination
        prepare(SimpleNamespace(manifest=args.manifest, predictions=getattr(args, runtime), runtime=runtime,
                                output=destination, evaluator=args.evaluator,
                                drop_dangling_truncated_relations=args.drop_dangling_truncated_relations))
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONHASHSEED="0")
    for runtime, destination in prepared.items():
        subprocess.run([str(args.evaluation_python), str(project / "scripts/run_evaluation_checked.py"),
                        "--prepared", str(destination), "--manifest", str(args.manifest), "--runtime", runtime,
                        "--evaluator", str(args.evaluator)], env=environment, check=True)
    report = compare(prepared["gpu"], prepared["cpu"], args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "pages": report["pages"], "report": str(args.report)}, indent=2))
    if report["all_prediction_text_bytes_equal"] and not report["all_evaluator_outputs_equal"]:
        raise SystemExit("Identical predictions produced different evaluator results")


if __name__ == "__main__":
    main()
