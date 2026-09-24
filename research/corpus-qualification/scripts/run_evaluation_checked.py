#!/usr/bin/env python3
"""Run the pinned evaluator with per-page execution and output completeness audits."""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

from compare_official_evaluation import check_execution_audit, read, require, sha, validate_prepared, validate_results
from prepare_official_evaluation import EVALUATOR_REVISION

ERROR_MARKERS = ("!!!WARNING: No prediction", "Traceback (most recent call last)", "JSON 序列化错误",
                 "TEDS score error", "TEDS_structure_only score error")


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--evaluator", type=Path, default=Path("artifacts/OmniDocBench-eval"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    prepared, evaluator = args.prepared.resolve(), args.evaluator.resolve()
    provenance, _, image_names = validate_prepared(prepared, args.manifest, args.runtime, require_audit=False)
    require(not (prepared / "execution-audit.json").exists(), "Execution audit already exists; use a new preparation directory")
    require(not any((prepared / "result").iterdir()), "Result directory must be empty before execution")
    require(subprocess.check_output(["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True).strip() == EVALUATOR_REVISION, "Evaluator revision changed")
    require(not subprocess.check_output(["git", "-C", str(evaluator), "status", "--porcelain", "--untracked-files=no"], text=True).strip(), "Evaluator contains tracked modifications")
    requirements = project / "requirements/evaluation-resolved.txt"
    package_versions = {}
    for line in requirements.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = importlib.metadata.version(name)
        require(actual == expected, f"Evaluator dependency differs: {name} {actual} != {expected}")
        package_versions[name] = actual
    require(sys.version_info[:3] == (3, 10, 20), "Evaluator requires the pinned Python 3.10.20 runtime")
    audit = {"schema_version": 1, "status": "running", "exit_code": None, "expected_pages": image_names,
             "entered_pages": [], "completed_pages": [], "failures": [], "detected_error_lines": [],
             "preparation_provenance_sha256": sha(prepared / "provenance.json"), "evaluator_revision": EVALUATOR_REVISION,
             "wrapper_sha256": sha(__file__), "requirements_sha256": sha(requirements),
             "python_version": sys.version, "packages": package_versions,
             "instrumentation": "Wrap process_get_matched_elements only to record entry/normal return; matching inputs, return values and metric arithmetic are unchanged."}
    log_path = prepared / "execution.log"
    old_cwd, old_argv = Path.cwd(), sys.argv
    original = None
    dataset_class = None
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            sys.path.insert(0, str(evaluator))
            from dataset.end2end_dataset import End2EndDataset
            dataset_class = End2EndDataset
            original = dataset_class.process_get_matched_elements

            def audited_process(self, sample, pred_content, img_name, save_time):
                audit["entered_pages"].append(img_name)
                result = original(self, sample, pred_content, img_name, save_time)
                audit["completed_pages"].append(img_name)
                return result

            dataset_class.process_get_matched_elements = audited_process
            os.chdir(prepared)
            sys.argv = [str(evaluator / "pdf_validation.py"), "--config", str(prepared / "configuration.json")]
            runpy.run_path(str(evaluator / "pdf_validation.py"), run_name="__main__")
            audit["status"], audit["exit_code"] = "complete", 0
            check_execution_audit(audit, image_names, sha(prepared / "provenance.json"))
            result = validate_results(prepared, image_names)
            audit["result_file_sha256"] = {name: sha(path) for name, path in result["files"].items()}
            audit["component_counts"] = {name: {"pages": value["page_count"], "samples": value["sample_count"]} for name, value in result["components"].items()}
        except BaseException as error:
            # Upstream sys.exit() without an argument is an error here, even
            # though its OS exit status would otherwise be zero.
            audit["status"], audit["exit_code"] = "failed", 1
            audit["failures"].append({"type": type(error).__name__, "message": str(error)})
            traceback.print_exc()
        finally:
            if dataset_class is not None and original is not None:
                dataset_class.process_get_matched_elements = original
            os.chdir(old_cwd)
            sys.argv = old_argv
    audit["detected_error_lines"] = [line for line in log_path.read_text(encoding="utf-8").splitlines() if any(marker in line for marker in ERROR_MARKERS)]
    if audit["detected_error_lines"]:
        audit["status"], audit["exit_code"] = "failed", 1
    audit["execution_log_sha256"] = sha(log_path)
    (prepared / "execution-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "entered_pages": len(audit["entered_pages"]), "completed_pages": len(audit["completed_pages"]), "expected_pages": len(image_names)}, indent=2))
    raise SystemExit(audit["exit_code"])


if __name__ == "__main__":
    main()
