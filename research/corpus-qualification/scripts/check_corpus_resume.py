#!/usr/bin/env python3
"""Exercise build-bound corpus resume semantics with one capped CPU inference."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--build-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", default="reference/corpus-v3-evaluation-lock.json")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("use new output/report paths; prior evidence is never overwritten")
    command = [str(args.binary.resolve()), "--manifest", args.manifest,
               "--limit", "1", "--max-dimension", "256", "--max-new-tokens", "1",
               "--backend", "avx2", "--threads", "4", "--cpu-label", "local validation CPU",
               "--environment-label", "functional resume regression; not performance",
               "--build-manifest", str(args.build_manifest), "--output", str(args.output)]
    checks = []

    def run(name, selected, accepted, message=None):
        result = subprocess.run(selected, capture_output=True, text=True, encoding="utf-8")
        passed = (result.returncode == 0) == accepted
        if message is not None:
            passed = passed and message in result.stderr
        checks.append({"name": name, "passed": passed, "exit_code": result.returncode,
                       "stdout": result.stdout, "stderr": result.stderr})
        if not passed:
            raise RuntimeError(f"{name} failed: {result.stdout}\n{result.stderr}")

    run("fresh_build_bound_inference", command, True)
    run_path = args.output / "run.json"
    before = sha(run_path)
    original = json.loads(run_path.read_text(encoding="utf-8"))
    assert original["contract"]["binary_sha256"] == sha(args.binary)
    assert original["contract"]["build_manifest_sha256"] == sha(args.build_manifest)
    run("same_binary_resume", command + ["--resume"], True)
    assert before == sha(run_path), "resume rewrote the original run record"
    invocations = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted((args.output / "invocations").glob("*.json"))]
    assert len(invocations) == 2
    assert invocations[0]["new_completed_pages"] == 1 and invocations[0]["resumed_completed_pages"] == 0
    assert invocations[1]["new_completed_pages"] == 0 and invocations[1]["resumed_completed_pages"] == 1
    run("existing_output_requires_resume", command, False, "output already contains a run")
    changed = list(command)
    changed[changed.index("--environment-label") + 1] = "different run identity"
    run("changed_contract_rejected", changed + ["--resume"], False, "resume contract changed")
    invalid = json.loads(args.build_manifest.read_text(encoding="utf-8"))
    invalid["binary_sha256"] = "0" * 64
    negative = args.output / "invalid-build-for-test.json"
    negative.write_text(json.dumps(invalid), encoding="utf-8")
    changed = list(command)
    changed[changed.index("--build-manifest") + 1] = str(negative)
    run("different_build_binary_rejected", changed + ["--resume"], False,
        "build snapshot belongs to a different executable")
    assert before == sha(run_path)
    assert len(list((args.output / "invocations").glob("*.json"))) == 2
    report = {"schema_version": 1, "passed": True, "checks": checks,
              "binary_sha256": sha(args.binary), "build_manifest_sha256": sha(args.build_manifest),
              "script_sha256": sha(pathlib.Path(__file__)), "manifest_sha256": sha(pathlib.Path(args.manifest)),
              "original_run_record_sha256": before, "run_record_unchanged_after_resume_and_rejections": True,
              "invocations": invocations,
              "scope": "One CPU page at 256px and one output token; build/resume bookkeeping only, no quality or performance qualification."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "checks": len(checks), "report": str(args.report)}))


if __name__ == "__main__":
    main()
