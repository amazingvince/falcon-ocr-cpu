#!/usr/bin/env python3
"""Check an isolated AOCL prefill-W2 intervention against the unchanged FP32 gate."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = ROOT / "artifacts/reference/smoke-fp32/trace.safetensors"
    policy = ROOT / "reference/tolerances-smoke-fp32-v1.json"
    comparator = ROOT / "research/gpu-reference/scripts/compare_traces.py"
    before = {str(p.relative_to(ROOT)): sha(p) for p in (reference, policy, comparator)}
    metadata = json.loads((args.traces / "interventions.json").read_text())
    variants = []
    for variant in ("production", "aocl_prefill_w2", "production_after"):
        trace = args.traces / f"{variant}.safetensors"
        report_path = args.output.with_name(f"{args.output.stem}-{variant}.json")
        if report_path.exists():
            raise FileExistsError(report_path)
        command = [sys.executable, str(comparator), str(reference), str(trace),
                   "--tolerances", str(policy), "--output", str(report_path)]
        with (args.traces / f"comparison-{variant}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode not in (0, 1) or not report_path.exists():
            raise RuntimeError(f"Comparison did not produce a valid report: {variant}")
        report = json.loads(report_path.read_text())
        if bool(report["passed"]) != (completed.returncode == 0):
            raise ValueError("Comparison exit status disagrees with report")
        if report["missing"] or report["compared_tensors"] != 1904:
            raise ValueError("Incomplete intervention trace")
        if report["reference_sha256"] != sha(reference) or report["candidate_sha256"] != sha(trace):
            raise ValueError("Comparison identity mismatch")
        decisions = report["logit_decisions"]
        variants.append({"variant": variant, "trace_sha256": sha(trace),
                         "report": str(report_path), "report_sha256": sha(report_path),
                         "compared_tensors": report["compared_tensors"],
                         "failed_tensors": len(report["failures"]), "failures": report["failures"],
                         "logit_checks": len(decisions),
                         "all_argmax_match": all(x["argmax_matches"] for x in decisions.values()),
                         "prefill_logits": report["measurements"]["logits"],
                         "final_prefill_hidden": report["measurements"]["layer.21.hidden"]})
    controls_equal = variants[0]["trace_sha256"] == variants[2]["trace_sha256"]
    expected = "e2dad223ab7afc252a4a02e6294848ce17258ee0aad76dbd56f441af5fd85309"
    if not controls_equal or variants[0]["trace_sha256"] != expected:
        raise ValueError("Production controls changed from the preserved FP32 baseline")
    after = {str(p.relative_to(ROOT)): sha(p) for p in (reference, policy, comparator)}
    if before != after:
        raise ValueError("Comparator/reference/policy changed during comparison")
    summary = {"schema_version": 1, "scope": "Test-only whole-model same-prefix intervention on one synthetic fixture; not free-running or corpus qualification",
               "intervention": "AOCL FP32 only for every prefill W2 with rows>8, K=2304,N=768; current Rust everywhere else",
               "platform": metadata["platform"], "threads": metadata["threads"],
               "aocl_library": metadata["aocl_library"], "test_binary_sha256": metadata["test_binary_sha256"],
               "interventions_manifest_sha256": sha(args.traces / "interventions.json"),
               "preservation_manifest_sha256": sha(args.traces / "preservation.json"),
               "comparator_inputs_unchanged": before, "summary_script_sha256": sha(pathlib.Path(__file__)),
               "production_controls_byte_identical": True, "policy_changed": False,
               "production_promoted": False, "performance_measured": False, "variants": variants}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "failed_tensors": {v["variant"]: v["failed_tensors"] for v in variants},
                      "all_argmax_match": all(v["all_argmax_match"] for v in variants)}, indent=2))


if __name__ == "__main__":
    main()
