"""Read-only join of one frozen mixed-b4 run to four cap4096 GPU goldens.

No inference, normalization, OCR metrics, timing aggregation or mutable repair.
The historical natural prose reference is not retroactively startup-attested.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from corpus_comparison import ASSETS, PROMPT, GREEDY, semantic_contract
from fetch_reference import REVISION, WEIGHT_SHA256
from functional_batch_regression import jobs_for, command, validate_results
from reference_run_identity import canonical_digest
from validate_gpu_reference_record import validate_gpu_reference_record
from validate_text_replay import resolve_recorded_path

FUNCTIONAL = ROOT / "artifacts/functional-batch/mixed-b4-v1"
SUBSET = ROOT / "reference/functional-originals-v1-lock.json"
GPU_ORIGINALS = ROOT / "artifacts/reference/functional-originals-fp32-4096"
GPU_PROSE = ROOT / "artifacts/reference/corpus-v3-fp32-4096"
ORDER = ["prose", "blank-white", "sparse-room", "receipt-cafe"]
PINS = {
    "reference/functional-batch-v1-lock.json": "c9ba1bcbadb7ffe8865aa8a149a75fb239053fb79955d86083c7a28b4de3fb72",
    "reference/functional-originals-v1-lock.json": "c2b56cf580b4edef8968bc78edf9c021905cc3ba41c5e8ee065f2f0c7150965a",
    "reference/gpu-corpus-v3-fp32-summary.json": "fe2b5c58c86bf27290819361381966d36930df5711849a2bcbba0062cabb3ab0",
    "artifacts/functional-batch/mixed-b4-v1/plan.json": "25572beea457125589fd55aee13fb228e3547b7b6697d1a752719d8f3bc466df",
    "artifacts/functional-batch/mixed-b4-v1/report.json": "7eeea54dfe47f6452239f516645d225b464c6dbe82fabc18136cbb6157bfbb77",
    "artifacts/functional-batch/mixed-b4-v1/execution-final.json": "6b03d17bd3411db8930ad38833e102a6fbc90f8e9f926390fc503765b287643f",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    bound, checks = {}, []

    def bind(path, expected=None):
        path = resolve_recorded_path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, "changed file: " + str(path))
        bound[path.relative_to(ROOT).as_posix()] = actual
        return actual

    def read(path, expected=None):
        bind(path, expected)
        return json.loads(resolve_recorded_path(path).read_text(encoding="utf-8"))

    def check(name, actual, expected):
        passed = actual == expected
        checks.append({"name": name, "passed": passed})
        require(passed, "invalid saved evidence: " + name)

    bind(Path(__file__))
    for module in ["corpus_comparison.py", "fetch_reference.py", "functional_batch_regression.py",
                   "capture_functional_cli.py", "capture_rust_build.py", "reference_run_identity.py",
                   "validate_gpu_reference_record.py", "validate_text_replay.py"]:
        bind(ROOT / "scripts" / module)
    for name, digest in PINS.items():
        bind(ROOT / name, digest)
    workload = read(ROOT / "reference/functional-batch-v1-lock.json")
    subset = read(SUBSET)
    source = read(ROOT / subset["source_manifest"], subset["source_manifest_sha256"])
    selected = {workload["inputs"][key]["id"] for key in ORDER[1:]}
    check("subset.complete_parent_objects_and_order", subset["pages"], [p for p in source["pages"] if p["id"] in selected])
    check("subset.pages", len(subset["pages"]), 3)
    check("subset.parent_count", subset["parent_count"], len({p["parent_id"] for p in subset["pages"]}))
    check("functional.order", workload["cases"], {"mixed-b4": ORDER})
    for key, expected in {"precision": "fp32", "backend": "avx2", "threads": 4,
                          "min_dimension": 64, "max_dimension": 1536, "max_new_tokens": 4096}.items():
        check("functional.runtime." + key, workload["runtime"][key], expected)
    for name, digest in workload["source_locks"].items():
        bind(ROOT / name, digest)
    for key, item in workload["inputs"].items():
        members = [p for p in read(ROOT / item["source_lock"])["pages"] if p["id"] == item["id"]]
        require(len(members) == 1 and canonical_digest(members[0]) == item["source_page_sha256"], key + ": source membership")
        for field in ["canonical_path", "canonical_png_sha256", "rgb_sha256", "width", "height", "ground_truth_sha256"]:
            check(key + ".source." + field, item[field], members[0][field])
        bind(ROOT / item["source_path"], item["source_sha256"])
        bind(ROOT / item["canonical_path"], item["canonical_png_sha256"])
        bind(ROOT / item["ground_truth_path"], item["ground_truth_sha256"])
    check("model.revision", workload["model"]["revision"], REVISION)
    for name, expected in {**ASSETS, "model.safetensors": WEIGHT_SHA256}.items():
        check("model.pin." + name, workload["model"]["assets"][name]["sha256"], expected)
        bind(ROOT / workload["model"]["directory"] / name, expected)

    plan, final, report = [read(FUNCTIONAL / name) for name in ["plan.json", "execution-final.json", "report.json"]]
    plan_hash = sha(FUNCTIONAL / "plan.json")
    check("final.pass", [final["status"], final["functional_gate_passed"]], ["passed", True])
    check("final.plan", final["plan_sha256"], plan_hash)
    check("final.report", final["report_sha256"], sha(FUNCTIONAL / "report.json"))
    check("report.pass", [report["status"], report["functional_gate_passed"], report["all_outputs_exact"]], ["passed", True, True])
    check("report.plan", report["plan_sha256"], plan_hash)
    for name, digest in report["artifact_sha256"].items():
        require(Path(name).name == name, "unexpected functional artifact path")
        bind(FUNCTIONAL / name, digest)
    execution = read(FUNCTIONAL / "execution.json")
    check("execution.identity_unchanged", execution["identity"], final["identity_at_end"])
    check("execution.plan", execution["plan_sha256"], plan_hash)
    check("plan.workload_snapshot", read(FUNCTIONAL / plan["workload_snapshot"], plan["workload_snapshot_sha256"]), workload)
    bind(plan["workload_source"], plan["workload_source_sha256"])
    build = read(plan["build_manifest"], plan["build_manifest_sha256"])
    check("build.complete", [build["status"], build["build_exit_code"], build["source_unchanged_during_build"]], ["complete", 0, True])
    build_dir = resolve_recorded_path(plan["build_manifest"]).parent
    binary = build_dir / build["binary"]
    bind(binary, plan["binary_sha256"])
    check("binary.build", build["binary_sha256"], plan["binary_sha256"])
    check("binary.execution", execution["identity"]["cli_executable_sha256"], plan["binary_sha256"])
    archive = build_dir / build["source_archive"]
    bind(archive, build["source_archive_sha256"])
    with zipfile.ZipFile(archive) as contents:
        check("build.source_inventory", sorted(contents.namelist()), sorted(build["source_sha256"]))
        for name, expected in build["source_sha256"].items():
            check("build.source." + name, hashlib.sha256(contents.read(name)).hexdigest(), expected)
    for name in ["scripts/functional_batch_regression.py", "scripts/corpus_comparison.py", "scripts/validate_text_replay.py"]:
        bind(ROOT / name, build["source_sha256"][name])
    check("plan.semantics", [plan["prompt"], plan["greedy_policy"], plan["teacher_forced"]], [PROMPT, GREEDY, False])
    expected_jobs = jobs_for(workload, "mixed-b4")
    for job in expected_jobs:
        job["command"] = command(binary, workload, job)
    check("plan.exact_requests_and_options", plan["jobs"], expected_jobs)
    check("report.invocation_count", len(report["invocations"]), 5)

    prose_summary = read(ROOT / "reference/gpu-corpus-v3-fp32-summary.json")
    prose_run = read(GPU_PROSE / "run.json", prose_summary["source_run_sha256"])
    natural = read(ROOT / "reference/corpus-v3-evaluation-lock.json", prose_summary["manifest_sha256"])
    prose_page = [p for p in natural["pages"] if p["id"] == workload["inputs"]["prose"]["id"]]
    require(len(prose_page) == 1, "prose parent membership")
    historical = read(ROOT / "reference/gpu-corpus-source-capture-after-launch.json", prose_summary["historical_source_capture"]["report_sha256"])
    bind(ROOT / historical["archive"], historical["archive_sha256"])
    with zipfile.ZipFile(ROOT / historical["archive"]) as contents:
        check("historical.archive_inventory", sorted(contents.namelist()), sorted([p["path"] for p in historical["files"]] + ["CAPTURE-SCOPE.txt"]))
        for item in historical["files"]:
            check("historical.source." + item["path"], hashlib.sha256(contents.read(item["path"])).hexdigest(), item["sha256"])

    new_validation = read(ROOT / "reference/gpu-functional-originals-fp32-4096-validation.json")
    check("new_gpu.validation_status", new_validation["status"], "complete_gpu_reference_validated")
    new_run = read(GPU_ORIGINALS / "run.json", new_validation["run_sha256"])
    check("new_gpu.complete", [new_run["status"], new_run["pages"]], ["complete", 3])
    check("new_gpu.validation_configuration", new_run["configuration"], new_validation["configuration"])
    check("new_gpu.manifest", new_run["configuration"]["manifest_sha256"], sha(SUBSET))
    check("new_gpu.selected_order", new_run["configuration"]["selected_page_ids"], [p["id"] for p in subset["pages"]])
    startup = read(GPU_ORIGINALS / "provenance/startup-identity.json", new_validation["startup_identity_sha256"])
    identity = startup["identity"]
    check("new_gpu.identity_digest", canonical_digest({k: v for k, v in identity.items() if k != "identity_sha256"}), identity["identity_sha256"])
    check("new_gpu.runtime_binding", identity["identity_sha256"], new_run["configuration"]["runtime_identity_sha256"])
    check("new_gpu.validation_runtime", identity["identity_sha256"], new_validation["runtime_identity_sha256"])
    check("new_gpu.source_count", len(identity["source_files"]), new_validation["archived_source_file_count"])
    for name, item in identity["source_files"].items():
        bind(GPU_ORIGINALS / "provenance/sources" / name, item["sha256"])

    gpu, gpu_metadata = {}, {}
    for key in ORDER:
        item = workload["inputs"][key]
        is_prose = key == "prose"
        run, folder = (prose_run, GPU_PROSE) if is_prose else (new_run, GPU_ORIGINALS)
        sample = "3f294b5e60a0c2d4" if is_prose else key
        expected_hash = prose_summary["record_snapshots"][sample]["sha256"] if is_prose else new_validation["record_sha256"][sample]
        record = read(folder / (sample + ".json"), expected_hash)
        config = run["configuration"]
        semantic = semantic_contract(config, sample, "fp32", check, gpu=True)
        for option, expected in [("min_dimension", 64), ("max_dimension", 1536), ("max_new_tokens", 4096), ("tf32", False), ("flex_float32_precision", "ieee"), ("compiled_blocks", False)]:
            check(sample + "." + option, config[option], expected)
        for level in [run, config, record]:
            require(level.get("teacher_forced") is not True and level.get("inference_reexecuted") is not False
                    and not any(marker in level for marker in ["postprocessing_replay", "derived_text_replay"]), sample + ": derived/teacher-forced reference")
            if not is_prose:
                check(sample + ".fresh_metadata", [level.get("teacher_forced"), level.get("inference_reexecuted")], [False, True])
        require(validate_gpu_reference_record(record, config, check, sample), "invalid GPU record")
        check(sample + ".id", record["id"], item["id"])
        check(sample + ".pixels", record["canonical_rgb_sha256"], item["rgb_sha256"])
        check(sample + ".prefix", record["prefix_length"], item["input_tokens_expected"])
        page = prose_page[0] if is_prose else next(p for p in subset["pages"] if p["id"] == item["id"])
        for field in ["canonical_path", "canonical_png_sha256", "rgb_sha256", "width", "height", "ground_truth_sha256"]:
            check(sample + ".parent." + field, page[field], item[field])
        gpu[key] = record
        gpu_metadata[key] = {"record_path": (folder / (sample + ".json")).relative_to(ROOT).as_posix(), "record_sha256": expected_hash,
                             "prepared_dimensions_observed": False, "startup_semantics": semantic, "startup_attested": not is_prose,
                             "free_running_evidence": "explicit fresh/non-teacher-forced run/config/page fields" if not is_prose else "historical free-greedy source preserved after launch plus saved per-token argmax decisions; absent startup fields remain absent"}

    comparisons = []
    for job, invocation in zip(plan["jobs"], report["invocations"]):
        saved = read(FUNCTIONAL / (job["id"] + ".invocation.json"))
        check(job["id"] + ".invocation", saved, invocation)
        check(job["id"] + ".job", saved["job"], job)
        check(job["id"] + ".plan", saved["plan_sha256"], plan_hash)
        check(job["id"] + ".exit", saved["exit_code"], 0)
        stdout = FUNCTIONAL / (job["id"] + ".jsonl")
        bind(stdout, saved["stdout_sha256"])
        bind(FUNCTIONAL / (job["id"] + ".stderr.txt"), saved["stderr_sha256"])
        values = [json.loads(line) for line in stdout.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_results(values, workload, job)
        rows = []
        for index, (key, result) in enumerate(zip(job["request_keys"], values)):
            reference = gpu[key]
            fields = {"token_ids_exact": result["token_ids"] == reference["token_ids"],
                      "literal_text_exact": result["text"] == reference["text"],
                      "finish_reason_exact": result["finish_reason"] == reference["finish_reason"],
                      "output_count_exact": result["output_tokens"] == len(reference["token_ids"]),
                      "input_count_exact": result["input_tokens"] == reference["prefix_length"],
                      "precision_exact": result["precision"] == reference["configuration"]["precision"]}
            rows.append({"request_index": index, "input_key": key, "id": workload["inputs"][key]["id"], **fields,
                         "exact_observed_output": all(fields.values()), "output_tokens": result["output_tokens"],
                         "finish_reason": result["finish_reason"], "input_tokens": result["input_tokens"],
                         "cpu_prepared_dimensions": [result["width"], result["height"]], "cpu_dimensions_match_frozen_expectation": True,
                         "gpu_prepared_dimensions": None,
                         "cpu_text_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
                         "gpu_text_sha256": hashlib.sha256(reference["text"].encode()).hexdigest(),
                         "cpu_token_ids_sha256": canonical_digest(result["token_ids"]), "gpu_token_ids_sha256": canonical_digest(reference["token_ids"])})
        comparisons.append({"job": job["id"], "batch_size": job["batch_size"], "mode": job["mode"], "requests": rows,
                            "all_outputs_exact": all(row["exact_observed_output"] for row in rows)})
    require(len(comparisons) == 5 and sum(len(c["requests"]) for c in comparisons) == 20, "incomplete join")
    for name, expected in bound.items():
        require(sha(ROOT / name) == expected, "source changed during join: " + name)
    passed = all(c["all_outputs_exact"] for c in comparisons)
    output = {"schema_version": 1, "status": "exact_saved_outputs_match" if passed else "output_mismatch",
              "saved_output_gate_passed": passed, "invocations": 5, "request_comparisons": 20,
              "unique_inputs": 4, "runtime": {k: workload["runtime"][k] for k in ["precision", "backend", "threads", "min_dimension", "max_dimension", "max_new_tokens"]},
              "gpu_evidence": gpu_metadata, "comparisons": comparisons, "checks": checks,
              "sources_unchanged_during_join": True, "checked_source_sha256": bound,
              "qualification": "Bounded same-request-contract saved-output join only. All CPU results are original CLI outputs, with no text replay. GPU pixels/options/prefix are checked; prepared GPU dimensions were not recorded and are not fabricated. Natural prose preserves its after-launch source qualification and missing startup fields. No new CPU inference, corpus quality, intermediate numerical parity, performance claim or backend promotion."}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "comparisons": 20, "checks": len(checks), "bound_files": len(bound), "sha256": sha(args.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
