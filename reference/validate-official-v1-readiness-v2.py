"""Read existing v1 records/results in WSL; never invoke inference/evaluation."""
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_official_evaluation import validate_source_run
from compare_official_evaluation import validate_results


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    import os
    os.chdir(ROOT)
    paths = ["scripts/prepare_official_evaluation.py", "scripts/validate_gpu_reference_record.py",
             "scripts/compare_official_evaluation.py", "scripts/validate_text_replay.py",
             "reference/corpus-smoke-lock-v1.json", "reference/official-components-v1-fp32.json",
             "reference/validate-official-v1-readiness-v2.py"]
    start = {path: sha(path) for path in paths}
    assert start["reference/official-components-v1-fp32.json"] == "3c451b0079221bdb90aecee210256f26c8218d0f138043615aae41af0ae34922"
    results = {}
    for side, directory in [("gpu", "artifacts/reference/corpus-smoke-fp32-4096"),
                            ("cpu", "artifacts/cpu/corpus-smoke-fp32-4096")]:
        manifest, run, contract, pages = validate_source_run(pathlib.Path("reference/corpus-smoke-lock-v1.json"), pathlib.Path(directory), side)
        prepared = pathlib.Path("artifacts/evaluation/v1-" + side)
        provenance = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
        # Legacy provenance has no image_name or execution audit. Read actual
        # prepared annotations for result identities; do not invent new fields.
        annotations = json.loads((prepared / "ground-truth.json").read_text(encoding="utf-8"))
        names = [pathlib.Path(page["page_info"]["image_path"]).name for page in annotations]
        assert len(names) == len(set(names)) == len(pages) == 24
        assert sha(prepared / "ground-truth.json") == provenance["ground_truth_sha256"]
        assert sha(prepared / "configuration.json") == provenance["configuration_sha256"]
        assert sha(pathlib.Path(directory) / "run.json") == provenance["run_sha256"]
        validated = validate_results(prepared, names)
        records = {page["id"]: (record_path, value) for page, record_path, value in pages}
        assert len(provenance["pages"]) == len(records)
        assert {page["id"] for page in provenance["pages"]} == set(records)
        for page in provenance["pages"]:
            record_path, value = records[page["id"]]
            prediction = prepared / "predictions" / page["prediction_filename"]
            assert sha(record_path) == page["record_sha256"]
            assert sha(prediction) == page["prediction_sha256"]
            assert prediction.read_bytes() == value["text"].encode("utf-8")
        results[side] = {"source_run_path": directory + "/run.json", "source_run_sha256": sha(pathlib.Path(directory) / "run.json"),
                         "validated_records": len(pages), "prepared_provenance_sha256": sha(prepared / "provenance.json"),
                         "literal_prepared_predictions_match_records": True,
                         "result_files": {name: sha(path) for name, path in validated["files"].items()},
                         "component_counts": {name: {"samples": item["sample_count"], "pages": item["page_count"]}
                                              for name, item in validated["components"].items()}}
    assert start == {path: sha(path) for path in paths}
    receipt = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "status": "passed", "source_sha256": start, "source_unchanged_during_validation": True,
               "python_version": sys.version, "python_executable": sys.executable, "validation": results,
               "inference_executed": False, "evaluator_executed": False,
               "qualification": "Read-only historical v1 validation in its original WSL preparation environment. Strict current source-record predicates, exact saved prepared text, and component aggregates/counts pass. Legacy v1 lacks the newer execution-audit schema; no retrospective audited-execution claim or rewritten report."}
    path = pathlib.Path("reference/official-v1-readiness-revalidation-v2.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(path), "sha256": sha(path), "gpu_records": 24,
                      "cpu_records": 24, "historical_report_unchanged": True}))


if __name__ == "__main__":
    main()
