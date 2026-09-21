#!/usr/bin/env python3
"""Prepare pinned OmniDocBench inputs without changing model output text."""
import argparse
import hashlib
import json
import pathlib
import subprocess

from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from validate_text_replay import validate_run_replay, validate_page_replay, validate_inference_result
from validate_gpu_reference_record import validate_gpu_reference_record

EVALUATOR_REVISION = "59b103c4b47d3a01fada83491585d6512a40c0bc"


def replay_check(name, actual, expected):
    if actual != expected:
        raise ValueError(f"Invalid saved-token replay provenance: {name}")


def validate_source_run(manifest_path, predictions, runtime):
    """Validate all records before any evaluator inputs are written."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = manifest["pages"]
    if not pages or len({p["id"] for p in pages}) != len(pages):
        raise ValueError("Manifest must contain nonempty unique page IDs")
    run = json.loads((predictions / "run.json").read_text(encoding="utf-8"))
    contract = run["configuration"] if runtime == "gpu" else run["contract"]
    for key, expected in [("manifest_sha256", sha256(manifest_path)), ("model_revision", REVISION), ("weights_sha256", WEIGHT_SHA256)]:
        if contract.get(key) != expected:
            raise ValueError(f"{runtime} inference provenance differs: {key}")
    if contract.get("precision") not in {"fp32", "bf16"}:
        raise ValueError(f"Invalid {runtime} precision")
    options = contract if runtime == "gpu" else contract["options"]
    if any(type(options.get(k)) is not int or options[k] <= 0 for k in ["min_dimension", "max_dimension", "max_new_tokens"]):
        raise ValueError(f"Invalid {runtime} inference options")
    if runtime == "gpu":
        if run.get("status") != "complete" or run.get("pages") != len(pages):
            raise ValueError("GPU inference run is incomplete or has a different page count")
        if run.get("teacher_forced", False) is not False or contract.get("teacher_forced", False) is not False:
            raise ValueError("GPU run is explicitly teacher-forced")
        if any("postprocessing_replay" in value or value.get("derived_text_replay", False)
               or value.get("inference_reexecuted") is False
               for value in (run, contract)):
            raise ValueError("GPU run must be original inference, not saved-token replay")
        if contract.get("tf32") is not False or contract.get("flex_float32_precision") != "ieee":
            raise ValueError("GPU reference did not use the strict precision contract")
    else:
        payload = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != run.get("contract_sha256") or run.get("teacher_forced") is not False:
            raise ValueError("CPU run contract hash differs or run is not free-running")
    replay_source = validate_run_replay(run, contract, replay_check) if runtime == "cpu" else None
    validated = []
    errors = []
    samples = set()
    for page in pages:
        sample = pathlib.Path(page["canonical_path"]).parent.name
        if sample in samples:
            raise ValueError(f"Duplicate input sample key: {sample}")
        samples.add(sample)
        record_path = predictions / (sample + ".json")
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if "error" in record:
                raise ValueError(f"failed inference: {record['error']}")
            if record.get("id") != page["id"] or record.get("category") != page["category"]:
                raise ValueError("prediction identity/category mismatch")
            if sha256(pathlib.Path(page["canonical_path"])) != page["canonical_png_sha256"]:
                raise ValueError("canonical image hash mismatch")
            if runtime == "gpu":
                if record.get("configuration") != contract or record.get("canonical_rgb_sha256") != page["rgb_sha256"]:
                    raise ValueError("GPU page contract differs")
                if record.get("teacher_forced", False) is not False:
                    raise ValueError("GPU page is explicitly teacher-forced")
                if ("postprocessing_replay" in record or record.get("derived_text_replay", False)
                        or record.get("inference_reexecuted") is False):
                    raise ValueError("GPU page must be original inference, not saved-token replay")
                if not validate_gpu_reference_record(record, contract, replay_check, sample + ".gpu_result"):
                    raise ValueError("Malformed GPU inference result")
                result = record
            else:
                if record.get("contract_sha256") != run["contract_sha256"] or record.get("input_sha256") != page["canonical_png_sha256"] or record.get("ground_truth_sha256") != page["ground_truth_sha256"]:
                    raise ValueError("CPU page contract differs")
                result = record["result"]
                if result.get("teacher_forced") is not False or result.get("precision") != contract["precision"]:
                    raise ValueError("CPU result precision/free-running contract differs")
                validate_page_replay(record, replay_source, replay_check, sample)
            ids = result.get("token_ids")
            if not isinstance(result.get("text"), str) or not isinstance(ids, list) or not ids or not all(type(i) is int and 0 <= i < 65536 for i in ids):
                raise ValueError("invalid output text/token schema")
            if len(ids) > options["max_new_tokens"] or any(i in {11, 263} for i in ids[:-1]):
                raise ValueError("invalid token budget or early stop sequence")
            if runtime == "cpu" and result.get("output_tokens") != len(ids):
                raise ValueError("CPU output-token count differs")
            finish = "eos" if ids[-1] in {11, 263} else "length"
            if result.get("finish_reason") != finish or (finish == "length" and len(ids) != options["max_new_tokens"]):
                raise ValueError("inconsistent finish reason/cap")
            if runtime == "cpu":
                validate_inference_result(result, contract, replay_check, sample + ".result")
            validated.append((page, record_path, result))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{sample}: {error}")
    if errors:
        raise ValueError(f"{runtime}: {len(errors)}/{len(pages)} missing, failed or invalid page records:\n" + "\n".join(errors))
    return manifest, run, contract, validated


def prepare(args):
    revision = subprocess.check_output(["git", "-C", str(args.evaluator), "rev-parse", "HEAD"], text=True).strip()
    if revision != EVALUATOR_REVISION:
        raise ValueError("Official evaluator revision differs from the pinned v1_5 branch snapshot")
    if subprocess.check_output(["git", "-C", str(args.evaluator), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Pinned evaluator has tracked modifications")
    manifest, run, contract, validated = validate_source_run(args.manifest, args.predictions, args.runtime)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Evaluation output must be new or empty; refusing stale results or prior provenance")
    annotations, records, names, adaptations, prediction_texts = [], [], set(), [], {}
    for page, record_path, result in validated:
        sample = pathlib.Path(page["canonical_path"]).parent.name
        annotation = json.loads(pathlib.Path(page["annotation_path"]).read_text(encoding="utf-8"))
        payload = json.dumps(annotation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != page["annotation_page_sha256"]:
            raise ValueError(f"Ground-truth annotation hash mismatch: {sample}")
        annotation_ids = {item["anno_id"] for item in annotation["layout_dets"]}
        retained_relations = []
        for relation in annotation["extra"]["relation"]:
            dangling = relation["relation_type"] == "truncated" and any(
                relation[key] not in annotation_ids for key in ("source_anno_id", "target_anno_id"))
            if dangling:
                if not args.drop_dangling_truncated_relations:
                    raise ValueError(f"Dangling truncated relation in {sample}: {relation}; explicit adaptation required")
                adaptations.append({"id": page["id"], "removed_relation": relation,
                                    "reason": "A merge endpoint does not exist among the page's annotation IDs."})
            else:
                retained_relations.append(relation)
        annotation["extra"]["relation"] = retained_relations
        image_name = pathlib.Path(annotation["page_info"]["image_path"]).name
        filename = image_name[:-4] + ".md"  # Exact naming rule in the pinned evaluator.
        if filename in names:
            raise ValueError(f"Ambiguous prediction filename: {filename}")
        names.add(filename)
        prediction_texts[filename] = result["text"].encode("utf-8")
        annotations.append(annotation)
        records.append({"id": page["id"], "category": page["category"], "image_name": image_name,
                        "record_path": str(record_path.resolve()), "record_sha256": sha256(record_path),
                        "prediction_filename": filename, "prediction_sha256": hashlib.sha256(prediction_texts[filename]).hexdigest(),
                        "finish_reason": result["finish_reason"], "output_tokens": len(result["token_ids"])})
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "result").mkdir()
    for filename, content in prediction_texts.items():
        (prediction_dir / filename).write_bytes(content)
    ground_truth = args.output / "ground-truth.json"
    ground_truth.write_text(json.dumps(annotations, ensure_ascii=False) + "\n", encoding="utf-8")
    configuration = {"end2end_eval": {"metrics": {
        "text_block": {"metric": ["Edit_dist"]},
        "display_formula": {"metric": ["Edit_dist", "CDM_plain"]},
        "table": {"metric": ["TEDS", "Edit_dist"]},
        "reading_order": {"metric": ["Edit_dist"]}}, "dataset": {
        "dataset_name": "end2end_dataset", "ground_truth": {"data_path": str(ground_truth.resolve())},
        "prediction": {"data_path": str(prediction_dir.resolve())}, "match_method": "quick_match"}}}
    config_path = args.output / "configuration.json"
    config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {"schema_version": 2, "evaluator_repository": "https://github.com/opendatalab/OmniDocBench", "evaluator_revision": revision,
        "manifest_path": str(args.manifest.resolve()), "manifest_name": args.manifest.name, "manifest_sha256": sha256(args.manifest),
        "expected_page_count": len(records), "expected_image_names": [p["image_name"] for p in records],
        "dataset": manifest.get("dataset"), "dataset_revision": manifest.get("revision"),
        "run_path": str((args.predictions / "run.json").resolve()), "run_sha256": sha256(args.predictions / "run.json"),
        "inference_contract": contract, "runtime": args.runtime,
        "prediction_origin": "saved_token_text_replay_original_inference_timings_retained" if run.get("derived_text_replay") else "original_free_running_inference",
        "ground_truth_sha256": sha256(ground_truth), "configuration_sha256": sha256(config_path),
        "preparation_script_sha256": sha256(pathlib.Path(__file__)), "pages": records,
        "annotation_adaptation": {"policy": "Drop dangling truncated relations only; retain all layout/text annotations unchanged.",
                                  "explicitly_enabled": args.drop_dangling_truncated_relations, "changes": adaptations},
        "qualification": "Official matcher and component metrics on exactly the selected manifest. Adapted-input results are not unmodified benchmark scores. CDM_plain exports formula pairs; rendered CDM and the official Overall score are not computed."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--predictions", type=pathlib.Path, required=True)
    parser.add_argument("--runtime", choices=["gpu", "cpu"], required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--evaluator", type=pathlib.Path, default=pathlib.Path("artifacts/OmniDocBench-eval"))
    parser.add_argument("--drop-dangling-truncated-relations", action="store_true",
                        help="Explicitly remove invalid merge links from evaluator inputs; preserve and report every annotation")
    args = parser.parse_args()
    provenance = prepare(args)
    print(json.dumps({"runtime": args.runtime, "pages": len(provenance["pages"]), "configuration": str((args.output / "configuration.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
