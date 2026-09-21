#!/usr/bin/env python3
"""Validate and compare pinned component outputs on an explicit frozen manifest."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import pathlib
import statistics

from prepare_official_evaluation import EVALUATOR_REVISION

COMPONENTS = ("text_block", "display_formula", "table", "reading_order")


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual), f"{label}: nonfinite/nonnumeric score")
    # Recompute metric summaries from serialized rows. This only allows
    # floating summation/serialization rounding, not changes to model tolerance.
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label}: aggregate differs: {actual} != {expected}")


def teds_score(value, label):
    # Pinned upstream TEDS is 1 - tree_distance / max(descendant_counts),
    # without clamping. Unit-cost insert/delete gives the safe range [-1, 1].
    require(type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1,
            f"Invalid {label} value")
    return value


def check_execution_audit(audit, expected_names, provenance_sha256):
    require(audit.get("status") == "complete" and audit.get("exit_code") == 0, "Evaluator execution did not complete successfully")
    require(audit.get("preparation_provenance_sha256") == provenance_sha256, "Execution audit belongs to different prepared inputs")
    for key in ("entered_pages", "completed_pages"):
        names = audit.get(key, [])
        require(len(names) == len(expected_names) and len(set(names)) == len(names) and set(names) == set(expected_names),
                f"Evaluator {key} is incomplete, duplicated or unexpected")
    require(not audit.get("failures") and not audit.get("detected_error_lines"), "Evaluator emitted an error or skipped input")


def validate_results(root, expected_names, prefix="predictions_quick_match_"):
    """Audit every output and recompute counts/aggregates from saved samples."""
    root = pathlib.Path(root)
    directory = root / "result"
    require(directory.is_dir(), f"Missing evaluator result directory: {directory}")
    metrics = read(directory / (prefix + "metric_result.json"))
    require(set(metrics) == set(COMPONENTS), "Missing or unexpected metric components")
    expected_files = {prefix + "metric_result.json", prefix + "display_formula_formula.json", prefix + "table_per_table_TEDS.json"}
    components, samples_by_component = {}, {}
    expected_names = set(expected_names)
    for component in COMPONENTS:
        sample_name = prefix + component + "_result.json"
        expected_files.add(sample_name)
        samples = read(directory / sample_name)
        require(isinstance(samples, list), f"{component}: sample output is not a list")
        samples_by_component[component] = samples
        groups = defaultdict(list)
        for sample in samples:
            require(isinstance(sample, dict) and sample.get("image_name") in expected_names, f"{component}: unknown/missing image name")
            upper, edits = sample.get("upper_len"), sample.get("Edit_num")
            require(type(upper) in (int, float) and upper > 0 and type(edits) in (int, float) and 0 <= edits <= upper, f"{component}: invalid edit count/denominator")
            close(sample.get("metric", {}).get("Edit_dist"), edits / upper, component + " sample edit")
            groups[sample["image_name"]].append(sample)
        if component == "reading_order":
            require(len(groups) == len(samples), "Reading-order output contains duplicate pages")
        score = metrics[component].get("all", {}).get("Edit_dist", {})
        per_page = {}
        if samples:
            filename = prefix + component + "_per_page_edit.json"
            expected_files.add(filename)
            per_page = read(directory / filename)
            require(isinstance(per_page, dict) and set(per_page) == set(groups), f"{component}: missing/extra per-page scores")
            for name, entries in groups.items():
                close(per_page[name], sum(s["Edit_num"] for s in entries) / sum(s["upper_len"] for s in entries), component + " page " + name)
            close(score.get("ALL_page_avg"), statistics.mean(per_page.values()), component + " page mean")
            close(score.get("edit_whole"), sum(s["Edit_num"] for s in samples) / sum(s["upper_len"] for s in samples), component + " whole edit")
            close(score.get("edit_sample_avg"), statistics.mean(s["Edit_num"] / s["upper_len"] for s in samples), component + " sample mean")
        else:
            require(score == {"ALL_page_avg": "NaN"}, f"{component}: empty component must retain explicit upstream NaN sentinel")
        components[component] = {"sample_count": len(samples), "page_count": len(groups), "per_page": per_page,
                                 "pages_without_component_score": sorted(expected_names - set(groups))}
    formulas = read(directory / (prefix + "display_formula_formula.json"))
    formula_samples = samples_by_component["display_formula"]
    require(isinstance(formulas, list) and len(formulas) == len(formula_samples), "CDM_plain formula pair count differs")
    for index, (exported, sample) in enumerate(zip(formulas, formula_samples)):
        require(exported.get("img_id") == str(index) and exported.get("img_name") == sample.get("img_id"), "CDM_plain formula pair identity differs")
    tables = samples_by_component["table"]
    teds = read(directory / (prefix + "table_per_table_TEDS.json"))
    expected_table_keys = {s["img_id"] + "_" + str(s.get("gt_idx", index)) for index, s in enumerate(tables)}
    require(len(expected_table_keys) == len(tables) and isinstance(teds, dict) and set(teds) == expected_table_keys, "TEDS missing/duplicate/extra table keys")
    for index, sample in enumerate(tables):
        key = sample["img_id"] + "_" + str(sample.get("gt_idx", index))
        for metric in ("TEDS", "TEDS_structure_only"):
            value = teds_score(sample.get("metric", {}).get(metric), metric + " sample")
            close(teds_score(teds[key].get(metric), metric + " per-table"), value, metric + " per-table")
    for metric in ("TEDS", "TEDS_structure_only"):
        scores = metrics["table"].get("all", {}).get(metric)
        if tables:
            value = teds_score(scores.get("all") if isinstance(scores, dict) else None, metric + " mean")
            close(value, statistics.mean(s["metric"][metric] for s in tables), metric + " mean")
        else:
            require(scores == {}, "Empty table component must not contain a numeric TEDS aggregate")
    files = {p.name: p for p in directory.iterdir() if p.is_file()}
    require(set(files) == expected_files, f"Evaluator output files differ; missing={sorted(expected_files-set(files))}, unexpected={sorted(set(files)-expected_files)}")
    return {"metrics": metrics, "components": components, "files": files, "expected_file_count": len(expected_files)}


def validate_prepared(root, manifest_path, runtime, require_audit=True):
    root = pathlib.Path(root)
    provenance = read(root / "provenance.json")
    manifest = read(manifest_path)
    require(provenance.get("evaluator_revision") == EVALUATOR_REVISION, "Unpinned evaluator revision")
    require(provenance.get("manifest_sha256") == sha(manifest_path), "Evaluation manifest hash differs")
    require(provenance.get("runtime") == runtime, "Evaluation runtime label differs")
    pages = provenance.get("pages", [])
    source_pages = manifest["pages"]
    page_by_id = {p["id"]: p for p in pages}
    require(len(page_by_id) == len(pages) == len(source_pages) and set(page_by_id) == {p["id"] for p in source_pages}, "Preparation page count/IDs differ from selected manifest")
    require(provenance.get("expected_page_count") == len(pages), "Preparation expected count missing/different")
    require(sha(root / "ground-truth.json") == provenance["ground_truth_sha256"], "Prepared annotations changed")
    require(sha(root / "configuration.json") == provenance["configuration_sha256"], "Evaluator configuration changed")
    config = read(root / "configuration.json")["end2end_eval"]
    require(pathlib.Path(config["dataset"]["ground_truth"]["data_path"]).resolve() == (root / "ground-truth.json").resolve()
            and pathlib.Path(config["dataset"]["prediction"]["data_path"]).resolve() == (root / "predictions").resolve(),
            "Configuration points outside the prepared inputs; run comparison in the preparation environment")
    require(config["dataset"].get("match_method") == "quick_match" and not config["dataset"].get("filter"), "Unexpected matcher or hidden dataset filter")
    required_metrics = {"text_block": ["Edit_dist"], "display_formula": ["Edit_dist", "CDM_plain"], "table": ["TEDS", "Edit_dist"], "reading_order": ["Edit_dist"]}
    require(config["metrics"] == {key: {"metric": value} for key, value in required_metrics.items()}, "Evaluation metric configuration differs")
    annotations = read(root / "ground-truth.json")
    image_names = [pathlib.Path(p["page_info"]["image_path"]).name for p in annotations]
    require(len(image_names) == len(pages) and len(set(image_names)) == len(image_names), "Prepared annotation image IDs duplicated/missing")
    require(set(image_names) == set(provenance.get("expected_image_names", [])), "Prepared annotation page IDs differ")
    run_path = pathlib.Path(provenance["run_path"])
    require(sha(run_path) == provenance["run_sha256"], "Inference run changed after preparation")
    run = read(run_path)
    contract = run["configuration"] if runtime == "gpu" else run["contract"]
    require(contract == provenance["inference_contract"], "Prepared inference contract differs from run")
    from prepare_official_evaluation import validate_source_run
    _, _, _, validated = validate_source_run(pathlib.Path(manifest_path), run_path.parent, runtime)
    valid_by_id = {p["id"]: (p, record_path, result) for p, record_path, result in validated}
    names = set()
    for page in pages:
        source_page, record_path, result = valid_by_id[page["id"]]
        require(page["category"] == source_page["category"], "Prepared category differs")
        require(sha(record_path) == page["record_sha256"], "Inference record changed after preparation")
        require(page["image_name"] in image_names and page["prediction_filename"] == page["image_name"][:-4] + ".md", "Prediction naming differs from upstream")
        name = page["prediction_filename"]
        require(pathlib.Path(name).name == name and name not in names, "Unsafe/duplicate prediction filename")
        names.add(name)
        prediction = root / "predictions" / name
        require(sha(prediction) == page["prediction_sha256"], "Prepared prediction changed")
        require(prediction.read_bytes() == result["text"].encode("utf-8"), "Prepared prediction is not untouched inference text")
    require({p.name for p in (root / "predictions").iterdir()} == names, "Missing/extra prepared prediction files")
    if require_audit:
        audit = read(root / "execution-audit.json")
        check_execution_audit(audit, image_names, sha(root / "provenance.json"))
        require(audit.get("evaluator_revision") == EVALUATOR_REVISION, "Execution evaluator revision differs")
        require(audit.get("requirements_sha256") == sha(pathlib.Path(__file__).resolve().parents[1] / "requirements/evaluation-resolved.txt"), "Execution requirements lock differs")
        require(audit.get("execution_log_sha256") == sha(root / "execution.log"), "Execution log changed")
        require(audit.get("result_file_sha256") == {p.name: sha(p) for p in (root / "result").iterdir() if p.is_file()}, "Result files changed after audited execution")
    return provenance, page_by_id, image_names


def compare(gpu, cpu, manifest_path):
    roots = {"gpu": pathlib.Path(gpu), "cpu": pathlib.Path(cpu)}
    prepared = {name: validate_prepared(root, manifest_path, name) for name, root in roots.items()}
    provenance = {name: value[0] for name, value in prepared.items()}
    for key in ("evaluator_revision", "manifest_sha256", "ground_truth_sha256", "annotation_adaptation"):
        require(provenance["gpu"][key] == provenance["cpu"][key], f"Evaluation contracts differ: {key}")
    contracts = {name: value["inference_contract"] for name, value in provenance.items()}
    for key in ("model_revision", "weights_sha256", "precision"):
        require(contracts["gpu"][key] == contracts["cpu"][key], f"CPU/GPU inference contracts differ: {key}")
    for key in ("min_dimension", "max_dimension", "max_new_tokens"):
        require(contracts["gpu"][key] == contracts["cpu"]["options"][key], f"CPU/GPU inference options differ: {key}")
    results = {name: validate_results(root, prepared[name][2]) for name, root in roots.items()}
    file_names = set(results["gpu"]["files"]) | set(results["cpu"]["files"])
    comparisons = {}
    for filename in sorted(file_names):
        paths = {name: results[name]["files"].get(filename) for name in roots}
        comparisons[filename] = {"equal_json": all(paths.values()) and read(paths["gpu"]) == read(paths["cpu"]),
                                 **{name + "_sha256": sha(path) if path else None for name, path in paths.items()}}
    pages = {name: value[1] for name, value in prepared.items()}
    equal_predictions = all(pages["gpu"][key]["prediction_sha256"] == pages["cpu"][key]["prediction_sha256"] for key in pages["gpu"])
    categories = sorted({page["category"] for page in pages["gpu"].values()})
    by_category = {}
    for category in categories:
        by_category[category] = {}
        for component in COMPONENTS:
            counts, means = {}, {}
            for name in roots:
                names = {p["image_name"] for p in pages[name].values() if p["category"] == category}
                per_page = results[name]["components"][component]["per_page"]
                values = [per_page[key] for key in names if key in per_page]
                counts[name], means[name] = len(values), statistics.mean(values) if values else None
            by_category[category][component] = {"page_counts": counts, "mean_page_normalized_edit_distance": means,
                "cpu_minus_gpu_percentage_points": 100 * (means["cpu"] - means["gpu"]) if all(x is not None for x in means.values()) else None}
    return {"schema_version": 2, "status": "complete", "scope": f"Official v1_5 component evaluation of all {len(pages['gpu'])} selected pages in {pathlib.Path(manifest_path).name}; no Overall score.",
            "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "pages": len(pages["gpu"]),
            "category_counts": dict(Counter(p["category"] for p in pages["gpu"].values())),
            "evaluator_revision": EVALUATOR_REVISION, "inference_contracts": contracts,
            "prediction_origins": {name: value.get("prediction_origin") for name, value in provenance.items()},
            "preparation_provenance_sha256": {name: sha(root / "provenance.json") for name, root in roots.items()},
            "execution_audit_sha256": {name: sha(root / "execution-audit.json") for name, root in roots.items()},
            "evaluator_requirements_sha256": sha(pathlib.Path(__file__).resolve().parents[1] / "requirements/evaluation-resolved.txt"),
            "comparison_script_sha256": sha(__file__), "annotation_adaptation": provenance["gpu"]["annotation_adaptation"],
            "all_prediction_text_bytes_equal": equal_predictions,
            "all_evaluator_outputs_equal": all(x["equal_json"] for x in comparisons.values()),
            "component_scores": {name: {component: value["all"] for component, value in result["metrics"].items()} for name, result in results.items()},
            "component_coverage": {name: {component: {k: v for k, v in value.items() if k != "per_page"} for component, value in result["components"].items()} for name, result in results.items()},
            "by_category": by_category, "result_files": comparisons,
            "limitations": ["CDM_plain exports formula pairs; rendered formula similarity and official Overall are not computed.",
                           "Annotation adaptations are explicit evaluator-input changes; frozen source annotations remain unchanged.",
                           "Component counts may differ because the official matcher can move formula content into text; missing components are reported, not assigned zero.",
                           "Completing this report does not by itself establish numerical tensor parity, independent corpus provenance, performance or an accuracy acceptance threshold."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=pathlib.Path, required=True)
    parser.add_argument("--cpu", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite a durable evaluation report; choose a new output")
    report = compare(args.gpu, args.cpu, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": report["pages"], "text_equal": report["all_prediction_text_bytes_equal"], "evaluation_equal": report["all_evaluator_outputs_equal"], "output": str(args.output)}))
    if report["all_prediction_text_bytes_equal"] and not report["all_evaluator_outputs_equal"]:
        raise SystemExit("Identical predictions produced different evaluator results")


if __name__ == "__main__":
    main()
