#!/usr/bin/env python3
"""Compare fresh targeted inference with preserved GPU and text-replay records."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image

from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from validate_text_replay import canonical_sha256, resolve_recorded_path, validate_inference_result, validate_page_replay, validate_run_replay


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_subset(subset, full, full_path, check):
    check("subset.source_manifest_sha256", subset.get("source_manifest_sha256"), sha256(full_path))
    check("subset.source_manifest_path", str(resolve_recorded_path(subset.get("source_manifest", "")).resolve()), str(Path(full_path).resolve()))
    full_pages = {page["id"]: page for page in full["pages"]}
    ids = [page["id"] for page in subset["pages"]]
    check("subset.nonempty_unique_ids", bool(ids) and len(ids) == len(set(ids)), True)
    for page in subset["pages"]:
        # Bind every selected field, including input hashes and annotation IDs,
        # rather than accepting a renamed or altered page under the same ID.
        check(page["id"] + ".exact_selected_page", canonical_sha256(page), canonical_sha256(full_pages.get(page["id"])))


def compare_result(actual, expected):
    left, right = actual["token_ids"], expected["token_ids"]
    first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    if first is None and len(left) != len(right):
        first = min(len(left), len(right))
    return {"tokens_exact": left == right, "text_exact": actual["text"] == expected["text"],
            "finish_reason_exact": actual["finish_reason"] == expected["finish_reason"],
            "output_count_exact": len(left) == len(right), "first_token_divergence": first,
            "actual_output_tokens": len(left), "expected_output_tokens": len(right),
            "actual_text_sha256": hashlib.sha256(actual["text"].encode()).hexdigest(),
            "expected_text_sha256": hashlib.sha256(expected["text"].encode()).hexdigest()}


def compare(args):
    checks, rows, missing, failures = [], [], [], []

    def check(name, actual, expected):
        checks.append({"name": name, "passed": actual == expected, "actual": actual, "expected": expected})

    subset, full = read(args.subset), read(args.full_manifest)
    validate_subset(subset, full, args.full_manifest, check)
    runs = {name: read(path / "run.json") for name, path in (("cpu", args.cpu), ("gpu", args.gpu), ("replay", args.replay))}
    contracts = {name: run["configuration"] if name == "gpu" else run["contract"] for name, run in runs.items()}
    for name, contract in contracts.items():
        for key, expected in (("model_revision", REVISION), ("weights_sha256", WEIGHT_SHA256), ("precision", "fp32")):
            check(name + "." + key, contract.get(key), expected)
        check(name + ".manifest_sha256", contract.get("manifest_sha256"), sha256(args.subset if name == "cpu" else args.full_manifest))
        if name != "gpu":
            check(name + ".contract_sha256", canonical_sha256(contract), runs[name].get("contract_sha256"))
            check(name + ".teacher_forced", runs[name].get("teacher_forced"), False)
        options = contract if name == "gpu" else contract["options"]
        for key, expected in (("min_dimension", 64), ("max_dimension", 1536), ("max_new_tokens", 4096)):
            check(name + ".options." + key, options.get(key), expected)
    check("cpu.original_free_inference", runs["cpu"].get("derived_text_replay", False), False)
    check("cpu.planned_pages", runs["cpu"].get("planned_pages"), len(subset["pages"]))
    check("cpu.threads", contracts["cpu"].get("threads"), 4)
    check("cpu.backend", contracts["cpu"].get("backend"), "avx2")
    check("cpu.model_config_sha256", contracts["cpu"].get("config_sha256"), sha256(args.model / "config.json"))
    pins = read(Path(__file__).resolve().parents[1] / "reference/manifest.json")
    check("model.pinned_config_sha256", sha256(args.model / "config.json"), pins["files"]["config.json"]["sha256"])
    check("cpu.prompt", contracts["cpu"].get("prompt"), "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>")
    check("gpu.tf32", contracts["gpu"].get("tf32"), False)
    check("gpu.flex_float32_precision", contracts["gpu"].get("flex_float32_precision"), "ieee")
    replay_context = validate_run_replay(runs["replay"], contracts["replay"], check)

    build_path = args.build / "build.json"
    build = read(build_path)
    check("cpu.build_manifest_sha256", contracts["cpu"].get("build_manifest_sha256"), sha256(build_path))
    check("build.status", build.get("status"), "complete")
    check("build.exit_code", build.get("build_exit_code"), 0)
    check("build.source_unchanged_during_build", build.get("source_unchanged_during_build"), True)
    check("build.binary_sha256", sha256(args.build / build["binary"]), build.get("binary_sha256"))
    check("cpu.binary_sha256", contracts["cpu"].get("binary_sha256"), build.get("binary_sha256"))
    archive = args.build / build["source_archive"]
    check("build.source_archive_sha256", sha256(archive), build.get("source_archive_sha256"))
    with zipfile.ZipFile(archive) as sources:
        for path, digest in build["source_sha256"].items():
            check("build.archived_source." + path, hashlib.sha256(sources.read(path)).hexdigest(), digest)
    special = {"lock": "Cargo.lock", "cargo_manifest": "Cargo.toml", "toolchain": "rust-toolchain.toml", "harness": "examples/corpus_eval.rs", "corpus_record": "examples/support/corpus_record.rs"}
    for key, digest in contracts["cpu"]["source_sha256"].items():
        check("cpu.compiled_source." + key, digest, build["source_sha256"].get(special.get(key, "src/" + key + ".rs")))
    check("cpu.decoder_matches_replay", contracts["cpu"]["source_sha256"]["tokenizer"], contracts["replay"]["postprocessing_replay"]["decoder_source_sha256"])

    for page in subset["pages"]:
        sample = Path(page["canonical_path"]).parent.name
        paths = {name: directory / (sample + ".json") for name, directory in (("cpu", args.cpu), ("gpu", args.gpu), ("replay", args.replay))}
        absent = [name for name, path in paths.items() if not path.is_file()]
        if absent:
            missing.append({"sample_id": sample, "runtimes": absent})
            continue
        before = len(checks)
        records = {name: read(path) for name, path in paths.items()}
        if any("error" in record for record in records.values()):
            failures.append({"sample_id": sample, "errors": {name: value.get("error") for name, value in records.items() if "error" in value}})
            continue
        check(sample + ".canonical_png_sha256", sha256(page["canonical_path"]), page["canonical_png_sha256"])
        with Image.open(page["canonical_path"]) as image:
            check(sample + ".canonical_rgb_sha256", hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(), page["rgb_sha256"])
        check(sample + ".ground_truth_sha256", sha256(page["ground_truth_path"]), page["ground_truth_sha256"])
        for name, record in records.items():
            check(sample + "." + name + ".id", record.get("id"), page["id"])
            check(sample + "." + name + ".category", record.get("category"), page["category"])
            if name == "gpu":
                check(sample + ".gpu.configuration", record.get("configuration"), contracts[name])
                check(sample + ".gpu.input", record.get("canonical_rgb_sha256"), page["rgb_sha256"])
            else:
                check(sample + "." + name + ".contract", record.get("contract_sha256"), runs[name]["contract_sha256"])
                check(sample + "." + name + ".input", record.get("input_sha256"), page["canonical_png_sha256"])
                check(sample + "." + name + ".ground_truth", record.get("ground_truth_sha256"), page["ground_truth_sha256"])
                validate_inference_result(record.get("result"), contracts[name], check, sample + "." + name)
                validate_page_replay(record, replay_context if name == "replay" else None, check, sample + "." + name)
        actual, expected, replay = records["cpu"]["result"], records["gpu"], records["replay"]["result"]
        check(sample + ".prefix_matches_gpu", actual["input_tokens"], expected.get("prefix_length"))
        for key in ("width", "height", "input_tokens"):
            check(sample + ".prefix_matches_replay." + key, actual[key], replay[key])
        rows.append({"sample_id": sample, "id": page["id"], "category": page["category"],
                     "record_sha256": {name: sha256(path) for name, path in paths.items()},
                     "record_paths": {name: str(path) for name, path in paths.items()},
                     "provenance_passed": all(item["passed"] for item in checks[before:]),
                     "fresh_cpu_vs_gpu": compare_result(actual, expected), "fresh_cpu_vs_replay": compare_result(actual, replay),
                     "old_text_was_different": actual["text"] != records["replay"]["postprocessing_replay"]["original_text"]})
    provenance = all(item["passed"] for item in checks)
    exact = sum(all(row[key][field] for key in ("fresh_cpu_vs_gpu", "fresh_cpu_vs_replay") for field in ("tokens_exact", "text_exact", "finish_reason_exact", "output_count_exact")) for row in rows)
    complete = not missing and not failures and len(rows) == len(subset["pages"])
    return {"schema_version": 1, "status": "failed" if failures or not provenance else ("complete" if complete else "partial"), "targeted_regression_passed": complete and provenance and exact == len(rows),
            "scope": "Fresh independent CPU inference on three cases selected for observed BPE cleanup failures; comparison with original GPU output and corrected saved-token replay. Not a 200-page quality or performance claim.",
            "subset_manifest": str(args.subset), "subset_manifest_sha256": sha256(args.subset), "full_manifest": str(args.full_manifest), "full_manifest_sha256": sha256(args.full_manifest),
            "selection_policy": subset.get("selection_policy"), "expected_pages": len(subset["pages"]), "compared_pages": len(rows), "exact_pages": exact,
            "total_compared_tokens": sum(row["fresh_cpu_vs_gpu"]["actual_output_tokens"] for row in rows),
            "missing": missing, "failed_records": failures, "provenance_passed": provenance, "checks": checks, "pages": rows,
            "run_snapshots": runs, "run_sha256": {name: sha256(path / "run.json") for name, path in (("cpu", args.cpu), ("gpu", args.gpu), ("replay", args.replay))},
            "build_manifest": str(build_path), "build_manifest_sha256": sha256(build_path), "build_snapshot": build,
            "comparison_script_sha256": sha256(__file__), "qualification": "GPU full-corpus status is recorded as observed; only selected completed pages are compared. Concurrent functional load, not a timing benchmark."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=Path("reference/tokenizer-regression-corpus-v1-lock.json"))
    parser.add_argument("--full-manifest", type=Path, default=Path("reference/corpus-v3-evaluation-lock.json"))
    parser.add_argument("--cpu", type=Path, default=Path("artifacts/cpu/tokenizer-regression-fp32-v1"))
    parser.add_argument("--gpu", type=Path, default=Path("artifacts/reference/corpus-v3-fp32-4096"))
    parser.add_argument("--replay", type=Path, default=Path("artifacts/cpu/corpus-v3-fp32-4096-redecoded-v2"))
    parser.add_argument("--build", type=Path, default=Path("artifacts/builds/corpus-eval-v5-windows"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing regression report")
    report = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "compared_pages", "exact_pages", "provenance_passed", "targeted_regression_passed")}))
    if not report["targeted_regression_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
