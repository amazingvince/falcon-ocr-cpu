"""Shared read-only corpus comparison checks; never repair source run metadata."""
import hashlib
import json
import math
import pathlib
import zipfile

from PIL import Image

from fetch_reference import REVISION, WEIGHT_SHA256, sha256
from validate_text_replay import canonical_sha256, resolve_recorded_path, validate_inference_result

PROMPT = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
GREEDY = "first token index attaining the maximum finite logit"
ASSETS = {
    "config.json": "ba4aec622ec2954e22c76d7ced80817c34d91e26970884e484c29a872e794adf",
    "tokenizer.json": "4a9892af2b1ef021a421f140c7e3c064f5b255f7d75ba18c883996d86e1cf15a",
    "tokenizer_config.json": "074e03d3fd56d190dac763ec5dfe75a728e15783fc5d919d4b5cbe72bcd24d26",
}
# Kept independently pinned, rather than accepting a caller-rewritten artifact manifest.


def read_object(path):
    return read_snapshot(path)[0]


def read_snapshot(path):
    """Parse and hash the identical bytes, even if a writer replaces the path."""
    data = pathlib.Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def sample_key(page):
    return pathlib.PurePosixPath(page["canonical_path"].replace("\\", "/")).parent.name


def page_index(manifest, label, check):
    pages = manifest.get("pages")
    check(label + ".nonempty_pages", isinstance(pages, list) and bool(pages), True)
    if not isinstance(pages, list):
        return {}
    result, keys = {}, set()
    for i, page in enumerate(pages):
        valid = isinstance(page, dict) and isinstance(page.get("id"), str) and bool(page["id"]) and isinstance(page.get("canonical_path"), str)
        check(f"{label}.page.{i}.identity", valid, True)
        if not valid:
            continue
        key = sample_key(page)
        check(f"{label}.page.{i}.unique_id", page["id"] not in result, True)
        check(f"{label}.page.{i}.unique_image_key", key not in keys and key not in ("", ".", ".."), True)
        result[page["id"]] = page
        keys.add(key)
    return result


def manifest_scope(selected_path, sources, check):
    """sources maps label to (explicit parent path or None, recorded run hash).

    None intentionally means the selected manifest itself: no implicit parent
    discovery and no substituted run hashes. Membership compares entire objects.
    """
    selected_path = pathlib.Path(selected_path)
    selected, selected_sha = read_snapshot(selected_path)
    page_index(selected, "selected_manifest", check)
    scope = {"qualification": "Only selected pages are compared; no claim of completion of a larger parent corpus.",
             "selected_manifest": {"path": str(selected_path), "sha256": selected_sha, "pages": len(selected["pages"])},
             "ordered_selected_ids": [p["id"] for p in selected["pages"]], "sources": {}}
    for label, (source_path, recorded_sha) in sources.items():
        source_path = pathlib.Path(source_path) if source_path is not None else selected_path
        parent, parent_sha = read_snapshot(source_path)
        index = page_index(parent, label + "_manifest", check)
        check(label + ".source_manifest_sha256", parent_sha, recorded_sha)
        for field in ["dataset", "revision", "annotation_sha256", "corpus_manifest_sha256", "rgb_policy", "ground_truth_policy"]:
            if field in selected or field in parent:
                check(label + ".manifest_metadata." + field, selected.get(field), parent.get(field))
        for page in selected["pages"]:
            check(label + ".selected_page_member." + page["id"], index.get(page["id"]), page)
        scope["sources"][label] = {"path": str(source_path), "sha256": parent_sha,
                                    "recorded_run_manifest_sha256": recorded_sha, "pages": len(parent["pages"]),
                                    "entire_source_selected": set(index) == {p["id"] for p in selected["pages"]}}
    return selected, scope


def file_check(path, expected, check, label):
    try:
        actual = sha256(resolve_recorded_path(path))
    except OSError as error:
        actual = "unreadable: " + str(error)
    check(label, actual, expected)
    return actual


def validate_page_assets(page, check):
    key = sample_key(page)
    file_check(page["canonical_path"], page["canonical_png_sha256"], check, key + ".canonical_png")
    with Image.open(resolve_recorded_path(page["canonical_path"])) as image:
        rgb = image.convert("RGB")
        check(key + ".canonical_rgb", hashlib.sha256(rgb.tobytes()).hexdigest(), page["rgb_sha256"])
        check(key + ".original_dimensions", list(rgb.size), [page["width"], page["height"]])
    file_check(page["ground_truth_path"], page["ground_truth_sha256"], check, key + ".ground_truth")
    if "annotation_path" in page:
        annotation = read_object(resolve_recorded_path(page["annotation_path"]))
        check(key + ".annotation", canonical_sha256(annotation), page["annotation_page_sha256"])


def model_asset_evidence(model_dir, check):
    assets = {}
    for name, expected in ASSETS.items():
        path = pathlib.Path(model_dir) / name
        assets[name] = {"path": str(path), "sha256": file_check(path, expected, check, "checked_asset." + name)}
    return {"scope": "Actual config/tokenizer bytes checked now against fixed model pins; this is not retroactive startup attestation. Checkpoint identity uses each run's recorded digest and loader contract, not a new 1 GB rehash.", "assets": assets}


def semantic_contract(config, label, precision, check, gpu=False):
    for key, expected in [("model_revision", REVISION), ("weights_sha256", WEIGHT_SHA256), ("precision", precision)]:
        check(label + "." + key, config.get(key), expected)
    missing = []
    for key, expected in [("prompt", PROMPT), ("config_sha256", ASSETS["config.json"])]:
        if key in config:
            check(label + "." + key, config[key], expected)
        else:
            missing.append(key)
    policy_key = "greedy_tie_rule" if gpu else "greedy_policy"
    expected = "torch.argmax: first vocabulary index among equal maxima" if gpu else GREEDY
    if policy_key in config:
        check(label + "." + policy_key, config[policy_key], expected)
    else:
        missing.append(policy_key)
    return {"missing_startup_fields": missing,
            "qualification": "Missing fields are not filled into old records. Available source evidence and checked assets are reported separately; output comparison is not complete startup attestation."}


def validate_cpu_result(result, config, check, label):
    valid = validate_inference_result(result, config, check, label)
    if not isinstance(result, dict):
        return False
    ids = result.get("token_ids")
    if isinstance(ids, list):
        no_early_eos = not any(i in (11, 263) for i in ids[:-1])
        check(label + ".no_tokens_after_eos", no_early_eos, True)
        valid &= no_early_eos
    # More recent runners add optional stage timings. Their values may differ
    # across platforms; malformed/nonfinite present values still are not valid.
    timings = result.get("timings")
    for name, value in (timings if isinstance(timings, dict) else {}).items():
        finite = value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0)
        check(label + ".finite_extra_timing." + name, finite, True)
        valid &= finite
    return valid


def cpu_build_evidence(path, config, check, label):
    if path is None:
        return {"verified": False, "qualification": "No build archive supplied for comparison; recorded binary/source fields, if any, are retained without an independent archive check."}
    path = pathlib.Path(path)
    build = read_object(path)
    check(label + ".build_manifest_sha256", sha256(path), config.get("build_manifest_sha256"))
    check(label + ".build_complete", build.get("status"), "complete")
    check(label + ".build_sources_unchanged", build.get("source_unchanged_during_build"), True)
    check(label + ".build_binary_contract", build.get("binary_sha256"), config.get("binary_sha256"))
    file_check(path.parent / build["binary"], build["binary_sha256"], check, label + ".binary_file")
    archive = path.parent / build["source_archive"]
    file_check(archive, build["source_archive_sha256"], check, label + ".source_archive")
    with zipfile.ZipFile(archive) as source:
        for name, expected in build["source_sha256"].items():
            check(label + ".archived_source." + name, hashlib.sha256(source.read(name)).hexdigest(), expected)
    aliases = {"harness": "examples/corpus_eval.rs", "corpus_record": "examples/support/corpus_record.rs", "lock": "Cargo.lock", "cargo_manifest": "Cargo.toml", "toolchain": "rust-toolchain.toml"}
    for name, expected in config["source_sha256"].items():
        check(label + ".embedded_source." + name, build["source_sha256"].get(aliases.get(name, "src/" + name + ".rs")), expected)
    return {"path": str(path), "sha256": sha256(path), "source_archive_sha256": build["source_archive_sha256"], "binary_sha256": build["binary_sha256"], "qualification": "Recorded startup build identity checked against preserved binary and archived sources; not a hermetic rebuild."}


def exact_output(left, right):
    fields = ["token_ids", "text", "finish_reason", "output_tokens", "input_tokens", "width", "height", "precision", "teacher_forced"]
    return {field + "_exact": left.get(field) == right.get(field) for field in fields}
