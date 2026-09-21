#!/usr/bin/env python3
"""Pure host-side input/output checks for new reproducible GPU corpus runs.

Prepared separately from the active historical driver. Importing this module
does not initialize CUDA or change an existing run.
"""
import hashlib
import json
import os
import pathlib
import re
import tempfile

from reference_run_identity import preserve_reference_identity
from validate_gpu_reference_record import validate_gpu_reference_record


def read_snapshot(path):
    data = pathlib.Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value, hashlib.sha256(data).hexdigest()


def sample_key(page):
    return pathlib.PurePosixPath(page["canonical_path"].replace("\\", "/")).parent.name


def selected_pages(manifest, limit=None):
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Manifest must contain a nonempty page list")
    if limit is not None and (type(limit) is not int or not 0 < limit <= len(pages)):
        raise ValueError("Limit must select between one and all manifest pages")
    ids, samples = set(), set()
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("id"), str) or not page["id"]:
            raise ValueError("Page is missing a nonempty ID")
        for key in ["canonical_path", "ground_truth_path", "canonical_png_sha256", "ground_truth_sha256", "rgb_sha256", "category"]:
            if not isinstance(page.get(key), str) or not page[key]:
                raise ValueError("Page is missing " + key)
        sample = sample_key(page)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample) or sample.lower() in ("run", "summary", "provenance") or page["id"] in ids or sample in samples:
            raise ValueError("Duplicate or invalid page identity")
        ids.add(page["id"])
        samples.add(sample)
    return pages[:limit]


def prepare_output(output, configuration, identity, contents, *, resume):
    """Check the whole configuration before creating or checking any archive."""
    output = pathlib.Path(output)
    run_path = output / "run.json"
    require_fresh_greedy(configuration, "configuration")
    if resume:
        if not run_path.is_file():
            raise ValueError("Resume requires an existing run and startup identity")
        previous, _ = read_snapshot(run_path)
        if previous.get("configuration") != configuration:
            raise ValueError("Output contains a different frozen run configuration")
        require_fresh_greedy(previous, "saved run")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("New output must be empty; use explicit resume for an existing run")
    output.mkdir(parents=True, exist_ok=True)
    evidence = preserve_reference_identity(output, identity, contents, new_run=not resume)
    if resume and previous.get("startup_identity") != evidence:
        raise ValueError("Run does not bind its preserved startup identity")
    return evidence


def require_fresh_greedy(value, label):
    if value.get("teacher_forced") is not False or value.get("inference_reexecuted") is not True:
        raise ValueError(label + " must explicitly identify fresh, non-teacher-forced inference")
    if any(key in value for key in ("postprocessing_replay", "derived_text_replay")):
        raise ValueError(label + " must not carry derived-text replay metadata")


def validate_completed_page(record, page, configuration):
    def check(name, actual, expected):
        if actual != expected:
            raise ValueError(f"Invalid completed record {name}: {actual!r} != {expected!r}")

    sample = sample_key(page)
    validate_gpu_reference_record(record, configuration, check, sample)
    require_fresh_greedy(configuration, "configuration")
    require_fresh_greedy(record, sample)
    for key, expected in [("id", page["id"]), ("sample_id", sample), ("category", page["category"]),
                          ("canonical_rgb_sha256", page["rgb_sha256"])]:
        check(sample + "." + key, record.get(key), expected)


def atomic_json(path, value):
    """Publish only complete UTF-8 JSON; never accept nonfinite numbers."""
    path = pathlib.Path(path)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=path.name + ".",
                                         suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
