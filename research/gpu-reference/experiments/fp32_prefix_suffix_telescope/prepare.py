#!/usr/bin/env python3
"""Prepare a fresh source-bound plan only after the quiet-window release."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from contract import (ROOT, KIND, PINS, RUNTIME, STAGES, BRANCHES, CONTROL_BRANCHES,
                      START_LAYER, ENTRY_STATE, ENDPOINT, SOURCE_FILES, resolve,
                      sha, require, read_bound, write_new, plan_window, stable)
from source_guard import verify_sources
from evidence import load_evidence


def prepare(args):
    require(args.quiet_window_released, "Preparation is held during timing")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Fresh project output required")
    manifest = read_bound(resolve(PINS["model_manifest"][0]), PINS["model_manifest"][1])
    files = sorted(set(SOURCE_FILES) | {"artifacts/model/" + n for n in manifest["files"]
                                      if n != "model.safetensors"})
    source_bytes = {name: resolve(name).read_bytes() for name in files}
    source_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()}
    source_guard = verify_sources()
    for name, (path, digest) in PINS.items():
        require(sha(resolve(path)) == digest, "Saved evidence changed: " + name)
    bound = {resolve(path): digest for path, digest in PINS.values()}
    bound.update({resolve(name): h for name, h in source_hashes.items()})
    evidence = load_evidence(bound)
    stable(bound)
    output.mkdir(parents=True)
    archive_path = output / "source.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in source_bytes.items():
            archive.writestr(name, raw)
    with zipfile.ZipFile(archive_path) as archive:
        require(set(archive.namelist()) == set(source_bytes), "Source archive inventory changed")
        require(all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in source_hashes.items()),
                "Source archive bytes differ")
    plan = {"kind": KIND, "runtime": RUNTIME, "stages": STAGES, "control_branches": CONTROL_BRANCHES,
            "branches": BRANCHES, "start_layer": START_LAYER, "entry_state": ENTRY_STATE,
            "endpoint": ENDPOINT, "state_raw_sha256": evidence["proof"]["state_raw_sha256"],
            "historical_evidence": evidence["proof"],
            "inputs": {k: {"path": p, "sha256": h} for k, (p, h) in PINS.items()},
            "source_sha256": source_hashes, "source_archive_sha256": sha(archive_path),
            "source_guard": source_guard, "model_directory": "artifacts/model",
            "model_files": manifest["files"], "execution_status": "prepared_not_executed",
            "limits": ["Three exact controls, then eight fixed unseen CPU-prefix/GPU-suffix endpoints, once each.",
                       "47/17/30 complete control tensors gate the new branches; 54 blocks and 11 native layer9 pre-QKV calls total.",
                       "No Rust build/inference, generation or full-model execution.",
                       "No numerical policy change, hidden-state qualification or performance claim.",
                       "Historical startup source gaps and changed observer/allocation history remain."]}
    write_new(output / "plan.json", plan)
    digest = sha(output / "plan.json")
    plan_window(output / "plan.json", digest)
    stable(bound)
    write_new(output / "preparation.json", {"kind": KIND + "-preparation", "status": "prepared_not_executed",
              "plan_sha256": digest, "source_archive_sha256": plan["source_archive_sha256"],
              "native_source_guard": source_guard, "model_or_cuda_work_performed": False})
    print(digest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quiet-window-released", action="store_true")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
