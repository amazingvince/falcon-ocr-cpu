#!/usr/bin/env python3
"""Copy an exact completed manifest prefix without changing saved inference bytes."""
import argparse
import hashlib
import json
import pathlib


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    run_bytes = (args.source / "run.json").read_bytes()
    run = json.loads(run_bytes)
    if not 0 < args.count <= len(manifest["pages"]):
        raise ValueError("Count must select a nonempty manifest prefix")
    if run["contract"]["manifest_sha256"] != digest(manifest_bytes):
        raise ValueError("Source run and manifest differ")
    selected = manifest["pages"][:args.count]
    blobs = {"run.json": run_bytes}
    records = []
    for page in selected:
        key = pathlib.Path(page["canonical_path"]).parent.name
        path = args.source / (key + ".json")
        raw = path.read_bytes()
        record = json.loads(raw)
        if (record.get("error") is not None or not isinstance(record.get("result"), dict)
                or record.get("id") != page["id"]
                or record.get("contract_sha256") != run["contract_sha256"]):
            raise ValueError("Selected source record is incomplete or inconsistent: " + str(path))
        blobs[path.name] = raw
        records.append({"id": page["id"], "source_path": str(path), "snapshot_name": path.name,
                        "sha256": digest(raw)})
    if (args.source / "binary-snapshot.json").is_file():
        blobs["binary-snapshot.json"] = (args.source / "binary-snapshot.json").read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in blobs.items():
        (args.output / name).write_bytes(raw)
    for row in records:
        if digest(pathlib.Path(row["source_path"]).read_bytes()) != row["sha256"]:
            raise RuntimeError("Source record changed during snapshot")
    if (args.source / "run.json").read_bytes() != run_bytes or args.manifest.read_bytes() != manifest_bytes:
        raise RuntimeError("Run or manifest changed during snapshot")
    script_copy = args.output / pathlib.Path(__file__).name
    script_copy.write_bytes(pathlib.Path(__file__).read_bytes())
    receipt = {"schema_version": 1, "source_run_path": str(args.source / "run.json"),
               "source_run_sha256": digest(run_bytes), "manifest_path": str(args.manifest),
               "manifest_sha256": digest(manifest_bytes), "selected_first_pages": args.count,
               "full_manifest_pages": len(manifest["pages"]), "records": records,
               "copied_files": {name: digest(raw) for name, raw in blobs.items()},
               "script_sha256": digest(script_copy.read_bytes()),
               "qualification": "After-inference byte-identical snapshot of the first requested manifest pages. No inference, no startup attestation, and no record changes."}
    (args.output / "snapshot.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pages": args.count, "output": str(args.output), "source_run_sha256": digest(run_bytes)}))


if __name__ == "__main__":
    main()
