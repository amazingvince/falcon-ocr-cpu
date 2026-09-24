#!/usr/bin/env python3
"""Run the pinned evaluator's annotation merger before model quality evaluation."""
import argparse
import copy
import hashlib
import json
import pathlib
import subprocess
import sys

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256
from prepare_official_evaluation import EVALUATOR_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--evaluator", type=pathlib.Path, default=pathlib.Path("artifacts/OmniDocBench-eval"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.evaluator), "rev-parse", "HEAD"], text=True).strip()
    if revision != EVALUATOR_REVISION:
        raise ValueError("Evaluator revision mismatch")
    sys.path.insert(0, str(args.evaluator.resolve()))
    from dataset.end2end_dataset import End2EndDataset
    evaluator = End2EndDataset.__new__(End2EndDataset)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for page in manifest["pages"]:
        annotation = json.loads(pathlib.Path(page["annotation_path"]).read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(annotation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != page["annotation_page_sha256"]:
            raise ValueError(f"Annotation hash mismatch: {page['id']}")
        ids = [item["anno_id"] for item in annotation["layout_dets"]]
        dangling = [relation for relation in annotation["extra"]["relation"]
            if relation["relation_type"] == "truncated" and any(relation[key] not in ids
                for key in ["source_anno_id", "target_anno_id"])]
        adapted = copy.deepcopy(annotation)
        adapted["extra"]["relation"] = [r for r in adapted["extra"]["relation"] if r not in dangling]
        record = {"id": page["id"], "annotation_page_sha256": digest,
                  "duplicate_annotation_ids": len(ids) != len(set(ids)), "dangling_truncated_relations": dangling}
        for name, value in [("original", annotation), ("adapted", adapted)]:
            try:
                elements = evaluator.get_page_elements(value)
                record[name] = {"merger_passed": True, "element_counts": {key: len(items) for key, items in elements.items()}}
            except Exception as error:
                record[name] = {"merger_passed": False, "error_type": type(error).__name__, "error": str(error)}
        records.append(record)
    report = {"manifest_sha256": sha256(args.manifest), "evaluator_revision": revision,
        "script_sha256": sha256(pathlib.Path(__file__)), "pages": len(records),
        "original_merger_failures": sum(not page["original"]["merger_passed"] for page in records),
        "adapted_merger_failures": sum(not page["adapted"]["merger_passed"] for page in records),
        "dangling_relations": sum(len(page["dangling_truncated_relations"]) for page in records),
        "duplicate_annotation_id_pages": sum(page["duplicate_annotation_ids"] for page in records),
        "records": records,
        "scope": "Annotation-merger compatibility only; no predictions, OCR scores or source annotations changed."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    if report["adapted_merger_failures"] or report["duplicate_annotation_id_pages"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
