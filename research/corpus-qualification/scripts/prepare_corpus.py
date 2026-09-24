#!/usr/bin/env python3
"""Materialize a frozen corpus selection and audit candidate duplicates.

Image data remains in ignored artifacts and is research-only. The derived lock
does not assert visual category correctness or published benchmark scores.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import shutil
import urllib.request

import PIL
from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256
from prepare_corpus_smoke import ground_truth


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def difference_hash(image):
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Refusing to overwrite changed corpus lock: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("reference/corpus-manifest-v2.json"))
    parser.add_argument("--annotations", type=pathlib.Path, default=pathlib.Path("artifacts/corpus-source-metadata/OmniDocBench.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/corpus/v2"))
    parser.add_argument("--lock-prefix", type=pathlib.Path, default=pathlib.Path("reference/corpus-v2"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-lock", type=pathlib.Path, action="append", default=[],
                        help="Reuse source images only after their frozen source hash is checked")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256(args.annotations) != manifest["annotation_sha256"]:
        raise ValueError("Annotation source hash mismatch")
    annotations = {row["page_info"]["image_path"]: row
                   for row in json.loads(args.annotations.read_text(encoding="utf-8"))}
    pages = manifest["pages"]
    if len(pages) != manifest["evaluation_count"] + manifest["calibration_count"]:
        raise ValueError("Corpus selection count mismatch")
    previous = {}
    cache_locks = [pathlib.Path("reference/corpus-smoke-lock-v1.json"), *args.cache_lock]
    for cache_lock in cache_locks:
        if not cache_lock.exists():
            if cache_lock in args.cache_lock:
                raise ValueError(f"Requested cache lock does not exist: {cache_lock}")
            continue
        for page in json.loads(cache_lock.read_text(encoding="utf-8"))["pages"]:
            old = previous.get(page["id"])
            if old and (old["source_url"], old["source_sha256"]) != (page["source_url"], page["source_sha256"]):
                raise ValueError(f"Conflicting cached source provenance: {page['id']}")
            previous[page["id"]] = page
    args.output.mkdir(parents=True, exist_ok=True)
    # Leave enough room for build outputs and traces on the shared system volume.
    if shutil.disk_usage(args.output).free < 2 * 1024**3:
        raise ValueError("Less than 2 GiB free before corpus preparation; select another output volume")

    def prepare(page):
        sample = hashlib.sha256(page["id"].encode()).hexdigest()[:16]
        output = args.output / sample
        output.mkdir(parents=True, exist_ok=True)
        source = output / ("source" + pathlib.Path(page["image_path"]).suffix)
        if not source.exists():
            temporary = source.with_suffix(source.suffix + ".partial")
            cached = previous.get(page["id"])
            reviewed_source = pathlib.Path(page["reviewed_source_path"]) if "reviewed_source_path" in page else None
            if reviewed_source and reviewed_source.exists():
                if sha256(reviewed_source) != page["reviewed_source_sha256"]:
                    raise ValueError(f"Visually reviewed source hash mismatch: {reviewed_source}")
                shutil.copyfile(reviewed_source, temporary)
            elif cached and cached["source_url"] == page["image_url"]:
                old_source = pathlib.Path(cached["source_path"])
                if sha256(old_source) != cached["source_sha256"]:
                    raise ValueError(f"Cached source hash mismatch: {old_source}")
                shutil.copyfile(old_source, temporary)
            else:
                with urllib.request.urlopen(page["image_url"], timeout=120) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            temporary.replace(source)
        if "reviewed_source_sha256" in page and sha256(source) != page["reviewed_source_sha256"]:
            raise ValueError(f"Materialized source differs from the visually reviewed image: {page['id']}")
        with Image.open(source) as loaded:
            mode = loaded.mode
            rgb = loaded.convert("RGB")
        if rgb.size != (page["width"], page["height"]):
            raise ValueError(f"Image dimensions disagree with selected annotation: {page['id']}")
        canonical = output / "canonical-rgb.png"
        rgb.save(canonical)
        annotation = annotations[page["ground_truth"]["image_path"]]
        annotation_sha = stable_hash(annotation)
        if annotation_sha != page["ground_truth"]["page_sha256"]:
            raise ValueError(f"Annotation page hash mismatch: {page['id']}")
        annotation_path = output / "annotation.json"
        annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        truth = ground_truth(annotation)
        if not truth:
            raise ValueError(f"No ordered text ground truth: {page['id']}")
        truth_path = output / "ground-truth.txt"
        truth_path.write_text(truth + "\n", encoding="utf-8")
        record = {"id": page["id"], "category": page["category"], "split": page["split"],
                  "smoke": page["smoke"], "source_family": page["source_family"],
                  "source_url": page["image_url"], "source_path": source.as_posix(),
                  "source_sha256": sha256(source), "source_bytes": source.stat().st_size, "source_mode": mode,
                  "canonical_path": canonical.as_posix(), "canonical_png_sha256": sha256(canonical),
                  "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                  "width": rgb.width, "height": rgb.height, "difference_hash_64": difference_hash(rgb),
                  "ground_truth_path": truth_path.as_posix(), "ground_truth_sha256": sha256(truth_path),
                  "normalized_ground_truth_sha256": hashlib.sha256(" ".join(truth.split()).encode()).hexdigest(),
                  "annotation_path": annotation_path.as_posix(), "annotation_page_sha256": annotation_sha,
                  "attributes": page["attributes"],
                  "visual_review": page.get("visual_review", "See source selection manifest and its visual-review report")}
        print(json.dumps({"prepared": sample, "split": record["split"], "category": record["category"]}), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        prepared = list(pool.map(prepare, pages))
    candidates, exact_cross_split = [], []
    for index, left in enumerate(prepared):
        for right in prepared[index + 1:]:
            distance = (int(left["difference_hash_64"], 16) ^ int(right["difference_hash_64"], 16)).bit_count()
            exact_rgb = left["rgb_sha256"] == right["rgb_sha256"]
            exact_text = left["normalized_ground_truth_sha256"] == right["normalized_ground_truth_sha256"]
            cross_split = left["split"] != right["split"]
            if exact_rgb or exact_text or (cross_split and distance <= 4):
                pair = {"left": left["id"], "right": right["id"], "cross_split": cross_split,
                        "exact_rgb": exact_rgb, "exact_normalized_ground_truth": exact_text,
                        "difference_hash_hamming": distance}
                candidates.append(pair)
                if cross_split and (exact_rgb or exact_text):
                    exact_cross_split.append(pair)
    common = {"schema_version": 2, "dataset": manifest["dataset"], "revision": manifest["revision"],
              "corpus_manifest_sha256": sha256(args.manifest), "annotation_sha256": sha256(args.annotations),
              "preparation_script_sha256": sha256(pathlib.Path(__file__)),
              "ground_truth_script_sha256": sha256(pathlib.Path(__file__).with_name("prepare_corpus_smoke.py")),
              "pillow_version": PIL.__version__,
              "source_terms": manifest["source_terms"],
              "rgb_policy": "Pinned Pillow decode, then RGB conversion before upstream preprocessing; no EXIF auto-orientation.",
              "ground_truth_policy": "Nonignored blocks with numeric order >= 0, sorted by order/index; text, HTML tables, LaTeX equations; NFC/LF, trim edges, join with two newlines.",
              "metrics_note": "Assembled text supports diagnostic normalized CER only; retain original blocks for text/table/formula evaluation.",
              "qualification": "Pixel/annotation freeze only. Visual category evidence belongs to the selection manifest/review report; newly found duplicate candidates require review. No quality or performance acceptance claim."}
    audit = {**common, "page_count": len(prepared), "duplicate_candidates": candidates,
             "exact_cross_split_matches": exact_cross_split,
             "difference_hash_note": "64-bit horizontal gradient hash is a review heuristic; similar layout can create false positives and dissimilar scans can evade it."}
    freeze(pathlib.Path(str(args.lock_prefix) + "-duplicates.json"), audit)
    if exact_cross_split:
        raise ValueError("Exact image/text duplicates cross evaluation and calibration; review and select a new manifest revision")
    for name, subset in [("evaluation", [p for p in prepared if p["split"] == "evaluation"]),
                         ("calibration", [p for p in prepared if p["split"] == "calibration"]),
                         ("smoke", [p for p in prepared if p["smoke"]])]:
        expected = manifest[name + "_count"]
        if len(subset) != expected:
            raise ValueError(f"Prepared {name} count differs from {expected}")
        freeze(pathlib.Path(str(args.lock_prefix) + "-" + name + "-lock.json"), {**common, "selection": name, "pages": subset})
    print(json.dumps({"prepared_pages": len(prepared), "downloaded_source_bytes": sum(p["source_bytes"] for p in prepared),
                      "duplicate_candidates": len(candidates), "exact_cross_split_matches": len(exact_cross_split)}, indent=2))


if __name__ == "__main__":
    main()
