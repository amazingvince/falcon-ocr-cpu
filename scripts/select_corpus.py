#!/usr/bin/env python3
"""Build a frozen research evaluation split from pinned OmniDocBench annotations.

Only downloads annotations, never image or PDF archives. Image downloading is a
separate explicitly sized stage. The source data is research-only/noncommercial.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import urllib.parse
import urllib.request

REVISION = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
DATASET = "opendatalab/OmniDocBench"
ANNOTATION_SHA256 = "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"


def blocks(row):
    return [b for b in row["layout_dets"] if not b.get("ignore", False)]


def char_count(row):
    return sum(len(b.get("text", "")) for b in blocks(row))


def tiny_height(row):
    heights = []
    for b in blocks(row):
        for line in b.get("line_with_spans", []):
            poly = line.get("poly", [])
            if poly and line.get("text"):
                heights.append((max(poly[1::2]) - min(poly[1::2])) / row["page_info"]["height"])
    if not heights:
        # A single-line text block is usable when span-level boxes are absent.
        for b in blocks(row):
            if b.get("text") and "\n" not in b["text"] and len(b["text"]) <= 120 and b.get("poly"):
                poly = b["poly"]
                heights.append((max(poly[1::2]) - min(poly[1::2])) / row["page_info"]["height"])
    return sorted(heights)[len(heights) // 2] if heights else 1.0


def family(name):
    note = re.search(r"^(notes_[0-9a-f]+)_\d+\.[^.]+$", name, re.IGNORECASE)
    if note:
        return note[1]
    match = re.search(r"^(.*\.pdf)_\d+\.[^.]+$", name, re.IGNORECASE)
    return match[1] if match else name


def eligible(row, category):
    attrs = row["page_info"]["page_attribute"]
    kinds = {b["category_type"] for b in blocks(row)}
    if not any(b.get("text") or b.get("latex") or b.get("html") for b in blocks(row)):
        return False
    if category == "degraded":
        return attrs["data_source"] == "historical_document" or bool(set(attrs.get("special_issue", [])) & {"fuzzy_scan", "fuzzy_content", "transparent_pages", "geometric_deformation"})
    if category == "handwriting":
        return attrs["data_source"] == "note" or "handwriting" in attrs.get("special_issue", [])
    if category == "formulas":
        return "equation_isolated" in kinds and attrs["data_source"] != "note"
    if category == "tables":
        return any(b["category_type"] == "table" and b.get("html") for b in blocks(row))
    if category == "tiny_text":
        return char_count(row) >= 1000 and tiny_height(row) <= 0.012
    if category == "multi_column":
        return attrs["layout"] in ["double_column", "three_column", "1andmore_column"]
    if category == "ordinary":
        return attrs["data_source"] in ["book", "research_report", "academic_literature", "magazine"] and attrs["layout"] == "single_column" and char_count(row) >= 100
    raise ValueError(category)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reference/corpus-manifest-v2.json"))
    args = p.parse_args()
    path = pathlib.Path("artifacts/corpus-source-metadata/OmniDocBench.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/OmniDocBench.json"
        print(f"Downloading annotations only (42,208,096 bytes): {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ANNOTATION_SHA256
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = {"pages": len(data), "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ["data_source", "language", "layout", "fuzzy_scan"]:
        stats[name] = dict(collections.Counter(str(row["page_info"]["page_attribute"].get(name)) for row in data))
    stats["special_issue"] = dict(collections.Counter(item for row in data for item in row["page_info"]["page_attribute"].get("special_issue", [])))
    stats["example_image_names"] = [row["page_info"]["image_path"] for row in data[300:310]]
    stats["handwriting_families"] = dict(collections.Counter(family(row["page_info"]["image_path"]) for row in data if eligible(row, "handwriting")))
    stats["sample_page_info"] = data[0]["page_info"]
    stats["sample_blocks"] = [{k: v for k, v in b.items() if k not in ["text", "latex", "html", "line_with_spans"]}
                               for b in data[0]["layout_dets"][:2]]
    print(json.dumps(stats, indent=2))
    if args.inspect_only:
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection {args.output}")
    counts = {"degraded": 25, "handwriting": 25, "formulas": 30, "tables": 30,
              "tiny_text": 30, "multi_column": 25, "ordinary": 35}
    smoke_counts = {"degraded": 3, "handwriting": 3, "formulas": 4, "tables": 4,
                    "tiny_text": 4, "multi_column": 3, "ordinary": 3}
    selected, used, used_families = [], set(), set()
    available = {c: sum(eligible(r, c) for r in data) for c in counts}
    # The first manifest missed notes_<hash>_<page> notebook families. Reserve a
    # whole notebook before evaluation selection; page shuffling is insufficient.
    notebook_counts = collections.Counter(family(r["page_info"]["image_path"]) for r in data
                                         if eligible(r, "handwriting") and family(r["page_info"]["image_path"]).startswith("notes_"))
    calibration_notebook = min((f for f, n in notebook_counts.items() if n >= 8), key=lambda f: (notebook_counts[f], f))

    def rank(row, split, category):
        return hashlib.sha256(("falcon-cpu-corpus-v2\0" + split + "\0" + category + "\0" + row["page_info"]["image_path"]).encode()).hexdigest()

    for split, allocation in [("evaluation", counts), ("calibration", {c: (16 if c == "ordinary" else 8) for c in counts})]:
        for category, count in allocation.items():
            rows = [r for r in data if eligible(r, category) and r["page_info"]["image_path"] not in used
                    and family(r["page_info"]["image_path"]) not in used_families
                    and (family(r["page_info"]["image_path"]) != calibration_notebook or (split == "calibration" and category == "handwriting"))]
            rows.sort(key=lambda r: rank(r, split, category))
            if category == "handwriting":
                if split == "calibration":
                    rows = [r for r in rows if family(r["page_info"]["image_path"]) == calibration_notebook]
                else:
                    # Interleave source families; cap each notebook at 13/25.
                    groups = collections.defaultdict(list)
                    for row in rows:
                        groups[family(row["page_info"]["image_path"])].append(row)
                    families = sorted(groups, key=lambda f: (not f.startswith("notes_"), rank(groups[f][0], split, category)))
                    rows = [groups[f][i] for i in range(13) for f in families if i < len(groups[f])]
            chosen = []
            chosen_families = set()
            for row in rows:
                info = row["page_info"]
                name = info["image_path"]
                if category != "handwriting" and family(name) in chosen_families:
                    continue
                used.add(name)
                chosen_families.add(family(name))
                chosen.append(row)
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f"Insufficient eligible pages/families in {split}/{category}: {len(chosen)} < {count}; raw pools {available}")
            used_families.update(chosen_families)
            smoke_names = set()
            if split == "evaluation":
                # Prefer genuine dense columns for this smoke stratum, then
                # distinct families (including distinct handwritten notebooks).
                smoke_rows = sorted(chosen, key=lambda r: (category == "multi_column" and r["page_info"]["page_attribute"]["layout"] not in ["double_column", "three_column"], chosen.index(r)))
                smoke_families = set()
                for row in smoke_rows:
                    name = row["page_info"]["image_path"]
                    if family(name) not in smoke_families:
                        smoke_names.add(name)
                        smoke_families.add(family(name))
                    if len(smoke_names) == smoke_counts[category]:
                        break
                if len(smoke_names) != smoke_counts[category]:
                    raise ValueError(f"Insufficient distinct smoke families for {category}")
            for i, row in enumerate(chosen):
                info = row["page_info"]
                name = info["image_path"]
                selected.append({"id": f"omnidocbench:{name}", "split": split, "category": category,
                                 "smoke": name in smoke_names,
                                 "image_path": "images/" + name, "image_url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/images/{urllib.parse.quote(name)}",
                                 "source_family": family(name), "width": info["width"], "height": info["height"],
                                 "attributes": info["page_attribute"], "ground_truth": {"annotation_file": "OmniDocBench.json", "image_path": name,
                                 "page_sha256": hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
                                 "annotation_text_characters": char_count(row), "median_line_height_ratio": tiny_height(row)})
    evaluation_families = {p["source_family"] for p in selected if p["split"] == "evaluation"}
    calibration_families = {p["source_family"] for p in selected if p["split"] == "calibration"}
    assert not evaluation_families & calibration_families
    manifest = {"schema_version": 2, "status": "selected_metadata_only_images_not_downloaded_or_visually_verified",
                "dataset": DATASET, "revision": REVISION, "annotation_sha256": ANNOTATION_SHA256,
                "source_terms": "Research purposes only; not for commercial use. Do not redistribute source images in runner releases.",
                "evaluation_count": 200, "smoke_count": 24, "calibration_count": 64,
                "selection": "Stable SHA-256 ranking with seed falcon-cpu-corpus-v2; ordered category allocation; exclusive images and known filename-derived document/notebook families across splits. Evaluation handwriting interleaves families with at most 13 pages per notebook. Other categories use distinct families. Smoke uses distinct families and prefers double/three-column layouts.",
                "supersedes": "corpus-manifest-v1.json has known handwritten notebook overlap across evaluation/calibration and is suitable only for qualitative smoke.",
                "handwriting_calibration_notebook": calibration_notebook,
                "handwriting_limitations": "Only three source notebook families are identifiable. Calibration uses 8 pages from the smallest notebook; evaluation uses the other two plus opaque-ID handwriting examples. Handwriting language/writer diversity remains limited.",
                "tiny_text_rule": "At least 1000 annotated characters and median line height <= 1.2% of page height; visual validation still required",
                "family_limitation": "Filename PDF page suffixes and notes_<hash>_<page> notebooks are grouped; opaque image IDs do not identify original document families. Content/perceptual duplicate checks remain required.",
                "known_family_overlap_between_splits": [],
                "available_counts_before_exclusive_assignment": available, "category_counts": counts, "smoke_category_counts": smoke_counts,
                "pages": selected}
    out = args.output
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out), "evaluation": 200, "smoke": 24, "calibration": 64, "available": available}, indent=2))


if __name__ == "__main__":
    main()
