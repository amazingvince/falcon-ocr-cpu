#!/usr/bin/env python3
"""Freeze a visually reviewed v3 while preserving every v1/v2 artifact.

Run from the repository root. Candidate review decisions are inputs, not inferred
from model output. --draft checks quotas and writes only an artifact preview.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import select_corpus as original

ROOT = Path(__file__).resolve().parents[1]
SEED = "falcon-cpu-corpus-v3"
CATEGORIES = {"degraded": 25, "handwriting": 25, "formulas": 30, "tables": 30, "tiny_text": 30, "multi_column": 25, "ordinary": 35}
SMOKE = {"degraded": 3, "handwriting": 3, "formulas": 4, "tables": 4, "tiny_text": 4, "multi_column": 3, "ordinary": 3}
CALIBRATION = {c: 16 if c == "ordinary" else 8 for c in CATEGORIES}


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def rank(split, category, key):
    return hashlib.sha256(f"{SEED}\0{split}\0{category}\0{key}".encode()).hexdigest()


def source_family(name, row, overrides):
    if name in overrides:
        return overrides[name]
    lower = name.lower().replace(" ", "")
    if "putnam-archive" in lower:
        return "publication:putnam-archive"
    for needle, family in [("theeconomist", "the-economist"), ("theguardian", "the-guardian"), ("dailymail", "daily-mail"), ("nydailynews", "ny-daily-news"), ("thenewyorker", "the-new-yorker")]:
        if needle in lower:
            return "publication:" + family
    text = " ".join(b.get("text", "") for b in row["layout_dets"])
    if re.search(r"Federal\s+Register", text, re.I):
        return "publication:federal-register"
    if re.search(r"Grzimek.{0,3}s Animal Life Encyclopedia", text, re.I):
        return "publication:grzimeks-animal-life-encyclopedia"
    for pattern in [r"^(.*)_page_\d+\.[^.]+$", r"^((?:notes|newspaper)_[0-9a-f]+)_\d+\.[^.]+$", r"^(.*\.pdf)_\d+\.[^.]+$"]:
        match = re.match(pattern, name, re.I)
        if match:
            return match[1]
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    annotation_path = "artifacts/corpus-source-metadata/OmniDocBench.json"
    assert sha(annotation_path) == original.ANNOTATION_SHA256
    rows = read(annotation_path)
    by_name = {r["page_info"]["image_path"]: r for r in rows}
    v2 = read("reference/corpus-manifest-v2.json")
    previous_calibration_ids = {p["id"] for p in v2["pages"] if p["split"] == "calibration"}
    v2_locks = read("reference/corpus-v2-evaluation-lock.json")["pages"] + read("reference/corpus-v2-calibration-lock.json")["pages"]
    v2_locked_by_id = {p["id"]: p for p in v2_locks}
    v2_review = read("reference/corpus-v2-visual-review.json")
    v2_reviews = {p["id"]: p for p in v2_review["pages"]}
    review_path = "reference/corpus-v3-candidate-review.json"
    # Once frozen, the durable review contains the complete candidate pool and
    # all decisions, so reproducing a draft does not depend on scratch TSV/indexes.
    if (ROOT / review_path).exists():
        frozen_review = read(review_path)
        candidates = frozen_review["pages"]
        observations = {p["review_id"]: (p["verdict"], p["visual_evidence"]) for p in candidates}
        review_input_sha256 = frozen_review["input_sha256"]
        for path, expected in review_input_sha256.items():
            if path.startswith("reference/"):
                assert sha(path) == expected, path
    else:
        candidates = read("artifacts/corpus-review-v3/candidate-index.json") + read("artifacts/corpus-review-v3/extra-candidate-index.json")
        observations = {}
        for line in (ROOT / "artifacts/corpus-review-v3/observations.tsv").read_text(encoding="utf-8").splitlines():
            label, verdict, evidence = line.split("\t")
            assert label not in observations
            observations[label] = (verdict, evidence)
        review_input_sha256 = {p: sha(p) for p in ["artifacts/corpus-review-v3/candidate-index.json", "artifacts/corpus-review-v3/extra-candidate-index.json", "artifacts/corpus-review-v3/observations.tsv", "reference/corpus-manifest-v2.json", "reference/corpus-v2-evaluation-lock.json", "reference/corpus-v2-calibration-lock.json", "reference/corpus-v2-visual-review.json"]}
    assert len(candidates) == len(observations) == 323
    assert all(verdict in {"accept", "reject", "uncertain"} for verdict, _ in observations.values())
    by_label = {p["review_id"]: p for p in candidates}
    assert set(by_label) == set(observations)
    overrides = {}
    visual_family_groups = {
        "publication:burning-wheel-books": ["P011", "P027", "P067"],
        "template:ebook-best-seller-promotion": ["P068", "P078"],
        "course:software-engineering-se-slides": ["P026", "P028", "P091", "S001"],
        "template:chinese-red-review-box-reading-workbook": ["P085", "P101"],
        "template:classical-chinese-literature-workbook": ["P074", "P088", "S052"],
        "template:jinghua-reading-comprehension": ["P014", "P017"],
        "publication:ey-reports": ["T013", "T041", "T050"],
    }
    for family, labels in visual_family_groups.items():
        for label in labels:
            overrides[by_label[label]["image_name"]] = family
    for p in candidates:
        verdict, evidence = observations[p["review_id"]]
        p.update({"verdict": verdict, "visual_evidence": evidence})
        if "Federal Register" in evidence:
            overrides[p["image_name"]] = "publication:federal-register"
    for p in v2_review["pages"]:
        if "Federal Register" in p["visual_evidence"]:
            overrides[p["id"].split(":", 1)[1]] = "publication:federal-register"
    family_by_name = {name: source_family(name, row, overrides) for name, row in by_name.items()}
    for p in candidates:
        p["source_family"] = family_by_name[p["image_name"]]
    candidate_by_id = defaultdict(list)
    for p in candidates:
        candidate_by_id[p["id"]].append(p)
    calibration_review = {p["id"]: p for p in candidates if p["group"] == "calibration_v2"}
    selected = []
    exclusions = []

    def add_existing(page, review_record):
        p = dict(page)
        p["source_family"] = family_by_name[p["ground_truth"]["image_path"]]
        p["visual_review"] = {
            "review_id": review_record["review_id"], "status": "supported",
            "evidence": review_record["visual_evidence"],
            "source": "reference/corpus-v2-visual-review.json" if page["split"] == "evaluation" else "reference/corpus-v3-candidate-review.json",
        }
        p["reviewed_source_sha256"] = review_record["source_sha256"]
        p["reviewed_source_path"] = v2_locked_by_id[p["id"]]["source_path"]
        selected.append(p)

    for p in v2["pages"]:
        if p["split"] != "evaluation" or p["category"] == "ordinary":
            continue
        reviewed = v2_reviews[p["id"]]
        if reviewed["declared_category_verdict"] == "supported":
            add_existing(p, reviewed)
        else:
            exclusions.append({"id": p["id"], "split": p["split"], "category": p["category"], "reason": "uncertain visual category", "review_id": reviewed["review_id"]})

    def selected_ids():
        return {p["id"] for p in selected}

    def families(split):
        return {p["source_family"] for p in selected if p["split"] == split}

    def make_page(candidate, split, category):
        row = by_name[candidate["image_name"]]
        info = row["page_info"]
        name = candidate["image_name"]
        return {"id": candidate["id"], "split": split, "category": category, "smoke": False,
                "image_path": "images/" + name, "image_url": candidate["source_url"],
                "source_family": candidate["source_family"], "width": info["width"], "height": info["height"],
                "attributes": info["page_attribute"],
                "ground_truth": {"annotation_file": "OmniDocBench.json", "image_path": name,
                                 "page_sha256": hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
                "annotation_text_characters": original.char_count(row), "median_line_height_ratio": original.tiny_height(row),
                "reviewed_source_sha256": candidate["source_sha256"],
                "reviewed_source_path": candidate["source_path"],
                "visual_review": {"review_id": candidate["review_id"], "status": "supported", "evidence": candidate["visual_evidence"], "source": "reference/corpus-v3-candidate-review.json"}}

    # Reserve retained calibration families before selecting replacement pages.
    for p in v2["pages"]:
        if p["split"] != "calibration" or p["category"] == "ordinary":
            continue
        reviewed = calibration_review[p["id"]]
        family = family_by_name[p["ground_truth"]["image_path"]]
        if reviewed["verdict"] == "accept" and family not in families("evaluation"):
            add_existing(p, reviewed)
        else:
            reason = "uncertain visual category" if reviewed["verdict"] != "accept" else "source/publication family overlaps evaluation"
            exclusions.append({"id": p["id"], "split": p["split"], "category": p["category"], "reason": reason, "source_family": family, "review_id": reviewed["review_id"]})

    groups_for_category = {"degraded": {"degraded"}, "handwriting": {"handwriting"}, "formulas": {"replacement_formulas"}, "tables": set(), "tiny_text": {"replacement_tiny_text"}, "multi_column": {"multi_column", "replacement_multi_column"}}

    def fill(split, category, target, pool):
        opposite = "calibration" if split == "evaluation" else "evaluation"
        approved = {p["id"]: p for p in pool if p["verdict"] == "accept"}
        ordered = sorted(approved.values(), key=lambda p: rank(split, category, p["id"]))
        for p in ordered:
            current = [q for q in selected if q["split"] == split and q["category"] == category]
            if len(current) == target:
                break
            if p["id"] in selected_ids() or p["source_family"] in families(opposite):
                continue
            if split == "evaluation" and p["id"] in previous_calibration_ids:
                continue
            if category != "handwriting" and p["source_family"] in {q["source_family"] for q in current}:
                continue
            if category == "handwriting" and Counter(q["source_family"] for q in current)[p["source_family"]] >= 13:
                continue
            selected.append(make_page(p, split, category))
        current = [q for q in selected if q["split"] == split and q["category"] == category]
        if len(current) != target:
            raise ValueError(f"Insufficient visually approved family-separated {split}/{category}: {len(current)} < {target}; raw approved candidates={len(approved)}")

    for split, counts in [("evaluation", CATEGORIES), ("calibration", CALIBRATION)]:
        for category, target in counts.items():
            if category != "ordinary":
                fill(split, category, target, [p for p in candidates if p["group"] in groups_for_category[category]])
    plain = [p for p in candidates if p["group"] in {"plain", "plain_extra_columns", "plain_extra_single"}]
    # Reserve sixteen ordinary calibration families first. All ordinary candidates
    # have direct image evidence; content and page layout are separate dimensions.
    fill("calibration", "ordinary", 16, plain)
    fill("evaluation", "ordinary", 35, plain)

    for p in selected:
        p["smoke"] = False
    for category, count in SMOKE.items():
        pool = [p for p in selected if p["split"] == "evaluation" and p["category"] == category]
        pool.sort(key=lambda p: rank("smoke", category, p["id"]))
        seen = set()
        for p in pool:
            if p["source_family"] in seen:
                continue
            p["smoke"] = True
            seen.add(p["source_family"])
            if len(seen) == count:
                break
        assert len(seen) == count, category
    selected.sort(key=lambda p: (p["split"] != "evaluation", list(CATEGORIES).index(p["category"]), rank(p["split"], p["category"], p["id"])))
    assert len(selected) == len(selected_ids()) == 264
    assert not families("evaluation") & families("calibration")
    for split, counts in [("evaluation", CATEGORIES), ("calibration", CALIBRATION)]:
        assert dict(Counter(p["category"] for p in selected if p["split"] == split)) == counts
    assert sum(p["smoke"] for p in selected) == 24
    assert not previous_calibration_ids & {p["id"] for p in selected if p["split"] == "evaluation"}
    # Bind every visual approval to the actual downloaded bytes before freezing.
    # The materializer must also verify these hashes when reusing or downloading.
    for p in selected:
        assert sha(p["reviewed_source_path"]) == p["reviewed_source_sha256"], p["id"]
    moved_between_splits = []
    v2_by_id = {p["id"]: p for p in v2["pages"]}
    for p in selected:
        previous = v2_by_id.get(p["id"])
        if previous is not None and previous["split"] != p["split"]:
            moved_between_splits.append({"id": p["id"], "previous_split": previous["split"], "new_split": p["split"]})

    # Detect shared substantial text blocks, including promotional paragraphs
    # whose page-level hashes differ. Keep the complete finding for adjudication.
    block_owners = defaultdict(list)
    for p in selected:
        for b in by_name[p["ground_truth"]["image_path"]]["layout_dets"]:
            if b.get("ignore") or b["category_type"] not in {"text_block", "list_group"}:
                continue
            text = " ".join(unicodedata.normalize("NFC", b.get("text", "")).split())
            if len(text) >= 160:
                block_owners[hashlib.sha256(text.encode()).hexdigest()].append(p)
    shared_blocks = [{"text_sha256": h, "page_ids": sorted({p["id"] for p in ps})} for h, ps in block_owners.items() if len({p["split"] for p in ps}) > 1]

    candidate_report = {"schema_version": 1, "dataset": original.DATASET, "revision": original.REVISION,
                        "status": "All 323 candidate/reused-calibration page views reviewed; rejected or uncertain candidates cannot be selected.",
                        "method": "Contact sheets inspected visually; image content determines approval. No selection decision uses model output quality or evaluator success. Single reviewer, no transcription or native-pixel accuracy claim.",
                        "ordinary_definition": "Sustained ordinary prose without meaningful tables, equation regions, code or infographics. Full prose paragraphs may occur in one or multiple columns or on slides. Decorative photos/art and incidental measurement quantities/abbreviations are allowed. Minimum metadata prefilter is 300 annotated text characters; visual approval is mandatory.",
                        "ordinary_layout_policy": "Content category and layout are separate; multiple columns do not disqualify pure prose.",
                        "source_family_rules": "Generic _page_N, .pdf_N and notes/newspaper_<hash>_N names grouped; known Putnam, Federal Register and identifiable named publications grouped conservatively. Visual template overrides listed explicitly. Opaque names cannot establish all original provenance.",
                        "unique_source_page_count": len({p["id"] for p in candidates}),
                        "visual_family_overrides": overrides,
                        "input_sha256": review_input_sha256,
                        "verdict_counts": dict(Counter(p["verdict"] for p in candidates)), "pages": candidates}
    report_bytes = (json.dumps(candidate_report, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = {"schema_version": 3, "status": "visually_reviewed_selection_frozen_materialization_and_duplicate_audit_pending",
                "dataset": original.DATASET, "revision": original.REVISION, "annotation_sha256": original.ANNOTATION_SHA256,
                "source_terms": v2["source_terms"], "evaluation_count": 200, "smoke_count": 24, "calibration_count": 64,
                "category_counts": CATEGORIES, "calibration_category_counts": CALIBRATION, "smoke_category_counts": SMOKE,
                "selection": "Retain confidently reviewed v2 nonordinary pages, replace uncertain labels and cross-split known document/publication families, fill from visually approved candidates by deterministic SHA256 rank with seed falcon-cpu-corpus-v3. Reserve ordinary calibration families before evaluation; one ordinary page per known family in each split. Former v2 calibration page IDs cannot enter evaluation.",
                "ordinary_definition": candidate_report["ordinary_definition"], "ordinary_layout_policy": candidate_report["ordinary_layout_policy"],
                "family_limitation": "Zero overlap for the explicit document/publication/template family rules is verified. Opaque source IDs and unknown publication provenance prevent claiming all pages or writers are statistically independent. Image/text/perceptual audits still required after materialization.",
                "handwriting_limitations": "Evaluation uses two main notebooks plus opaque examples; calibration uses a distinct notebook. These are page counts, not independent writer counts.",
                "tiny_text_rule": v2["tiny_text_rule"], "known_family_overlap_between_splits": [],
                "shared_substantial_text_blocks_between_splits": shared_blocks,
                "review": {"path": review_path, "sha256": hashlib.sha256(report_bytes).hexdigest(), "inherited_v2_review_sha256": sha("reference/corpus-v2-visual-review.json"), "selector_sha256": sha("scripts/select_corpus_v3.py")},
                "replacement_exclusions": exclusions,
                "split_migrations_from_v2": moved_between_splits,
                "calibration_version_policy": "No former v2 calibration page IDs enter v3 evaluation. Splits have no known family overlap within v3, but v2 had documented document/publication overlap; candidates tuned on earlier calibration must disclose that historical family exposure rather than claim complete v3 held-out independence.",
                "selected_source_hash_verification": {"verified_files": len(selected), "algorithm": "SHA256", "policy": "Each local reviewed_source_path was rehashed and matched reviewed_source_sha256 before selection freeze; materialization must verify the same digest."},
                "missing_planned_domains": ["receipts", "truly blank pages", "dedicated full-page rotation coverage", "scripts beyond English/Chinese"],
                "supplemental_fixture_policy": "Address missing domains in a separately versioned original fixture set with explicit text/geometry/rotation recipes and parent grouping. Do not claim the natural v3 corpus already covers them.",
                "ordinary_layout_counts": {split: dict(Counter(p["attributes"]["layout"] for p in selected if p["split"] == split and p["category"] == "ordinary")) for split in ["evaluation", "calibration"]},
                "ordinary_layout_count_basis": "Pinned source layout annotations, with content/layout caveats recorded in each visual_review.evidence. Layout was visually checked when approving ordinary content.",
                "ordinary_source_type_counts": {split: dict(Counter(p["attributes"]["data_source"] for p in selected if p["split"] == split and p["category"] == "ordinary")) for split in ["evaluation", "calibration"]},
                "pages": selected}
    output = ROOT / ("artifacts/corpus-review-v3/selection-draft.json" if args.draft else "reference/corpus-manifest-v3.json")
    if not args.draft and output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection {output}")
    if not args.draft and shared_blocks:
        raise ValueError(f"Unadjudicated cross-split shared substantial text blocks: {shared_blocks}")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_output = ROOT / ("artifacts/corpus-review-v3/candidate-review-draft.json" if args.draft else review_path)
    review_output.write_bytes(report_bytes)
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "pages": len(selected), "exclusions": len(exclusions), "shared_blocks": shared_blocks, "ordinary_layout_counts": manifest["ordinary_layout_counts"], "candidate_verdicts": candidate_report["verdict_counts"]}, indent=2))


if __name__ == "__main__":
    main()
