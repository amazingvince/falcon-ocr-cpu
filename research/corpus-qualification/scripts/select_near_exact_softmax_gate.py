#!/usr/bin/env python3
"""Select the near-exact softmax gate v1 pages from the pinned OmniDocBench annotations.

Rules (fixed before any run on these pages with either near-exact arm):

- Spent pages: every page that a file of SPENT_SOURCES names, by image name
  or by stem (the name without its extension), URL-quoted or not, in a JSON
  string split at `/`, `\\` and `:`. The sources are the corpus v1, v2 and
  v3 manifests, the English gate v1, the router development set, the
  fast-defaults gate v1, the corpus v2 and v3 visual reviews, and the
  evaluator line-ending receipt (it names 18 pages of the OmniDocBench
  evaluator's own demo predictions). `--audit-reference` checks that no
  other JSON file under reference/ names a page outside that set.
- Document families: select_fast_defaults_gate.document_families, with the
  corpus v3 review's assignments and the source_family that every spent
  manifest records, plus two keys for opaque image names (most Chinese
  pages are `page-<uuid>` or `yanbao..._<n>`): each running header or footer
  text of at least MIN_FURNITURE characters once digits, punctuation and
  whitespace are removed (a newspaper's name, a broker's, a distributor's
  watermark; generic ones such as a weekday date line merge more than one
  publication, which only excludes more), and the TEMPLATES that the pixel
  audit of the first draft found.
- Pool: every page, in any language, that is not spent, has text ground
  truth (prepare_corpus_smoke.ground_truth) and has no spent page in its
  family. No English page qualifies: corpus v1-v3, the English gate and the
  router development set hold every English page with text ground truth.
- Categories: select_fast_defaults_gate.gate_category (the English gate's
  priority tiny_text > formulas > tables > multi_column > ordinary > slides >
  exam, then `other` for a page with at least 100 ground-truth characters);
  pages with neither are dropped.
- One page per family: the one with the highest-priority category, ties
  broken by the lowest SHA-256 of
  "falcon-cpu-near-exact-softmax-gate-v1\\0<category>\\0omnidocbench:<name>".
- At most QUOTA pages per category, those of lowest rank, so that one arm
  runs in about an hour and the rest of the pool stays fresh for later
  gates. Manifest order: category priority, then rank.

Selection reads the annotations and the Hub tree listing of the pinned
revision only (no local images). Each page's `reviewed_source_sha256` is the
image's LFS SHA-256 from that listing, and `reviewed_source_path` points at
the local download, so prepare_corpus.py copies that file and checks its
hash.

From the repository root:

  python3 research/corpus-qualification/scripts/select_near_exact_softmax_gate.py --audit-reference
  python3 research/corpus-qualification/scripts/select_near_exact_softmax_gate.py --print-images > names.txt
  xargs -d '\\n' hf download opendatalab/OmniDocBench --repo-type dataset \\
      --revision aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec --local-dir artifacts/omnidocbench < names.txt
  python3 research/corpus-qualification/scripts/select_near_exact_softmax_gate.py \\
      --output reference/near-exact-softmax-gate-v1-manifest.json
  python3 research/corpus-qualification/scripts/prepare_corpus.py \\
      --manifest reference/near-exact-softmax-gate-v1-manifest.json \\
      --output artifacts/corpus/near-exact-softmax-gate-v1 --lock-prefix reference/near-exact-softmax-gate-v1
  python3 research/corpus-qualification/scripts/mask_excluded.py \\
      --lock reference/near-exact-softmax-gate-v1-evaluation-lock.json \\
      --output artifacts/corpus/near-exact-softmax-gate-v1-masked \\
      --masked-lock reference/near-exact-softmax-gate-v1-masked-lock.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.parse
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from prepare_corpus import stable_hash
from prepare_corpus_smoke import ground_truth
from select_corpus import ANNOTATION_SHA256, DATASET, REVISION, char_count, tiny_height
from select_english_gate import gate_family
from select_fast_defaults_gate import (CATEGORIES, CORPORA, ENGLISH_GATE, ROUTER_DEV, V3_REVIEW, document_families,
                                       file_sha256, gate_category, pages_of, pools)

SEED = "falcon-cpu-near-exact-softmax-gate-v1"
PREFIX = "near-exact-softmax-gate-v1"
QUOTA = 40
SPENT_MANIFESTS = [*CORPORA, ENGLISH_GATE, ROUTER_DEV, Path("reference/fast-defaults-gate-v1-manifest.json")]
SPENT_SOURCES = [*SPENT_MANIFESTS, Path("reference/corpus-v2-visual-review.json"), V3_REVIEW,
                 Path("reference/official-evaluator-line-endings-v1.json")]
MIN_FURNITURE = 4
# Pages whose difference hash lies within 4 of a spent page's in the first
# draft and whose annotations show one template: 2098 (header 全球科研实力对比)
# and the corpus v2/v3 page 5877 (科技人才储备实力研究) are the same
# seven-level topic-word table at the same size, hash distance 0.
TEMPLATES = {
    "template:topic-word-taxonomy-table": ["yanbaor2_yanbaoPPT_2098.jpg", "yanbaor2_yanbaoPPT_5877.jpg"],
}


def strings(value) -> Iterator[str]:
    """Every string of a JSON value, object keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def named_pages(path: Path, by_stem: dict[str, str]) -> set[str]:
    """The pages a JSON file names: a part of one of its strings, split at
    `/`, `\\` and `:`, that is a page's image name or stem, with or without
    one more extension, URL-quoted or not."""
    out = set()
    for text in strings(json.loads(path.read_text(encoding="utf-8"))):
        for part in re.split(r"[/\\:]", text):
            for p in {part, urllib.parse.unquote(part)}:
                if name := by_stem.get(p) or by_stem.get(p.rsplit(".", 1)[0]):
                    out.add(name)
    return out


def furniture(row: dict) -> set[str]:
    """A page's running header and footer texts as family keys: NFKC, then
    digits, punctuation and whitespace removed, at least MIN_FURNITURE
    characters left."""
    out = set()
    for block in row["layout_dets"]:
        if block["category_type"] in ("header", "footer"):
            text = re.sub(r"[\d\W_]+", "", unicodedata.normalize("NFKC", block.get("text") or "")).lower()
            if len(text) >= MIN_FURNITURE:
                out.add(f"furniture:{text}")
    return out


def rank(cat: str, name: str) -> str:
    return hashlib.sha256(f"{SEED}\0{cat}\0omnidocbench:{name}".encode()).hexdigest()


def audit_reference(spent: set[str], by_stem: dict[str, str]) -> None:
    """Fail if a JSON file under reference/ (other than this gate's own)
    names a page that SPENT_SOURCES leaves unspent."""
    files = [p for p in sorted(Path("reference").rglob("*.json")) if not p.name.startswith(PREFIX)]
    outside = {}
    for path in files:
        if extra := named_pages(path, by_stem) - spent:
            outside[path.as_posix()] = sorted(extra)
    print(json.dumps({"json_files": len(files), "spent_pages": len(spent), "outside_spent": outside}, indent=2,
                     ensure_ascii=False))
    if outside:
        sys.exit("reference/ names pages outside SPENT_SOURCES")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=Path("artifacts/corpus-source-metadata/OmniDocBench.json"))
    ap.add_argument("--tree", type=Path,
                    default=Path(f"artifacts/omnidocbench/.cache/huggingface/trees/{REVISION}.json"),
                    help="Hub tree listing of the pinned revision (written by `hf download --local-dir`)")
    ap.add_argument("--images", type=Path, default=Path("artifacts/omnidocbench/images"),
                    help="where the downloaded images live (recorded as reviewed_source_path)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-reference", action="store_true",
                      help="check that reference/ names no page outside the spent sources")
    mode.add_argument("--print-images", action="store_true", help="print images/<name> of each chosen page")
    mode.add_argument("--output", type=Path, help="manifest to write (must not exist)")
    a = ap.parse_args()
    raw = a.annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANNOTATION_SHA256:
        sys.exit("annotation source hash mismatch")
    rows = json.loads(raw.decode("utf-8"))
    by_stem = {}
    for row in rows:
        name = row["page_info"]["image_path"]
        by_stem[name] = by_stem[name.rsplit(".", 1)[0]] = name
    spent = set().union(*(named_pages(path, by_stem) for path in SPENT_SOURCES))
    if a.audit_reference:
        audit_reference(spent, by_stem)
        return
    if a.output and a.output.exists():
        sys.exit(f"refusing to overwrite {a.output}")
    tree = json.loads(a.tree.read_text(encoding="utf-8"))["files"]
    if tree["OmniDocBench.json"].get("lfs_sha256") != ANNOTATION_SHA256:
        sys.exit("tree listing is not of the pinned revision")

    recorded: dict[str, set[str]] = {}
    for path in SPENT_MANIFESTS:
        for page in pages_of(path):
            recorded.setdefault(Path(page["image_path"]).name, set()).add(page["source_family"])
    labels = {name: set(keys) for name, keys in recorded.items()}
    for row in rows:
        labels.setdefault(row["page_info"]["image_path"], set()).update(furniture(row))
    for key, members in TEMPLATES.items():
        for name in members:
            labels[name].add(key)
    reviewed = {page["image_name"]: page["source_family"] for page in pages_of(V3_REVIEW)}
    family = document_families(rows, reviewed, labels)
    excluded_families = {family[name] for name in spent}
    fresh = [r for r in rows if r["page_info"]["image_path"] not in spent]
    with_truth = [r for r in fresh if ground_truth(r)]
    outside = [r for r in with_truth if family[r["page_info"]["image_path"]] not in excluded_families]
    # The same exclusion without the furniture and template keys, for the record.
    plain = document_families(rows, reviewed, recorded)
    plain_excluded = {plain[name] for name in spent}
    by_plain_rule = sum(plain[r["page_info"]["image_path"]] in plain_excluded for r in with_truth)
    by_category = pools(outside, gate_category)
    eligible = [(row, cat) for cat, ps in by_category.items() for row in ps]
    best: dict[str, tuple[tuple[int, str], str, dict]] = {}
    for row, cat in eligible:
        name = row["page_info"]["image_path"]
        key = (CATEGORIES.index(cat), rank(cat, name))
        fam = family[name]
        if fam not in best or key < best[fam][0]:
            best[fam] = (key, cat, row)
    winners = sorted(best.values(), key=lambda x: x[0])
    taken: Counter[str] = Counter()
    chosen = []
    for key, cat, row in winners:
        if taken[cat] < QUOTA:
            taken[cat] += 1
            chosen.append((key, cat, row))

    pages = []
    for _, cat, row in chosen:
        name = row["page_info"]["image_path"]
        lfs = tree[f"images/{name}"]
        pages.append({
            "id": f"omnidocbench:{name}", "split": "evaluation", "category": cat, "smoke": False,
            "image_path": f"images/{name}",
            "image_url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/images/{urllib.parse.quote(name)}",
            "source_family": family[name], "width": row["page_info"]["width"],
            "height": row["page_info"]["height"], "attributes": row["page_info"]["page_attribute"],
            "ground_truth": {"annotation_file": "OmniDocBench.json", "image_path": name,
                             "page_sha256": stable_hash(row)},
            "annotation_text_characters": char_count(row),
            "median_line_height_ratio": tiny_height(row),
            "visual_review": "none: near-exact softmax gate v1, category from the corpus rules only",
            # prepare_corpus.py copies this local download and checks it against the Hub's LFS hash.
            "reviewed_source_path": (a.images / name).as_posix(),
            "reviewed_source_sha256": lfs["lfs_sha256"],
            "source_bytes": lfs["lfs_size"],
        })
    if a.print_images:
        for page in pages:
            print(page["image_path"])
        total = sum(p["source_bytes"] for p in pages)
        print(f"{len(pages)} images, {total} bytes ({total / 1e6:.1f} MB)", file=sys.stderr)
        return

    counts = {cat: taken[cat] for cat in CATEGORIES}
    languages = Counter(p["attributes"].get("language") for p in pages)
    by_language = {cat: dict(sorted(Counter(p["attributes"].get("language") for p in pages
                                            if p["category"] == cat).items())) for cat in CATEGORIES if taken[cat]}
    family_winners = {cat: sum(c == cat for _, c, _ in winners) for cat in CATEGORIES}
    spent_gate_families = {gate_family(n) for n in spent}
    manifest = {
        "schema_version": 3, "status": PREFIX, "dataset": DATASET, "revision": REVISION,
        "annotation_sha256": ANNOTATION_SHA256,
        "source_terms": ("Research purposes only; not for commercial use. "
                         "Do not redistribute source images in runner releases."),
        "evaluation_count": len(pages), "smoke_count": 0, "calibration_count": 0,
        "category_counts": counts,
        "language_counts": dict(sorted(languages.items())),
        "category_language_counts": by_language,
        "selection": (f"Every OmniDocBench page, in any language, that no spent source names (corpus v1/v2/v3, the "
                      f"English gate v1, the router development set, the fast-defaults gate v1, the corpus v2/v3 "
                      f"visual reviews, the evaluator line-ending receipt), with text ground truth and no spent page "
                      f"in its document family; categories by priority {CATEGORIES} (select_english_gate.category, "
                      f"then other with >= 100 characters; pages with neither dropped); one page per family by "
                      f"category priority then SHA-256 rank with seed {SEED}; at most {QUOTA} pages per category, "
                      f"lowest rank first."),
        "family_rule": ("select_fast_defaults_gate.document_families: connected components over every annotated page "
                        "of shared keys (gate_family, select_corpus_v3.source_family with the corpus v3 review's "
                        "assignments, the title of book_en_<title>_N pages), the source_family recorded by corpus "
                        "v1/v2/v3, the English gate v1, the router development set and the fast-defaults gate v1, "
                        f"every header or footer text with at least {MIN_FURNITURE} characters once NFKC-normalized "
                        f"and stripped of digits, punctuation and whitespace, and the templates {TEMPLATES}; "
                        "source_family below is the component's smallest key"),
        "pool": {
            "annotated_pages": len(rows),
            "spent_pages": len(spent),
            "unspent_pages": len(fresh),
            "unspent_with_text_ground_truth": len(with_truth),
            "excluded_by_gate_family_alone": sum(gate_family(r["page_info"]["image_path"]) in spent_gate_families
                                                 for r in with_truth),
            "excluded_by_family_without_furniture_or_templates": by_plain_rule,
            "excluded_by_family": len(with_truth) - len(outside),
            "after_family_exclusion": len(outside),
            "after_family_exclusion_languages": dict(sorted(Counter(
                r["page_info"]["page_attribute"].get("language") for r in outside).items())),
            "dropped_no_category_under_100_characters": len(outside) - len(eligible),
            "eligible": len(eligible),
            "families": len(best),
            "family_winners_by_category": family_winners,
            "left_after_quota": len(best) - len(pages),
        },
        "pool_sizes": {cat: len(by_category.get(cat, [])) for cat in CATEGORIES},
        "exclusion_sources": {path.as_posix(): file_sha256(path) for path in SPENT_SOURCES},
        "image_hashes": f"LFS SHA-256 from the Hub tree listing of {REVISION}",
        "source_bytes": sum(p["source_bytes"] for p in pages),
        "pages": pages,
    }
    a.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("evaluation_count", "category_counts", "language_counts",
                                                     "category_language_counts", "pool", "pool_sizes",
                                                     "source_bytes")}, indent=2))


if __name__ == "__main__":
    main()
