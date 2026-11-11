#!/usr/bin/env python3
"""Select the fast-defaults gate v1 pages from the pinned OmniDocBench annotations.

Rules (fixed before any run on these pages with either fast-mode arm):

- Pool: the English pages of the resolution-router development set
  (reference/router-dev-v1-manifest.json), that is every English page that no
  corpus (v1, v2, v3) manifest and not the English gate v1 uses and that has
  text ground truth (prepare_corpus_smoke.ground_truth). The script derives
  the pool from the annotations and checks that it equals the router-dev-v1
  page set.
- Document families: pages join one family when they share any of these keys
  (connected components over every annotated page):
  - `gate_family` (the `.pdf_N`, `notes_<hash>_N` and `_page_N` series);
  - corpus v3's family (select_corpus_v3.source_family: named publications
    such as the Putnam archive and The Economist, Federal Register pages,
    `newspaper_<hash>_N`), with the families its visual review assigned
    (reference/corpus-v3-candidate-review.json, templates among them);
  - for `book_en_<title>_N` pages (numbered without `_page_`), the title;
  - the `source_family` that a corpus v1/v2/v3 or English gate v1 manifest
    records for its page.
- Family exclusion: drop every page whose family holds a corpus v1, v2 or v3
  or English gate v1 page.
- Categories: the English gate's priority (tiny_text > formulas > tables >
  multi_column > ordinary > slides > exam; select_english_gate.category), then
  `other` for a page that no category matches with at least 100 ground-truth
  characters (select_corpus.char_count). Pages with neither are dropped.
- One page per family, no quotas: every family left contributes one page, the
  one with the highest-priority category, ties broken by the lowest SHA-256 of
  "falcon-cpu-fast-defaults-gate-v1\\0<category>\\0omnidocbench:<name>".
  Manifest order: category priority, then that rank.

Selection reads the annotations and the Hub tree listing of the pinned
revision only (no local images). Each page's `reviewed_source_sha256` is the
image's LFS SHA-256 from that listing (checked against the router-dev-v1
manifest's own hash), and `reviewed_source_path` points at the local download,
so prepare_corpus.py copies that file and checks its hash.

`--check-english-gate-v1` runs the same pool and category code with the
English gate v1 rules (outside corpus v1/v2/v3 and every v3 `gate_family`, no
`other`) and compares the pool sizes with the ones
reference/english-gate-v1-manifest.json records.

From the repository root:

  python3 research/corpus-qualification/scripts/select_fast_defaults_gate.py --check-english-gate-v1
  python3 research/corpus-qualification/scripts/select_fast_defaults_gate.py --print-images > names.txt
  xargs -d '\\n' hf download opendatalab/OmniDocBench --repo-type dataset \\
      --revision aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec --local-dir artifacts/omnidocbench < names.txt
  python3 research/corpus-qualification/scripts/select_fast_defaults_gate.py \\
      --output reference/fast-defaults-gate-v1-manifest.json
  python3 research/corpus-qualification/scripts/prepare_corpus.py \\
      --manifest reference/fast-defaults-gate-v1-manifest.json \\
      --output artifacts/corpus/fast-defaults-gate-v1 --lock-prefix reference/fast-defaults-gate-v1
  python3 research/corpus-qualification/scripts/mask_excluded.py \\
      --lock reference/fast-defaults-gate-v1-evaluation-lock.json \\
      --output artifacts/corpus/fast-defaults-gate-v1-masked \\
      --masked-lock reference/fast-defaults-gate-v1-masked-lock.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path

from prepare_corpus import stable_hash
from prepare_corpus_smoke import ground_truth
from select_corpus import ANNOTATION_SHA256, DATASET, REVISION, char_count, tiny_height
from select_corpus_v3 import source_family as v3_family
from select_english_gate import PRIORITY, category, gate_family

SEED = "falcon-cpu-fast-defaults-gate-v1"
CATEGORIES = [*PRIORITY, "other"]
CORPORA = [Path(f"reference/corpus-manifest-v{version}.json") for version in (1, 2, 3)]
ENGLISH_GATE = Path("reference/english-gate-v1-manifest.json")
ROUTER_DEV = Path("reference/router-dev-v1-manifest.json")
V3_REVIEW = Path("reference/corpus-v3-candidate-review.json")
BOOK_PAGES = re.compile(r"^(book_en_.+)_\d+\.[^.]+$")


def pages_of(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["pages"]


def names(manifest: Path) -> list[str]:
    """Image names of a selection manifest's pages."""
    return [Path(p["image_path"]).name for p in pages_of(manifest)]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys(row: dict, reviewed: dict[str, str]) -> list[str]:
    """A page's own family keys: gate_family, corpus v3's family and, for
    numbered book pages, the book title."""
    name = row["page_info"]["image_path"]
    out = [gate_family(name), v3_family(name, row, reviewed)]
    if match := BOOK_PAGES.match(gate_family(name)):
        out.append(match[1])
    return out


def document_families(rows: list[dict], reviewed: dict[str, str], labels: dict[str, set[str]]) -> dict[str, str]:
    """Each page's family: the smallest key of its connected component, where
    a page links to its own `keys` and to every label recorded for it."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    page_names = [row["page_info"]["image_path"] for row in rows]
    for name, row in zip(page_names, rows, strict=True):
        for key in (*keys(row, reviewed), *sorted(labels.get(name, ()))):
            union(f"name:{name}", f"key:{key}")
    smallest: dict[str, str] = {}
    for x in list(parent):
        if x.startswith("key:"):
            root = find(x)
            smallest[root] = min(smallest.get(root, x), x)
    return {name: smallest[find(f"name:{name}")].removeprefix("key:") for name in page_names}


def gate_category(row: dict) -> str | None:
    """The English gate's category, else `other` with >= 100 characters."""
    return category(row) or ("other" if char_count(row) >= 100 else None)


def choose_family_pages(
    eligible: list[tuple[dict, str]], family: dict[str, str], seed: str
) -> list[tuple[str, dict]]:
    """Choose one page per family by category priority and seeded rank."""
    best: dict[str, tuple[tuple[int, str], str, dict]] = {}
    for row, cat in eligible:
        name = row["page_info"]["image_path"]
        rank = hashlib.sha256(f"{seed}\0{cat}\0omnidocbench:{name}".encode()).hexdigest()
        key = (CATEGORIES.index(cat), rank)
        fam = family[name]
        if fam not in best or key < best[fam][0]:
            best[fam] = (key, cat, row)
    return [(cat, row) for _, cat, row in sorted(best.values(), key=lambda x: x[0])]


def page_record(row: dict, cat: str, family: str, lfs: dict, images: Path, gate: str) -> dict:
    """Build a manifest page from its annotation and pinned Hub tree entry."""
    info = row["page_info"]
    name = info["image_path"]
    return {
        "id": f"omnidocbench:{name}", "split": "evaluation", "category": cat, "smoke": False,
        "image_path": f"images/{name}",
        "image_url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/images/{urllib.parse.quote(name)}",
        "source_family": family, "width": info["width"],
        "height": info["height"], "attributes": info["page_attribute"],
        "ground_truth": {"annotation_file": "OmniDocBench.json", "image_path": name,
                         "page_sha256": stable_hash(row)},
        "annotation_text_characters": char_count(row),
        "median_line_height_ratio": tiny_height(row),
        "visual_review": f"none: {gate}, category from the corpus rules only",
        # prepare_corpus.py copies this local download and checks it against the Hub's LFS hash.
        "reviewed_source_path": (images / name).as_posix(),
        "reviewed_source_sha256": lfs["lfs_sha256"],
        "source_bytes": lfs["lfs_size"],
    }


def print_image_list(pages: list[dict]) -> None:
    """Print the selected image paths and report their total download size."""
    for page in pages:
        print(page["image_path"])
    total = sum(p["source_bytes"] for p in pages)
    print(f"{len(pages)} images, {total} bytes ({total / 1e6:.1f} MB)", file=sys.stderr)


def pools(rows: list[dict], categorize) -> dict[str, list[dict]]:
    """The rows by category; rows that `categorize` leaves out are dropped."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        cat = categorize(row)
        if cat:
            out.setdefault(cat, []).append(row)
    return out


def check_english_gate_v1(rows: list[dict]) -> None:
    """Reproduce the English gate v1 pool sizes with the pool code above."""
    used = {name for path in CORPORA for name in names(path)}
    v3_families = {gate_family(name) for name in names(CORPORA[2])}
    english = [r for r in rows if r["page_info"]["page_attribute"].get("language") == "english"]
    kept = [r for r in english if r["page_info"]["image_path"] not in used
            and gate_family(r["page_info"]["image_path"]) not in v3_families]
    by_category = pools(kept, category)
    sizes = {cat: len(by_category.get(cat, [])) for cat in PRIORITY}
    recorded = json.loads(ENGLISH_GATE.read_text(encoding="utf-8"))["pool_sizes"]
    print(json.dumps({"reproduced": sizes, "recorded": recorded, "equal": sizes == recorded}, indent=2))
    if sizes != recorded:
        sys.exit("English gate v1 pool sizes not reproduced")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=Path("artifacts/corpus-source-metadata/OmniDocBench.json"))
    ap.add_argument("--tree", type=Path,
                    default=Path(f"artifacts/omnidocbench/.cache/huggingface/trees/{REVISION}.json"),
                    help="Hub tree listing of the pinned revision (written by `hf download --local-dir`)")
    ap.add_argument("--images", type=Path, default=Path("artifacts/omnidocbench/images"),
                    help="where the downloaded images live (recorded as reviewed_source_path)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-english-gate-v1", action="store_true")
    mode.add_argument("--print-images", action="store_true", help="print images/<name> of each chosen page")
    mode.add_argument("--output", type=Path, help="manifest to write (must not exist)")
    a = ap.parse_args()
    raw = a.annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANNOTATION_SHA256:
        sys.exit("annotation source hash mismatch")
    rows = json.loads(raw.decode("utf-8"))
    if a.check_english_gate_v1:
        check_english_gate_v1(rows)
        return
    if a.output and a.output.exists():
        sys.exit(f"refusing to overwrite {a.output}")
    tree = json.loads(a.tree.read_text(encoding="utf-8"))["files"]
    if tree["OmniDocBench.json"].get("lfs_sha256") != ANNOTATION_SHA256:
        sys.exit("tree listing is not of the pinned revision")

    spent_manifests = [*CORPORA, ENGLISH_GATE]
    labels: dict[str, set[str]] = {}
    for path in spent_manifests:
        for page in pages_of(path):
            labels.setdefault(Path(page["image_path"]).name, set()).add(page["source_family"])
    reviewed = {page["image_name"]: page["source_family"] for page in pages_of(V3_REVIEW)}
    family = document_families(rows, reviewed, labels)
    used = {name for path in spent_manifests for name in names(path)}
    excluded_families = {family[name] for name in used}
    pool = [r for r in rows if r["page_info"]["page_attribute"].get("language") == "english"
            and r["page_info"]["image_path"] not in used and ground_truth(r)]
    dev = {p["id"].split(":", 1)[1]: p for p in pages_of(ROUTER_DEV)}
    if {r["page_info"]["image_path"] for r in pool} != set(dev):
        sys.exit("pool differs from the router-dev-v1 page set")
    outside = [r for r in pool if family[r["page_info"]["image_path"]] not in excluded_families]
    by_category = pools(outside, gate_category)
    eligible = [(row, cat) for cat, ps in by_category.items() for row in ps]
    chosen = choose_family_pages(eligible, family, SEED)

    pages = []
    for cat, row in chosen:
        name = row["page_info"]["image_path"]
        lfs = tree[f"images/{name}"]
        if lfs["lfs_sha256"] != dev[name]["reviewed_source_sha256"]:
            sys.exit(f"{name}: tree listing and router-dev-v1 disagree on the image hash")
        pages.append(page_record(row, cat, family[name], lfs, a.images, "fast-defaults gate v1"))
    if a.print_images:
        print_image_list(pages)
        return

    counts = {cat: sum(p["category"] == cat for p in pages) for cat in CATEGORIES}
    pool_sizes = {cat: len(by_category.get(cat, [])) for cat in CATEGORIES}
    spent_gate_families = {gate_family(n) for n in used}
    by_gate_family = [r for r in pool if gate_family(r["page_info"]["image_path"]) in spent_gate_families]
    manifest = {
        "schema_version": 3, "status": "fast-defaults-gate-v1", "dataset": DATASET, "revision": REVISION,
        "annotation_sha256": ANNOTATION_SHA256,
        "source_terms": ("Research purposes only; not for commercial use. "
                         "Do not redistribute source images in runner releases."),
        "evaluation_count": len(pages), "smoke_count": 0, "calibration_count": 0,
        "category_counts": counts,
        "selection": (f"The English pages of the router development set (reference/router-dev-v1-manifest.json: every "
                      f"English OmniDocBench page outside corpus v1/v2/v3 and the English gate v1 with text ground "
                      f"truth), minus every page whose document family holds a corpus v1/v2/v3 or English gate v1 "
                      f"page; categories by priority {CATEGORIES} (select_english_gate.category, then other with "
                      f">= 100 characters; pages with neither dropped); one page per family, no quotas, chosen by "
                      f"category priority then SHA-256 rank with seed {SEED}."),
        "family_rule": ("connected components over every annotated page of shared keys: gate_family, "
                        "select_corpus_v3.source_family with the corpus v3 review's assignments, the title of "
                        "book_en_<title>_N pages, and the source_family recorded by corpus v1/v2/v3 and English gate "
                        "v1; source_family below is the component's smallest key"),
        "pool": {
            "router_dev_v1_pages": len(pool),
            "excluded_by_gate_family_alone": len(by_gate_family),
            "excluded_by_family": len(pool) - len(outside),
            "after_family_exclusion": len(outside),
            "dropped_no_category_under_100_characters": len(outside) - len(eligible),
            "eligible": len(eligible),
            "families": len(chosen),
        },
        "pool_sizes": pool_sizes,
        "exclusion_sources": {path.as_posix(): file_sha256(path) for path in [*spent_manifests, V3_REVIEW, ROUTER_DEV]},
        "image_hashes": f"LFS SHA-256 from the Hub tree listing of {REVISION} (equal to router-dev-v1's)",
        "source_bytes": sum(p["source_bytes"] for p in pages),
        "pages": pages,
    }
    a.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": len(pages), "category_counts": counts, "pool": manifest["pool"],
                      "pool_sizes": pool_sizes}, indent=2))


if __name__ == "__main__":
    main()
