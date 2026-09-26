#!/usr/bin/env python3
"""Build the English page set for draft-head training from the raw sources.

Every page is stored as a JPEG at most `--size` pixels on its long side (the
model's own maximum is 1536) with one manifest line: id, source, document,
category, license, path, 64-bit perceptual hash. Sampling caps pages per
source document; a small share gets rotation or colour-inversion augmentation.
Pages within `--hash-distance` of any blocklist image (the 1,651 OmniDocBench
pages and our evaluation corpus) are dropped.

  python research/draft-head/data/build_pages.py --raw D:/falcon-draft/raw \
      --out D:/falcon-draft/pages/stage1 --blocklist D:/falcon-draft/blocklist.json
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import tarfile
import zipfile
from pathlib import Path

import imagehash
import pyarrow.parquet as pq
import pypdfium2 as pdfium
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

# Pages per source for stage 1 (about 5,000 in total).
TARGETS = {
    "olmocr_documents": 1400,
    "olmocr_books": 300,
    "olmocr_loc": 250,
    "olmocr_archives": 250,
    "pdfa": 800,
    "idl": 600,
    "doclaynet": 600,
    "loc_newspapers": 300,
    "zenodo_slides": 250,
    "notes_notesbank": 200,
    "notes_humyn": 100,
}
# Stage 2 (about 25k pages in total, a superset of stage 1).
TARGETS_STAGE2 = {
    "olmocr_documents": 9000,
    "olmocr_books": 2000,
    "olmocr_loc": 1200,
    "olmocr_archives": 1200,
    "pdfa": 5000,
    "idl": 3000,
    "doclaynet": 2000,
    "loc_newspapers": 500,
    "zenodo_slides": 250,
    "notes_notesbank": 400,
    "notes_humyn": 400,
}
# Stage 3 (about 48k pages): the stage-2 set plus more of the large sources.
TARGETS_STAGE3 = TARGETS_STAGE2 | {
    "olmocr_documents": 20000,
    "olmocr_books": 3500,
    "pdfa": 12000,
    "idl": 7000,
}
PER_DOCUMENT = 2
# Handwritten notebooks are few, so more of their pages are used.
PER_NOTEBOOK = 15
LICENSES = {
    "olmocr": "ODC-BY-1.0",
    "pdfa": "Common Crawl terms / Digital Corpora",
    "idl": "UCSF Industry Documents Library terms",
    "doclaynet": "CDLA-Permissive-1.0",
    "loc_newspapers": "CC0-1.0",
    "notes_notesbank": "Apache-2.0",
    "notes_humyn": "CC-BY-4.0",
}
# Zenodo records keep their own license; only these are used.
OPEN_LICENSES = {"cc-by-4.0", "cc-by", "cc0-1.0", "cc0", "cc-by-3.0", "cc-by-2.0", "public-domain"}


def fit(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = size / max(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    return image


def render_pdf(data: bytes, page: int, size: int) -> Image.Image:
    pdf = pdfium.PdfDocument(data)
    try:
        p = pdf[page]
        width, height = p.get_size()
        return p.render(scale=size / max(width, height, 1)).to_pil().convert("RGB")
    finally:
        pdf.close()


def pdf_pages(data: bytes) -> int:
    pdf = pdfium.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


# ----- sources: each yields (source, document, category, license, image)

def olmocr(raw: Path, subset: str, parquet: str, rng: random.Random, want: int, size: int):
    root = raw / "allenai__olmOCR-mix-1025"
    rows = pq.read_table(root / parquet, columns=["url", "pdf_relpath", "primary_language", "is_table",
                                                   "rotation_correction", "id"]).to_pylist()
    rows = [r for r in rows if r["primary_language"] == "en"]
    local = {p.name for p in (root / "pdf_tarballs").glob("*.tar.gz")}
    rows = [r for r in rows if r["pdf_relpath"].split(":")[0].split("/")[-1] in local]
    rng.shuffle(rows)
    # Tables are a large share of real workloads: take them first up to a third.
    tables = [r for r in rows if r["is_table"] == "True"][: want // 3]
    chosen, per_doc = [], collections.Counter()
    for r in tables + rows:
        if len(chosen) >= int(want * 1.3):
            break
        if per_doc[r["url"]] >= PER_DOCUMENT or r in chosen:
            continue
        per_doc[r["url"]] += 1
        chosen.append(r)
    by_tar = collections.defaultdict(dict)
    for r in chosen:
        tar, member = r["pdf_relpath"].split(":", 1)
        by_tar[tar][member] = r
    for tar, members in by_tar.items():
        with tarfile.open(root / tar, "r:gz") as t:
            for info in t:
                r = members.get(info.name)
                if r is None:
                    continue
                data = t.extractfile(info).read()
                try:
                    image = render_pdf(data, 0, size)
                except Exception:
                    continue
                category = "table" if r["is_table"] == "True" else subset
                yield f"olmocr_{subset}", r["url"], category, LICENSES["olmocr"], image


def webdataset_pdfs(raw: Path, repo: str, source: str, rng: random.Random, size: int):
    root = raw / repo
    for shard in sorted(root.glob("*.tar")):
        with tarfile.open(shard, "r") as t:
            for info in t:
                if not info.name.endswith(".pdf"):
                    continue
                data = t.extractfile(info).read()
                try:
                    pages = pdf_pages(data)
                    picks = rng.sample(range(pages), min(PER_DOCUMENT, pages)) if pages else []
                    for page in picks:
                        yield source, info.name, source, LICENSES[source], render_pdf(data, page, size)
                except Exception:
                    continue


def doclaynet(raw: Path, rng: random.Random, size: int, already: collections.Counter | None = None):
    per_category = collections.Counter(already or {})
    cap = TARGETS["doclaynet"] // 6 + 1
    for f in sorted((raw / "docling-project__DocLayNet-v1.2" / "data").glob("train-*.parquet")):
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                meta = row.get("metadata") or {}
                category = (meta.get("doc_category") if isinstance(meta, dict) else None) or row.get("doc_category") or "unknown"
                if per_category[category] >= cap or rng.random() > 0.35:
                    continue
                image = row.get("image")
                pdf = row.get("pdf")
                try:
                    if pdf:
                        img = render_pdf(pdf if isinstance(pdf, bytes) else pdf["bytes"], 0, size)
                    else:
                        img = Image.open(io.BytesIO(image["bytes"] if isinstance(image, dict) else image))
                except Exception:
                    continue
                per_category[category] += 1
                doc = str(meta.get("original_filename", row.get("page_hash", ""))) if isinstance(meta, dict) else ""
                yield "doclaynet", doc, f"doclaynet_{category}", LICENSES["doclaynet"], img
        if sum(per_category.values()) >= TARGETS["doclaynet"] * 1.3:
            break


def loc_newspapers(raw: Path, rng: random.Random, size: int):
    archive = raw / "biglam__loc_beyond_words" / "data" / "images.zip"
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
        rng.shuffle(names)
        for n in names:
            try:
                yield "loc_newspapers", n, "newspaper", LICENSES["loc_newspapers"], Image.open(io.BytesIO(z.read(n)))
            except Exception:
                continue


def zenodo(raw: Path, rng: random.Random, size: int):
    rows = []
    for f in sorted((raw / "PDFPages__zenodo-presentations-open" / "data").glob("*.parquet")):
        rows += pq.read_table(f).to_pylist()
    rng.shuffle(rows)
    per_deck = collections.Counter()
    for row in rows:
        if per_deck[row.get("Title")] >= 5:
            continue
        per_deck[row.get("Title")] += 1
        if True:
            lic = str(row.get("License") or "").lower()
            if lic not in OPEN_LICENSES:
                continue
            image = row.get("image")
            try:
                img = Image.open(io.BytesIO(image["bytes"] if isinstance(image, dict) else image))
            except Exception:
                continue
            yield "zenodo_slides", str(row.get("Title")), "slides", row.get("License"), img


def notesbank(raw: Path, rng: random.Random, size: int):
    root = raw / "NoTeS-Bank__ICDAR_2025_Handwritten_Notes_Understanding_Challenge"
    files = sorted(f for f in root.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg") and "Images" in f.parts)
    rng.shuffle(files)
    per_doc = collections.Counter()
    for f in files:
        if per_doc[f.parent.name] >= PER_NOTEBOOK:
            continue
        per_doc[f.parent.name] += 1
        yield "notes_notesbank", f.parent.name, "handwriting", LICENSES["notes_notesbank"], Image.open(f)


def humyn(raw: Path, rng: random.Random, size: int):
    """Every page of the CC-BY HumynLabs handwritten-notes PDFs (math,
    physics, chemistry, biology, computer science)."""
    files = sorted(f for d in raw.glob("HumynLabs__*Notes*") for f in d.glob("*.pdf"))
    rng.shuffle(files)
    for f in files:
        try:
            data = f.read_bytes()
            for page in range(min(pdf_pages(data), PER_NOTEBOOK)):
                yield "notes_humyn", f.stem.rsplit("-", 1)[0], "handwriting", LICENSES["notes_humyn"], render_pdf(data, page, size)
        except Exception:
            continue


def augment(image: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    """About 6% rotated (90/180/270) and 2% colour-inverted pages."""
    roll = rng.random()
    if roll < 0.02:
        return ImageOps.invert(image.convert("RGB")), "invert"
    if roll < 0.08:
        angle = rng.choice([90, 180, 270])
        return image.rotate(angle, expand=True), f"rotate{angle}"
    return image, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocklist", type=Path, required=True, help="JSON list of hex perceptual hashes")
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--hash-distance", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--targets", choices=["stage1", "stage2", "stage3"], default="stage1")
    ap.add_argument("--base-manifest", type=Path, help="start a new set from an existing manifest (a superset)")
    a = ap.parse_args()
    if a.targets != "stage1":
        TARGETS.update(TARGETS_STAGE2 if a.targets == "stage2" else TARGETS_STAGE3)
    blocked = [imagehash.hex_to_hash(h) for h in json.loads(a.blocklist.read_text())]
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out / "manifest.jsonl"
    if a.base_manifest and not manifest.exists():
        manifest.write_text(a.base_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    done = collections.Counter()
    categories = collections.Counter()
    seen = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["source"]] += 1
            seen.add(r["hash"])
            if r["source"] == "doclaynet":
                categories[r["category"].removeprefix("doclaynet_")] += 1
    sources = {
        "olmocr_documents": lambda rng: olmocr(a.raw, "documents", "00_documents_train.parquet", rng, TARGETS["olmocr_documents"], a.size),
        "olmocr_books": lambda rng: olmocr(a.raw, "books", "01_books_train.parquet", rng, TARGETS["olmocr_books"], a.size),
        "olmocr_loc": lambda rng: olmocr(a.raw, "loc", "02_loc_transcripts_train.parquet", rng, TARGETS["olmocr_loc"], a.size),
        "olmocr_archives": lambda rng: olmocr(a.raw, "archives", "03_national_archives_train.parquet", rng, TARGETS["olmocr_archives"], a.size),
        "pdfa": lambda rng: webdataset_pdfs(a.raw, "pixparse__pdfa-eng-wds", "pdfa", rng, a.size),
        "idl": lambda rng: webdataset_pdfs(a.raw, "pixparse__idl-wds", "idl", rng, a.size),
        "doclaynet": lambda rng: doclaynet(a.raw, rng, a.size, categories),
        "loc_newspapers": lambda rng: loc_newspapers(a.raw, rng, a.size),
        "zenodo_slides": lambda rng: zenodo(a.raw, rng, a.size),
        "notes_notesbank": lambda rng: notesbank(a.raw, rng, a.size),
        "notes_humyn": lambda rng: humyn(a.raw, rng, a.size),
    }
    with manifest.open("a", encoding="utf-8") as out:
        for name, make in sources.items():
            if a.only and name not in a.only:
                continue
            want = TARGETS[name] - done[name]
            if want <= 0:
                continue
            rng = random.Random(f"{a.seed}-{name}")
            dropped = kept = 0
            directory = a.out / name
            directory.mkdir(exist_ok=True)
            try:
                for source, document, category, license_, image in make(rng):
                    image = fit(image, a.size)
                    h = imagehash.phash(image)
                    if str(h) in seen or any(h - b <= a.hash_distance for b in blocked):
                        dropped += 1
                        continue
                    seen.add(str(h))
                    image, aug = augment(image, rng)
                    index = done[name] + kept
                    path = directory / f"{index:06d}.jpg"
                    image.save(path, quality=92)
                    out.write(json.dumps({"id": f"{name}/{index:06d}", "source": source, "document": document,
                                          "category": category, "license": license_, "augment": aug,
                                          "width": image.width, "height": image.height,
                                          "path": str(path), "hash": str(h)}) + "\n")
                    out.flush()
                    kept += 1
                    if kept >= want:
                        break
            except FileNotFoundError as e:
                print(f"{name}: missing raw data ({e})", flush=True)
            print(f"{name}: kept {kept} (total {done[name] + kept}/{TARGETS[name]}), dropped {dropped} near-duplicates", flush=True)


if __name__ == "__main__":
    main()
