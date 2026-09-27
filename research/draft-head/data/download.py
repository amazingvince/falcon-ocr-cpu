#!/usr/bin/env python3
"""Download the stage-1 raw sources for the draft-head training set to D:.

Only permissively licensed, English sources (the head will be published):
olmOCR-mix-1025 (ODC-BY), PDFA English (Common Crawl / Digital Corpora
terms), IDL (Industry Documents Library terms), DocLayNet v1.2 (CDLA-P),
LoC Beyond Words (CC0), Zenodo open presentations (per-record CC licenses,
filtered later), NoTeS-Bank (Apache-2.0), HumynLabs notes (CC-BY-4.0).
OmniDocBench is downloaded only as the decontamination blocklist.

  python research/draft-head/data/download.py --root D:/falcon-draft/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# (repo, files or glob prefixes). Train splits only.
SOURCES = [
    ("allenai/olmOCR-mix-1025", [
        "00_documents_train.parquet", "01_books_train.parquet",
        "02_loc_transcripts_train.parquet", "03_national_archives_train.parquet",
        "pdf_tarballs/00_documents_train_00000.tar.gz", "pdf_tarballs/00_documents_train_00001.tar.gz",
        "pdf_tarballs/01_books_train_00000.tar.gz",
        "pdf_tarballs/02_loc_transcripts_train_00000.tar.gz",
        "pdf_tarballs/03_national_archives_train_00000.tar.gz",
    ]),
    ("docling-project/DocLayNet-v1.2", ["data/train-00000-", "data/train-00001-"]),
    ("biglam/loc_beyond_words", ["data/images.zip", "README.md"]),
    ("PDFPages/zenodo-presentations-open", ["data/"]),
    ("NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge", ["Task_1/Images/"]),
    ("HumynLabs/English-Handwritten-Math-Notes-Dataset", [""]),
    ("pixparse/pdfa-eng-wds", ["pdfa-eng-train-0000.tar", "pdfa-eng-train-0001.tar"]),
    ("pixparse/idl-wds", ["idl-train-00000.tar", "idl-train-00001.tar"]),
    ("opendatalab/OmniDocBench", ["OmniDocBench.json", "images/"]),
]

# Stage 2 (about 25k pages in total): more shards of the same English sources
# plus the other CC-BY HumynLabs handwritten-notes subjects.
DOCLAYNET_MORE = [3, 9, 12, 21, 24, 33, 36, 45, 48, 57, 60, 69]
SOURCES_STAGE2 = [
    ("allenai/olmOCR-mix-1025", [f"pdf_tarballs/00_documents_train_{i:05d}.tar.gz" for i in range(2, 6)] + [
        "pdf_tarballs/01_books_train_00001.tar.gz",
        "pdf_tarballs/02_loc_transcripts_train_00001.tar.gz",
        "pdf_tarballs/03_national_archives_train_00001.tar.gz",
        "pdf_tarballs/03_national_archives_train_00002.tar.gz",
    ]),
    ("pixparse/pdfa-eng-wds", [f"pdfa-eng-train-{i:04d}.tar" for i in range(2, 6)]),
    ("pixparse/idl-wds", [f"idl-train-{i:05d}.tar" for i in range(2, 6)]),
    ("docling-project/DocLayNet-v1.2", [f"data/train-{i:05d}-" for i in DOCLAYNET_MORE]),
    ("NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge", ["Task_2/Images/"]),
    ("HumynLabs/Handwritten-Physics-Notes-Dataset", [""]),
    ("HumynLabs/Handwritten-Chemistry-Notes-Dataset", [""]),
    ("HumynLabs/Handwritten-Biology-Notes-Dataset", [""]),
    ("HumynLabs/Handwritten-Computer-Science-Notes-Dataset", [""]),
]


# Stage 3 (about 48k pages in total): more shards of the large English
# document sources (the head's acceptance was still rising with data).
SOURCES_STAGE3 = [
    ("allenai/olmOCR-mix-1025", [f"pdf_tarballs/00_documents_train_{i:05d}.tar.gz" for i in range(6, 14)] + [
        f"pdf_tarballs/01_books_train_{i:05d}.tar.gz" for i in range(2, 4)]),
    ("pixparse/pdfa-eng-wds", [f"pdfa-eng-train-{i:04d}.tar" for i in range(6, 15)]),
    ("pixparse/idl-wds", [f"idl-train-{i:05d}.tar" for i in range(6, 14)]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="repo ids to fetch (default: all)")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    a = ap.parse_args()
    api = HfApi()
    for repo, wanted in {1: SOURCES, 2: SOURCES_STAGE2, 3: SOURCES_STAGE3}[a.stage]:
        if a.only and repo not in a.only:
            continue
        files = api.list_repo_files(repo, repo_type="dataset")
        chosen = [f for f in files if any(f == w or (w.endswith(("/", "-")) or w == "") and f.startswith(w) for w in wanted)]
        if not chosen:
            # Webdataset shard names differ between repos: list what exists.
            print(f"{repo}: no match for {wanted}; first files: {files[:8]}", flush=True)
            continue
        target = a.root / repo.replace("/", "__")
        print(f"{repo}: {len(chosen)} files -> {target}", flush=True)
        for i, name in enumerate(chosen):
            if name.startswith(".git"):
                continue
            hf_hub_download(repo, name, repo_type="dataset", local_dir=target)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(chosen)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
