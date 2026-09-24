#!/usr/bin/env python3
"""Inspect primary dataset revision, license statement and small metadata only."""
import json
import pathlib
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    out = pathlib.Path("artifacts/corpus-source-metadata")
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset in ["opendatalab/OmniDocBench", "allenai/olmOCR-bench"]:
        meta = get_json(f"https://huggingface.co/api/datasets/{dataset}")
        refs = get_json(f"https://huggingface.co/api/datasets/{dataset}/refs")
        revision = meta["sha"]
        tree = get_json(f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}?recursive=false&expand=false")
        name = dataset.replace("/", "--")
        (out / f"{name}.json").write_text(json.dumps({"metadata": meta, "refs": refs, "tree": tree}, indent=2) + "\n")
        with urllib.request.urlopen(f"https://huggingface.co/datasets/{dataset}/raw/{revision}/README.md", timeout=60) as response:
            (out / f"{name}-README.md").write_bytes(response.read())
        reports.append({"dataset": dataset, "revision": revision, "card_data": meta.get("cardData"),
                        "refs": refs, "root_files": [{"path": f["path"], "type": f["type"], "size": f.get("size")} for f in tree]})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
