#!/usr/bin/env python3
"""Fetch immutable Falcon-OCR assets, recording hashes and checking the weight digest."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import urllib.request

REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"
OFFICIAL_REVISION = "c457916c9974efbacfa91f0f6ecc2c49c6543e56"
WEIGHT_SHA256 = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"
FILES = ["attention.py", "config.json", "configuration_falcon_ocr.py", "model_args.json",
         "modeling_falcon_ocr.py", "processing_falcon_ocr.py", "rope.py", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "README.md"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(out, filename):
    target = out / filename
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        url = f"https://huggingface.co/tiiuae/Falcon-OCR/resolve/{REVISION}/{filename}"
        subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "6", "--retry-delay", "2",
                        "--continue-at", "-", "--output", str(partial), url], check=True)
        partial.replace(target)
    digest = sha256(target)
    if filename == "model.safetensors" and digest != WEIGHT_SHA256:
        raise RuntimeError(f"Weight checksum mismatch: {digest}")
    print(f"Verified {filename}: {digest}", flush=True)
    return filename, {"sha256": digest, "bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/model"))
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--official-output", type=pathlib.Path, default=pathlib.Path("artifacts/Falcon-Perception"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(lambda name: fetch(args.output, name), FILES))
    if not args.skip_weights:
        name, info = fetch(args.output, "model.safetensors")
        hashes[name] = info
    manifest = {"schema_version": 1, "model_id": "tiiuae/Falcon-OCR", "model_revision": REVISION,
                "weights_sha256": WEIGHT_SHA256,
                "official_repository": "https://github.com/tiiuae/Falcon-Perception",
                "official_revision": OFFICIAL_REVISION, "files": hashes}
    (args.output / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not (args.official_output / ".git").exists():
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/tiiuae/Falcon-Perception", str(args.official_output)], check=True)
    subprocess.run(["git", "-C", str(args.official_output), "checkout", "--detach", OFFICIAL_REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(args.official_output), "rev-parse", "HEAD"], text=True).strip()
    assert actual == OFFICIAL_REVISION, actual


if __name__ == "__main__":
    main()
