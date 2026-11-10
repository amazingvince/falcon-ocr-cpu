#!/usr/bin/env python3
"""Fetch the verification fixtures the weights-backed gates compare against.

`scripts/fetch_reference.py` brings the pinned checkpoint, and the packed files
come from the published catalogue. This brings the two things that are neither:
the FP32 GPU smoke record and the original GPTQ overlay. Both were made on one
machine and could not be remade elsewhere, which is why the gates that read them
ran in one place (issue #8); they are files, and a GPU is needed only to record
them, not to check against them.

What they unlock, once `artifacts/model` holds the checkpoint:

    metadata.json      tests/gpu_parity.rs, the exact-mode test in tests/modes.rs
    canonical-rgb.png  the input page for most of the weights-backed gates
    trace.safetensors  tests/cache_layout.rs, tests/weight_layout.rs,
                       the tensor-trace test in tests/modes.rs, tools/check.sh --smoke
    w8-gptq.safetensors  fast mode, and the overlay digest that
                         tests/fixtures/modes/fast-journal-300.json pins

Every file is checked against the digest recorded here, so a fixture that
changed upstream fails loudly instead of quietly moving a gate's baseline.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

REPO = "BEE-spoke-data/falcon-ocr-v1.5-cpu"
REVISION = "a6eeacfd0ac03989c2f814fb8f807dc1376323d4"

# Remote path -> (path under --output, sha256, bytes).
FIXTURES = {
    "fixtures/smoke-fp32/metadata.json": (
        "reference/smoke-fp32/metadata.json",
        "c84b1fcc4526135677579392d2c2c52b1566c1f361ac950a6409a5145a8fc65e",
        2335,
    ),
    "fixtures/smoke-fp32/canonical-rgb.png": (
        "reference/smoke-fp32/canonical-rgb.png",
        "d62f8db10dca06c65c99ae4dde59e4a68b9548c56a42c76caa2bf7044ccca0db",
        5280,
    ),
    "fixtures/smoke-fp32/teacher-tokens.json": (
        "reference/smoke-fp32/teacher-tokens.json",
        "7cb1179d6190b81b22e7c3d552bb732d75f1a040f530bf74f098c5981f422d0e",
        92,
    ),
    "fixtures/smoke-fp32/trace.safetensors": (
        "reference/smoke-fp32/trace.safetensors",
        "30dca24da26b6a42b5f6e65c0f0a3efd02f845c54710ddb626e624b32c4395d4",
        74002424,
    ),
    "fixtures/w8-gptq.safetensors": (
        "model/w8-gptq.safetensors",
        "60808bbf24e506497da3a828318085a1e3211bb974f02a3c5375721d1b22f996",
        179253080,
    ),
}
# The overlay is 171 MB and only fast mode and the overlay-digest assertion need it.
OVERLAY = "fixtures/w8-gptq.safetensors"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(out, remote, local, expected, size):
    target = out / local
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
        subprocess.run(
            ["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "6", "--retry-delay", "2",
             "--continue-at", "-", "--output", str(partial), url],
            check=True,
        )
        partial.replace(target)
    digest = sha256(target)
    if digest != expected:
        raise RuntimeError(
            f"{local}: expected sha256 {expected}, got {digest}. Delete it and rerun; if it persists, the "
            f"fixture moved and this script's pin is stale."
        )
    actual = target.stat().st_size
    if actual != size:
        raise RuntimeError(f"{local}: expected {size} bytes, got {actual}")
    print(f"Verified {local}: {digest}", flush=True)
    return local, {"sha256": digest, "bytes": actual, "remote": remote}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts"),
                        help="the artifacts directory the gates read (default: artifacts)")
    parser.add_argument("--skip-overlay", action="store_true",
                        help="leave out the 171 MB GPTQ overlay (fast mode and the digest assertion need it)")
    args = parser.parse_args()
    if REVISION == "PENDING":
        raise SystemExit("REVISION is unset: pin the fixture commit before using this script")
    wanted = {k: v for k, v in FIXTURES.items() if not (args.skip_overlay and k == OVERLAY)}
    hashes = dict(fetch(args.output, remote, *spec) for remote, spec in wanted.items())
    manifest = {
        "schema_version": 1,
        "fixtures_repo": REPO,
        "fixtures_revision": REVISION,
        "files": hashes,
    }
    (args.output / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output / 'fixture-manifest.json'}")


if __name__ == "__main__":
    main()
