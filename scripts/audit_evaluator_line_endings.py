#!/usr/bin/env python3
"""Prove pinned evaluator working bytes differ only by CRLF before Git interop."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

REVISION = "59b103c4b47d3a01fada83491585d6512a40c0bc"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = Path(__file__)
    source_hash = sha(source.read_bytes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, default=Path("artifacts/OmniDocBench-eval"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing audit")

    def git(*argv, **kwargs):
        return subprocess.check_output(["git", "-C", str(args.evaluator), *argv],
                                       env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"), **kwargs)

    if git("rev-parse", "HEAD").decode().strip() != REVISION:
        raise ValueError("Evaluator commit differs")
    tree = git("ls-tree", "-r", "-z", "HEAD")
    flags = {entry[2:].decode("utf-8"): entry[:1] for entry in git("ls-files", "-t", "-z").split(b"\0") if entry}
    entries = []
    sparse_absent = []
    for item in tree.split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, kind, blob = metadata.split()
        if kind != b"blob" or mode not in [b"100644", b"100755"]:
            raise ValueError("Unexpected nonregular evaluator source")
        name = name.decode("utf-8")
        if not (args.evaluator / name).is_file():
            if flags.get(name) != b"S":
                raise ValueError("Missing nonsparse tracked file: " + name)
            sparse_absent.append(name)
            continue
        entries.append((name, blob))
    stream = io.BytesIO(git("cat-file", "--batch", input=b"".join(blob + b"\n" for _, blob in entries)))
    rows = []
    for name, blob in entries:
        actual_blob, kind, size = stream.readline().split()
        if (actual_blob, kind) != (blob, b"blob"):
            raise ValueError("Unexpected Git batch object")
        expected = stream.read(int(size))
        if len(expected) != int(size) or stream.read(1) != b"\n":
            raise ValueError("Truncated Git batch object")
        raw = (args.evaluator / name).read_bytes()
        if raw == expected:
            change = "exact_git_blob_bytes"
        elif b"\0" not in expected and raw.replace(b"\r\n", b"\n") == expected:
            change = "CRLF_to_LF_only"
        else:
            raise ValueError("Non-line-ending source change: " + name)
        rows.append({"path": name, "git_blob_sha1": blob.decode(), "git_blob_sha256": sha(expected),
                     "working_sha256": sha(raw), "comparison": change})
    if stream.read():
        raise ValueError("Unexpected extra Git output")
    for row in rows:
        if sha((args.evaluator / row["path"]).read_bytes()) != row["working_sha256"]:
            raise ValueError("Source changed during audit")
    if any((args.evaluator / name).exists() for name in sparse_absent):
        raise ValueError("Sparse file inventory changed")
    if git("ls-tree", "-r", "-z", "HEAD") != tree or git("rev-parse", "HEAD").decode().strip() != REVISION:
        raise ValueError("Git tree changed during audit")
    if sha(source.read_bytes()) != source_hash:
        raise ValueError("Audit source changed")
    report = {"schema_version": 1, "status": "all_materialized_tracked_files_exact_or_only_crlf", "revision": REVISION,
              "evaluator": str(args.evaluator), "files": len(rows),
              "crlf_only_files": sum(r["comparison"] == "CRLF_to_LF_only" for r in rows),
              "absent_skip_worktree_paths": sparse_absent,
              "script_sha256": source_hash, "tracked_files": rows,
              "scope": "Read-only byte comparison of every materialized tracked file against its pinned Git blob. Absent skip-worktree entries are listed separately and not downloaded. No evaluator, tracked source, local Git config, input prediction or numerical result was changed. Allows a separately recorded per-process core.autocrlf=true setting while preserving clean-tree validation and raw-byte source audit."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({k: report[k] for k in ["status", "files", "crlf_only_files"]}))


if __name__ == "__main__":
    main()
