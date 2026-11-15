#!/usr/bin/env python3
"""Turn `falcon-ocr run` JSON lines into the report shape check_heldout.py reads.

research/phase4-hillclimb/attempt3/check_heldout.py reads a `bench` report's
`inputs[i].path` (the page is the path's parent directory name) and, at the
same index, `samples[0].outputs[i]`: `text`, `token_ids`, `finish_reason`,
`output_tokens` and `timings.total_ms`. A run record carries all of them, so
each output is the record itself without `path` and `page`.

The expected pages come from --lock (each page's `canonical_path`, in lock
order) or --pages (one path per line, as `falcon-ocr run --list` reads them,
blank and `#` lines skipped). The run must hold exactly those pages, in that
order, each once, with `page` counting up from 0 and no error record;
anything else is an error. Each input file is hashed into `inputs[].sha256`
as `bench` does, and with --lock the hash must equal the page's
`canonical_png_sha256`. Run from the directory the run was started in (its
paths are relative to it). The output must not exist.

  python3 research/corpus-qualification/scripts/bench_report_from_run.py \\
      --run artifacts/fast-defaults-gate-v1/old.jsonl --lock reference/fast-defaults-gate-v1-masked-lock.json \\
      --output artifacts/fast-defaults-gate-v1/old-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_pages(a: argparse.Namespace) -> list[tuple[Path, str | None]]:
    """The expected input paths, each with the hash its file must have."""
    if a.lock:
        pages = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
        return [(Path(p["canonical_path"]), p["canonical_png_sha256"]) for p in pages]
    lines = [line.strip() for line in a.pages.read_text(encoding="utf-8").splitlines()]
    return [(Path(line), None) for line in lines if line and not line.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="`falcon-ocr run` JSON lines")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--lock", type=Path, help="corpus lock whose canonical paths the run read, in order")
    source.add_argument("--pages", type=Path, help="page list the run read, in order")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        sys.exit(f"refusing to overwrite {a.output}")
    expected = expected_pages(a)
    records = []
    for number, line in enumerate(a.run.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            sys.exit(f"{a.run}:{number}: not a JSON record ({error})")
    errors = [f"{r.get('path')}: {r['error']}" for r in records if "error" in r]
    if errors:
        sys.exit(f"{len(errors)} failed page(s): {errors[:5]}")
    seen = [Path(r["path"]) for r in records]
    if seen != [path for path, _ in expected]:
        missing = [str(p) for p, _ in expected if p not in set(seen)]
        extra = [str(p) for p in seen if p not in {p for p, _ in expected}]
        detail = (f"missing {len(missing)} {missing[:5]}, unexpected {len(extra)} {extra[:5]}" if missing or extra
                  else "same pages in another order or with repeats")
        sys.exit(f"run holds {len(seen)} pages, expected {len(expected)}: {detail}")
    if [r["page"] for r in records] != list(range(len(records))):
        sys.exit("page indices do not count up from 0")
    inputs, outputs = [], []
    for record, (path, want) in zip(records, expected, strict=True):
        digest = file_sha256(path)
        if want is not None and digest != want:
            sys.exit(f"{path}: sha256 {digest} differs from the lock's {want}")
        inputs.append({"path": record["path"], "sha256": digest})
        outputs.append({k: v for k, v in record.items() if k not in ("path", "page")})
    report = {
        "schema": "falcon-ocr-run-report-v1",
        "source": {"run": a.run.as_posix(), "run_sha256": file_sha256(a.run),
                   "expected": (a.lock or a.pages).as_posix(), "expected_sha256": file_sha256(a.lock or a.pages)},
        "inputs": inputs,
        "samples": [{"index": 0, "outputs": outputs}],
    }
    a.output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": a.output.as_posix(), "pages": len(outputs),
                      "finish_reasons": {k: sum(o["finish_reason"] == k for o in outputs)
                                         for k in sorted({o["finish_reason"] for o in outputs})}}))


if __name__ == "__main__":
    main()
