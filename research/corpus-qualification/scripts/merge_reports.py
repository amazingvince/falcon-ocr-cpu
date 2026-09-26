#!/usr/bin/env python3
"""Replace pages of a `bench` report with those of another run (for example
the figure-masked copies from mask_excluded.py), matching pages by their
directory name, so a full set can be checked with check_heldout.py when only
some of its pages changed. Pages whose input did not change keep the base
run's output (the runner is deterministic for a given input).

  python research/corpus-qualification/scripts/merge_reports.py --base fast.json --override masked/fast.json \\
      --output masked/fast-merged.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--override", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    base = json.loads(a.base.read_text(encoding="utf-8"))
    over = json.loads(a.override.read_text(encoding="utf-8"))
    # Reports written on Windows carry backslash paths; key pages by directory name either way.
    for i in over["inputs"]:
        i["path"] = i["path"].replace("\\", "/")
    replaced = {Path(i["path"]).parent.name: (i, o)
                for i, o in zip(over["inputs"], over["samples"][0]["outputs"], strict=True)}
    inputs, outputs = [], []
    for i, o in zip(base["inputs"], base["samples"][0]["outputs"], strict=True):
        i, o = replaced.pop(Path(i["path"]).parent.name, (i, o))
        inputs.append(i)
        outputs.append(o)
    assert not replaced, f"pages missing from the base report: {sorted(replaced)}"
    merged = dict(base)
    merged["inputs"] = inputs
    merged["samples"] = [{**base["samples"][0], "outputs": outputs}]
    merged["merged_from"] = {"base": str(a.base), "override": str(a.override),
                             "replaced_pages": len(over["inputs"])}
    a.output.write_text(json.dumps(merged), encoding="utf-8")
    print(f"{a.output}: {len(inputs)} pages, {len(over['inputs'])} from {a.override.name}")


if __name__ == "__main__":
    main()
