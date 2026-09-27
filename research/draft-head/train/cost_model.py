#!/usr/bin/env python3
"""CPU decode speedup of a draft head from its measured acceptance.

`reach[j-1]` = P(the first j drafts are all accepted), from `eagle3.py eval`.
A step drafting k tokens costs `single + k * (row + draft)` ms and yields
`1 + sum(reach[:k])` tokens; the best k is chosen per configuration.

  python research/draft-head/train/cost_model.py D:/falcon-draft/runs/*/calib.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SINGLE_MS = 9.0
ROWS_MS = (1.7, 1.0)  # verify cost per extra row: measured today / after more kernel work


def best(reach: list[float], draft_ms: float, row_ms: float) -> tuple[int, float]:
    options = [(SINGLE_MS * (1 + sum(reach[:k])) / (SINGLE_MS + k * (row_ms + draft_ms)), k)
               for k in range(1, len(reach) + 1)]
    speedup, k = max(options)
    return (k, speedup) if speedup > 1 else (0, 1.0)


def main() -> None:
    for path in sys.argv[1:]:
        report = json.loads(Path(path).read_text())
        reach = report["reach"]
        image = "text" not in path
        draft_ms = 0.6 if image else 0.45
        cells = []
        for row in ROWS_MS:
            k, s = best(reach, draft_ms, row)
            cells.append(f"row {row} ms: k={k} {s:.2f}x")
        print(f"{Path(path).parent.name:24} step-1 {100 * reach[0]:.1f}%  mean accepted {report['mean_accepted']:.2f}  "
              + "  |  ".join(cells))


if __name__ == "__main__":
    main()
