#!/usr/bin/env python3
"""Router parity fixtures (tests/fixtures/router.json) for src/router: the
26 statistics, both raw tree scores and the route of synthetic pages (the
formula below, mirrored by `synthetic` in src/router/mod.rs) and of the
decode fixtures in tests/fixtures/images (after the processor's first resize
at the 1536 cap), computed by the specification
research/resolution-router/router_features.py and the exported trees.

  wsl.exe -d Ubuntu-24.04-CUDA --cd /mnt/c/Users/amazi/Documents/ChatGPT/falcon-ocr -- \\
      /home/amazi/falcon-ocr-rust-reference/.venv/bin/python tests/generate_router_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "resolution-router"))
from router_features import CAP, capped_page, features  # noqa: E402
from train_trees import raw_score  # noqa: E402

SYNTHETIC = [(1187, 1536, 0), (1536, 1086, 1), (1200, 1553, 2), (900, 1200, 3), (700, 950, 1), (1536, 400, 0),
             (1100, 1536, 1), (1536, 1187, 3)]


def synthetic(width: int, height: int, seed: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    pitch, glyph = 18 + 5 * seed, 9 + 3 * seed
    out = np.repeat((250 - (x * 7 + y * 13 + seed) % 6)[..., None], 3, axis=2)
    text = (x >= 60) & (x < max(0, width - 60)) & (y >= 80) & (y < max(0, height - 80))
    gutter = (seed == 2) & (np.abs(x - width // 2) < 20)
    row, col = (y - 80) // pitch, (x - 60) // 7
    ink = text & ~gutter & ((y - 80) % pitch < glyph) & ((col + row * 3) % 11 != 0) \
        & ((x * 31 + y * 17 + row * 7 + col * 13 + seed) % 23 < 11)
    v = 20 + (x + y) % 30
    out[ink] = np.stack([v, v, v + seed * 10], axis=2)[ink]
    header = (y >= 20) & (y < 60) & ((x // 40) % 2 == 0)
    out[header] = [200, 40 + seed * 30, 40]
    return out.astype(np.uint8)


def case(name: str, rgb: np.ndarray, trees: dict) -> dict:
    f = features(rgb)
    raw = {s: raw_score(trees["models"][s], f) for s in ("768", "1024")}
    long_side = max(f[0], f[1])
    route = CAP
    for s in (768, 1024):
        if raw[str(s)] >= 0 and f[15] * min(1.0, s / long_side) >= 8:
            route = s
            break
    return {"name": name, "features": f, "raw_768": raw["768"], "raw_1024": raw["1024"], "route": route}


def main() -> None:
    trees = json.loads((ROOT / "src" / "router" / "trees.json").read_text(encoding="utf-8"))
    out = {"pillow": PIL.__version__, "numpy": np.__version__, "synthetic": [], "files": []}
    for w, h, seed in SYNTHETIC:
        c = case(f"synthetic-{w}x{h}-{seed}", synthetic(w, h, seed), trees)
        out["synthetic"].append({"width": w, "height": h, "seed": seed, **c})
    for path in sorted((ROOT / "tests" / "fixtures" / "images").iterdir()):
        if path.suffix in (".png", ".jpg"):
            out["files"].append(case(path.name, capped_page(str(path)), trees))
    (ROOT / "tests" / "fixtures" / "router.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print({c["name"]: c["route"] for c in out["synthetic"]})
    print(f"{len(out['files'])} decode fixtures")


if __name__ == "__main__":
    main()
