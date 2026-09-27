#!/usr/bin/env python3
"""CNN routing probabilities for the router development set, trained on
exactly the labels the image-statistics trees use (dev_image_probs.py: the
noise-aware labels of the stable Phase 0 pages), so the three routers compare
fairly on ground truth with dev_policy.py.

  python research/resolution-router/dev_cnn_probs.py --model cnn --router /mnt/d/falcon-draft/router \\
      --lock reference/router-dev-v1-evaluation-lock.json --out /mnt/d/falcon-draft/router-dev/cnn-probs.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import crops, native_path, statistics, thumbnail  # noqa: E402
from sample_pages import resize_image_if_necessary  # noqa: E402
from train_router import CropCnn, TinyCnn  # noqa: E402


def dev_inputs(path: str):
    with Image.open(native_path(path)) as img:
        img = resize_image_if_necessary(img.convert("RGB"), 64, 1536)
        g = np.asarray(img.convert("L"), dtype=np.float32)
        return thumbnail(img), crops(g), statistics(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cnn", "crops"], required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, default=Path("/mnt/d/falcon-draft/outputs/stage3"))
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args()
    info = json.loads((a.router / "phase1-meta.json").read_text(encoding="utf-8"))
    meta, data = info["pages"], np.load(a.router / "phase1-data.npz")
    noise, stop_b = [], []
    for m in meta:
        name = m["id"].replace("/", "__") + ".json"
        first = json.loads((a.outputs / name).read_text(encoding="utf-8"))
        second = json.loads((a.router / "outputs-1536-retest" / name).read_text(encoding="utf-8"))
        noise.append(Levenshtein.distance(second["text"], first["text"]) / max(1, len(first["text"])))
        stop_b.append(second["finish_reason"])
    stable = np.array([nz <= a.tolerance and m["stop"]["1536"] == sb for m, nz, sb in zip(meta, noise, stop_b)])
    y = np.stack([[m["disagreement"][s] <= max(a.tolerance, nz + a.margin) and m["stop"][s] in (m["stop"]["1536"], sb)
                   for m, nz, sb in zip(meta, noise, stop_b)] for s in ("768", "1024")], 1).astype(np.float32)
    feats = data["features"]
    size = np.stack([feats[:, 0] / 1536, feats[:, 1] / 1536, feats[:, 2], feats[:, 3] / 2.4], 1).astype(np.float32)
    x = data["thumbs"][:, None] if a.model == "cnn" else data["crops"][:, :, None]
    train = np.nonzero(stable)[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    model = (TinyCnn if a.model == "cnn" else CropCnn)().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * (len(train) // 64 + 1))
    bce = nn.BCEWithLogitsLoss()
    xt, st, yt = torch.from_numpy(x), torch.from_numpy(size), torch.from_numpy(y)
    for _ in range(a.epochs):
        model.train()
        perm = torch.from_numpy(train)[torch.randperm(len(train))]
        for b in range(0, len(perm), 64):
            j = perm[b:b + 64]
            route, _ = model((xt[j].float().to(device) / 255.0 - 0.5) / 0.5, st[j].to(device))
            loss = bce(route, yt[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    with concurrent.futures.ProcessPoolExecutor(8) as pool:
        dev = list(pool.map(dev_inputs, [p["canonical_path"] for p in lock]))
    out = {}
    with torch.no_grad():
        for entry, (thumb, tiles, f) in zip(lock, dev):
            inp = thumb[None, None] if a.model == "cnn" else tiles[None, :, None]
            s = np.array([[f[0] / 1536, f[1] / 1536, f[2], f[3] / 2.4]], dtype=np.float32)
            p = torch.sigmoid(model((torch.from_numpy(inp).float().to(device) / 255.0 - 0.5) / 0.5,
                                    torch.from_numpy(s).to(device))[0])[0].cpu().numpy()
            pid = Path(entry["canonical_path"]).parent.name
            out[pid] = {"768": float(p[0]), "1024": float(p[1]), "width": float(f[0]), "height": float(f[1]),
                        "line_height_px": float(f[15])}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{a.model}: {len(out)} pages; mean p768 {np.mean([v['768'] for v in out.values()]):.2f}, "
          f"p1024 {np.mean([v['1024'] for v in out.values()]):.2f}")


if __name__ == "__main__":
    main()
