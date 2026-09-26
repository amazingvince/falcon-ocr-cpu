#!/usr/bin/env python3
"""Phase 1: train and compare two CPU-cheap routers on the Phase 0 labels.

Labels per page: routable at 1024 / 768 = the text stays within `--tolerance`
of the 1536 transcript and the stop reason is unchanged. Split by source
document (70/15/15). Candidates:

- `gbdt`: gradient-boosted trees on the ~20 image statistics of
  build_dataset.py (about 1 ms to compute, trivial inference).
- `cnn`: a plain CNN on the 384-px grayscale thumbnail, five stride-2 3x3
  convolutions (16-32-64-64-96 channels, ~120M multiply-adds) plus the page
  size, with two routability heads and a page-type head.

Policy: the lowest resolution whose predicted routability clears its
threshold, else 1536. Thresholds are tuned on validation to maximise modelled
page time saved (router plan cost model, fast mode with the draft head)
with at most `--budget` of pages routed below their label ("mis-routed"),
then reported on test, with the oracle and flat policies for reference.

  python research/resolution-router/train_router.py --router D:/falcon-draft/router
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch import nn

SIZES = (768, 1024)
TYPES = ["prose", "formulas", "tables", "slides", "newspaper", "handwriting"]


def page_seconds(n, t):
    return (0.33 * n + 0.000049 * n * n + t * (4.3 + 0.00064 * n) / 1.47) / 1000


class TinyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        chans, layers, c_in = [16, 32, 64, 64, 96], [], 1
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * c_in + 4, 64), nn.ReLU(inplace=True))
        self.route = nn.Linear(64, len(SIZES))
        self.kind = nn.Linear(64, len(TYPES))

    def forward(self, x, size):
        f = self.body(x)
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3)), size], 1)
        h = self.head(f)
        return self.route(h), self.kind(h)


class CropCnn(nn.Module):
    """A shared trunk over native-resolution crops (glyph size is what decides
    routability), pooled over crops, plus the page size."""

    def __init__(self):
        super().__init__()
        chans, layers, c_in = [16, 32, 64, 96], [], 1
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(4 * c_in + 4, 64), nn.ReLU(inplace=True))
        self.route = nn.Linear(64, len(SIZES))
        self.kind = nn.Linear(64, len(TYPES))

    def forward(self, x, size):
        b, k = x.shape[:2]
        f = self.body(x.flatten(0, 1))
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1).view(b, k, -1)
        f = torch.cat([f.mean(1), f.amax(1), size], 1)
        h = self.head(f)
        return self.route(h), self.kind(h)


def split_of(group: str) -> str:
    v = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
    return "train" if v < 70 else "val" if v < 85 else "test"


def policy(probs, thresholds):
    """Lowest resolution whose probability clears its threshold, else 1536."""
    choice = np.full(len(probs[768]), 1536)
    for size in sorted(SIZES, reverse=True):
        choice = np.where(probs[size] >= thresholds[size], size, choice)
    return choice


def evaluate(choice, meta, idx, labels):
    base = sum(page_seconds(meta[i]["image_tokens"]["1536"], meta[i]["tokens"]["1536"]) for i in idx)
    routed, bad = 0.0, 0
    for c, i in zip(choice, idx):
        routed += page_seconds(meta[i]["image_tokens"][str(c)], meta[i]["tokens"][str(c)])
        if c != 1536 and not labels[c][i]:
            bad += 1
    return 1 - routed / base, bad / len(idx)


def tune(probs, meta, idx, labels, budget):
    best = (-1.0, None, None)
    grid = np.append(np.linspace(0.3, 0.99, 24), [0.995, 0.999, 1.01])  # 1.01: never route there
    for t768 in grid:
        for t1024 in grid:
            th = {768: t768, 1024: t1024}
            saved, bad = evaluate(policy(probs, th), meta, idx, labels)
            if bad <= budget and saved > best[0]:
                best = (saved, th, bad)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    data = np.load(a.router / "phase1-data.npz")
    info = json.loads((a.router / "phase1-meta.json").read_text(encoding="utf-8"))
    meta, feats, thumbs = info["pages"], data["features"], data["thumbs"]
    n = len(meta)
    labels = {s: np.array([m["disagreement"][str(s)] <= a.tolerance and m["stop"][str(s)] == m["stop"]["1536"]
                           for m in meta]) for s in SIZES}
    kinds = np.array([TYPES.index(m["type"]) for m in meta])
    splits = np.array([split_of(m["group"]) for m in meta])
    idx = {s: np.nonzero(splits == s)[0] for s in ("train", "val", "test")}
    print({k: len(v) for k, v in idx.items()}, {s: f"{labels[s].mean():.1%} routable" for s in SIZES})
    probs = {}

    # Gradient-boosted trees on the image statistics.
    gb = {}
    for s in SIZES:
        gb[s] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        gb[s].fit(feats[idx["train"]], labels[s][idx["train"]])
    probs["gbdt"] = {part: {s: gb[s].predict_proba(feats[idx[part]])[:, 1] for s in SIZES} for part in ("val", "test")}

    # CNNs: the thumbnail and the native crops (+ page size).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_feats = np.stack([feats[:, 0] / 1536, feats[:, 1] / 1536, feats[:, 2], feats[:, 3] / 2.4], 1).astype(np.float32)
    s_all = torch.from_numpy(size_feats)
    y_all = torch.from_numpy(np.stack([labels[s] for s in SIZES], 1).astype(np.float32))
    k_all = torch.from_numpy(kinds).long()
    inputs = {"cnn": torch.from_numpy(thumbs).unsqueeze(1), "crops": torch.from_numpy(data["crops"]).unsqueeze(2)}
    models = {}
    for name, make in (("cnn", TinyCnn), ("crops", CropCnn)):
        torch.manual_seed(0)
        x_all = inputs[name]
        model = make().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = a.epochs * (len(idx["train"]) // 64 + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
        train = torch.from_numpy(idx["train"])
        for epoch in range(a.epochs):
            model.train()
            perm = train[torch.randperm(len(train))]
            for b in range(0, len(perm), 64):
                j = perm[b:b + 64]
                x = (x_all[j].float().to(device) / 255.0 - 0.5) / 0.5
                route, kind = model(x, s_all[j].to(device))
                loss = bce(route, y_all[j].to(device)) + 0.3 * ce(kind, k_all[j].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        model.eval()

        def predict(part):
            out = []
            with torch.no_grad():
                for b in range(0, len(idx[part]), 256):
                    j = torch.from_numpy(idx[part][b:b + 256])
                    x = (x_all[j].float().to(device) / 255.0 - 0.5) / 0.5
                    out.append(torch.sigmoid(model(x, s_all[j].to(device))[0]).cpu().numpy())
            p = np.concatenate(out)
            return {s: p[:, k] for k, s in enumerate(SIZES)}
        probs[name] = {part: predict(part) for part in ("val", "test")}
        torch.save(model.state_dict(), a.router / f"phase1-{name}.pt")
        models[name] = model.cpu()

    # Report.
    for name in ("gbdt", "cnn", "crops"):
        aucs = {s: roc_auc_score(labels[s][idx["test"]], probs[name]["test"][s]) for s in SIZES}
        saved_val, th, bad_val = tune(probs[name]["val"], meta, idx["val"], labels, a.budget)
        saved, bad = evaluate(policy(probs[name]["test"], th), meta, idx["test"], labels)
        choice = policy(probs[name]["test"], th)
        mix = {int(c): f"{(choice == c).mean():.0%}" for c in (768, 1024, 1536)}
        print(f"{name}: test AUC 768 {aucs[768]:.3f} / 1024 {aucs[1024]:.3f}; thresholds "
              f"{ {k: round(v, 2) for k, v in th.items()} }; test time saved {saved:.1%} with {bad:.1%} "
              f"mis-routed (val {saved_val:.1%}, {bad_val:.1%}); routes {mix}")
        by_type = []
        for t in TYPES:
            sel = np.array([meta[i]["type"] == t for i in idx["test"]])
            if sel.sum():
                s_t, b_t = evaluate(choice[sel], meta, idx["test"][sel], labels)
                by_type.append(f"{t} {s_t:.0%} saved/{b_t:.0%} bad (n={sel.sum()})")
        print("   " + "; ".join(by_type))
    oracle = np.array([next((s for s in sorted(SIZES) if labels[s][i]), 1536) for i in idx["test"]])
    print(f"oracle: test time saved {evaluate(oracle, meta, idx['test'], labels)[0]:.1%}, 0 mis-routed")
    for flat in SIZES:
        s, b = evaluate(np.full(len(idx["test"]), flat), meta, idx["test"], labels)
        print(f"flat {flat}: test time saved {s:.1%} with {b:.1%} mis-routed")

    # CPU cost per page.
    shapes = {"cnn": (1, 1, 384, 384), "crops": (1, 4, 1, 192, 192)}
    for name, model in models.items():
        x1, s1 = torch.zeros(shapes[name]), torch.zeros(1, 4)
        for threads in (1, 4):
            torch.set_num_threads(threads)
            with torch.no_grad():
                for _ in range(3):
                    model(x1, s1)
                t0 = time.perf_counter()
                for _ in range(20):
                    model(x1, s1)
            print(f"{name} forward on CPU, {threads} thread(s): {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms "
                  f"(PyTorch, BN unfused)")
    t0 = time.perf_counter()
    for _ in range(50):
        gb[768].predict_proba(feats[:1])
        gb[1024].predict_proba(feats[:1])
    print(f"gbdt predict (both heads, sklearn call overhead included): {(time.perf_counter() - t0) / 50 * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
