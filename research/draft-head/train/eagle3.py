#!/usr/bin/env python3
"""EAGLE-3-style draft head for Falcon-OCR, trained on the model's own outputs.

Target: the pinned Falcon-OCR v1.5 (artifacts/model), BF16 on CUDA, frozen.
Per page one teacher-forced forward over prompt + generated tokens gives the
hidden states after layers 2, 11 and 19 at every position (computed online,
never stored). The draft block:

  step 1 input  (fc([h2, h11, h19]) at text position i, emb(y_i))  -> predicts y_{i+1}
  step j input  (draft hidden of step j-1 at the same base i, emb(y_{i+j-1})) -> y_{i+j}

  a = [rmsnorm(emb), rmsnorm(hidden)]                        (2 x 768)
  h = hidden + SelfAttn(a)      causal over text (RoPE), EAGLE-3 training-time-test chain
  h = h + CrossAttn(h, image)   over the page's fused image features (no RoPE; --no-image drops it)
  h = h + MLP(h)                squared-ReLU gate, as in Falcon
  logits = head(rmsnorm(h))     over the draft vocabulary (most frequent output tokens)

Self-attention mask (training-time test): the query of step j at base i sees
the step-1 keys of bases <= i and its own chain keys (steps 2..j at base i).

  python research/draft-head/train/eagle3.py train --outputs /mnt/d/falcon-draft/outputs/stage1 \
      --manifest /mnt/d/falcon-draft/pages/stage1/manifest.jsonl --run /mnt/d/falcon-draft/runs/s1-image
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "phase4-hillclimb" / "attempt3"))
from gpu_harness import Harness  # noqa: E402

LAYERS = (2, 11, 19)
DIM, HEADS, KV_HEADS, HEAD_DIM, FFN = 768, 16, 8, 64, 2304


def wsl_path(p: str) -> str:
    if len(p) > 2 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Rotary embedding on (..., seq, heads, 64) with positions (seq,)."""
    half = x.shape[-1] // 2
    freqs = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = pos.float()[:, None] * freqs[None]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class DraftHead(nn.Module):
    def __init__(self, vocab_ids: torch.Tensor, embedding: torch.Tensor, head: torch.Tensor, image: bool,
                 final_norm: torch.Tensor | None = None):
        super().__init__()
        self.image = image
        self.register_buffer("vocab_ids", vocab_ids, persistent=True)
        self.embedding = nn.Embedding.from_pretrained(embedding.float(), freeze=True)
        self.fc = nn.Linear(3 * DIM, DIM, bias=False)
        self.qkv = nn.Linear(2 * DIM, (HEADS + 2 * KV_HEADS) * HEAD_DIM, bias=False)
        self.o = nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        if image:
            self.q2 = nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
            self.kv2 = nn.Linear(DIM, 2 * KV_HEADS * HEAD_DIM, bias=False)
            self.o2 = nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.w13 = nn.Linear(DIM, 2 * FFN, bias=False)
        self.w2 = nn.Linear(FFN, DIM, bias=False)
        self.norm = nn.RMSNorm(DIM, eps=1e-5)
        self.head = nn.Linear(DIM, len(vocab_ids), bias=False)
        with torch.no_grad():
            self.head.weight.copy_(head[vocab_ids].float())
            if final_norm is not None:
                self.norm.weight.copy_(final_norm.float())
            for m in (self.o, self.w2) + ((self.o2,) if image else ()):
                m.weight.mul_(0.1)

    @staticmethod
    def _norm(x):
        return F.rms_norm(x, (x.shape[-1],))

    @torch.no_grad()
    def factorize_head(self, rank: int):
        """Replace the vocabulary head by `up(down(x))` of rank `rank`,
        initialised from its truncated SVD (sqrt(S) on each side); state keys
        become head.0.weight (rank x DIM) and head.1.weight (V x rank)."""
        w = self.head.weight.double()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        root = s[:rank].sqrt()
        down = nn.Linear(DIM, rank, bias=False).to(w.device)
        up = nn.Linear(rank, w.shape[0], bias=False).to(w.device)
        down.weight.copy_(root[:, None] * vh[:rank])
        up.weight.copy_(u[:, :rank] * root[None])
        self.head = nn.Sequential(down, up)

    def fuse(self, feats):
        """fc over the three layers' features, each RMS-normalised first: the
        residual stream carries a few massive-activation channels that would
        otherwise dominate the fused input."""
        parts = feats.view(*feats.shape[:-1], 3, DIM)
        return self.fc(self._norm(parts).flatten(-2))

    @torch.no_grad()
    def init_from_layer(self, layer):
        """Start the draft block from a target block: its QKV on the hidden
        half of the input, its output projection and its MLP (the embedding
        half of QKV keeps a small random init)."""
        self.qkv.weight[:, DIM:].copy_(layer.attention.wqkv.weight.float())
        self.qkv.weight[:, :DIM].mul_(0.1)
        self.o.weight.copy_(layer.attention.wo.weight.float())
        self.w13.weight.copy_(layer.feed_forward.w13.weight.float())
        self.w2.weight.copy_(layer.feed_forward.w2.weight.float())

    def image_kv(self, feats: torch.Tensor):
        """Keys/values of the page's image positions: feats (N, 3*DIM)."""
        kv = self.kv2(self._norm(self.fuse(feats))).view(-1, 2, KV_HEADS, HEAD_DIM)
        k, v = kv[:, 0], kv[:, 1]
        return self._norm(k), v

    def step(self, hidden, tokens, pos, keys, values, mask, image_kv):
        """One draft step for W queries. `keys`/`values`: previously gathered
        self-attention keys (K, KV_HEADS, 64) or None; the step's own keys are
        appended and returned for later steps."""
        a = torch.cat([self._norm(self.embedding(tokens)), self._norm(hidden)], -1)
        qkv = self.qkv(a).view(a.shape[0], HEADS + 2 * KV_HEADS, HEAD_DIM)
        q, k, v = qkv[:, :HEADS], qkv[:, HEADS:HEADS + KV_HEADS], qkv[:, HEADS + KV_HEADS:]
        q, k = rope(self._norm(q), pos), rope(self._norm(k), pos)
        return q, k, v, a

    def attend(self, q, k_all, v_all, mask):
        # q (W, H, D), k/v (K, KVH, D), mask (W, K) bool
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k_all.transpose(0, 1)[None], v_all.transpose(0, 1)[None],
            attn_mask=mask[None, None], enable_gqa=True)
        return out[0].transpose(0, 1).reshape(q.shape[0], HEADS * HEAD_DIM)

    def finish(self, hidden, attn_out, image_kv):
        h = hidden + self.o(attn_out)
        if self.image and image_kv is not None:
            q2 = self._norm(self.q2(self._norm(h)).view(-1, HEADS, HEAD_DIM))
            k2, v2 = image_kv
            out = F.scaled_dot_product_attention(q2.transpose(0, 1)[None], k2.transpose(0, 1)[None],
                                                 v2.transpose(0, 1)[None], enable_gqa=True)
            h = h + self.o2(out[0].transpose(0, 1).reshape(-1, HEADS * HEAD_DIM))
        g = self.w13(self._norm(h)).view(-1, FFN, 2)
        h = h + self.w2(F.relu(g[..., 0]).square() * g[..., 1])
        return h, self.head(self.norm(h))


def unroll(model: DraftHead, text_feats, tokens, image_feats, start, width, steps):
    """Training-time test over bases [start, start+width). text_feats (n, 3*DIM)
    for bases 0..n-1, tokens (n + steps,) with y_i at index i (padded with -1).
    Returns logits per step (width, V) and the target token ids per step."""
    device = text_feats.device
    end = start + width
    image_kv = model.image_kv(image_feats) if model.image else None
    # Step 1 keys for every base up to the window end (the visible prefix).
    base_hidden = model.fuse(text_feats[:end])
    base_tokens = tokens[:end].clamp(min=0)
    pos = torch.arange(end, device=device)
    _, k1, v1, _ = model.step(base_hidden, base_tokens, pos, None, None, None, None)
    window = torch.arange(start, end, device=device)
    hidden = base_hidden[start:end]
    keys, values = [k1], [v1]
    logits_out, targets = [], []
    for j in range(1, steps + 1):
        tok = tokens[window + j - 1].clamp(min=0)
        q, k, v, _ = model.step(hidden, tok, window + j - 1, None, None, None, None)
        if j > 1:
            keys.append(k)
            values.append(v)
        k_all, v_all = torch.cat(keys), torch.cat(values)
        # Visibility: step-1 keys of bases <= i; chain keys (steps 2..j) of base i.
        base_vis = torch.arange(end, device=device)[None, :] <= window[:, None]
        chains = [torch.eye(width, dtype=torch.bool, device=device)] * (j - 1)
        mask = torch.cat([base_vis] + chains, 1)
        out = model.attend(q, k_all, v_all, mask)
        hidden, logits = model.finish(hidden, out, image_kv)
        logits_out.append(logits)
        targets.append(tokens[window + j])
    return logits_out, targets


class Pages:
    """Training pages (vLLM outputs) with features from the frozen target."""

    def __init__(self, harness: Harness):
        self.h = harness
        self.captured = {}
        # Keep the target's head for the draft head's initialisation, then skip
        # it during feature extraction (only the hooked layers are needed).
        self.head_weight = harness.model.output.weight.detach().clone()
        # BF16 casting also hit the golden RoPE table; torch.polar needs FP32.
        harness.model.freqs_cis_golden = harness.model.freqs_cis_golden.float()
        harness.model.output = nn.Identity()
        harness.source = None  # the harness's FP32 weight copy (1.2 GB) is not needed here
        for layer in LAYERS:
            self.h.model.layers[str(layer)].register_forward_hook(self._hook(layer))

    def _hook(self, layer):
        def fn(_module, _inputs, output):
            self.captured[layer] = output[0]
        return fn

    @torch.inference_mode()
    def features(self, image_path: str, generated: list[int]):
        batch = self.h.prepare(image_path)
        tokens, pos_t, pos_hw = batch["tokens"], batch["pos_t"], batch["pos_hw"]
        n = len(generated)
        extra = torch.tensor([generated[:-1]], dtype=tokens.dtype)
        last = int(pos_t[0, -1])
        tokens_all = torch.cat([tokens, extra], 1).to(self.h.device)
        pos_t_all = torch.cat([pos_t, torch.arange(last + 1, last + n, dtype=pos_t.dtype)[None]], 1).to(self.h.device)
        pos_hw_all = torch.cat([pos_hw, torch.full((1, n - 1, pos_hw.shape[2]), float("nan"), dtype=pos_hw.dtype)],
                               1).to(self.h.device)
        self.h._forward_prefix(batch, tokens_all, pos_t_all, pos_hw_all)
        feats = torch.cat([self.captured[layer] for layer in LAYERS], -1)  # (S, 3*DIM)
        prompt = tokens.shape[1]
        text = feats[prompt - 1:prompt - 1 + n].float().clone()
        image_mask = (tokens[0] == self.h.config.img_id).to(self.h.device)
        image = feats[:prompt][image_mask].float().clone()
        self.h._batches.clear()
        self.captured.clear()
        return text, image


def load_records(outputs: pathlib.Path, manifest: pathlib.Path):
    pages = {json.loads(l)["id"]: json.loads(l) for l in manifest.read_text().splitlines() if l.strip()}
    records = []
    for f in sorted(outputs.glob("*.json")):
        r = json.loads(f.read_text())
        page = pages.get(r["id"])
        if page is None or len(r["token_ids"]) < 16:
            continue
        r["path"] = wsl_path(page["path"])
        records.append(r)
    return records


EXCLUDED = set()
_decontam = pathlib.Path("/mnt/d/falcon-draft/decontam-13gram.json")
if _decontam.exists():
    EXCLUDED = set(json.loads(_decontam.read_text()))


def looping(y: list[int]) -> bool:
    """The last tokens repeat one block of at most 64 tokens three times."""
    tail = y[-192:]
    return any(len(tail) >= 3 * p and tail[-p:] == tail[-2 * p:-p] == tail[-3 * p:-2 * p] for p in range(1, 65))


def keep(r: dict) -> bool:
    """English pages without a repetition loop (loops are the n-gram
    drafter's job); length-capped dense pages (newspapers) are kept."""
    if r["id"] in EXCLUDED or (r["finish_reason"] != "stop" and looping(r["token_ids"])):
        return False
    words = [w.lower() for w in r["text"].split() if w.isalpha()]
    common = {"the", "and", "of", "to", "in", "a", "is", "for", "that", "on", "with", "as", "by", "be", "this", "are", "or", "it", "from", "at"}
    return len(words) < 30 or sum(w in common for w in words) / len(words) > 0.04


def draft_vocab(records, size: int, specials=(11, 263)):
    counts = collections.Counter(t for r in records for t in r["token_ids"])
    ids = [t for t, _ in counts.most_common(size - len(specials)) if t not in specials]
    ids = list(specials) + ids
    covered = sum(counts[t] for t in ids) / max(sum(counts.values()), 1)
    return torch.tensor(sorted(ids)), covered


def evaluate(model, pages, records, t2d, steps, device, limit=None):
    """Per-step top-1 accuracy and mean accepted chain length (leading correct steps)."""
    model.eval()
    correct = torch.zeros(steps)
    counted = torch.zeros(steps)
    accepted = []
    confs, hits = [], []  # per position: (steps,) draft confidence and correctness
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r in records[:limit]:
            y = r["token_ids"]
            with torch.autocast("cuda", enabled=False):
                text, image = pages.features(r["path"], y)
            n = len(y)
            tokens = torch.tensor(y + [-1] * (steps + 1), device=device)
            for start in range(0, n - 1, 1024):
                width = min(1024, n - 1 - start)
                logits, targets = unroll(model, text, tokens, image, start, width, steps)
                ok_prev = torch.ones(width, dtype=torch.bool, device=device)
                run = torch.zeros(width, device=device)
                step_conf, step_hit = [], []
                for j, (lg, tg) in enumerate(zip(logits, targets)):
                    valid = tg >= 0
                    pred = model.vocab_ids[lg.argmax(-1)]
                    hit = (pred == tg) & valid
                    step_conf.append(lg.float().softmax(-1).amax(-1))
                    step_hit.append(hit)
                    correct[j] += hit.sum().item()
                    counted[j] += valid.sum().item()
                    ok_prev = ok_prev & hit
                    run += ok_prev.float()
                accepted.append(run.cpu())
                confs.append(torch.stack(step_conf, 1).cpu())
                hits.append(torch.stack(step_hit, 1).cpu())
    model.train()
    acc = (correct / counted.clamp(min=1)).tolist()
    if not accepted:
        return acc, 0.0
    runs = torch.cat(accepted)
    evaluate.reach = [round((runs >= j).float().mean().item(), 4) for j in range(1, steps + 1)]
    evaluate.gated = gated_speedups(torch.cat(confs), torch.cat(hits))
    return acc, runs.mean().item()


def gated_speedups(conf, hit, single=9.0, rows=(1.7, 1.0), draft_ms=0.6):
    """EAGLE-2-style gating on CPU: keep drafting while the head's confidence
    is at least tau (at most `steps` drafts); a step costs single + drafted *
    (row + draft) and yields 1 + the leading accepted drafts. Best tau per row cost."""
    out = {}
    for row in rows:
        best = (1.0, None, 0.0, 0.0)
        for tau in [x / 20 for x in range(4, 20)]:
            confident = conf >= tau
            drafted = torch.cumprod(confident.int(), 1).sum(1)
            leading = torch.cumprod((confident & hit).int(), 1).sum(1)
            speedup = (single * (1 + leading.float())).mean() / (single + drafted.float() * (row + draft_ms)).mean()
            if speedup > best[0]:
                best = (speedup.item(), tau, drafted.float().mean().item(), leading.float().mean().item())
        out[f"row_{row}"] = {"speedup": round(best[0], 3), "tau": best[1],
                             "drafted_per_step": round(best[2], 3), "accepted_per_step": round(best[3], 3)}
    return out


def cmd_train(a):
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda"
    harness = Harness(ROOT / "artifacts" / "model", "bf16", device)
    pages = Pages(harness)
    records = [{"id": r["id"], "path": r["path"], "token_ids": r["token_ids"], "source": r["source"]}
               for r in load_records(a.outputs, a.manifest) if keep(r)]
    random.shuffle(records)
    val, train = records[:a.val], records[a.val:]
    start_from = None
    if a.init_from:
        start_from = torch.load(a.init_from, map_location="cpu", weights_only=False)
        vocab_ids = start_from["state"]["vocab_ids"].cpu()
        counts = collections.Counter(t for r in train for t in r["token_ids"])
        covered = sum(counts[int(t)] for t in vocab_ids) / max(sum(counts.values()), 1)
    else:
        vocab_ids, covered = draft_vocab(train, a.vocab)
    t2d = torch.full((harness.config.vocab_size,), -1, dtype=torch.long)
    t2d[vocab_ids] = torch.arange(len(vocab_ids))
    t2d = t2d.to(device)
    print(f"{len(train)} train / {len(val)} val pages; draft vocab {len(vocab_ids)} covers {100 * covered:.2f}% of tokens",
          flush=True)
    params = harness.model
    model = DraftHead(vocab_ids.to(device), params.tok_embeddings.weight.detach(), pages.head_weight,
                      image=not a.no_image, final_norm=params.norm.weight.detach()).to(device)
    if start_from is not None:
        if "head.0.weight" in start_from["state"]:
            model.factorize_head(start_from["state"]["head.0.weight"].shape[0])
        model.load_state_dict(start_from["state"])
        print(f"draft head initialised from {a.init_from}", flush=True)
    elif a.init_layer is not None:
        model.init_from_layer(params.layers[str(a.init_layer)])
        print(f"draft block initialised from target layer {a.init_layer}", flush=True)
    if a.head_rank and not isinstance(model.head, nn.Sequential):
        model.factorize_head(a.head_rank)
        print(f"vocabulary head factorised to rank {a.head_rank}", flush=True)
    if a.head_only:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
    if a.limit_pages:
        train = train[:a.limit_pages]
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"draft parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M trainable", flush=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total = a.epochs * len(train) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0))))
    a.run.mkdir(parents=True, exist_ok=True)
    log = (a.run / "log.jsonl").open("a")
    step = 0
    started = time.time()
    weights = torch.tensor([a.step_decay ** j for j in range(a.steps)], device=device)
    for epoch in range(a.epochs):
        random.shuffle(train)
        for idx, r in enumerate(train):
            y = r["token_ids"]
            n = len(y)
            text, image = pages.features(r["path"], y)
            tokens = torch.tensor(y + [-1] * (a.steps + 1), device=device)
            width = min(a.window, n - 1)
            text, image = text.clone(), image.clone()
            # Several windows per target forward (the expensive part), each
            # with its own backward so attention memory stays bounded.
            for _ in range(a.windows_per_page if n - 1 > width else 1):
                start = random.randint(0, n - 1 - width)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, targets = unroll(model, text, tokens, image, start, width, a.steps)
                    loss = 0.0
                    for j, (lg, tg) in enumerate(zip(logits, targets)):
                        d = t2d[tg.clamp(min=0)]
                        d = torch.where(tg >= 0, d, torch.full_like(d, -1))
                        loss = loss + weights[j] * F.cross_entropy(lg.float(), d, ignore_index=-1)
                    loss = loss / weights.sum()
                (loss / a.accum).backward()
            if (idx + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 50 == 0:
                    rate = (idx + 1 + epoch * len(train)) / (time.time() - started)
                    print(f"epoch {epoch} step {step}/{total} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{rate:.2f} pages/s", flush=True)
                    log.write(json.dumps({"epoch": epoch, "step": step, "loss": loss.item()}) + "\n")
                    log.flush()
        acc, mean_accept = evaluate(model, pages, val, t2d, a.steps, device, limit=a.val)
        print(f"epoch {epoch} val per-step top-1 {[round(x, 3) for x in acc]} mean accepted {mean_accept:.2f} "
              f"P(accepted>=j) {getattr(evaluate, 'reach', None)}", flush=True)
        log.write(json.dumps({"epoch": epoch, "val_acc": acc, "val_accept": mean_accept}) + "\n")
        log.flush()
        args = {k: v for k, v in vars(a).items() if k != "func"}
        torch.save({"state": model.state_dict(), "args": args | {"run": str(a.run), "outputs": str(a.outputs),
                                                                  "manifest": str(a.manifest)}},
                   a.run / "draft.pt")


def cmd_eval(a):
    """Acceptance on reference token sequences (e.g. the FP32 calibration outputs)."""
    device = "cuda"
    harness = Harness(ROOT / "artifacts" / "model", "bf16", device)
    pages = Pages(harness)
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)  # our own checkpoint
    saved = ckpt["args"]
    params = harness.model
    vocab_ids = ckpt["state"]["vocab_ids"]
    model = DraftHead(vocab_ids.to(device), params.tok_embeddings.weight.detach(), pages.head_weight,
                      image=not saved["no_image"], final_norm=params.norm.weight.detach()).to(device)
    if "head.0.weight" in ckpt["state"]:
        model.factorize_head(ckpt["state"]["head.0.weight"].shape[0])
    model.load_state_dict(ckpt["state"])
    t2d = torch.full((harness.config.vocab_size,), -1, dtype=torch.long)
    t2d[vocab_ids.cpu()] = torch.arange(len(vocab_ids))
    ref = json.loads(pathlib.Path(a.reference).read_text())
    records = []
    for inp, out in zip(ref["inputs"], ref["samples"][0]["outputs"]):
        records.append({"path": wsl_path(str(inp["path"])), "token_ids": out["token_ids"],
                        "finish_reason": out["finish_reason"], "id": pathlib.Path(inp["path"]).parent.name})
    if a.english_only:
        english = set(a.english_only.read_text().split())
        records = [r for r in records if r["id"] in english]
    acc, mean_accept = evaluate(model, pages, records, t2d.to(device), saved["steps"], device)
    report = {"checkpoint": str(a.checkpoint), "pages": len(records), "per_step_top1": acc,
              "mean_accepted": mean_accept, "reach": getattr(evaluate, "reach", None),
              "confidence_gated": getattr(evaluate, "gated", None)}
    print(json.dumps(report), flush=True)
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--outputs", type=pathlib.Path, required=True)
    t.add_argument("--manifest", type=pathlib.Path, required=True)
    t.add_argument("--run", type=pathlib.Path, required=True)
    t.add_argument("--no-image", action="store_true")
    t.add_argument("--vocab", type=int, default=16384)
    t.add_argument("--steps", type=int, default=7)
    t.add_argument("--window", type=int, default=512)
    t.add_argument("--windows-per-page", type=int, default=2)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--val", type=int, default=100)
    t.add_argument("--init-layer", type=int, help="initialise the draft block from this target layer (e.g. 21)")
    t.add_argument("--init-from", type=pathlib.Path, help="start from a draft checkpoint (keeps its draft vocabulary)")
    t.add_argument("--step-decay", type=float, default=0.8, help="loss weight of draft step j is decay**j")
    t.add_argument("--head-rank", type=int, help="factorise the vocabulary head to this rank (SVD initialised)")
    t.add_argument("--head-only", action="store_true", help="train only the vocabulary head")
    t.add_argument("--limit-pages", type=int, help="use only this many training pages per epoch")
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", type=pathlib.Path, required=True)
    e.add_argument("--reference", required=True)
    e.add_argument("--english-only", type=pathlib.Path)
    e.add_argument("--report")
    e.set_defaults(func=cmd_eval)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
