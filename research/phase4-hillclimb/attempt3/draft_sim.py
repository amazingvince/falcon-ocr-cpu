#!/usr/bin/env python3
"""Offline simulation of speculative drafting on reference token sequences.

Replays each page's reference tokens (the model's own greedy output) through
a copy of `src/draft.rs` (n-gram drafter + adaptive `DraftPolicy`) and prices
every step with a cost model: a single step costs `--single-ms`, a verify
step with d drafts `single + d * --extra-ms`. Verification keeps the output
unchanged, so only time differs. With `--document`, pages of one source
document (in page order) also draft from the earlier pages' tokens.

  python research/phase4-hillclimb/attempt3/draft_sim.py --reference artifacts/phase4/checks/calibration-reference.json \
      --group notes_9e951846094758afac08c620144e3a76 --extra-ms 3.7 1.1
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

MAX_N = 4
PROBE_EVERY = 16
ALPHA = 0.15
SEPARATOR = -1
# How history drafts compete with current-page ones: "longest" (per n, page
# first), "fallback" (only without any page match), "full" (4-gram only).
HISTORY_MODE = "longest"


class Drafter:
    """`NgramDrafter` over the current page, plus an optional history index."""

    def __init__(self, min_match: int, history: list[int] | None = None):
        self.min_match = min_match
        self.index: dict[tuple, int] = {}
        self.indexed = 0
        self.history = history or []
        self.history_index: dict[tuple, int] = {}
        for end in range(1, len(self.history)):
            for n in range(1, min(MAX_N, end) + 1):
                gram = tuple(self.history[end - n:end])
                if SEPARATOR not in gram:
                    self.history_index[gram] = end

    def propose(self, tokens: list[int], limit: int) -> list[int]:
        length = len(tokens)
        while self.indexed + 1 < length:
            end = self.indexed + 1
            for n in range(1, min(MAX_N, end) + 1):
                self.index[tuple(tokens[end - n:end])] = end
            self.indexed = end
        if limit == 0:
            return []
        for n in range(min(MAX_N, length), self.min_match - 1, -1):
            gram = tuple(tokens[length - n:length])
            start = self.index.get(gram)
            if start is not None:
                period = length - start
                return [tokens[start + j % period] for j in range(limit)]
            if HISTORY_MODE == "fallback":
                continue
            if HISTORY_MODE == "full" and n < MAX_N:
                continue
            start = self.history_index.get(gram)
            if start is not None:
                out = []
                for j in range(limit):
                    if start + j >= len(self.history) or self.history[start + j] == SEPARATOR:
                        break
                    out.append(self.history[start + j])
                if out:
                    return out
        if HISTORY_MODE == "fallback":
            for n in range(min(MAX_N, length), self.min_match - 1, -1):
                start = self.history_index.get(tuple(tokens[length - n:length]))
                if start is not None:
                    out = []
                    for j in range(limit):
                        if start + j >= len(self.history) or self.history[start + j] == SEPARATOR:
                            break
                        out.append(self.history[start + j])
                    if out:
                        return out
        return []


class Policy:
    def __init__(self):
        self.single_ms = 0.0
        self.extra_ms = 0.0
        self.rate = 0.6
        self.since_probe = 0

    @staticmethod
    def average(old, new):
        return new if old == 0.0 else old + ALPHA * (new - old)

    def should_draft(self):
        pays = self.single_ms == 0.0 or self.extra_ms == 0.0 or self.rate > self.extra_ms / self.single_ms
        if pays or self.since_probe >= PROBE_EVERY:
            self.since_probe = 0
            return True
        self.since_probe += 1
        return False

    def single_step(self, ms):
        self.single_ms = self.average(self.single_ms, ms)

    def verify_step(self, ms, drafted, accepted):
        if drafted == 0:
            return
        if self.single_ms > 0.0:
            self.extra_ms = self.average(self.extra_ms, max((ms - self.single_ms) / drafted, 0.0))
        self.rate += ALPHA * (accepted / drafted - self.rate)


def simulate(reference: list[int], single_ms: float, extra_ms: float, max_draft: int, min_match: int,
             history: list[int] | None) -> dict:
    drafter = Drafter(min_match, history)
    policy = Policy()
    generated = [reference[0]]
    ms = 0.0
    steps = verifies = drafted = accepted = history_drafts = 0
    while len(generated) < len(reference):
        limit = min(max_draft, len(reference) - len(generated) - 1)
        draft = drafter.propose(generated, limit) if policy.should_draft() else []
        if draft:
            present = len(generated)
            ok = 0
            for token in draft:
                if present + ok < len(reference) and reference[present + ok] == token:
                    ok += 1
                else:
                    break
            cost = single_ms + len(draft) * extra_ms
            policy.verify_step(cost, len(draft), ok)
            generated.extend(reference[present:present + ok + 1])
            verifies += 1
            drafted += len(draft)
            accepted += ok
        else:
            cost = single_ms
            policy.single_step(cost)
            generated.append(reference[len(generated)])
        ms += cost
        steps += 1
    return {"tokens": len(reference), "steps": steps, "ms": ms, "verifies": verifies,
            "drafted": drafted, "accepted": accepted}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/corpus/v3"))
    ap.add_argument("--group", action="append", required=True, help="source-document prefix of image_path")
    ap.add_argument("--single-ms", type=float, default=8.5)
    ap.add_argument("--extra-ms", type=float, nargs="+", default=[3.7, 1.1])
    ap.add_argument("--max-draft", type=int, default=4)
    ap.add_argument("--min-match", type=int, default=2)
    ap.add_argument("--history-mode", choices=["longest", "fallback", "full"], default="longest")
    a = ap.parse_args()
    global HISTORY_MODE
    HISTORY_MODE = a.history_mode
    ref = json.loads(a.reference.read_text(encoding="utf-8"))
    tokens = {}
    for inp, out in zip(ref["inputs"], ref["samples"][0]["outputs"]):
        tokens[Path(inp["path"]).parent.name] = out["token_ids"]
    docs = defaultdict(list)
    for f in a.corpus.glob("*/annotation.json"):
        info = json.loads(f.read_text(encoding="utf-8"))["page_info"]
        m = re.match(r"(.*)_(\d+)\.\w+$", info["image_path"])
        doc = m.group(1) if m else info["image_path"]
        if f.parent.name in tokens and any(doc.startswith(g) for g in a.group):
            docs[doc].append((info.get("page_no") or 0, f.parent.name))
    for doc, pages in docs.items():
        pages.sort()
        print(f"{doc}: {len(pages)} pages, {sum(len(tokens[p]) for _, p in pages)} tokens")
        base = sum(len(tokens[p]) for _, p in pages) * a.single_ms
        print(f"  no speculation: {base / 1e3:.2f} s")
        for extra in a.extra_ms:
            for document in (False, True):
                history: list[int] = []
                total = {"ms": 0.0, "drafted": 0, "accepted": 0, "verifies": 0, "steps": 0}
                for _, page in pages:
                    r = simulate(tokens[page], a.single_ms, extra, a.max_draft, a.min_match,
                                 history if document else None)
                    for k in total:
                        total[k] += r[k]
                    history = history + [SEPARATOR] + tokens[page]
                rate = total["accepted"] / max(total["drafted"], 1)
                print(f"  extra {extra:.1f} ms/draft, {'document' if document else 'page    '}: "
                      f"{total['ms'] / 1e3:.2f} s ({100 * (total['ms'] / base - 1):+.1f}%), "
                      f"{total['verifies']} verifies, acceptance {100 * rate:.0f}%")


if __name__ == "__main__":
    main()
