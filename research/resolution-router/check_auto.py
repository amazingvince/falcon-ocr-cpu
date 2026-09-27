#!/usr/bin/env python3
"""Phase 2 check: the runner's own router (`bench --max-dimension auto`) on the
development set against the fixed-resolution runs of the same pages.

Per page: the route and whether the safety net reran it; the tokens must equal
the fixed run at the chosen resolution (at 1536 after a rerun), since a routed
page is the same input a fixed run of that size sees; CER against ground truth
and measured CPU time against 1536, as dev_policy.py computes them (its
simulated policy, from the fixed runs, is printed alongside for comparison).

  python research/resolution-router/check_auto.py --runs D:/falcon-draft/router-dev
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def outputs(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    return {Path(i["path"]).parent.name: o for i, o in zip(r["inputs"], r["samples"][0]["outputs"], strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--lock", type=Path, default=Path("reference/router-dev-v1-evaluation-lock.json"))
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text(encoding="utf-8"))["pages"]
    fixed = {s: outputs(a.runs / f"fast-{s}.json") for s in (768, 1024, 1536)}
    auto = outputs(a.runs / "fast-auto.json")
    routes, reruns, token_mismatch = Counter(), [], []
    chars = err_base = err_auto = err_sim = 0
    t_base = t_auto = t_sim = router_ms = 0.0
    for entry in lock:
        pid = Path(entry["canonical_path"]).parent.name
        out = auto[pid]
        route = out["route"]
        chosen = route["max_dimension"]
        routes[chosen] += 1
        net = route.get("safety_net")
        final = 1536 if net else chosen
        if net:
            reruns.append((pid, chosen, net["finish_reason"]))
        if out["token_ids"] != fixed[final][pid]["token_ids"]:
            token_mismatch.append((pid, chosen, final))
        truth = Path(entry["ground_truth_path"]).read_text(encoding="utf-8")
        chars += len(truth)
        err_base += Levenshtein.distance(truth, fixed[1536][pid]["text"])
        err_auto += Levenshtein.distance(truth, out["text"])
        t_base += fixed[1536][pid]["timings"]["total_ms"]
        t_auto += out["timings"]["total_ms"]
        router_ms += route["statistics_ms"]
        # The simulation from the fixed runs (dev_policy.py's safety-net accounting).
        sim_net = chosen != 1536 and fixed[chosen][pid]["finish_reason"] in ("repetition", "length")
        err_sim += Levenshtein.distance(truth, fixed[1536 if sim_net else chosen][pid]["text"])
        t_sim += fixed[chosen][pid]["timings"]["total_ms"] + (fixed[1536][pid]["timings"]["total_ms"] if sim_net else 0)
    n = len(lock)
    print(f"{n} pages; routes {dict(sorted(routes.items()))}; safety-net reruns {len(reruns)} {reruns}")
    print(f"tokens differ from the fixed run at the final resolution on {len(token_mismatch)} pages "
          f"{token_mismatch[:10]}")
    change = 100 * (err_auto - err_base) / chars
    print(f"CER 1536 {err_base / chars:.2%} -> auto {err_auto / chars:.2%} ({change:+.2f} pt); "
          f"simulated {err_sim / chars:.2%}")
    print(f"CPU time 1536 {t_base / 1000 / 60:.1f} min -> auto {t_auto / 1000 / 60:.1f} min "
          f"(saved {1 - t_auto / t_base:.1%}); simulated saved {1 - t_sim / t_base:.1%}; "
          f"router {router_ms / n:.1f} ms per page")


if __name__ == "__main__":
    main()
