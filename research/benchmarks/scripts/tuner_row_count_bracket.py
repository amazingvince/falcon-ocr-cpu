#!/usr/bin/env python3
"""Control/candidate/control bracket for the decode-tuner row-count fix.

The official promotion_bracket.py drives ocr_bench against the FP32 checkpoint,
whose decode runs on the whole pool rather than on the decode team, so it cannot
see this change. This applies the same protocol - fresh processes, three rounds
in the frozen order control-before/candidate/control-after, medians, a control
drift limit and exact-output checks - to the path that changed: the quantized
near-exact profile under continuous batching, where the tuner's samples carry a
row count that moves with the schedule.
"""
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\amazi\Documents\ChatGPT\falcon-ocr")
ARMS = {"control": REPO / "artifacts/builds/pr10-control-fb8176d/falcon-ocr.exe",
        "candidate": REPO / "artifacts/builds/pr10-candidate-335aece/falcon-ocr.exe"}
PAGES = [str(REPO / f"artifacts/corpus/smoke/{h}/canonical-rgb.png") for h in
         ("02efc5e800fa8102", "15ef968b6700f557", "23aab39cd5bf7391", "417692fb6e2a625b")]
LANES = [
    {"name": "batch4", "batch": 4, "pages": PAGES,
     "why": "Rows fall from 4 to 1 as pages finish: the row count moves with the rotation."},
    {"name": "batch1", "batch": 1, "pages": PAGES[:1],
     "why": "One row per step: the candidate compares the control's samples. A guard."},
]
ROUNDS, TOKENS, THREADS, DRIFT_LIMIT, REGRESSION_LIMIT = 3, 192, 16, 0.05, 0.05
TEAM = re.compile(r"decode threads: (\d+) \(auto; median ms/step ([^)]*)\)")
DONE = re.compile(r"pages: .* in ([\d.]+) s \(([\d.]+) pages/hour\)")

def run(binary, lane, out):
    cmd = [str(binary), "--model", str(REPO / "artifacts/packed"), "--mode", "near-exact",
           "--threads", str(THREADS), "run", *lane["pages"], "--batch-size", str(lane["batch"]),
           "--max-new-tokens", str(TOKENS), "--output", str(out)]
    if out.exists():
        out.unlink()
    started = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=REPO)
    wall = time.perf_counter() - started
    assert p.returncode == 0, p.stderr[-2000:]
    team = TEAM.search(p.stderr)
    done = DONE.search(p.stderr)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return {"wall_s": wall, "run_s": float(done.group(1)), "pages_per_hour": float(done.group(2)),
            "decode_threads": int(team.group(1)) if team else None,
            "step_medians": team.group(2) if team else None,
            "signature": [[r["token_ids"], r["text"], r["finish_reason"]] for r in records]}

def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": "tuner-row-count-bracket-v1", "rounds": ROUNDS, "tokens": TOKENS,
              "threads": THREADS, "arms": {k: str(v) for k, v in ARMS.items()}, "lanes": []}
    for lane in LANES:
        samples = {"control-before": [], "candidate": [], "control-after": []}
        signature = None
        identical = True
        for r in range(ROUNDS):
            for slot in ("control-before", "candidate", "control-after"):
                binary = ARMS["candidate" if slot == "candidate" else "control"]
                got = run(binary, lane, out_dir / f"{lane['name']}-{slot}-{r}.jsonl")
                if signature is None:
                    signature = got["signature"]
                identical &= got["signature"] == signature
                samples[slot].append(got)
                print(f"  {lane['name']} round {r+1} {slot:14} "
                      f"run {got['run_s']:6.2f}s  {got['pages_per_hour']:8.1f} pages/h  "
                      f"team {got['decode_threads']}  [{got['step_medians']}]", flush=True)
        med = {k: statistics.median(s["run_s"] for s in v) for k, v in samples.items()}
        drift = abs(med["control-before"] - med["control-after"]) / min(med["control-before"], med["control-after"])
        deltas = {k: (med[k] - med["candidate"]) / med[k] for k in ("control-before", "control-after")}
        teams = {k: sorted({s["decode_threads"] for s in v}) for k, v in samples.items()}
        lane_report = {"lane": lane["name"], "why": lane["why"], "median_run_s": med,
                       "control_drift": drift, "drift_within_limit": drift <= DRIFT_LIMIT,
                       "candidate_gain_vs": deltas,
                       "regression_within_limit": all(d >= -REGRESSION_LIMIT for d in deltas.values()),
                       "decode_threads_chosen": teams,
                       "same_team_as_control": teams["candidate"] == teams["control-before"] == teams["control-after"],
                       "outputs_identical": identical, "samples": samples}
        report["lanes"].append(lane_report)
        print(f"  -> {lane['name']}: medians {med}, drift {drift:.3%}, "
              f"candidate vs controls {[f'{d:+.2%}' for d in deltas.values()]}, "
              f"teams {teams['candidate']} vs {teams['control-before']}, identical={identical}\n", flush=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_dir / "report.json")

main()
