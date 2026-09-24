#!/usr/bin/env python3
"""Fair head-to-head driver: our `falcon-ocr` runner versus pszemraj's `focr`.

Both implementations recognize the same pages with identical preprocessing
settings (`--min-dimension`, `--max-dimension`, no token-budget shrink) and an
identical generation budget. Every measured recognition runs in a fresh
process. Warm latency comes from each side's own warm harness
(`examples/ocr_bench.rs` for ours, `focr bench` for theirs) with the same
number of dropped warmups and measured repetitions; one cold run per side
records model load / backend preparation and the generated token IDs used for
output agreement. Everything is written as JSON evidence plus a Markdown table.

Only the standard library is used. The script never modifies either checkout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import glob
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_KIND = "focr-comparison-v1"
SCHEMA_VERSION = 1
OURS = "ours"
FOCR = "focr"
OUR_NON_PATCH_PROMPT_TOKENS = 16
LEVENSHTEIN_CELL_LIMIT = 40_000_000


# --------------------------------------------------------------------------
# Pure helpers (unit tested)
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_json_object(text: str) -> dict:
    """Decodes the first JSON object found in `text` (stdout may carry noise)."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    value, _ = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def last_json_line(text: str) -> dict:
    """Decodes the last non-empty stdout line (our `run` prints JSON lines)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("no JSON line in output")


def strip_stop_tokens(ids: list[int], stop_ids: set[int]) -> list[int]:
    out = list(ids)
    while out and out[-1] in stop_ids:
        out.pop()
    return out


def levenshtein_normalized(a, b) -> tuple[float, str]:
    """Normalized edit distance in [0, 1]; falls back to difflib on huge inputs."""
    if len(a) == 0 and len(b) == 0:
        return 0.0, "levenshtein"
    if len(a) * len(b) > LEVENSHTEIN_CELL_LIMIT:
        ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        return 1.0 - ratio, "difflib-ratio"
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / max(len(a), len(b)), "levenshtein"


def first_divergence(a: list, b: list) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def agreement(ours_ids: list[int], focr_ids: list[int], ours_text: str, focr_text: str,
              stop_ids: set[int]) -> dict:
    a = strip_stop_tokens(ours_ids, stop_ids)
    b = strip_stop_tokens(focr_ids, stop_ids)
    divergence = first_divergence(a, b)
    common = len(a) if divergence is None else divergence
    token_distance, token_method = levenshtein_normalized(a, b)
    text_distance, text_method = levenshtein_normalized(ours_text, focr_text)
    return {
        "token_ids_exact": a == b,
        "first_divergence_index": divergence,
        "common_prefix_fraction": (common / max(len(a), len(b), 1)),
        "ours_len": len(a),
        "focr_len": len(b),
        "token_levenshtein_normalized": token_distance,
        "token_similarity_method": token_method,
        "text_levenshtein_normalized": text_distance,
        "text_similarity_method": text_method,
        "text_exact": ours_text == focr_text,
        "ours_text_sha256": hashlib.sha256(ours_text.encode("utf-8")).hexdigest(),
        "focr_text_sha256": hashlib.sha256(focr_text.encode("utf-8")).hexdigest(),
    }


def clamp_max_new_tokens(requested: int, focr_prompt_tokens: int, focr_max_seq_len: int) -> int:
    """focr refuses prompt + max_new_tokens > max_seq_len; apply to both sides."""
    budget = focr_max_seq_len - focr_prompt_tokens
    if budget <= 0:
        raise ValueError(
            f"focr prompt of {focr_prompt_tokens} tokens leaves no generation budget "
            f"inside its {focr_max_seq_len}-token context")
    return min(requested, budget)


def bracket_order(processes: int) -> list[str]:
    """A B B A ... so neither side always runs first."""
    order = []
    for index in range(processes):
        pair = [OURS, FOCR] if index % 2 == 0 else [FOCR, OURS]
        order.extend(pair)
    return order


def drop_warmups(values: list, warmup: int) -> list:
    return list(values[warmup:])


def samples_from_ocr_bench(report: dict, warmup_already_dropped: bool = True) -> list[dict]:
    """Per-sample metrics from our `examples/ocr_bench.rs` schema-2 report."""
    case = report["cases"][0]
    samples = []
    for sample in case["samples"]:
        request = sample["per_request"][0]
        timings = request["timings"]
        prefill = float(timings["prefill_ms"])
        decode = float(timings["decode_ms"])
        samples.append({
            "prefill_ms": prefill,
            "decode_ms": decode,
            "model_ms": prefill + decode,
            "e2e_ms": float(sample["wall_ms"]),
            "preprocessing_ms": float(timings.get("preprocessing_ms", 0.0)),
            "output_tokens": int(request["output_tokens"]),
            "stop_reason": str(request["finish_reason"]),
            "decode_tok_s": (int(request["output_tokens"]) / decode * 1000.0) if decode > 0 else None,
        })
    return samples


def samples_from_focr_bench(report: dict, warmup: int) -> list[dict]:
    """Per-sample metrics from `focr bench --json`, dropping the first `warmup` repeats."""
    prefill = drop_warmups(report["prefill_ms"], warmup)
    decode = drop_warmups(report["decode_ms"], warmup)
    elapsed = drop_warmups(report["elapsed_ms"], warmup)
    generated = drop_warmups(report["generated_tokens"], warmup)
    total = drop_warmups(report["total_ocr_latency_ms"], warmup)
    if not prefill:
        raise ValueError("focr bench produced no measured repeats after dropping warmups")
    samples = []
    for p, d, e, g, t in zip(prefill, decode, elapsed, generated, total):
        samples.append({
            "prefill_ms": float(p),
            "decode_ms": float(d),
            "model_ms": float(p) + float(d),
            "e2e_ms": float(e),
            "total_ocr_latency_ms": float(t),
            "output_tokens": int(g),
            "stop_reason": None,
            "decode_tok_s": (int(g) / float(d) * 1000.0) if float(d) > 0 else None,
        })
    return samples


def summarize_samples(samples: list[dict]) -> dict:
    def median_of(key):
        values = [s[key] for s in samples if s.get(key) is not None]
        return statistics.median(values) if values else None

    return {
        "count": len(samples),
        "median_prefill_ms": median_of("prefill_ms"),
        "median_decode_ms": median_of("decode_ms"),
        "median_model_ms": median_of("model_ms"),
        "median_e2e_ms": median_of("e2e_ms"),
        "median_decode_tok_s": median_of("decode_tok_s"),
        "output_tokens": sorted({s["output_tokens"] for s in samples}),
    }


def control_drift(medians: list[float]) -> float | None:
    if len(medians) < 2:
        return None
    return (max(medians) - min(medians)) / min(medians)


def focr_stop_reason(ids: list[int], max_new_tokens: int, stop_ids: set[int]) -> str:
    if ids and ids[-1] in stop_ids:
        return "eos"
    if len(ids) < max_new_tokens:
        return "eos"
    return "length"


def parse_cmake_isa(cache_text: str) -> dict:
    keys = ("GGML_NATIVE", "GGML_AVX", "GGML_AVX2", "GGML_AVX512", "GGML_AVX512_VBMI",
            "GGML_AVX512_VNNI", "GGML_AVX512_BF16", "GGML_FMA", "GGML_F16C", "GGML_BMI2",
            "GGML_CPU_REPACK", "GGML_OPENMP", "GGML_LLAMAFILE", "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_COMPILER", "CMAKE_GENERATOR")
    found = {}
    for line in cache_text.splitlines():
        match = re.match(r"^([A-Z0-9_]+):[A-Z]+=(.*)$", line.strip())
        if match and match.group(1) in keys:
            found[match.group(1)] = match.group(2)
    return found


def geometry_matches(ours: dict, focr: dict) -> tuple[bool, list[str]]:
    problems = []
    if ours["width"] != focr["width"] or ours["height"] != focr["height"]:
        problems.append(f"resized dims differ: ours {ours['width']}x{ours['height']} vs focr {focr['width']}x{focr['height']}")
    if ours["patches"] != focr["patches"]:
        problems.append(f"patch counts differ: ours {ours['patches']} vs focr {focr['patches']}")
    if focr.get("resized_by_token_budget"):
        problems.append("focr shrank the image to its token budget; raise --focr-max-image-tokens")
    return (not problems), problems


def render_markdown(report: dict) -> str:
    lines = []
    settings = report["settings"]
    lines.append(f"# {SCHEMA_KIND}: {report['host']['cpu_label']}, {report['host']['environment_label']}")
    lines.append("")
    lines.append(f"Generated {report['created_utc']}. max-dimension {settings['max_dimension']}, "
                 f"min-dimension {settings['min_dimension']}, requested max-new-tokens "
                 f"{settings['max_new_tokens_requested']}, warmup {settings['warmup']}, "
                 f"repetitions {settings['repetitions']}, {settings['processes']} fresh processes per side "
                 f"in {settings['order']} order.")
    lines.append("")
    ours = report["implementations"][OURS]
    focr = report["implementations"][FOCR]
    lines.append(f"- ours: `{ours.get('binary_sha256', '?')[:16]}` commit `{str(ours.get('git_commit'))[:12]}`, "
                 f"backend {ours.get('backend')}, cache {ours.get('cache_layout')}, weights {ours.get('weight_layout')}, "
                 f"threads {ours.get('threads')}, lane {ours.get('precision_lane')}")
    lines.append(f"- focr: `{focr.get('binary_sha256', '?')[:16]}` commit `{str(focr.get('git_commit'))[:12]}` "
                 f"ggml `{str(focr.get('ggml_submodule_commit'))[:12]}`, build lane **{focr.get('build_lane')}**, "
                 f"backend {focr.get('backend')}, weight mode {focr.get('weight_mode')}, threads {focr.get('threads')}, "
                 f"lane {focr.get('precision_lane')}, ISA {focr.get('cmake_isa', {})}")
    lines.append("")
    lines.append("## Warm latency (medians of measured repetitions; best process per side)")
    lines.append("")
    lines.append("| page | dims / patches | impl | prefill s | decode s | model s | e2e s | decode tok/s | out tokens | drift % | focr/ours model |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for page in report["pages"]:
        geometry = page.get("geometry", {})
        dims = "?"
        side = geometry.get(OURS) or geometry.get(FOCR)
        if side:
            dims = f"{side['width']}x{side['height']} / {side['patches']}"
        warm = page.get("warm")
        if not warm:
            lines.append(f"| {page['id']} | {dims} | (no warm runs: {page.get('error', 'skipped')}) | | | | | | | | |")
            continue
        summary = warm["summary"]
        ratio = summary.get("ratio_focr_over_ours", {}).get("model")
        for impl in (OURS, FOCR):
            s = summary.get(impl)
            if not s:
                continue
            best = s["best"]

            def fmt(v, scale=1000.0, digits=3):
                return "-" if v is None else f"{v / scale:.{digits}f}"

            drift = s.get("control_drift_pct")
            tok_s = "-" if best["median_decode_tok_s"] is None else f"{best['median_decode_tok_s']:.2f}"
            drift_text = "-" if drift is None else f"{drift:.2f}"
            ratio_text = "-" if (impl != FOCR or ratio is None) else f"{ratio:.3f}"
            lines.append(
                f"| {page['id']} | {dims} | {impl} | {fmt(best['median_prefill_ms'])} | {fmt(best['median_decode_ms'])} | "
                f"{fmt(best['median_model_ms'])} | {fmt(best['median_e2e_ms'])} | {tok_s} | "
                f"{'/'.join(str(t) for t in best['output_tokens'])} | {drift_text} | {ratio_text} |")
    lines.append("")
    lines.append("## Output agreement (cold runs, stop tokens stripped)")
    lines.append("")
    lines.append("| page | exact IDs | first divergence | common prefix | ours len | focr len | token dist | text dist |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for page in report["pages"]:
        a = page.get("agreement")
        if not a:
            lines.append(f"| {page['id']} | (no cold runs) | | | | | | |")
            continue
        lines.append(f"| {page['id']} | {a['token_ids_exact']} | {a['first_divergence_index']} | "
                     f"{a['common_prefix_fraction']:.4f} | {a['ours_len']} | {a['focr_len']} | "
                     f"{a['token_levenshtein_normalized']:.4f} | {a['text_levenshtein_normalized']:.4f} ({a['text_similarity_method']}) |")
    lines.append("")
    lines.append("## Cold start (one fresh process per side, includes model load)")
    lines.append("")
    lines.append("| page | impl | process wall s | load / prepare ms | prefill s | decode s | out tokens | stop |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for page in report["pages"]:
        cold = page.get("cold") or {}
        for impl in (OURS, FOCR):
            c = cold.get(impl)
            if not c:
                continue
            load = c.get("model_load_ms") if impl == OURS else c.get("backend_prepare_ms")
            lines.append(f"| {page['id']} | {impl} | {c['process_wall_ms'] / 1000:.2f} | "
                         f"{'-' if load is None else f'{load:.0f}'} | {c['prefill_ms'] / 1000:.3f} | "
                         f"{c['decode_ms'] / 1000:.3f} | {c['output_tokens']} | {c['stop_reason']} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append("- `model s` = prefill + decode on both sides; it excludes file decode and image preprocessing.")
    lines.append("- `e2e s`: ours is the warm RGB-buffer wall time (no PNG decode); focr's `elapsed_ms` includes PNG decode and preprocessing.")
    lines.append("- focr's context is pinned to 8192 tokens, so max-new-tokens is clamped to `8192 - prompt` per page on both sides when needed.")
    lines.append("- focr runs with `--max-image-tokens` raised so its default pixel budget never shrinks a page below ours.")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    if report.get("thread_sweep"):
        lines.append("")
        lines.append("## Thread sweep")
        lines.append("")
        lines.append("| impl | threads | median model s | median prefill s | median decode s |")
        lines.append("|---|---:|---:|---:|---:|")
        for impl in (OURS, FOCR):
            for row in report["thread_sweep"].get(impl, []):
                lines.append(f"| {impl} | {row['threads']} | {row['median_model_ms'] / 1000:.3f} | "
                             f"{row['median_prefill_ms'] / 1000:.3f} | {row['median_decode_ms'] / 1000:.3f} |")
        selected = report["thread_sweep"].get("selected", {})
        lines.append("")
        lines.append(f"Selected: ours {selected.get(OURS)}, focr {selected.get(FOCR)}.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Process execution
# --------------------------------------------------------------------------


class Runner:
    def __init__(self, args, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def run(self, label: str, command: list[str], timeout: float) -> tuple[str, str, float, int]:
        self.counter += 1
        stem = f"{self.counter:03d}-{label}"
        printable = subprocess.list2cmdline(command)
        (self.run_dir / f"{stem}.command.txt").write_text(printable + "\n", encoding="utf-8")
        if self.args.dry_run:
            print(f"[dry-run] {printable}")
            return "{}", "", 0.0, 0
        print(f"[{stem}] {printable}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout)
        wall_ms = (time.perf_counter() - start) * 1000.0
        (self.run_dir / f"{stem}.stdout.json").write_text(completed.stdout, encoding="utf-8")
        (self.run_dir / f"{stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"{stem} exited with {completed.returncode}; see {self.run_dir / (stem + '.stderr.log')}\n"
                               f"{completed.stderr[-2000:]}")
        return completed.stdout, completed.stderr, wall_ms, completed.returncode


def ours_common_flags(args) -> list[str]:
    return ["--model", str(args.ours_model), "--mode", "exact", "--threads", str(args.ours_threads),
            "--backend", args.ours_backend, "--cache-layout", args.ours_cache_layout,
            "--weight-layout", args.ours_weight_layout]


def ours_cold_command(args, page: Path, max_new: int) -> list[str]:
    return [str(args.ours_run), *ours_common_flags(args), "run", str(page),
            "--max-new-tokens", str(max_new), "--min-dimension", str(args.min_dimension),
            "--max-dimension", str(args.max_dimension)]


def ours_warm_command(args, page: Path, max_new: int, output: Path, warmup: int, repetitions: int,
                      threads: int | None = None) -> list[str]:
    flags = ours_common_flags(args)
    if threads is not None:
        flags[flags.index("--threads") + 1] = str(threads)
    return [str(args.ours_bench), *flags, "--execution", "sequential", "--batches", "1",
            "--warmup", str(warmup), "--repetitions", str(repetitions),
            "--max-new-tokens", str(max_new), "--min-dimension", str(args.min_dimension),
            "--max-dimension", str(args.max_dimension), "--cpu-label", args.cpu_label,
            "--environment-label", args.environment_label, "--output", str(output), str(page)]


def focr_common_flags(args, threads: int | None = None) -> list[str]:
    return ["--model", str(args.focr_model), "--category", "plain", "--prompt-mode", args.focr_prompt_mode,
            "--max-length", str(args.focr_max_length), "--min-dimension", str(args.min_dimension),
            "--max-dimension", str(args.max_dimension), "--max-image-tokens", str(args.focr_max_image_tokens)]


def focr_model_flags(args) -> list[str]:
    return ["--backend", args.focr_backend, "--weight-mode", args.focr_weight_mode,
            "--activation-dtype", "f32"]


def focr_preprocess_command(args, page: Path) -> list[str]:
    return [str(args.focr), "preprocess-dump", *focr_common_flags(args), "--image", str(page)]


def focr_cold_command(args, page: Path, max_new: int) -> list[str]:
    return [str(args.focr), "run", *focr_common_flags(args), *focr_model_flags(args), "--image", str(page),
            "--max-new-tokens", str(max_new), "--threads", str(args.focr_threads), "--json"]


def focr_warm_command(args, page: Path, max_new: int, repeats: int, threads: int | None = None) -> list[str]:
    return [str(args.focr), "bench", *focr_common_flags(args), *focr_model_flags(args), "--image", str(page),
            "--max-new-tokens", str(max_new), "--threads", str(threads if threads is not None else args.focr_threads),
            "--repeats", str(repeats), "--json"]


def parse_ours_cold(stdout: str, stderr: str, wall_ms: float) -> dict:
    result = last_json_line(stdout)
    load = re.search(r"model loaded in ([0-9.]+) ms", stderr)
    timings = result["timings"]
    return {
        "process_wall_ms": wall_ms,
        "model_load_ms": float(load.group(1)) if load else None,
        "prefill_ms": float(timings["prefill_ms"]),
        "decode_ms": float(timings["decode_ms"]),
        "total_ms": float(timings["total_ms"]),
        "preprocessing_ms": float(timings.get("preprocessing_ms", 0.0)),
        "output_tokens": int(result["output_tokens"]),
        "input_tokens": int(result["input_tokens"]),
        "stop_reason": str(result["finish_reason"]),
        "width": int(result["width"]),
        "height": int(result["height"]),
        "token_ids": [int(t) for t in result["token_ids"]],
        "text": result["text"],
        "backend": result.get("backend"),
        "cache_layout": result.get("cache_layout"),
        "weight_layout": result.get("weight_layout"),
    }


def parse_focr_cold(stdout: str, wall_ms: float, max_new: int, stop_ids: set[int]) -> dict:
    result = first_json_object(stdout)
    ids = [int(t) for t in result["generated_ids"]]
    return {
        "process_wall_ms": wall_ms,
        "backend_prepare_ms": float(result.get("backend_prepare_ms", 0.0)),
        "prefill_ms": float(result["prefill_ms"]),
        "decode_ms": float(result["decode_ms"]),
        "total_ocr_latency_ms": float(result["total_ocr_latency_ms"]),
        "output_tokens": len(ids),
        "prompt_tokens": int(result["prompt_tokens"]),
        "stop_reason": focr_stop_reason(ids, max_new, stop_ids),
        "width": int(result["image_width"]),
        "height": int(result["image_height"]),
        "grid_w": int(result["image_grid_w"]),
        "grid_h": int(result["image_grid_h"]),
        "patches": int(result["image_patches"]),
        "resized_by_token_budget": bool(result.get("resized_by_token_budget", False)),
        "token_ids": ids,
        "text": result["text"],
        "backend_report": result.get("backend_report"),
        "prompt_mode": result.get("prompt_mode"),
    }


def parse_focr_preprocess(stdout: str) -> dict:
    result = first_json_object(stdout)
    return {
        "prompt_tokens": len(result["tokens"]),
        "width": int(result["image_width"]),
        "height": int(result["image_height"]),
        "grid_w": int(result["image_grid_w"]),
        "grid_h": int(result["image_grid_h"]),
        "patches": int(result["image_patches"]),
        "resized_by_token_budget": bool(result["resized_by_token_budget"]),
        "image_rgb_crc32": result.get("image_rgb_crc32"),
        "max_pixels": result.get("max_pixels"),
        "prompt_mode": result.get("prompt_mode"),
    }


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def git_output(repo: Path | None, *argv: str) -> str | None:
    if repo is None:
        return None
    try:
        return subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def dirty_entries(repo: Path | None) -> int | None:
    status = git_output(repo, "status", "--porcelain")
    if status is None:
        return None
    return len([line for line in status.splitlines() if line.strip()])


def tool_version(*argv: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(list(argv), capture_output=True, text=True, check=True,
                              cwd=str(cwd) if cwd else None).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def find_cmake_cache(focr_exe: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    target_dir = focr_exe.resolve().parent  # <target>/release
    pattern = str(target_dir / "build" / "falcon-ocr-ggml-sys-*" / "out" / "ggml-build" / "CMakeCache.txt")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return Path(matches[-1]) if matches else None


def ours_identity(args) -> dict:
    repo = Path(__file__).resolve().parents[3]
    return {
        "implementation": "falcon-ocr (this repository)",
        "binary_sha256": sha256_file(args.ours_run) if args.ours_run.exists() else None,
        "bench_binary_sha256": sha256_file(args.ours_bench) if args.ours_bench.exists() else None,
        "git_commit": git_output(repo, "rev-parse", "HEAD"),
        "git_dirty_entries": dirty_entries(repo),
        "cargo_lock_sha256": sha256_file(repo / "Cargo.lock") if (repo / "Cargo.lock").exists() else None,
        "rustc": tool_version("rustc", "-V", cwd=repo),
        "backend": args.ours_backend,
        "threads": args.ours_threads,
        "cache_layout": args.ours_cache_layout,
        "weight_layout": args.ours_weight_layout,
        "precision_lane": "fp32-weights/fp32-compute",
        "build_lane": args.ours_build_lane,
        "binary_commit_note": args.ours_commit_note,
        "model": str(args.ours_model),
    }


def focr_identity(args) -> dict:
    repo = args.focr_repo
    cache_path = find_cmake_cache(args.focr, args.focr_cmake_cache)
    cmake_isa = parse_cmake_isa(cache_path.read_text(encoding="utf-8", errors="replace")) if cache_path and cache_path.exists() else {}
    submodule = git_output(repo, "submodule", "status") if repo else None
    return {
        "implementation": "focr (pszemraj/falcon-ocr.rs)",
        "binary_sha256": sha256_file(args.focr) if args.focr.exists() else None,
        "git_commit": git_output(repo, "rev-parse", "HEAD"),
        "git_dirty_entries": dirty_entries(repo),
        "git_diff_sha256": hashlib.sha256((git_output(repo, "diff") or "").encode("utf-8")).hexdigest() if repo else None,
        "ggml_submodule_status": submodule,
        "ggml_submodule_commit": (submodule or "").strip().split(" ")[0].lstrip("+-") if submodule else None,
        "cargo_lock_sha256": sha256_file(repo / "Cargo.lock") if repo and (repo / "Cargo.lock").exists() else None,
        "rustc": tool_version("rustc", "-V", cwd=repo) if repo else tool_version("rustc", "-V"),
        "cargo_features": args.focr_features,
        "cmake_cache_path": str(cache_path) if cache_path else None,
        "cmake_isa": cmake_isa,
        "backend": args.focr_backend,
        "weight_mode": args.focr_weight_mode,
        "activation_dtype": "f32",
        "prompt_mode": args.focr_prompt_mode,
        "max_length": args.focr_max_length,
        "max_seq_len": args.focr_max_seq_len,
        "max_image_tokens": args.focr_max_image_tokens,
        "threads": args.focr_threads,
        "precision_lane": args.focr_precision_lane,
        "build_lane": args.focr_build_lane,
        "patch_path": str(args.focr_patch) if args.focr_patch else None,
        "model": str(args.focr_model),
    }


# --------------------------------------------------------------------------
# Page flow
# --------------------------------------------------------------------------


def page_id(page: Path) -> str:
    parent = page.parent.name
    return parent if page.name.startswith("canonical") else page.stem


def compare_page(args, runner: Runner, page: Path, stop_ids: set[int]) -> dict:
    entry = {"id": page_id(page), "path": str(page), "png_sha256": sha256_file(page) if page.exists() else None}
    timeout = args.process_timeout

    # 1. focr preprocessing pre-flight (no weights) -> prompt length and geometry.
    stdout, _, _, _ = runner.run(f"{entry['id']}-focr-preprocess", focr_preprocess_command(args, page), timeout)
    pre = parse_focr_preprocess(stdout) if not args.dry_run else {"prompt_tokens": 0, "width": 0, "height": 0, "grid_w": 0, "grid_h": 0, "patches": 0, "resized_by_token_budget": False}
    entry["focr_preprocess"] = pre
    max_new = clamp_max_new_tokens(args.max_new_tokens, pre["prompt_tokens"], args.focr_max_seq_len) if not args.dry_run else args.max_new_tokens
    entry["max_new_tokens_effective"] = max_new
    if max_new != args.max_new_tokens:
        entry.setdefault("notes", []).append(
            f"max-new-tokens clamped from {args.max_new_tokens} to {max_new} (focr prompt {pre['prompt_tokens']} + budget must fit {args.focr_max_seq_len})")

    # 2. Cold runs.
    cold = {}
    if not args.skip_cold:
        if not args.skip_ours:
            stdout, stderr, wall, _ = runner.run(f"{entry['id']}-ours-cold", ours_cold_command(args, page, max_new), timeout)
            if not args.dry_run:
                cold[OURS] = parse_ours_cold(stdout, stderr, wall)
        if not args.skip_focr:
            stdout, _, wall, _ = runner.run(f"{entry['id']}-focr-cold", focr_cold_command(args, page, max_new), timeout)
            if not args.dry_run:
                cold[FOCR] = parse_focr_cold(stdout, wall, max_new, stop_ids)
    entry["cold"] = cold

    # 3. Geometry check (ours from the cold run; focr from preprocess-dump).
    geometry = {}
    if OURS in cold:
        geometry[OURS] = {"width": cold[OURS]["width"], "height": cold[OURS]["height"],
                          "input_tokens": cold[OURS]["input_tokens"],
                          "patches": cold[OURS]["input_tokens"] - OUR_NON_PATCH_PROMPT_TOKENS}
    geometry[FOCR] = {"width": pre["width"], "height": pre["height"], "grid_w": pre["grid_w"], "grid_h": pre["grid_h"],
                      "patches": pre["patches"], "prompt_tokens": pre["prompt_tokens"],
                      "resized_by_token_budget": pre["resized_by_token_budget"]}
    if OURS in geometry:
        ok, problems = geometry_matches(geometry[OURS], geometry[FOCR])
        geometry["match"] = ok
        geometry["problems"] = problems
        if not ok and not args.allow_geometry_mismatch:
            entry["geometry"] = geometry
            entry["error"] = "geometry mismatch: " + "; ".join(problems)
            print(f"[{entry['id']}] {entry['error']}", file=sys.stderr)
            return entry
    entry["geometry"] = geometry

    # 4. Agreement from cold outputs.
    if OURS in cold and FOCR in cold:
        entry["agreement"] = agreement(cold[OURS]["token_ids"], cold[FOCR]["token_ids"],
                                       cold[OURS]["text"], cold[FOCR]["text"], stop_ids)

    # 5. Warm bracket.
    if not args.skip_warm:
        order = [impl for impl in bracket_order(args.processes)
                 if not ((impl == OURS and args.skip_ours) or (impl == FOCR and args.skip_focr))]
        processes = []
        for index, impl in enumerate(order):
            label = f"{entry['id']}-{impl}-warm-{index}"
            if impl == OURS:
                output = runner.run_dir / f"{runner.counter + 1:03d}-{label}.ocr_bench.json"
                stdout, _, wall, _ = runner.run(label, ours_warm_command(args, page, max_new, output, args.warmup, args.repetitions), timeout)
                if args.dry_run:
                    continue
                report = json.loads(output.read_text(encoding="utf-8"))
                samples = samples_from_ocr_bench(report)
                ids = [int(t) for t in report["cases"][0]["token_ids"][0]]
                text = report["cases"][0]["samples"][0]["per_request"][0]["text"]
                processes.append({
                    "impl": impl, "index": index, "output_file": str(output), "sha256": sha256_file(output),
                    "process_wall_ms": wall, "samples": samples, "summary": summarize_samples(samples),
                    "peak_rss_bytes": report["cases"][0].get("memory_after", {}).get("peak_resident_bytes"),
                    "token_ids": ids, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "verified_model_load_ms": report.get("verified_model_load_ms"),
                })
            else:
                stdout, _, wall, _ = runner.run(label, focr_warm_command(args, page, max_new, args.warmup + args.repetitions), timeout)
                if args.dry_run:
                    continue
                report = first_json_object(stdout)
                samples = samples_from_focr_bench(report, args.warmup)
                for sample in samples:
                    sample["stop_reason"] = "eos" if sample["output_tokens"] < max_new else "length"
                processes.append({
                    "impl": impl, "index": index, "process_wall_ms": wall, "samples": samples,
                    "summary": summarize_samples(samples),
                    "peak_rss_bytes": (report.get("peak_rss_kib") or 0) * 1024 or None,
                    "text_sha256": hashlib.sha256(report.get("last_text", "").encode("utf-8")).hexdigest(),
                    "backend_prepare_ms": report.get("backend_prepare_ms"),
                    "backend_report": report.get("backend_report"),
                    "geometry": {"width": report.get("image_width"), "height": report.get("image_height"),
                                 "patches": report.get("image_patches"), "prompt_tokens": report.get("prompt_tokens")},
                })
        if not args.dry_run:
            summary = {}
            for impl in (OURS, FOCR):
                rows = [p for p in processes if p["impl"] == impl]
                if not rows:
                    continue
                medians = [p["summary"]["median_model_ms"] for p in rows]
                best = min(rows, key=lambda p: p["summary"]["median_model_ms"])
                drift = control_drift(medians)
                summary[impl] = {
                    "process_median_model_ms": medians,
                    "control_drift_pct": None if drift is None else drift * 100.0,
                    "drift_ok": None if drift is None else drift <= 0.05,
                    "best": best["summary"],
                    "peak_rss_bytes": max((p["peak_rss_bytes"] or 0) for p in rows) or None,
                    "deterministic": len({p["text_sha256"] for p in rows}) == 1,
                }
                if impl == OURS and OURS in cold:
                    summary[impl]["warm_matches_cold"] = all(
                        p["token_ids"] == cold[OURS]["token_ids"] for p in rows)
                if impl == FOCR and FOCR in cold:
                    cold_sha = hashlib.sha256(cold[FOCR]["text"].encode("utf-8")).hexdigest()
                    summary[impl]["warm_matches_cold"] = all(p["text_sha256"] == cold_sha for p in rows)
            if OURS in summary and FOCR in summary:
                o, f = summary[OURS]["best"], summary[FOCR]["best"]
                summary["ratio_focr_over_ours"] = {
                    key: (f[f"median_{key}_ms"] / o[f"median_{key}_ms"]) if o.get(f"median_{key}_ms") else None
                    for key in ("model", "prefill", "decode")}
            entry["warm"] = {"order": order, "processes": processes, "summary": summary}
    return entry


def thread_sweep(args, runner: Runner, page: Path, stop_ids: set[int]) -> dict:
    stdout, _, _, _ = runner.run(f"{page_id(page)}-focr-preprocess", focr_preprocess_command(args, page), args.process_timeout)
    pre = parse_focr_preprocess(stdout) if not args.dry_run else {"prompt_tokens": 0}
    max_new = clamp_max_new_tokens(args.sweep_max_new_tokens, pre["prompt_tokens"], args.focr_max_seq_len) if not args.dry_run else args.sweep_max_new_tokens
    ours_values = [int(v) for v in args.ours_thread_sweep.split(",") if v]
    focr_values = [int(v) for v in args.thread_sweep.split(",") if v]
    plan = []
    for index in range(max(len(ours_values), len(focr_values))):
        if index < len(ours_values) and not args.skip_ours:
            plan.append((OURS, ours_values[index]))
        if index < len(focr_values) and not args.skip_focr:
            plan.append((FOCR, focr_values[index]))
    rows = {OURS: [], FOCR: []}
    for impl, threads in plan:
        label = f"{page_id(page)}-{impl}-sweep-t{threads}"
        if impl == OURS:
            output = runner.run_dir / f"{runner.counter + 1:03d}-{label}.ocr_bench.json"
            runner.run(label, ours_warm_command(args, page, max_new, output, args.sweep_warmup, args.sweep_repetitions, threads=threads), args.process_timeout)
            if args.dry_run:
                continue
            samples = samples_from_ocr_bench(json.loads(output.read_text(encoding="utf-8")))
        else:
            stdout, _, _, _ = runner.run(label, focr_warm_command(args, page, max_new, args.sweep_warmup + args.sweep_repetitions, threads=threads), args.process_timeout)
            if args.dry_run:
                continue
            samples = samples_from_focr_bench(first_json_object(stdout), args.sweep_warmup)
        summary = summarize_samples(samples)
        rows[impl].append({"threads": threads, "median_model_ms": summary["median_model_ms"],
                           "median_prefill_ms": summary["median_prefill_ms"],
                           "median_decode_ms": summary["median_decode_ms"], "samples": samples})
    selected = {impl: (min(r, key=lambda x: x["median_model_ms"])["threads"] if r else None) for impl, r in rows.items()}
    return {"page": str(page), "max_new_tokens": max_new, OURS: rows[OURS], FOCR: rows[FOCR], "selected": selected}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new directory for all evidence (must not exist)")
    parser.add_argument("--pages", type=Path, nargs="+", required=True)
    parser.add_argument("--min-dimension", type=int, default=64)
    parser.add_argument("--max-dimension", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--processes", type=int, default=2, help="fresh warm processes per side, A B B A order")
    parser.add_argument("--process-timeout", type=float, default=4 * 3600.0, help="seconds per child process")
    parser.add_argument("--stop-ids", default="11,263", help="token IDs stripped from the end before agreement")
    parser.add_argument("--allow-geometry-mismatch", action="store_true")
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--skip-warm", action="store_true")
    parser.add_argument("--skip-ours", action="store_true")
    parser.add_argument("--skip-focr", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    parser.add_argument("--cpu-label", default="AMD Ryzen 9 7950X")
    parser.add_argument("--environment-label", default="native Windows")
    parser.add_argument("--quiet-attestation", default="", help="operator statement about the idle machine")
    parser.add_argument("--note", action="append", default=[], help="free-text notes copied into the report")

    ours = parser.add_argument_group("ours")
    ours.add_argument("--ours-run", type=Path, default=Path("target/release/falcon-ocr.exe"))
    ours.add_argument("--ours-bench", type=Path, default=Path("target/release/examples/ocr_bench.exe"))
    ours.add_argument("--ours-model", type=Path, default=Path("artifacts/model"))
    ours.add_argument("--ours-threads", type=int, default=16)
    ours.add_argument("--ours-backend", default="avx2")
    ours.add_argument("--ours-cache-layout", default="compact", help="the runtime default is compact; pass expanded for the previous layout")
    ours.add_argument("--ours-weight-layout", default="unpacked")
    ours.add_argument("--ours-build-lane", default="head", help="label for the binary under test, e.g. head or control-<commit>")
    ours.add_argument("--ours-commit-note", default="", help="which commit the ours binaries were built from, if not HEAD")

    focr = parser.add_argument_group("focr")
    focr.add_argument("--focr", type=Path, default=Path(".tmp/friend-target-baseline/release/focr.exe"))
    focr.add_argument("--focr-repo", type=Path, default=Path(".tmp/friend-falcon-ocr.rs"))
    focr.add_argument("--focr-model", type=Path, default=Path(".tmp/compare-model"))
    focr.add_argument("--focr-threads", type=int, default=16)
    focr.add_argument("--focr-backend", default="ggml-linear")
    focr.add_argument("--focr-weight-mode", default="resident-f32")
    focr.add_argument("--focr-prompt-mode", default="hf-split")
    focr.add_argument("--focr-max-length", type=int, default=8192)
    focr.add_argument("--focr-max-seq-len", type=int, default=8192)
    focr.add_argument("--focr-max-image-tokens", type=int, default=39200)
    focr.add_argument("--focr-build-lane", default="unmodified")
    focr.add_argument("--focr-precision-lane", default="fp32-weights/fp32-compute")
    focr.add_argument("--focr-features", default="default (ggml-cpu, turbojpeg)")
    focr.add_argument("--focr-cmake-cache", type=Path, default=None)
    focr.add_argument("--focr-patch", type=Path, default=None, help="diff applied for a modified lane")

    sweep = parser.add_argument_group("thread sweep (runs instead of the main matrix)")
    sweep.add_argument("--thread-sweep", default="", help="focr thread values, e.g. 8,16,24,32")
    sweep.add_argument("--ours-thread-sweep", default="8,16,32")
    sweep.add_argument("--sweep-max-new-tokens", type=int, default=256)
    sweep.add_argument("--sweep-warmup", type=int, default=1)
    sweep.add_argument("--sweep-repetitions", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite existing output directory {args.output}", file=sys.stderr)
        return 2
    stop_ids = {int(v) for v in args.stop_ids.split(",") if v}
    runner = Runner(args, args.output / "runs")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": SCHEMA_KIND,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "cpu_label": args.cpu_label,
            "environment_label": args.environment_label,
            "quiet_attestation": args.quiet_attestation,
        },
        "settings": {
            "min_dimension": args.min_dimension,
            "max_dimension": args.max_dimension,
            "max_new_tokens_requested": args.max_new_tokens,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "processes": args.processes,
            "order": "ABBA",
            "stop_ids": sorted(stop_ids),
            "focr_max_seq_len": args.focr_max_seq_len,
            "focr_max_image_tokens": args.focr_max_image_tokens,
            "focr_prompt_mode": args.focr_prompt_mode,
        },
        "implementations": {OURS: ours_identity(args), FOCR: focr_identity(args)},
        "thread_sweep": None,
        "pages": [],
        "notes": list(args.note),
    }
    try:
        if args.thread_sweep:
            report["thread_sweep"] = thread_sweep(args, runner, args.pages[0], stop_ids)
        else:
            for page in args.pages:
                try:
                    report["pages"].append(compare_page(args, runner, page, stop_ids))
                except Exception as error:  # keep evidence of the other pages
                    report["pages"].append({"id": page_id(page), "path": str(page), "error": str(error)})
                    print(f"[{page_id(page)}] failed: {error}", file=sys.stderr)
    finally:
        report["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
        print(f"wrote {args.output / 'report.json'} and report.md")
    failed = [p for p in report["pages"] if p.get("error")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
