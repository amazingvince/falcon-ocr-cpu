"""Read-only independent audit of the frozen 200-page and platform output evidence.

This does not import the original comparators, execute inference, or inspect quality
scores. Receipt and inventory contain identities/counts only, never OCR text.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reviews/completed-corpus-independent-v1"
RECEIPT = ROOT / "reference/completed-corpus-independent-review-v1.json"
REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"
WEIGHTS = "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16"
checks = 0
inputs: dict[str, dict] = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def path(value):
    p = Path(str(value).replace("\\", "/"))
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def rel(value):
    return path(value).relative_to(ROOT).as_posix()


def require(condition, label):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(label)  # Never print compared text/IDs.


def read(value, expected=None):
    p = path(value)
    data = p.read_bytes()
    identity = {"sha256": sha(data), "bytes": len(data)}
    key = rel(p)
    if key in inputs:
        require(inputs[key] == identity, "input changed during audit: " + key)
    inputs[key] = identity
    if expected is not None:
        require(identity["sha256"] == expected, "recorded hash: " + key)
    return data


def js(value, expected=None):
    return json.loads(read(value, expected))


def canonical(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode())


def exact(value):
    """Type-aware equality, including every inherited IEEE binary64 timing bit."""
    if type(value) is float:
        require(math.isfinite(value), "nonfinite inherited float")
        return ("float64", struct.pack(">d", value).hex())
    if isinstance(value, dict):
        return ("dict", tuple((k, exact(v)) for k, v in sorted(value.items())))
    if isinstance(value, list):
        return ("list", tuple(exact(v) for v in value))
    return (type(value).__name__, value)


def same(a, b, label):
    require(exact(a) == exact(b), label)


def strip(value, *keys):
    return {k: v for k, v in value.items() if k not in keys}


def archive(value, expected, members):
    with zipfile.ZipFile(io.BytesIO(read(value, expected))) as z:
        require(len(z.namelist()) == len(set(z.namelist())), "duplicate archive members")
        for member, digest in members.items():
            require(sha(z.read(member)) == digest, "archive member: " + member)


def build(value):
    b = js(value)
    require(b["status"] == "complete" and b["build_exit_code"] == 0
            and b["source_unchanged_during_build"] is True, "build completion")
    base = path(value).parent
    read(base / b["binary"], b["binary_sha256"])
    archive(base / b["source_archive"], b["source_archive_sha256"], b["source_sha256"])
    return b


def valid_result(r, c):
    ids = r["token_ids"]
    require(isinstance(ids, list) and bool(ids), "nonempty IDs")
    require(all(type(i) is int and 0 <= i < 65536 for i in ids), "ID range")
    require(r["output_tokens"] == len(ids) <= c["options"]["max_new_tokens"], "token count")
    require(r["teacher_forced"] is False and r["precision"] == "fp32", "free FP32 output")
    require(isinstance(r["text"], str), "text type")
    require((r["finish_reason"] == "eos" and ids[-1] in (11, 263)) or
            (r["finish_reason"] == "length" and len(ids) == 4096 and ids[-1] not in (11, 263)),
            "stop semantics")
    require(not any(i in (11, 263) for i in ids[:-1]), "no earlier EOS")
    require(all(type(r[k]) is int and 64 <= r[k] <= 1536 and r[k] % 16 == 0
                for k in ("width", "height")), "prepared dimensions")
    require(r["input_tokens"] == r["width"] * r["height"] // 256 + 16,
            "patch/prefix count")
    require(r["input_tokens"] + 4096 <= 16384, "request context budget")


def main():
    require(not OUT.exists() and not RECEIPT.exists(), "refuse overwrite")
    started = datetime.now(timezone.utc).isoformat()
    read(__file__)
    full_path = "reference/windows-rust-corpus-v3-fp32-redecoded-200.json"
    join_path = "reference/windows-linux-corpus-v3-smoke-output-agreement-v1.json"
    f = js(full_path, "a362b42625a5ee594212135ff4a0e36bb64cfcbb64296b91bc19a7712030f4d0")
    j = js(join_path)
    linux_comparison_path = "artifacts/cpu/linux-v3-gpu-comparison-complete-v1.json"
    lc = js(linux_comparison_path)
    for recorded, digest in j["source_sha256"].items():
        read(recorded, digest)
    manifest = js(f["manifest"], f["manifest_sha256"])
    smoke = js("reference/corpus-v3-smoke-lock.json")
    ordered = [p["id"] for p in manifest["pages"]]
    require(len(ordered) == len(set(ordered)) == 200, "full scope unique200")
    same(f["comparison_scope"]["ordered_selected_ids"], ordered, "full comparison ordered scope")
    by_id = {p["id"]: p for p in manifest["pages"]}
    require(len(smoke["pages"]) == 24, "smoke scope24")
    for p in smoke["pages"]:
        same(p, by_id[p["id"]], "exact subset page object")
    gpu_run = js(f["gpu_run"], f["gpu_run_sha256"])
    replay_run = js(f["cpu_run"], f["cpu_run_sha256"])
    original_run = js(f["token_inference_source_run"], f["token_inference_source_run_sha256"])
    linux_run = js(lc["cpu_run"], lc["cpu_run_sha256"])
    c = replay_run["contract"]
    o = original_run["contract"]
    l = linux_run["contract"]
    replay = c["postprocessing_replay"]
    require(replay_run["derived_text_replay"] is True, "replay label")
    require(not original_run.get("derived_text_replay", False), "original not replay")
    require(not linux_run.get("derived_text_replay", False), "Linux fresh inference label")
    same(strip(c, "postprocessing_replay"), o, "inherited inference contract")
    same(strip(replay_run, "contract", "contract_sha256", "qualification", "derived_text_replay"),
         strip(original_run, "contract", "contract_sha256", "qualification"), "inherited run fields")
    for r in (original_run, replay_run, linux_run):
        require(canonical(r["contract"]) == r["contract_sha256"], "run contract digest")
    for contract in (o, l, gpu_run["configuration"]):
        require(contract["model_revision"] == REVISION and contract["weights_sha256"] == WEIGHTS,
                "fixed model identity")
    same(o["options"], l["options"], "CPU options")
    same(o["options"], {k: gpu_run["configuration"][k] for k in o["options"]}, "GPU options")
    require(o["threads"] == 16 and l["threads"] == 4, "separate platform threads")
    require("Windows" in o["environment_label"] and "WSL" in l["environment_label"], "platform labels")
    require(gpu_run["configuration"]["tf32"] is False and
            gpu_run["configuration"]["flex_float32_precision"] == "ieee", "GPU strict math metadata")
    read(replay["source_run_path"], replay["source_run_sha256"])
    require(replay["source_run_sha256"] == f["token_inference_source_run_sha256"], "same replay source run")
    require(replay["source_contract_sha256"] == original_run["contract_sha256"], "replay source contract")
    read(replay["decoder_binary_path"], replay["decoder_binary_sha256"])
    read(replay["decoder_source_path"], replay["decoder_source_sha256"])
    for p, digest in replay["tokenizer_asset_sha256"].items():
        read(p, digest)
    decoder_build = build("artifacts/builds/redecode-corpus-v3-windows/build.json")
    require(decoder_build["binary_sha256"] == replay["decoder_binary_sha256"], "decoder build link")
    for source_name, field in [("src/tokenizer.rs", "decoder_source_sha256"),
                               ("examples/redecode_corpus.rs", "harness_source_sha256"),
                               ("examples/support/corpus_record.rs", "record_validator_sha256"),
                               ("Cargo.lock", "cargo_lock_sha256")]:
        require(decoder_build["source_sha256"][source_name] == replay[field], "decoder archived source join")
    lb = lc["contract_evidence"]["cpu"]["build"]
    read(lb["path"], lb["sha256"])
    linux_build = build(lb["path"])
    require(linux_build["binary_sha256"] == l["binary_sha256"] == lb["binary_sha256"], "Linux build join")
    require(lb["sha256"] == l["build_manifest_sha256"], "Linux startup build digest")
    for asset in f["checked_model_assets"]["assets"].values():
        read(asset["path"], asset["sha256"])
    historical = js("reference/gpu-corpus-source-capture-after-launch.json")
    require("AFTER" in historical["timing"] and "not startup" in historical["timing"], "historical capture qualifier")
    archive(historical["archive"], historical["archive_sha256"],
            {x["path"]: x["sha256"] for x in historical["files"]})
    original_dir = path(f["token_inference_source_run"]).parent
    replay_dir = path(f["cpu_run"]).parent
    gpu_dir = path(f["gpu_run"]).parent
    snap = js(original_dir / "snapshot.json")
    read(snap["source_run_path"], snap["source_run_sha256"])
    read(original_dir / "snapshot_cpu_corpus.py", snap["script_sha256"])
    require(snap["selected_first_pages"] == snap["full_manifest_pages"] == 200, "snapshot scope")
    same([r["id"] for r in snap["records"]], ordered, "snapshot ordered scope")
    for p, digest in snap["copied_files"].items():
        read(original_dir / p, digest)
    report_pages = {r["id"]: r for r in f["pages"]}
    require(len(report_pages) == len(f["pages"]) == 200, "reported full records")
    tokens = changed = 0
    stops = Counter()
    full_records = {}
    for p, snapshot in zip(manifest["pages"], snap["records"]):
        sid = Path(p["canonical_path"]).parent.name
        rep = report_pages[p["id"]]
        require(rep["sample_id"] == sid, "sample identity")
        read(p["canonical_path"], p["canonical_png_sha256"])
        read(p["ground_truth_path"], p["ground_truth_sha256"])
        read(snapshot["source_path"], snapshot["sha256"])
        require(snapshot["snapshot_name"] == sid + ".json", "snapshot filename")
        old = js(original_dir / (sid + ".json"), snapshot["sha256"])
        new = js(replay_dir / (sid + ".json"), rep["cpu_record_sha256"])
        gpu = js(gpu_dir / (sid + ".json"), rep["gpu_record_sha256"])
        require("error" not in old and "error" not in new and "error" not in gpu, "no error record")
        meta = new["postprocessing_replay"]
        require(meta["inference_reexecuted"] is False, "text-only replay")
        require(path(meta["source_record_path"]) == original_dir / (sid + ".json"), "source record location")
        require(meta["source_record_sha256"] == snapshot["sha256"], "source record hash")
        require(meta["source_contract_sha256"] == original_run["contract_sha256"], "source page contract")
        require(old["contract_sha256"] == original_run["contract_sha256"] and
                new["contract_sha256"] == replay_run["contract_sha256"], "page contract identity")
        require("postprocessing_replay" not in old and not old.get("derived_text_replay", False), "no replay of replay")
        require(meta["original_text"] == old["result"]["text"] and
                meta["original_text_sha256"] == sha(old["result"]["text"].encode()), "original text binding")
        same(strip(old, "result", "contract_sha256"),
             strip(new, "result", "contract_sha256", "postprocessing_replay"), "all inherited record fields")
        same(strip(old["result"], "text"), strip(new["result"], "text"), "all inherited result/float bits")
        valid_result(new["result"], c)
        require(new["teacher_forced"] is False, "page free run")
        require(new["id"] == gpu["id"] == p["id"] and
                new["category"] == gpu["category"] == p["category"], "page lock identity")
        require(new["input_sha256"] == p["canonical_png_sha256"] and
                new["ground_truth_sha256"] == p["ground_truth_sha256"] and
                gpu["canonical_rgb_sha256"] == p["rgb_sha256"], "input identities")
        same(gpu["configuration"], gpu_run["configuration"], "GPU page contract")
        for field in ("token_ids", "text", "finish_reason"):
            same(new["result"][field], gpu[field], "direct full output " + field)
        require(new["result"]["input_tokens"] == gpu["prefix_length"], "prefix equality")
        require(rep["tokens_exact"] and rep["text_exact"] and rep["finish_reason_exact"] and
                rep["provenance_passed"], "full report flags")
        require(rep["gpu_tokens"] == rep["cpu_tokens"] == len(gpu["token_ids"]), "report token counts")
        changed += old["result"]["text"] != new["result"]["text"]
        tokens += len(gpu["token_ids"])
        stops[gpu["finish_reason"]] += 1
        full_records[p["id"]] = (new, gpu)
    require(tokens == f["compared_gpu_tokens"] == 275903 and changed == 12, "full totals")
    require(stops == {"eos": 181, "length": 19}, "full stops")
    require(all(f[k] == 0 for k in ("missing_cpu_count", "missing_gpu_count", "failed_cpu_count", "failed_gpu_count")), "full missing/error counts")
    require(f["completed_output_parity_passed"] is True and f["compared_pages"] == f["exact_pages"] == 200, "full completion flags")
    selected = {p["id"]: p for p in smoke["pages"]}
    require(len(j["records"]) == len(selected) == j["pages"] == 24, "platform record count")
    require({r["id"] for r in j["records"]} == set(selected), "platform exact selected set")
    platform_tokens = 0
    for r in j["records"]:
        sid = r["sample_id"]
        win, gpu = full_records[r["id"]]
        lin = js(path(lc["cpu_run"]).parent / (sid + ".json"), r["linux_record_sha256"])
        read(replay_dir / (sid + ".json"), r["windows_record_sha256"])
        read(gpu_dir / (sid + ".json"), r["gpu_record_sha256"])
        require(lin["contract_sha256"] == linux_run["contract_sha256"] and lin["teacher_forced"] is False,
                "Linux page contract/free run")
        require(lin["id"] == r["id"] and lin["input_sha256"] == selected[r["id"]]["canonical_png_sha256"], "Linux page/input identity")
        valid_result(lin["result"], l)
        for field in j["compared_cpu_result_fields"]:
            same(win["result"][field], lin["result"][field], "direct platform " + field)
        for field in ("token_ids", "text", "finish_reason"):
            same(lin["result"][field], gpu[field], "direct Linux GPU " + field)
        require(r["generated_ids"] == len(gpu["token_ids"]), "platform token count")
        platform_tokens += len(gpu["token_ids"])
    require(platform_tokens == j["matching_token_ids"] == 29205, "platform total")
    require(f["startup_semantic_fields_complete"] is False and
            j["startup_semantic_fields_complete"] == {"windows": False, "linux": False}, "startup limits retained")
    require(f["contract_evidence"]["gpu"]["missing_startup_fields"] == ["prompt", "config_sha256", "greedy_tie_rule"], "GPU startup gap")
    require(f["contract_evidence"]["cpu"]["missing_startup_fields"] == ["prompt", "config_sha256", "greedy_policy"], "Windows startup gap")
    require(f["contract_evidence"]["cpu"]["build"]["verified"] is False, "Windows inference build not upgraded")
    before = dict(inputs)
    for p, identity in before.items():
        require(sha(path(p).read_bytes()) == identity["sha256"], "end hash: " + p)
    inventory = {"schema_version": 1, "files": before, "all_end_hashes_equal_start": True,
                 "scope": "Metadata-only inventory; parse and initial digest used the same bytes."}
    OUT.mkdir(parents=True)
    inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    inventory_path = OUT / "input-manifest.json"
    with inventory_path.open("xb") as out:
        out.write(inventory_bytes)
    receipt = {
        "schema_version": 1, "status": "passed_independent_saved_evidence_review",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "review_source": rel(__file__), "review_source_sha256": before[rel(__file__)]["sha256"],
        "checks_passed": checks, "input_files": len(before),
        "inventory": rel(inventory_path), "inventory_sha256": sha(inventory_bytes),
        "all_input_hashes_unchanged_at_end": True,
        "reviewed_reports": {full_path: before[full_path]["sha256"], join_path: before[join_path]["sha256"]},
        "windows_full": {"pages": 200, "exact_ids": tokens, "exact_texts": 200, "exact_stops": 200,
                         "finish_reasons": dict(stops), "derived_text_changes": changed,
                         "all_original_ids_and_nontext_result_fields_inherited_exactly": True,
                         "all_inherited_float64_timing_bits_exact": True,
                         "original_run_and_records_match_completed_source_and_snapshot": True},
        "platform_subset": {"pages": 24, "exact_ids": platform_tokens, "exact_texts": 24,
                            "exact_stops": 24, "windows_threads": 16, "linux_threads": 4,
                            "same_selected_page_objects": True, "same_binary_or_source_claim": False},
        "build_checks": {"corrected_decoder_binary_and_archived_sources": True,
                         "fresh_linux_binary_and_archived_sources": True,
                         "original_windows_inference_build_independently_verified": False},
        "scope_limits": [
            "Windows text is a saved-ID replay; all IDs, inference source fields and timing bits are inherited. No inference was reexecuted.",
            "GPU startup records omit prompt/config digest/greedy tie policy. Windows inference startup omits prompt/config digest/greedy policy and has no supplied build archive verification.",
            "GPU source ZIP is verified as an AFTER-launch preservation only; no retroactive startup attestation.",
            "The corrected decoder build attests text replay, not original Windows inference. Builds are preserved artifacts, not hermetic rebuilds.",
            "GPU processed width/height were not recorded; pixel/options/prefix identities are joined without inventing dimensions.",
            "Canonical PNG and ground-truth file bytes were rehashed; raw RGB digests are joined to the frozen manifest, not freshly decoded here.",
            "Checkpoint identity is recorded digest consistency, not a new 1 GB weight rehash.",
            "WSL Linux and native Windows are separate functional runs with different threads/binaries/sources; no bare-metal performance qualification.",
            "Output agreement does not qualify intermediate tensor parity, BF16, OCR accuracy, or quality scores. No official evaluator output was inspected."
        ],
        "inference_executed": False, "gpu_work_executed": False, "raw_ocr_text_in_receipt": False,
    }
    receipt_bytes = (json.dumps(receipt, indent=2) + "\n").encode()
    with RECEIPT.open("xb") as out:
        out.write(receipt_bytes)
    print(json.dumps({"status": receipt["status"], "receipt": rel(RECEIPT),
                      "receipt_sha256": sha(receipt_bytes), "checks": checks,
                      "input_files": len(before), "windows_ids": tokens, "platform_ids": platform_tokens}))


if __name__ == "__main__":
    main()
