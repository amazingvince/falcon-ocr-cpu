#!/usr/bin/env python3
"""Shared, fail-closed provenance checks for text-only replay of saved Rust IDs.

The caller supplies check(name, actual, expected), either collecting failures for
a report or raising immediately. No original or derived artifact is modified.
"""
import hashlib
import json
import math
import os
import pathlib
import re

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))  # frozen GPU-reference closure (scripts/README.md)

from fetch_reference import sha256


REPLAY_QUALIFICATION = "Original free-running inference token IDs with separately verified Rust text-only decoding replay; no new model inference was executed. Original source hashes and timings describe the original inference."


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def resolve_recorded_path(value):
    """Resolve recorded native Windows or WSL drive paths on either local host."""
    value = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", value):
        value = "/mnt/" + value[0].lower() + value[2:]
    elif os.name == "nt" and re.match(r"^/mnt/[A-Za-z]/", value):
        value = value[5].upper() + ":" + value[6:]
    return pathlib.Path(value)


def _hashed_file(path_value, expected, check, label):
    if not isinstance(path_value, str) or not isinstance(expected, str):
        check(label + ".path_and_hash_present", False, True)
        return None
    path = resolve_recorded_path(path_value)
    try:
        actual = sha256(path)
    except (OSError, ValueError) as error:
        check(label + ".readable", type(error).__name__ + ": " + str(error), "readable file")
        return None
    check(label + ".sha256", actual, expected)
    return path


def _read_json(path, check, label):
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        check(label + ".json", type(error).__name__ + ": " + str(error), "readable JSON object")
        return None
    check(label + ".object", isinstance(value, dict), True)
    return value if isinstance(value, dict) else None


def validate_inference_result(result, contract, check, label):
    """Validate the saved plain-OCR result before treating it as completed."""
    valid = True

    def require(name, passed):
        nonlocal valid
        valid = valid and bool(passed)
        check(label + "." + name, bool(passed), True)

    require("result_object", isinstance(result, dict))
    if not isinstance(result, dict):
        return False
    options = contract.get("options", {})
    require("options_object", isinstance(options, dict))
    if not isinstance(options, dict):
        return False
    cap, maximum = options.get("max_new_tokens"), options.get("max_dimension")
    require("valid_output_budget", type(cap) is int and cap > 0)
    require("valid_dimension_limit", type(maximum) is int and maximum > 0)
    ids = result.get("token_ids")
    ids_valid = isinstance(ids, list) and bool(ids) and all(type(i) is int and 0 <= i < 65536 for i in ids)
    require("valid_token_ids", ids_valid)
    if ids_valid and type(cap) is int:
        require("output_count", type(result.get("output_tokens")) is int and result["output_tokens"] == len(ids) <= cap)
        stop = ids[-1] in [11, 263]
        require("finish_reason", (result.get("finish_reason") == "eos" and stop) or
                (result.get("finish_reason") == "length" and not stop and len(ids) == cap))
    require("text_string", isinstance(result.get("text"), str))
    require("free_running", result.get("teacher_forced") is False)
    require("precision", result.get("precision") == contract.get("precision") and result.get("precision") in ["fp32", "bf16"])
    width, height, prefix = result.get("width"), result.get("height"), result.get("input_tokens")
    dimensions = type(maximum) is int and all(type(n) is int and 0 < n <= maximum and n % 16 == 0 for n in [width, height])
    require("image_dimensions", dimensions)
    require("prefix_integer", type(prefix) is int and prefix > 0)
    if dimensions and type(prefix) is int:
        require("prefix_patch_shape", prefix == (width // 16) * (height // 16) + 16)
    if type(prefix) is int and type(cap) is int:
        require("context_budget", prefix + cap <= 16384)
    timings = result.get("timings")
    require("timings_object", isinstance(timings, dict))
    if isinstance(timings, dict):
        for name in ["image_decode_ms", "preprocessing_ms", "prefill_ms", "decode_ms", "total_ms", "time_to_first_token_ms"]:
            value = timings.get(name)
            require("timing." + name, type(value) in [int, float] and math.isfinite(value) and value >= 0)
    return valid


def validate_run_replay(run, contract, check):
    """Return None for original inference, or context for validate_page_replay."""
    replay = contract.get("postprocessing_replay")
    if replay is None:
        check("cpu.derived_text_replay", run.get("derived_text_replay", False), False)
        return None
    check("cpu.derived_text_replay", run.get("derived_text_replay"), True)
    check("cpu.replay.metadata_object", isinstance(replay, dict), True)
    context = {"metadata": replay, "source_run": None, "source_run_path": None, "source_contract_sha256": None}
    if not isinstance(replay, dict):
        return context
    check("cpu.replay.policy_present", isinstance(replay.get("policy"), str) and bool(replay["policy"]), True)
    check("cpu.replay.inference_reexecuted", replay.get("inference_reexecuted", False), False)
    check("cpu.replay.run_inference_reexecuted", run.get("inference_reexecuted", False), False)
    source_path = _hashed_file(replay.get("source_run_path"), replay.get("source_run_sha256"), check, "cpu.replay.source_run")
    original = _read_json(source_path, check, "cpu.replay.source_run")
    context["source_run"] = original
    context["source_run_path"] = source_path
    for item in ["decoder_binary", "decoder_source"]:
        _hashed_file(replay.get(item + "_path"), replay.get(item + "_sha256"), check, "cpu.replay." + item)
    for key in ["harness_source_sha256", "cargo_lock_sha256"]:
        check("cpu.replay." + key + ".valid_digest", isinstance(replay.get(key), str) and bool(re.fullmatch(r"[0-9a-f]{64}", replay.get(key, ""))), True)
    assets = replay.get("tokenizer_asset_sha256")
    check("cpu.replay.tokenizer_assets_object", isinstance(assets, dict) and bool(assets), True)
    if isinstance(assets, dict):
        pins_path = pathlib.Path(__file__).resolve().parents[3] / "reference/manifest.json"
        pins = _read_json(pins_path, check, "cpu.replay.reference_pins")
        if pins is not None:
            check("cpu.replay.tokenizer_revision", contract.get("model_revision"), pins.get("model_revision"))
        names = {resolve_recorded_path(p).name for p in assets}
        check("cpu.replay.required_tokenizer_assets", {"tokenizer.json", "tokenizer_config.json"}.issubset(names), True)
        for path, digest in assets.items():
            _hashed_file(path, digest, check, "cpu.replay.tokenizer_asset." + str(path))
            if pins is not None:
                check("cpu.replay.pinned_tokenizer_asset." + str(path), digest,
                      pins.get("files", {}).get(resolve_recorded_path(path).name, {}).get("sha256"))
    if original is not None:
        original_contract = original.get("contract")
        check("cpu.replay.source_contract_object", isinstance(original_contract, dict), True)
        check("cpu.replay.original_not_derived", original.get("derived_text_replay", False), False)
        check("cpu.replay.original_teacher_forced", original.get("teacher_forced"), False)
        context["source_contract_sha256"] = original.get("contract_sha256")
        check("cpu.replay.source_contract_sha256", replay.get("source_contract_sha256"), original.get("contract_sha256"))
        if isinstance(original_contract, dict):
            check("cpu.replay.original_has_no_replay", "postprocessing_replay" in original_contract, False)
            check("cpu.replay.original_contract_hash", canonical_sha256(original_contract), original.get("contract_sha256"))
            check("cpu.replay.inference_contract_unchanged", {k: v for k, v in contract.items() if k != "postprocessing_replay"}, original_contract)
        allowed_run_changes = {"contract", "contract_sha256", "derived_text_replay", "qualification", "inference_reexecuted"}
        check("cpu.replay.inherited_run_fields", {k: v for k, v in run.items() if k not in allowed_run_changes},
              {k: v for k, v in original.items() if k not in allowed_run_changes})
    return context


def validate_page_replay(page, context, check, sample):
    """Validate original page binding and exact inheritance except decoded text."""
    prefix = sample + ".replay"
    replay = page.get("postprocessing_replay")
    if context is None:
        check(prefix + ".metadata_absent_for_original_inference", replay is None, True)
        return
    check(prefix + ".metadata_object", isinstance(replay, dict), True)
    if not isinstance(replay, dict):
        return
    check(prefix + ".inference_reexecuted", replay.get("inference_reexecuted"), False)
    check(prefix + ".source_contract_sha256", replay.get("source_contract_sha256"), context["source_contract_sha256"])
    path = _hashed_file(replay.get("source_record_path"), replay.get("source_record_sha256"), check, prefix + ".source_record")
    if path is not None and context["source_run_path"] is not None:
        check(prefix + ".source_directory", str(path.resolve().parent), str(context["source_run_path"].resolve().parent))
    original = _read_json(path, check, prefix + ".source_record")
    if original is None:
        return
    check(prefix + ".original_has_no_replay", "postprocessing_replay" in original, False)
    check(prefix + ".original_contract_sha256", original.get("contract_sha256"), context["source_contract_sha256"])
    check(prefix + ".original_success", "error" in original, False)
    source_result, result = original.get("result"), page.get("result")
    check(prefix + ".source_result_object", isinstance(source_result, dict), True)
    check(prefix + ".result_object", isinstance(result, dict), True)
    if isinstance(source_result, dict) and isinstance(result, dict):
        if context["source_run"] is not None and isinstance(context["source_run"].get("contract"), dict):
            validate_inference_result(source_result, context["source_run"]["contract"], check, prefix + ".original_result")
        check(prefix + ".source_teacher_forced", source_result.get("teacher_forced"), False)
        check(prefix + ".original_text_matches", replay.get("original_text") == source_result.get("text"), True)
        text = source_result.get("text")
        check(prefix + ".source_text_string", isinstance(text, str), True)
        if isinstance(text, str):
            check(prefix + ".original_text_sha256", hashlib.sha256(text.encode("utf-8")).hexdigest(), replay.get("original_text_sha256"))
        check(prefix + ".all_inference_result_fields_sha256",
              canonical_sha256({k: v for k, v in result.items() if k != "text"}),
              canonical_sha256({k: v for k, v in source_result.items() if k != "text"}))
    excluded = {"contract_sha256", "postprocessing_replay", "result"}
    check(prefix + ".original_top_level_fields_unchanged", {k: v for k, v in page.items() if k not in excluded},
          {k: v for k, v in original.items() if k not in excluded})
