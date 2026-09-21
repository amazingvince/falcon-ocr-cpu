#!/usr/bin/env python3
"""Fail-closed validation of completed free-greedy GPU page records."""
import math
import re
import struct


def validate_gpu_reference_record(record, configuration, check, label):
    """Report checks through check(name, actual, expected), returning validity.

This validates record structure and inference invariants; callers separately
verify manifest membership, actual file hashes and startup source identity.
"""
    valid = True

    def require(name, condition):
        nonlocal valid
        good = bool(condition)
        valid = valid and good
        check(label + "." + name, good, True)
        return good

    def integer(value):
        return type(value) is int

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    if not require("record_object", isinstance(record, dict)):
        return False
    require("no_error", "error" not in record)
    require("configuration", record.get("configuration") == configuration)
    cap = configuration.get("max_new_tokens")
    cap_valid = require("positive_output_cap", integer(cap) and cap > 0)
    ids = record.get("token_ids")
    ids_valid = require("token_ids", isinstance(ids, list) and bool(ids)
                        and all(integer(x) and 0 <= x < 65536 for x in ids))
    require("text", isinstance(record.get("text"), str))
    if ids_valid and cap_valid:
        require("within_output_cap", len(ids) <= cap)
        require("no_early_stop_token", all(x not in [11, 263] for x in ids[:-1]))
        reason = record.get("finish_reason")
        require("finish_reason", (reason == "eos" and ids[-1] in [11, 263])
                or (reason == "length" and len(ids) == cap and ids[-1] not in [11, 263]))
    prefix, capacity = record.get("prefix_length"), record.get("cache_capacity")
    prefix_valid = require("prefix_length", integer(prefix) and 16 < prefix <= 16384)
    capacity_valid = require("cache_capacity", integer(capacity) and 128 <= capacity <= 16384 and capacity % 128 == 0)
    if prefix_valid and capacity_valid and cap_valid:
        require("context_budget", prefix + cap <= 16384)
        require("cache_capacity_rounding", capacity == ((prefix + cap + 127) // 128) * 128)
    require("canonical_rgb_sha256", isinstance(record.get("canonical_rgb_sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", record["canonical_rgb_sha256"]) is not None)
    decisions = record.get("logit_decisions")
    decisions_valid = require("logit_decisions_length", isinstance(decisions, list) and ids_valid and len(decisions) == len(ids))
    if decisions_valid:
        bad = []
        for index, (token, item) in enumerate(zip(ids, decisions)):
            if not isinstance(item, dict):
                bad.append(index)
                continue
            winner, runner, margin = (item.get(key) for key in ["winner_logit", "runner_up_logit", "winner_margin"])
            runner_id = item.get("runner_up")
            margin_valid = False
            if all(finite(x) for x in [winner, runner, margin]):
                difference = winner - runner
                # Original FP32 smoke/long records subtract in the FP32 tensor;
                # later runs subtract the two exactly promoted Python floats.
                # Both are known saved schemas, with at most that one rounding.
                try:
                    rounded = struct.unpack("f", struct.pack("f", difference))[0]
                    margin_valid = margin in [difference, rounded]
                except OverflowError:
                    margin_valid = False
            good = (integer(item.get("step")) and item["step"] == index
                    and integer(item.get("argmax")) and item["argmax"] == token
                    and integer(runner_id) and 0 <= runner_id < 65536 and runner_id != token
                    and all(finite(x) for x in [winner, runner, margin])
                    and winner >= runner and margin >= 0 and margin_valid)
            if not good:
                bad.append(index)
        require("logit_decision_values", not bad)
        if bad:
            check(label + ".invalid_decision_indices", bad[:32], [])
    for name in ["prefill_seconds_including_compile", "decode_seconds_including_python_diagnostics", "elapsed_seconds"]:
        require(name, finite(record.get(name)) and record[name] >= 0)
    peak = record.get("peak_gpu_allocated_bytes")
    require("peak_gpu_allocated_bytes", integer(peak) and peak >= 0)
    return valid
