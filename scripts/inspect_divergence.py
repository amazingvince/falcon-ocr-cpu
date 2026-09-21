#!/usr/bin/env python3
"""Locate peak errors and distinguish image/register tokens from output positions."""
import argparse
import json

import torch
from safetensors.torch import load_file

from compare_traces import normalized


p = argparse.ArgumentParser()
p.add_argument("reference")
p.add_argument("candidate")
args = p.parse_args()
a, b = normalized(load_file(args.reference)), normalized(load_file(args.candidate))
rows = []
for key in ["embedding"] + [f"layer.{i}.hidden" for i in range(22)] + ["logits"]:
    delta = (a[key].float() - b[key].float()).abs()
    flat = int(delta.flatten().argmax())
    row = flat // a[key].shape[-1]
    rows.append({"stage": key, "max_error": float(delta.max()), "max_position": row,
                 "token_id": int(a["tokens"][row]) if row < len(a["tokens"]) else None,
                 "last_position_error": float(delta[-1].max()), "rms_error": float(delta.square().mean().sqrt())})
print(json.dumps(rows, indent=2))
