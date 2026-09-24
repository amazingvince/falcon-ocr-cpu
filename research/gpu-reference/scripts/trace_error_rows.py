#!/usr/bin/env python3
"""Track failed tensor locations back through identical token positions."""
import json
import math
import pathlib
import re

import torch
from safetensors.torch import load_file

from compare_traces import normalized


root = pathlib.Path(__file__).resolve().parents[3]
report = json.loads((root / "reference/windows-rust-smoke-fp32-sinks-pairwise.json").read_text())
ref = normalized(load_file(str(root / report["reference"])))
candidate = normalized(load_file(str(root / report["candidate"])))
rows = []
for key in report["failures"]:
    delta = (ref[key] - candidate[key]).abs()
    flat = int(delta.flatten().argmax())
    row_index = flat // math.prod(delta.shape[1:])
    layer = int(re.search(r"layer\.(\d+)", key)[1])
    decode = re.match(r"decode\.(\d+)\.", key)
    prefix = decode[0] if decode else ""
    history = []
    for previous in range(layer + 1):
        stage = prefix + ("embedding" if previous == 0 else f"layer.{previous - 1}.hidden")
        a, b = ref[stage][row_index], candidate[stage][row_index]
        d = (a - b).abs()
        history.append({"input_to_layer": previous, "max_abs_error": float(d.max()),
                        "rms_error": float(d.square().mean().sqrt()), "reference_peak": float(a.abs().max())})
    rows.append({"failed_stage": key, "row": row_index, "shape": list(delta.shape),
                 "token_id": int(ref["teacher_tokens"][int(decode[1])]) if decode else int(ref["tokens"][row_index]),
                 "coordinate_flat": flat, "max_abs_error": float(delta.max()), "hidden_input_history": history})
out = root / "reference/failed-stage-row-evolution.json"
out.write_text(json.dumps({"source_comparison": "reference/windows-rust-smoke-fp32-sinks-pairwise.json", "rows": rows}, indent=2) + "\n")
print(json.dumps([{k: v for k, v in row.items() if k != "hidden_input_history"} for row in rows], indent=2))
