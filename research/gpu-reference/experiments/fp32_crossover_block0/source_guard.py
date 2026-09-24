"""Text-only guard against the preserved layer7 observer; no imports or I/O at load."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ENTRY_CHECK = '''                    require(torch.equal(tensors[branch + ".input"].view(torch.int32), input_state.view(torch.int32)),
                            "Captured full entry bits differ")
'''


def unique_slice(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("Native fragment anchor missing or ambiguous")
    begin, finish = source.index(start), source.index(end)
    if finish <= begin:
        raise ValueError("Native fragment anchors reversed")
    return source[begin:finish]


def check_text(old_adapter, adapter, old_gpu, gpu):
    expected_adapter = old_adapter.replace("layer7", "block0").replace("layer.7.", "layer.0.").replace("7..=7", "0..=0")
    if adapter != expected_adapter:
        raise ValueError("Adapter differs beyond single selected block and observation labels")
    begin = "        module = import_model(model_dir)\n"
    end = "        stable()\n        report.update(status="
    original = unique_slice(old_gpu, begin, end)
    expected = original.replace("layer7", "block0").replace("layer.7.", "layer.0.").replace('model.layers["7"]', 'model.layers["0"]')
    actual = unique_slice(gpu, begin, end)
    if actual.count(ENTRY_CHECK) != 1:
        raise ValueError("Expected exactly one passive full-entry bit check")
    if actual.replace(ENTRY_CHECK, "") != expected:
        raise ValueError("Native GPU setup, hooks, arithmetic, controls or branch order changed")
    if gpu.index("cpu_gate = accepted_cpu_execution(args, plan, bound)") >= gpu.index("from reference_preflight import preflight"):
        raise ValueError("Accepted CPU receipt must precede any Torch/CUDA preflight")
    for literal in ('gpu_state = saved(g, "embedding")', 'cpu_state = saved(c, "embedding")'):
        if gpu.count(literal) != 1:
            raise ValueError("Original complete embedding selection changed")
    return {"rust_adapter_scope_only": True, "gpu_native_fragment_preserved": True,
            "cpu_receipt_precedes_cuda_preflight": True, "gpu_branch_count": 2,
            "rust_selected_block_count": 1}


def check_sources():
    prior = ROOT / "experiments/fp32_crossover_layer7"
    current = ROOT / "experiments/fp32_crossover_block0"
    return check_text((prior / "adapter.py").read_text(encoding="utf-8"),
                      (current / "adapter.py").read_text(encoding="utf-8"),
                      (prior / "export_gpu.py").read_text(encoding="utf-8"),
                      (current / "export_gpu.py").read_text(encoding="utf-8"))
