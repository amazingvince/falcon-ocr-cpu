# Held downstream FP32 crossover

This new experiment implements the prospective scope in
`reference/fp32-crossover-downstream-next-v1.md`. Its sources are prepared during
a live quiet benchmark. **No import, test, plan preparation, payload hash/load,
build, model call or GPU work has been executed for this version.** Existing
experiment sources and results remain frozen. Independent source review and
root's scheduling release are required before any commands below.

The original target is `layer.9.v[112,14,2]`. Existing captures provide A=GPU89(G7),
C=GPU89(C7), D=Rust89(C7), and S=GPU7(CPU6). Complete entry states are `[144,768]`.
This experiment performs precisely two GPU suffix evaluations:

1. `control_c`: replay F(C7). All 17 captured stages, including the input, must
   reproduce the earlier GPU `cpu_state.*` raw bits before the next branch.
2. `substitution_s`: run F(S), once, then stop. S is loaded unchanged from the
   completed layer-7 GPU archive. No layer-7 or Rust call is performed.

F contains unchanged native layer8 and layer9 `_pre_attention_qkv`. The latter
retains input RMS, QKV projection, Q/K head RMS and GQA expansion. It ends at V;
there is no layer9 rotary/attention/FFN or later model execution. Pass-through
observations and native setup/call/cleanup fragments are copied from the pinned
prior exporter. A prospective source guard compares those fragments exactly
after indentation normalization and bounds the two native call sites. This
does not prove emitted machine-code identity. Runtime control equality remains
mandatory; observation/allocation history differences remain explicit.

The immutable contract fixes full144 rows, original token/position/BlockMask,
fresh cache256 per branch, pinned native checkpoint/dependencies, strict FP32,
TF32 and reduced-precision reductions off, Flex IEEE, OMP/MKL8 and isolated4090
UUID. Actual Torch settings, device/driver/package/interpreter metadata and host
thread environment must match the prior GPU capture (free memory may differ).
The 5090 is not used. No production files or frozen tolerance are changed.

The evidence loader validates 93 earlier stage payloads, 22 original controls,
eight complete state joins, the selected G7/C7/S raw hashes and original
mask/position bytes. It rechecks preserved CPU execution/build archives and
accepted independent-review identities. Old current live Rust source is not
treated as retrospective historical startup evidence. Existing gaps persist.

After review and root release, the following are separate, manually scheduled
steps. Flags record authorization; they do not create it. Every output path
must be fresh; there is no retry or variant loop.

```text
python -m unittest discover -s experiments/fp32_crossover_downstream -p test_source.py -v
python experiments/fp32_crossover_downstream/prepare.py --quiet-window-released --output artifacts/diagnostics/fp32-crossover-downstream-plan-v1
```

Preparation checks existing input hashes and source fragments, archives source
bytes, freezes plan/runtime/asset identities, then rechecks the closure. It does
not create a Rust build. Run the reviewed `export_gpu.py --plan <plan>
--plan-sha256 <hash> --output <fresh-output> --execute-reviewed-plan` only through
the isolated pinned WSL reference environment. The exporter invokes project
preflight before native work and preserves rejected outputs without starting
the substitution if the control fails. Do not execute during benchmark timing.

Finally `compare.py` takes explicit plan/GPU report paths and hashes, a fresh
`--output` and `--execute-reviewed-plan`. It independently revalidates historical
evidence, all34 new arrays, the complete C7/S entries and all17 actual control
arrays, then computes FP64:

`D-A = (D-C) + (C-B) + (B-A)`.

Reports contain the fixed endpoint, every stage/row, signed terms, RMS/maxima,
accounting residuals and cancellation. D-C is an accumulated downstream path
difference. C-B and B-A are conditional substitutions through a nonlinear map;
they are not independent additive kernel causes. Signed fractions can be
negative or exceed one. The existing absolute bound is reused only as a labeled
diagnostic comparison; no new qualification gate is invented. Stop after one
valid decomposition or any failed control. No further intervention follows.

Remaining before execution: independent source review; host tests (including
exact copied-fragment checks); authorized plan preparation and hash freeze;
reviewed isolated launch; actual 17-stage control success. These sources do not
claim any of those future results.
