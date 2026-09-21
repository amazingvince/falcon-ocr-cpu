# Independent source-only review

Reviewed the proposal and the text of `contract.py`, `adapter.py`, `rust_arm.rs`, `capture.py`, `export_gpu.py`, `compare.py`, `test_source.py` and `README.md` during the quiet benchmark. This review ran no imports, tests, builds, hashing, tensor loading, inference or model tools. It does not attest runtime success or immutable source hashes; those remain for the subsequent authorized freeze and validation.

One concrete issue was found and corrected by the Rust author: the initial state/control loaders looked up unprefixed CPU keys literally, while the original producer can use `prefill.` and the baseline comparison normalizes that prefix. Rust now requires exactly one of `name` and `prefill.name`, rejecting both missing and ambiguous matches. The GPU arm and offline comparer use the same rule. The new source tests cover the resolver, but were not executed in this review.

Root's earlier GPU runtime-dictionary mismatch is fixed by importing the common contract. The GPU arm now records and checks effective precision/reduction settings and compiled-block state; those source changes were inspected without executing Torch.

The retained operations and control boundaries are consistent in the reviewed source:

- The Rust adapter extracts the original forward text, limits the layer loop to 8 and 9, adds buffer captures, and stops after layer-9 V expansion before its Q/K head normalization. It retains the original layer-8 numerical calls, rotary/residual expressions, expanded cache and four-thread AVX2 path. The live project receives no edit.
- The GPU hooks pass through original operators. Layer-9 `_pre_attention_qkv` contains its native input RMS normalization. Layer-8 mask, positions, full state shape and GPU cache capacity remain those of the original prefill. All six saved GPU-state controls must match by bits before the GPU-on-CPU-state branch executes.
- The Rust capture requires the corresponding six original CPU controls. The offline comparer independently rechecks those actual arrays and all three original entry states, full stage inventories, finite FP32 values and payload hashes before decomposition. It keeps all 144 rows, including row 112.
- The FP64 decomposition reports signed engine/state terms and cancellation without changing the numerical policy. Its later-stage engine term is explicitly not an equal-input isolated-kernel error bound. Historical startup provenance, skipped earlier-layer execution and changed hook/allocation history remain limitations.
- The native build requires a new dedicated Cargo target and checks the emitted manifest, source, test artifact, freshness and target containment. The copied source scope, source archive, binary, logs and final artifact checks are preserved. CPU/GPU failures cannot qualify the offline comparison, and the drivers do not automatically retry them.

No additional concrete source-level blocker was identified after the resolver fix. Original-control bit equality is still a mandatory execution gate: this review does not predict it will pass. Explicit root release after the quiet benchmark is still required for any tests, preparation, compilation, tensor work or GPU execution. No combined numerical variant, tolerance relaxation, performance claim or production change is authorized by this note.
