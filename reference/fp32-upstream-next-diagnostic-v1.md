# Next bounded FP32 state crossover

Run **layer 7 only**, starting from the complete saved GPU/CPU **layer-6 hidden
states `[144,768]`**, and stop at **layer-7 hidden**. This partitions the exact
boundary already shown to supply 99.424% of the signed layer-9 V discrepancy.
Do not repeat layer 8 or layer 9 for this follow-up.

Saved row 112 is image token 227, temporal position 0. Its hidden RMS difference
progresses from `2.08586e-5` after layer 3 to `5.95790e-5` after layer 4,
`1.44189e-4` after layer 5, `2.14499e-4` after layer 6, and `6.72061e-4` after
layer 7. The layer-7 row maximum is `0.00229644775390625` at channel 249.
Layer-7 attention RMS difference is `2.46546e-5`; those unequal-input,
differently scaled stage errors do **not** identify a faulty FFN or reduction.
Other rows also carry larger errors, so attention mixing makes a cropped-row
experiment inappropriate.

The smallest controlled continuation has three branches: native GPU on GPU
entry, native Rust on CPU entry, then native GPU on CPU entry. Require the five
available saved layer-7 Q/K/V/attention/hidden tensors to match bit for bit on
both original-state branches. Preserve original mask/positions, all 144 rows,
GPU cache capacity 256, Rust expanded capacity 161, and the established FP32
runtime. Capture the native attention normalization/QKV/rotary/attention/output
projection, residual, FFN normalization/W13/gate/W2 and hidden stages with
pass-through observations. Decompose the output using the existing three-arm
identity, including fixed row 112/channel 249 and complete tensor summaries.

An earlier row-112 increase appears at layer 4, but jumping there would skip
unresolved inter-row mixing. The one-block layer-7 crossover gives a causal
partition immediately upstream of the validated boundary. Reject any failed
control and stop after one valid decomposition. No arithmetic variants,
tolerance changes, model qualification or execution authorization follow from
this note.

This analysis used read-only NumPy views of saved tensors and small FP64 error
summaries. No Torch import, GPU/model run or build occurred. Payload pins,
precise metrics, available controls, proposed fields and provenance limits are
recorded in `fp32-upstream-next-diagnostic-v1.json`.
