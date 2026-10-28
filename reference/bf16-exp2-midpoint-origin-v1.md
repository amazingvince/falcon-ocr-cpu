The sole BF16 exp2 disagreement is a **mixed counterfactual**, not either saved actual argument. It occurs exactly once in the frozen argument inventory: record42 (zero based), AVX512, `prefill.layer.17.row54.head10`, tile1, key22, condition `cpu_score_gpu_max`.

At this lane, the saved independent GPU-oracle score/max produce argument `c0c9799e`; the saved CPU score/max produce `c0c979a0`. Native GPU exp2 and the preserved Rust exp2 agree exactly at both arguments. Their different BF16 probabilities therefore remain attributable to different arguments in this saved comparison.

| Score / maximum source | Argument bits | Native GPU exp2 → BF16 | Rust exp2 → BF16 |
|---|---|---|---|
| GPU oracle / GPU oracle | `c0c9799e` | `3c508005` → `3c51` | `3c508005` → `3c51` |
| GPU oracle / CPU | `c0c9799e` | `3c508005` → `3c51` | `3c508005` → `3c51` |
| CPU / GPU oracle | `c0c9799f` | `3c508000` → `3c50` | `3c508001` → `3c51` |
| CPU / CPU | `c0c979a0` | `3c507ffc` → `3c50` | `3c507ffc` → `3c50` |

For `c0c9799f` (−6.296096324920654), native GPU exp2 lands exactly at the BF16 midpoint; round-to-nearest-even selects `3c50`. Rust returns one FP32 step above it and selects `3c51`. Thus the earlier **Rust-only counterfactual** observation that replacing the CPU maximum with the GPU-oracle maximum restores the GPU BF16 probability does not transfer to native GPU exp2 at that mixed argument.

The standalone replay matched native `ex2.approx.ftz` to Torch exp2 at all1,876 FP32/BF16 outputs; native versus Rust had1,180 FP32 differences and this one BF16 crossing. This result does not establish native fused attention's arguments, denominator, or P×V behavior. The rejected v4 fused capture is excluded; any later accepted observer must be assessed separately.

`bf16-exp2-midpoint-origin-v1.json` binds the replay, argument manifest/records, preserved Rust binary, generating sources, GPU compiled artifacts and original tile tensors. Its SHA is `7e0d246fa04fd07d3d5045dfece9b25c3a1d1add4e0545ff7627d8197909351a`. The CPU-only `experiments/bf16_attention/map_exp2_midpoint.py` independently reconstructed all four same-lane argument bit patterns and checked source hashes before/after. No GPU execution, kernel change or tolerance change occurred.
