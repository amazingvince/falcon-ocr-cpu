# Attempt 3: first measured screen (Stage 1)

**Status:** measured on one page. Not quality-qualified beyond token agreement on that page.

## Setup

- **Host:** Ryzen 9 7950X, 64 GB DDR5-4800, Windows 11, 16 threads, `--backend avx2`.
- **Binary:** `falcon-ocr-attempt` built from commit `49d7c19` plus the memory probe and the `w8-body-kv-q8` profile. The prebuilt copy is `artifacts/phase4/bin/falcon-ocr-attempt-stage1.exe`.
- **Code under test:** the external patch's kernels as delivered. None of the later rework was included.
- **Workload:** journal page `artifacts/corpus/v3/3f294b5e60a0c2d4`, max-dim 1536 (6,544 prompt tokens), 4,096-token budget.
- **Run schedule:** `attempt3/bench.py --schedule interleaved --control-every 2`. Each arm is one fresh process with 1 warmup and 2 samples. Arms ran in this order:

  | Position | Arms |
  |---|---|
  | 1 | reference |
  | 2 | hygiene, split-f32 |
  | 3 | reference |
  | 4 | w8-body, kv-bf16 |
  | 5 | reference |
  | 6 | w8-body-kv-bf16, w8-all |
  | 7 | reference |
  | 8 | kv-q8, w8-body-kv-q8 |
  | 9 | reference |

- **Weights:** W8 profiles used the offline overlays in `artifacts/phase4/w8/`.
  - The overlay traces are byte-identical to runtime quantization for body and body+head.
  - This means the Python converter and the Rust quantizer agree on the full checkpoint.
- **Raw data:** `artifacts/phase4/stage1-journal/` (`summary.json`, `summary.txt`, and per-arm reports).

## Results

"Ctrl" is the mean of the neighbouring reference controls. The five reference controls measured 65.8–68.9 s wall time and 47.1–49.2 ms/token. Neighbouring controls drifted by at most 2%.

| Profile | Ctrl s | Cand s | Δ | Decode ms/tok | Tokens vs FP32 | CER vs FP32 | CER vs truth (FP32 → cand) | Peak RSS |
|---|---:|---:|---:|---:|---|---:|---|---:|
| hygiene | 67.5 | 76.1 | +12.8% | 55.4 | identical | 0 | 14.81% → 14.81% | 2.29 GB |
| split-f32 | 67.5 | 82.8 | +22.7% | 61.3 | identical | 0 | same | 2.31 GB |
| w8-body | 66.8 | 56.1 | **−16.1%** | 37.5 | identical | 0 | same | 2.47 GB |
| kv-bf16 | 66.8 | 75.1 | +12.5% | 54.5 | identical | 0 | same | 2.30 GB |
| w8-body-kv-bf16 | 67.0 | 58.4 | −12.9% | 41.3 | identical | 0 | same | 2.48 GB |
| w8-all | 67.0 | 52.3 | −22.0% | 35.4 | first diff at token 609; token edit distance 10 | 0.24% | 14.81% → 14.55% | 2.52 GB |
| kv-q8 | 66.7 | 59.2 | −11.2% | 42.5 | identical | 0 | same | 2.29 GB |
| w8-body-kv-q8 | 66.7 | 46.9 | **−29.7%** | 31.2 | identical | 0 | same | 2.47 GB |

## Reading

- **W8 transformer weights:** about 10 ms/token saved even with the patch's naive kernel (single accumulator, per-call finite scans). No token changed on this page.
- **Quantizing the vocabulary head** is the only change that altered tokens. The replacement is the exact screened head, which gives the same byte saving with no token change; see `src/head_screen.rs`.
- **The patch's split-prefix consumer is slower than the compact kernel.** It rebuilds each key and value on the stack and makes indirect calls per key.
  - `split-f32` costs +23%.
  - The byte saving of BF16 does not recover that cost; the bigger Q8 saving does.
  - The group-major pair kernel in `src/attempt/prefix.rs` replaces this consumer.
- **Hygiene:** its +13% is not attributed. Its first sample overlapped editor and script activity on the host, and its two samples disagree by 11%. It is re-measured on a quiet host.
- **Prefill** varies 10.3–12.7 s even among the reference controls. Compare prefill only within a bracket.
