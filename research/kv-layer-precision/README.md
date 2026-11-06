# KV precision by layer and kind

A mixed Q8/Q16 cache qualifies as a default candidate on this ten-page
pack, while the setting stays opt-in. The chosen set saves 20.275482 percent
of near-exact prefix-cache bytes (77.062144 MB on the journal page) and
reduces CPU decode time by 9.282 percent without speculation and 5.850
percent with the draft head. Micro CER is 21.96551084739477 percent for
both candidate and baseline, with no new non-EOS endings. Nine of ten
pages have identical tokens; one changes tokens with the same CER.

The qualified setting includes both `--tune decode-exp=exact` and the
chosen `kv-layers` recipe below. These results use a Ryzen 7 7700X on Linux;
there is no held-out qualification or default change in this experiment.

This experiment measures each key and value cache's sensitivity before
changing the CPU runner. It uses the checkpoint's own FP32 PyTorch model,
TF32 disabled, and IEEE FlexAttention on the ten pages selected for this
experiment. Each page has at most 512 generated reference tokens; the
teacher-forced passes exclude the first token, which prefill produces.

Every 32-element cache block uses an absmax step rounded upward to BF16.
The signed code limits are 32,767, 127 and 7 for q16, q8 and unrotated q4.
The FP32 recomputation checks the simulation against the model's own
FlexAttention decode. Near-exact policies use q16 everywhere except one
layer's keys or values at q8. Fast sensitivity uses q8 everywhere except
one layer and kind at q4; this is report-only and keeps model weights in
FP32. For a single changed pair,
output error is the mean per-step relative L2 error of that layer's
attention output against the FP32 cache using the same query.

The cumulative curve ranks the 44 near-exact policies by step-weighted
mean KL and takes the shortest prefix saving at least 10, 15, 20, 30 and
40 percent of prefix-cache bytes. All-q8 is the 48.485 percent endpoint.
A key change saves 768 bytes per position across eight KV groups; a value
change saves 512. At 6,544 positions those savings are 5.026 and 3.351 MB.

## Running the simulation

Run from the repository root with the checkpoint and canonical page images
already present. The script needs PyTorch with CUDA, transformers,
safetensors, Pillow and NumPy. Each output is retained by page and policy;
repeating a command runs only missing policies.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 python \
  research/kv-layer-precision/kv_layer_precision.py run \
  --model artifacts/model --output artifacts/kv-layer-run \
  --policies fp32,all-q16,all-q8,all-q4,ne --threads 2 \
  --max-new-tokens 512 <page.png> ...
python research/kv-layer-precision/kv_layer_precision.py rank \
  artifacts/kv-layer-run --output artifacts/kv-layer-sets.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 python \
  research/kv-layer-precision/kv_layer_precision.py run \
  --model artifacts/model --output artifacts/kv-layer-run \
  --policies sets:artifacts/kv-layer-sets.json --threads 2 \
  --max-new-tokens 512 <page.png> ...
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 python \
  research/kv-layer-precision/kv_layer_precision.py run \
  --model artifacts/model --output artifacts/kv-layer-run \
  --policies fast --threads 2 --max-new-tokens 512 <page.png> ...
python research/kv-layer-precision/kv_layer_precision.py summarize \
  artifacts/kv-layer-run --sets artifacts/kv-layer-sets.json \
  --cpu-reference artifacts/verify/pr/agree/reference.json \
  --output artifacts/kv-layer-summary.json
```

The run and summarize files record the GPU identity, FP32 dtype,
FlexAttention check, token IDs, finish reason, forced steps, flips, mean
and maximum KL, output error for single-pair policies, and elapsed time.
The summary compares GPU and CPU reference tokens and applies the curve
exit rule. The CPU calibration and any later CPU runner measurements are
separate from that simulated result.

## Decision rules

The simulated all-q8 mean KL must agree with the CPU FP32-weight q8-cache
profile within a factor of three. Among curve points with mean KL at most
1e-6 and flips at most all-q16 plus one, the largest must save at least
15 percent of near-exact cache bytes. Otherwise there is no Rust change.
A qualifying set proceeds to CPU testing with its next smaller and larger
curve neighbors. In descending saved-byte order, a set needs CPU mean KL
at most 1e-6, flips at most the same-binary baseline plus one, and at least
5 percent faster median decode without speculation. The first admitted set
passing the ten-page text check is the candidate; later sets are not checked.
Text passes with at most 0.25 percentage points extra micro CER and no
new non-EOS ending. No token-changing setting becomes a default here.

## GPU sensitivity and CPU calibration

Measured on 2026-11-06 with a GeForce RTX 5090 for the FP32 simulation
and a Ryzen 7 7700X on Linux for CPU calibration. All ten FP32 passes
had zero flips and absolute mean KL below 1e-8; all 4,674 GPU reference
tokens match the CPU anchors exactly. The simulation excludes ten
prefill-produced first tokens, leaving 4,664 forced decode steps.
Tiny negative FP32/q16 KL values are retained from the FP32 accumulation.

Raw results are in `artifacts/kv-layers-2026-11-05/`. Per-policy `seconds`
measure simulation-loop elapsed time ending with CUDA synchronization;
they are not GPU variant latency measurements or CPU speed outcomes.
CPU calibration used the unchanged `a6b1107` eval binary, `kv-q8`,
`--exp fast --threads 16 --tune decode-exp=exact`, 512 steps per page,
and the existing FP32 reference and top-k files.

### Calibration

| Policy | Sim flips | Sim mean KL | CPU profile | CPU flips | CPU mean KL | Sim / CPU KL |
|---|---|---|---|---|---|---|
| all-q16 | 0 | -3.24686188e-10 | - | - | - | - |
| all-q8 | 1 | 8.19696791e-06 | kv-q8 | 1 | 6.6253167e-06 | 1.237219 |
| all-q4 | 15 | 0.00229701183 | - | - | - | - |

### Near-exact sensitivity

| Rank | Layer | Kind | MB saved, journal | Flips | Mean KL | Output error | Pages | Steps |
|---|---|---|---|---|---|---|---|---|
| 1 | 7 | k | 5.025792 | 0 | 2.6304276e-08 | 0.000741572511 | 10 | 4664 |
| 2 | 10 | v | 3.350528 | 0 | 2.69838109e-08 | 0.0012317176 | 10 | 4664 |
| 3 | 7 | v | 3.350528 | 0 | 3.09617832e-08 | 0.000891752509 | 10 | 4664 |
| 4 | 0 | k | 5.025792 | 0 | 3.99610235e-08 | 0.00159893511 | 10 | 4664 |
| 5 | 2 | v | 3.350528 | 1 | 4.2257704e-08 | 0.00198081056 | 10 | 4664 |
| 6 | 5 | k | 5.025792 | 0 | 4.39315807e-08 | 0.00101113845 | 10 | 4664 |
| 7 | 10 | k | 5.025792 | 1 | 4.76600704e-08 | 0.00124033596 | 10 | 4664 |
| 8 | 6 | k | 5.025792 | 1 | 5.09934833e-08 | 0.000942572789 | 10 | 4664 |
| 9 | 3 | v | 3.350528 | 0 | 5.8000376e-08 | 0.00186271429 | 10 | 4664 |
| 10 | 8 | k | 5.025792 | 1 | 6.06927905e-08 | 0.00070117562 | 10 | 4664 |
| 11 | 6 | v | 3.350528 | 0 | 6.22592653e-08 | 0.00124244802 | 10 | 4664 |
| 12 | 20 | v | 3.350528 | 0 | 6.72974284e-08 | 0.00131373159 | 10 | 4664 |
| 13 | 2 | k | 5.025792 | 1 | 6.77658244e-08 | 0.00182892733 | 10 | 4664 |
| 14 | 9 | v | 3.350528 | 0 | 6.78983695e-08 | 0.00108460726 | 10 | 4664 |
| 15 | 9 | k | 5.025792 | 0 | 6.89726787e-08 | 0.000875894778 | 10 | 4664 |
| 16 | 0 | v | 3.350528 | 0 | 6.99951367e-08 | 0.00230416616 | 10 | 4664 |
| 17 | 1 | v | 3.350528 | 0 | 7.66167239e-08 | 0.0015108684 | 10 | 4664 |
| 18 | 4 | v | 3.350528 | 0 | 7.71839821e-08 | 0.00147181953 | 10 | 4664 |
| 19 | 5 | v | 3.350528 | 0 | 8.51779961e-08 | 0.00125461226 | 10 | 4664 |
| 20 | 8 | v | 3.350528 | 0 | 9.11041336e-08 | 0.00130051898 | 10 | 4664 |
| 21 | 18 | v | 3.350528 | 0 | 1.07399642e-07 | 0.00157227411 | 10 | 4664 |
| 22 | 17 | v | 3.350528 | 0 | 1.10879592e-07 | 0.00174764924 | 10 | 4664 |
| 23 | 4 | k | 5.025792 | 0 | 1.14066011e-07 | 0.00126520016 | 10 | 4664 |
| 24 | 20 | k | 5.025792 | 0 | 1.14305613e-07 | 0.00126453934 | 10 | 4664 |
| 25 | 3 | k | 5.025792 | 1 | 1.15556952e-07 | 0.00154894907 | 10 | 4664 |
| 26 | 18 | k | 5.025792 | 1 | 1.298655e-07 | 0.0016555071 | 10 | 4664 |
| 27 | 11 | v | 3.350528 | 1 | 1.41273786e-07 | 0.00143916242 | 10 | 4664 |
| 28 | 19 | k | 5.025792 | 1 | 1.47304556e-07 | 0.000972195548 | 10 | 4664 |
| 29 | 11 | k | 5.025792 | 0 | 1.5038164e-07 | 0.00126546649 | 10 | 4664 |
| 30 | 14 | k | 5.025792 | 0 | 1.5773425e-07 | 0.000773876133 | 10 | 4664 |
| 31 | 12 | v | 3.350528 | 0 | 1.73750073e-07 | 0.00278639623 | 10 | 4664 |
| 32 | 19 | v | 3.350528 | 0 | 1.98503803e-07 | 0.0011035556 | 10 | 4664 |
| 33 | 21 | v | 3.350528 | 0 | 2.06902526e-07 | 0.00326791895 | 10 | 4664 |
| 34 | 1 | k | 5.025792 | 1 | 2.4365337e-07 | 0.00125737231 | 10 | 4664 |
| 35 | 17 | k | 5.025792 | 1 | 2.7053321e-07 | 0.00199835914 | 10 | 4664 |
| 36 | 13 | v | 3.350528 | 1 | 2.76999713e-07 | 0.00191158046 | 10 | 4664 |
| 37 | 12 | k | 5.025792 | 1 | 2.81639269e-07 | 0.00373900176 | 10 | 4664 |
| 38 | 13 | k | 5.025792 | 0 | 3.44893332e-07 | 0.0017770539 | 10 | 4664 |
| 39 | 15 | k | 5.025792 | 0 | 3.72526532e-07 | 0.000938033134 | 10 | 4664 |
| 40 | 14 | v | 3.350528 | 0 | 4.68311999e-07 | 0.00152368031 | 10 | 4664 |
| 41 | 15 | v | 3.350528 | 1 | 5.30231198e-07 | 0.00121880238 | 10 | 4664 |
| 42 | 16 | v | 3.350528 | 1 | 5.84067865e-07 | 0.00167106448 | 10 | 4664 |
| 43 | 21 | k | 5.025792 | 1 | 6.1076218e-07 | 0.00375719238 | 10 | 4664 |
| 44 | 16 | k | 5.025792 | 1 | 1.31113815e-06 | 0.00167811109 | 10 | 4664 |

### Curve

| Target | Entries | Knob | MB saved, journal | KV bytes saved % | Sim flips | Sim mean KL | Exit rule |
|---|---|---|---|---|---|---|---|
| 10 | 9 | 7k,10v,7v,0k,2v,5k,10k,6k,3v | 38.531072 | 10.137741 | 1 | 3.786689928615287e-07 | passes fidelity |
| 15 | 14 | 7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v | 58.634240 | 15.426997 | 1 | 6.937630010560472e-07 | passes fidelity |
| 20 | 19 | 7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v,9k,0v,1v,4v,5v | 77.062144 | 20.275482 | 1 | 9.98050886168736e-07 | passes fidelity |
| 30 | 28 | 7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v,9k,0v,1v,4v,5v,8v,18v,17v,4k,20k,3k,18k,11v,19k | 115.593216 | 30.413223 | 1 | 1.9961947857164413e-06 | fails fidelity |
| 40 | 37 | 7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v,9k,0v,1v,4v,5v,8v,18v,17v,4k,20k,3k,18k,11v,19k,11k,14k,12v,19v,21v,1k,17k,13v,12k | 154.124288 | 40.550964 | 1 | 3.930914924993017e-06 | fails fidelity |
| all | 44 | 0k,0v,1k,1v,2k,2v,3k,3v,4k,4v,5k,5v,6k,6v,7k,7v,8k,8v,9k,9v,10k,10v,11k,11v,12k,12v,13k,13v,14k,14v,15k,15v,16k,16v,17k,17v,18k,18v,19k,19v,20k,20v,21k,21v | 184.279040 | 48.484848 | 1 | 8.196967912409889e-06 | fails fidelity |

### FP32 sanity and CPU reference tokens

| Page | Input tokens | GPU outputs | Forced steps | FP32 flips | FP32 mean KL | FP32 sanity | CPU common prefix | CPU outputs | Finish |
|---|---|---|---|---|---|---|---|---|---|
| 02a1f828bad3aae2 | 7024 | 512 | 511 | 0 | 4.27679864e-10 | PASS | 512 | 512 | length |
| 204617788e13cc1f | 5968 | 409 | 408 | 0 | 3.60419028e-10 | PASS | 409 | 409 | eos |
| 20cb71d7734f053a | 7216 | 512 | 511 | 0 | -9.71770442e-10 | PASS | 512 | 512 | length |
| 3207aa8d68652b99 | 6544 | 512 | 511 | 0 | -2.99900799e-10 | PASS | 512 | 512 | length |
| 3549bba42cf61ba8 | 6352 | 512 | 511 | 0 | -3.59493324e-09 | PASS | 512 | 512 | length |
| 539f9422d3c1a563 | 6928 | 205 | 204 | 0 | -6.62780178e-11 | PASS | 205 | 205 | eos |
| 696f39cdc7378704 | 6928 | 476 | 475 | 0 | -8.64321614e-10 | PASS | 476 | 476 | eos |
| 754444afd4659b07 | 7120 | 512 | 511 | 0 | -6.89491853e-10 | PASS | 512 | 512 | length |
| 92fe997f26aa2304 | 6544 | 512 | 511 | 0 | -3.11033976e-09 | PASS | 512 | 512 | length |
| aa8e1119117b1822 | 6448 | 512 | 511 | 0 | -2.81111773e-10 | PASS | 512 | 512 | length |

## CPU mixed-cache results

All four agreement arms use `w16-body-kv-q16`, `--exp fast --threads 16`
and `--tune decode-exp=exact`, over 4,674 steps. The chosen set gives one
flip and mean KL 9.888471963753022e-7, below the fixed 1e-6 limit. The
larger set fails fidelity. The chosen and smaller sets pass speed and
fidelity, so the chosen set is checked first and passes; the smaller set
receives no free-running check.

The qualified recipe is:

```text
--tune decode-exp=exact
--tune kv-layers=7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v,9k,0v,1v,4v,5v
```

The plan labels `kv q16 (q8 keys=0x7e5, values=0x1006ff)` and reports
46,304 cache bytes per position instead of 58,080. The masks appear in
serialized plans and resumed-book configuration matching. Unset masks
use the existing cache path. The unset journal fixture gate and real-page
tokens in exact, near-exact and fast all reproduce the baseline.

Timing uses the packed near-exact route, 6,544 journal input tokens, a
300-token cap, eight decode threads, three interleaved rounds and fresh
processes. The required host snapshot has no unrelated process above
2.1 percent CPU. Each arm's speculation-off and default-head token IDs
match. The median total with drafting is 9,523.320357 ms for the chosen
set versus 9,638.316751 ms for baseline; the decode gain does not describe
the whole page's gain. The free-running text check uses the normal
product defaults, the ten selected pages and a 4,096-token cap.

### Agreement, ten pages

| Arm | Knob | KV bytes saved % | Flips | Mean KL | Mean KL / base |
|---|---|---|---|---|---|
| base | unset | 0.000000 | 0 | 7.890447467891552e-09 | 1 |
| chosen | 7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v,9k,0v,1v,4v,5v | 20.275482 | 1 | 9.888471963753022e-07 | 125.322068 |
| smaller | 7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v | 15.426997 | 1 | 6.694997024242541e-07 | 84.8493961 |
| larger | 7k,10v,7v,0k,2v,5k,10k,6k,3v,8k,6v,20v,2k,9v,9k,0v,1v,4v,5v,8v,18v,17v,4k,20k,3k,18k,11v,19k | 30.413223 | 2 | 2.062078811695605e-06 | 261.33864 |

### Timing, journal page

| Arm | Decode ms/token, --speculate 0, median [range] | Against base % | Decode ms/token, draft head, median [range] | Against base % | Peak RSS GB, 0 / draft |
|---|---|---|---|---|---|
| base | 13.527876103 [13.518871187, 13.641757217] | +0.000000 | 7.487585723 [7.478222927, 7.509846677] | +0.000000 | 1.840680960 / 1.864192000 |
| chosen | 12.272225567 [12.219238163, 12.311681413] | -9.281949 | 7.049554940 [7.013084110, 7.084802507] | -5.850094 | 1.833025536 / 1.855987712 |
| smaller | 12.629086377 [12.515174257, 12.652274847] | -6.643983 | 7.156441387 [7.138167903, 7.295440940] | -4.422578 | 1.861271552 / 1.884602368 |
| larger | 11.574448303 [11.569657980, 11.691065103] | -14.440018 | 6.801929560 [6.790585727, 6.821125547] | -9.157240 | 1.833082880 / 1.856249856 |

RSS is the maximum observed peak across the three fresh processes for each
speculation setting, in decimal GB. Negative against-base percentages mean
faster decode. All six token sequences per arm are identical.

### Ten-page text check

| Arm | Micro CER % | Baseline micro CER % | New non-EOS stops | Result |
|---|---|---|---|---|
| chosen | 21.96551084739477 | 21.96551084739477 | 0 | PASS |

## Fast sensitivity (report only)

All 44 fast cache policies cover ten pages and 4,664 forced decode steps
per policy. These passes change one layer's keys or values from q8 to
unrotated q4 with model weights in FP32. No q4 Rust work follows them.

The complete GPU study has 97 policies on all ten pages: four baseline
policies, 44 near-exact sensitivities, five curve sets and 44 fast
sensitivities. The all-q8 curve endpoint reuses its baseline. There are
970 completed policy outputs plus ten reference records. Total active
GPU worker wall is 6,476.87 seconds (107m 56.87s), including the initial
interrupted startup and excluding lock waiting, within the 7,200-second
cap. The fast batches take 1,287.53 and 1,502.96 seconds and both exit
successfully. No fast pass is dropped.

| Layer | Kind | Flips | Mean KL | Output error | Pages | Steps |
|---|---|---|---|---|---|---|
| 0 | k | 1 | 2.32339944e-05 | 0.0268874374 | 10 | 4664 |
| 0 | v | 1 | 2.54813858e-05 | 0.0431042667 | 10 | 4664 |
| 1 | k | 1 | 4.93565361e-05 | 0.0223181284 | 10 | 4664 |
| 1 | v | 1 | 3.75245931e-05 | 0.0319741355 | 10 | 4664 |
| 2 | k | 1 | 4.04514672e-05 | 0.0295393001 | 10 | 4664 |
| 2 | v | 1 | 2.81734901e-05 | 0.0372024858 | 10 | 4664 |
| 3 | k | 2 | 3.10838856e-05 | 0.0267789512 | 10 | 4664 |
| 3 | v | 2 | 2.85086027e-05 | 0.0346103064 | 10 | 4664 |
| 4 | k | 1 | 3.61046287e-05 | 0.0200111258 | 10 | 4664 |
| 4 | v | 3 | 3.86234637e-05 | 0.0270806155 | 10 | 4664 |
| 5 | k | 1 | 2.47063014e-05 | 0.0161961714 | 10 | 4664 |
| 5 | v | 2 | 3.73732002e-05 | 0.0233686349 | 10 | 4664 |
| 6 | k | 2 | 1.74265669e-05 | 0.0156739242 | 10 | 4664 |
| 6 | v | 3 | 3.75312196e-05 | 0.0227398045 | 10 | 4664 |
| 7 | k | 2 | 1.7964383e-05 | 0.0116559749 | 10 | 4664 |
| 7 | v | 2 | 2.37542429e-05 | 0.0163579846 | 10 | 4664 |
| 8 | k | 2 | 2.71590957e-05 | 0.0123399774 | 10 | 4664 |
| 8 | v | 4 | 4.92545036e-05 | 0.0258326422 | 10 | 4664 |
| 9 | k | 1 | 3.08801125e-05 | 0.014855314 | 10 | 4664 |
| 9 | v | 1 | 4.04599494e-05 | 0.0197795444 | 10 | 4664 |
| 10 | k | 1 | 1.71741753e-05 | 0.0220722408 | 10 | 4664 |
| 10 | v | 3 | 2.1671882e-05 | 0.0253047654 | 10 | 4664 |
| 11 | k | 1 | 4.91276278e-05 | 0.0192648229 | 10 | 4664 |
| 11 | v | 4 | 7.8978043e-05 | 0.0307735929 | 10 | 4664 |
| 12 | k | 1 | 5.79394889e-05 | 0.0424630408 | 10 | 4664 |
| 12 | v | 4 | 5.73716757e-05 | 0.0515884784 | 10 | 4664 |
| 13 | k | 2 | 0.000128587628 | 0.0300520827 | 10 | 4664 |
| 13 | v | 2 | 7.62341866e-05 | 0.0352420716 | 10 | 4664 |
| 14 | k | 3 | 6.39678995e-05 | 0.0145124151 | 10 | 4664 |
| 14 | v | 5 | 0.000190491184 | 0.0283303211 | 10 | 4664 |
| 15 | k | 1 | 0.000174749266 | 0.0176554395 | 10 | 4664 |
| 15 | v | 5 | 0.000151001214 | 0.0229787806 | 10 | 4664 |
| 16 | k | 7 | 0.000315371396 | 0.0268029118 | 10 | 4664 |
| 16 | v | 4 | 0.000226295226 | 0.0316052156 | 10 | 4664 |
| 17 | k | 1 | 6.67391425e-05 | 0.0295198493 | 10 | 4664 |
| 17 | v | 1 | 4.10529514e-05 | 0.0340932209 | 10 | 4664 |
| 18 | k | 2 | 3.95150272e-05 | 0.0253241323 | 10 | 4664 |
| 18 | v | 2 | 4.38296355e-05 | 0.0297843981 | 10 | 4664 |
| 19 | k | 2 | 4.3915724e-05 | 0.0151464513 | 10 | 4664 |
| 19 | v | 3 | 0.000145983441 | 0.0286898645 | 10 | 4664 |
| 20 | k | 1 | 3.70855873e-05 | 0.0191920111 | 10 | 4664 |
| 20 | v | 2 | 3.1563287e-05 | 0.0249350004 | 10 | 4664 |
| 21 | k | 6 | 0.000128799094 | 0.0571362865 | 10 | 4664 |
| 21 | v | 1 | 7.78110969e-05 | 0.0620737634 | 10 | 4664 |
