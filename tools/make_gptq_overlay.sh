#!/usr/bin/env bash
# Build the GPTQ W8G64 overlay used by `falcon-ocr --mode fast --w8-artifact`.
#   1. Capture the mean XᵀX of every body projection input on 12 calibration
#      pages (tools/gptq-calibration-pages.txt), running the FP32 model.
#   2. GPTQ-quantize each body matrix with those Grams, columns in descending
#      activation order with static G64 scales (tools/w8_variants.py).
# About 10 min for step 1 and 10 min for step 2 on a 16-core desktop.
#   BIN=... GRAM_DIR=... OUT=... bash tools/make_gptq_overlay.sh
set -eu
bin=${BIN:-target/release/falcon-ocr-eval.exe}
pages=${PAGES:-tools/gptq-calibration-pages.txt}
gram=${GRAM_DIR:-artifacts/w8/gram}
out=${OUT:-artifacts/model/w8-gptq.safetensors}  # the default overlay of --mode fast
mapfile -t P < <(tr -d '\r' < "$pages")
"$bin" --threads "${THREADS:-16}" --profile reference capture-gram "${P[@]}" \
    --max-new-tokens 512 --output "$gram"
python tools/w8_variants.py --output "$out" --gram-dir "$gram" --method gptq --group 64 --act-order
sha256sum "$out"
