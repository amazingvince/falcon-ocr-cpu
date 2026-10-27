#!/usr/bin/env bash
# Build the GPTQ W8G64 overlay used by `falcon-ocr --mode fast --w8-artifact`.
#   1. Capture the mean XᵀX of every body projection input on 12 calibration
#      pages (tools/gptq-calibration-pages.txt), running the FP32 model.
#      REUSE_GRAMS=1 skips this and reuses an existing GRAM_DIR, whatever
#      pages it was captured from.
#   2. GPTQ-quantize each body matrix with those Grams, columns in descending
#      activation order with static G64 scales (tools/w8_variants.py).
# About 10 min for step 1 and 10 min for step 2 on a 16-core desktop.
#   BIN=... GRAM_DIR=... OUT=... bash tools/make_gptq_overlay.sh
# Experimental: EXCEPTIONS=N keeps the N highest-energy input columns of each
# matrix matching EXCEPTIONS_INCLUDE (default the W2 projections) unquantized
# in FP32 (format v2, docs/MODES.md); OUT then defaults to
# w8-gptq-excN.safetensors, which fast mode reads only through --w8-artifact.
set -eu
bin=${BIN:-target/release/falcon-ocr-eval.exe}
pages=${PAGES:-tools/gptq-calibration-pages.txt}
gram=${GRAM_DIR:-artifacts/w8/gram}
exceptions=${EXCEPTIONS:-0}
if [[ "$exceptions" == 0 ]]; then
    out=${OUT:-artifacts/model/w8-gptq.safetensors}  # the default overlay of --mode fast
    extra=()
else
    out=${OUT:-artifacts/model/w8-gptq-exc$exceptions.safetensors}
    extra=(--exceptions "$exceptions" --exceptions-include "${EXCEPTIONS_INCLUDE:-feed_forward\.w2}")
fi
if [[ "${REUSE_GRAMS:-0}" == 1 ]]; then
    [[ -d "$gram" ]] || { echo "REUSE_GRAMS=1, but $gram does not exist" >&2; exit 1; }
    echo "reusing the Grams in $gram ($pages is not read)"
else
    mapfile -t P < <(tr -d '\r' < "$pages")
    "$bin" --threads "${THREADS:-16}" --profile reference capture-gram "${P[@]}" \
        --max-new-tokens 512 --output "$gram"
fi
python tools/w8_variants.py --output "$out" --gram-dir "$gram" --method gptq --group 64 --act-order \
    ${extra[@]+"${extra[@]}"}
sha256sum "$out"
