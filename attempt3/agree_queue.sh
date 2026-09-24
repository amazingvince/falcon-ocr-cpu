#!/usr/bin/env bash
# Run `falcon-ocr-attempt agree` arms one after another (one machine job at a time).
#   attempt3/agree_queue.sh <out-dir> <pages-file> <max-steps> <name>=<profile>[=<w8-artifact>] ...
# Every arm uses the fast prefill exp and the FP32 calibration reference tokens.
set -u
out=$1 pages=$2 steps=$3
shift 3
bin=${AGREE_BIN:-./target/release/falcon-ocr-eval.exe}
ref=${AGREE_REFERENCE:-artifacts/phase4/checks/calibration-reference.json}
mkdir -p "$out"
mapfile -t PAGES < <(tr -d '\r' < "$pages")
for arm in "$@"; do
  IFS='=' read -r name profile artifact <<< "$arm"
  extra=()
  [ -n "${artifact:-}" ] && extra=(--w8-artifact "$artifact")
  rm -f "$out/$name.json"
  start=$(date +%s)
  FALCON_OCR_EXP=fast "$bin" --threads "${AGREE_THREADS:-16}" --backend avx2 --profile "$profile" "${extra[@]}" \
    agree "${PAGES[@]}" --reference "$ref" --max-steps "$steps" --report "$out/$name.json" \
    > "$out/$name.stdout" 2> "$out/$name.stderr"
  echo "$name exit=$? seconds=$(( $(date +%s) - start )) $(tail -c 200 "$out/$name.stdout")"
done
