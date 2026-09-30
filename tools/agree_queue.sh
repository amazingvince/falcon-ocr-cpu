#!/usr/bin/env bash
# Run `falcon-ocr-attempt agree` arms one after another (one machine job at a time).
#   tools/agree_queue.sh <out-dir> <pages-file> <max-steps> <name>=<profile>[=<w8-artifact>] ...
# Every arm uses the fast prefill exp and the FP32 calibration reference tokens.
# AGREE_EXTRA adds runner flags before `agree` (e.g. "--tune decode-exp=exact"); AGREE_ARGS adds `agree`
# arguments after its own (e.g. "--reference-topk fp32-topk.json" to score KL).
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
  # AGREE_EXTRA: more runner flags for every arm (e.g. "--tune decode-exp=fast"); AGREE_ARGS: more
  # `agree` arguments (e.g. "--reference-topk <file>").
  # shellcheck disable=SC2086
  "$bin" --exp fast --threads "${AGREE_THREADS:-16}" --backend avx2 --profile "$profile" "${extra[@]}" ${AGREE_EXTRA:-} \
    agree "${PAGES[@]}" --reference "$ref" --max-steps "$steps" --report "$out/$name.json" ${AGREE_ARGS:-} \
    > "$out/$name.stdout" 2> "$out/$name.stderr"
  echo "$name exit=$? seconds=$(( $(date +%s) - start )) $(tail -c 200 "$out/$name.stdout")"
done
