#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$ROOT/artifacts/reference/linear-logger-fp32"
if [[ -e "$OUT" ]]; then
    echo "Preserve previous logger attempts: $OUT" >&2
    exit 1
fi
mkdir "$OUT"
cp -- "$0" "$OUT/run_linear_logging_reference.sh"
export CUBLAS_LOGINFO_DBG=1
export CUBLAS_LOGDEST_DBG="$OUT/cublas.log"
export CUBLASLT_LOG_LEVEL=5
export CUBLASLT_LOG_MASK=31
export CUBLASLT_LOG_FILE="$OUT/cublaslt_%i.log"
exec bash "$ROOT/scripts/run_reference.sh" --linear-logger >"$OUT/stdout.log" 2>"$OUT/stderr.log"
