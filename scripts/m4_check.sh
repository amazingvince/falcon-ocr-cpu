#!/usr/bin/env bash
# One-shot portability check for Apple Silicon (M-series) and other aarch64
# machines. Run from the repository root on the phase4-attempt3 branch:
#
#   bash scripts/m4_check.sh            # full run (~30-60 min)
#   QUICK=1 bash scripts/m4_check.sh    # build + tests + one short bench
#
# Prerequisites: rustup (the repo pins 1.94.0), cmake, python3 with numpy and
# safetensors, the pinned model in artifacts/model (python3
# scripts/fetch_reference.py), and the four benchmark pages copied from the
# Windows host (see docs/CPU_PORTABILITY.md, "Running on Apple Silicon").
set -euo pipefail

OUT="artifacts/portability/$(hostname -s)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
log() { echo "== $*" | tee -a "$OUT/summary.txt"; }

log "machine"
{
  uname -a
  if command -v sysctl >/dev/null; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize 2>/dev/null || true
    sysctl hw.optional.arm.FEAT_DotProd hw.optional.arm.FEAT_I8MM \
      hw.optional.arm.FEAT_BF16 hw.optional.arm.FEAT_SME 2>/dev/null || true
  fi
  rustc --version
} | tee -a "$OUT/summary.txt"

PERF_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)
ALL_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)

log "build (release)"
cargo build --release --locked --bins 2>&1 | tail -3 | tee -a "$OUT/summary.txt"

log "unit tests (includes NEON == portable bitwise checks)"
cargo test --release --locked --lib 2>&1 | tee "$OUT/unit-tests.log" | grep -E "test result|FAILED" | tee -a "$OUT/summary.txt"

log "doctor"
./target/release/falcon-ocr doctor | tee -a "$OUT/summary.txt"
./target/release/falcon-ocr-eval doctor | tee -a "$OUT/summary.txt"

if [ -f artifacts/reference/smoke-fp32/trace.safetensors ]; then
  log "model gates: GPU smoke tokens, cache layouts, screened head, zero decode allocations"
  cargo test --release --locked --test gpu_parity --test cache_layout --test head_screen \
    --test decode_allocations -- --ignored 2>&1 | tee "$OUT/integration.log" \
    | grep -E "test result|FAILED|panicked" | tee -a "$OUT/summary.txt"
else
  log "skipping model gates: artifacts/reference/smoke-fp32 not present"
fi

log "memory bandwidth probe"
python3 - <<'PY' | tee -a "$OUT/summary.txt"
import time, numpy as np
a = np.ones(512 * 1024 * 1024 // 4, dtype=np.float32)  # 512 MiB
best = 0.0
for _ in range(5):
    t = time.perf_counter(); s = a.sum(dtype=np.float32); dt = time.perf_counter() - t
    best = max(best, a.nbytes / dt / 1e9)
print(f"single-thread numpy read: {best:.1f} GB/s (multi-thread decode can exceed this)")
PY

IMG=artifacts/corpus/v3/3f294b5e60a0c2d4/canonical-rgb.png
if [ ! -f "$IMG" ]; then
  log "journal page missing ($IMG); copy the benchmark pages first"; exit 0
fi

log "prefill + decode phase profile (FP32 exact and W8+Q8), ${PERF_CORES} and ${ALL_CORES} threads"
for threads in "$PERF_CORES" "$ALL_CORES"; do
  for profile in reference w8-body-kv-q8; do
    ./target/release/falcon-ocr-eval --tune phases=1 --model artifacts/model \
      --threads "$threads" --profile "$profile" --head screened \
      bench "$IMG" --max-new-tokens 200 --warmup 0 --samples 1 \
      --report "$OUT/phases-$profile-t$threads.json" 2>&1 \
      | grep -E "phases" | sed "s/^/$profile t$threads: /" | tee -a "$OUT/summary.txt"
  done
done

if [ "${QUICK:-0}" = "1" ]; then log "QUICK=1: stopping before the bracket"; exit 0; fi

log "bracket with token agreement (journal page, full output)"
python3 attempt3/make_manifest.py "$IMG" --sibling-ground-truth --output "$OUT/cases.json"
python3 attempt3/bench.py --binary target/release/falcon-ocr-eval --model artifacts/model \
  --manifest "$OUT/cases.json" --candidate-args "--head screened" \
  --profiles reference w16-body-kv-q16 w8-body-kv-q8 --schedule interleaved --control-every 2 \
  --warmup 1 --samples 2 --threads "$ALL_CORES" --output "$OUT/bracket" 2>&1 | tail -8 \
  | tee -a "$OUT/summary.txt"

log "done: $OUT/summary.txt"
