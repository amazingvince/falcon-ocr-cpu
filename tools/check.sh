#!/usr/bin/env bash
# Format, lint and unit-test gate; `--smoke` also rebuilds the release binary and
# checks the exact-mode smoke trace hash (needs artifacts/model and the fixture).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all -- --check
cargo clippy --release --all-targets --locked -- -D warnings
cargo test --release --lib --locked
if [[ "${1:-}" == "--smoke" ]]; then
  cargo build --release --locked --bins
  out="$(mktemp -d)"
  target/release/falcon-ocr --threads 4 --backend avx2 trace \
    --fixture artifacts/reference/smoke-fp32/trace.safetensors \
    --output "$out/trace.safetensors" --max-new-tokens 17 > /dev/null
  hash="$(sha256sum "$out/trace.safetensors" | cut -c1-8)"
  rm -rf "$out"
  if [[ "$hash" != "e2dad223" ]]; then
    echo "smoke trace hash $hash, expected e2dad223" >&2
    exit 1
  fi
  echo "smoke trace ok ($hash)"
fi
