#!/usr/bin/env bash
set -euo pipefail
set -C
rustc -Vv
cargo -V
log=artifacts/diagnostics/toolchain-1.94-linux-tests-v1.log
set +e
bash tools/build_linux.sh test --locked --release --lib --bins --tests --jobs 2 > "$log" 2>&1
result=$?
set -e
tail -n 25 "$log"
exit "$result"
