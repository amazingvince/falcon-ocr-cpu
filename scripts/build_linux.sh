#!/usr/bin/env bash
# Keep build prerequisites local. No sudo, global packages, or shell startup edits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${FOCR_TOOL_DIR:-$ROOT/artifacts/tools/linux}"
mkdir -p "$TOOLS"

fetch() {
    local url="$1" target="$2" expected="$3"
    if [[ ! -f "$target" ]]; then
        curl --fail --location --retry 3 "$url" --output "$target"
    fi
    printf '%s  %s\n' "$expected" "$target" | sha256sum --check --status
}

if ! command -v cargo >/dev/null && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v cargo >/dev/null || { printf 'Install Rust rustup first.\n' >&2; exit 1; }
command -v cc >/dev/null || { printf 'A C compiler is required.\n' >&2; exit 1; }
command -v make >/dev/null || { printf 'GNU make is required.\n' >&2; exit 1; }

if ! command -v cmake >/dev/null; then
    CMAKE_DIR="$TOOLS/cmake-3.31.10-linux-x86_64"
    if [[ ! -x "$CMAKE_DIR/bin/cmake" ]]; then
        [[ "$(uname -m)" == x86_64 ]] || { printf 'Install CMake for this architecture.\n' >&2; exit 1; }
        ARCHIVE="$TOOLS/cmake-3.31.10-linux-x86_64.tar.gz"
        fetch 'https://github.com/Kitware/CMake/releases/download/v3.31.10/cmake-3.31.10-linux-x86_64.tar.gz' "$ARCHIVE" '3cb3dd247b6a1de2d0f4b20c6fd4326c9024e894cebc9dc8699758887e566ca7'
        tar -xzf "$ARCHIVE" -C "$TOOLS"
    fi
    export PATH="$CMAKE_DIR/bin:$PATH"
fi
if [[ -z "${ASM_NASM:-}" ]] && ! command -v nasm >/dev/null; then
    NASM_PREFIX="$TOOLS/nasm-installed"
    if [[ ! -x "$NASM_PREFIX/bin/nasm" ]]; then
        ARCHIVE="$TOOLS/nasm-2.16.03.tar.xz"
        fetch 'https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.xz' "$ARCHIVE" '1412a1c760bbd05db026b6c0d1657affd6631cd0a63cddb6f73cc6d4aa616148'
        tar -xJf "$ARCHIVE" -C "$TOOLS"
        printf 'Building local NASM; log: %s\n' "$TOOLS/nasm-build.log"
        if ! (
            cd "$TOOLS/nasm-2.16.03"
            ./configure --prefix="$NASM_PREFIX"
            make -j "${CARGO_BUILD_JOBS:-4}"
            make install
        ) >"$TOOLS/nasm-build.log" 2>&1; then
            tail -n 60 "$TOOLS/nasm-build.log" >&2
            exit 1
        fi
    fi
    export ASM_NASM="$NASM_PREFIX/bin/nasm"
fi

# Keep Windows and Linux build products separate when using a shared checkout.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target/linux}"
cd "$ROOT"
if [[ $# -eq 0 ]]; then set -- build --release; fi
cargo "$@"
