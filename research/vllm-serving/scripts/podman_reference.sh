#!/usr/bin/env bash
# Project-specific rootful Podman storage. This WSL VHD is backed by D:, not C:.
set -euo pipefail
storage=/home/amazi/falcon-ocr-vllm
[[ $(findmnt -T /home -n -o FSTYPE) == ext4 ]] || { echo 'Expected ext4 runtime storage' >&2; exit 1; }
sudo -n mkdir -p "$storage/storage" "$storage/run" "$storage/tmp"
exec sudo -n env TMPDIR="$storage/tmp" podman \
  --root "$storage/storage" --runroot "$storage/run" \
  --storage-driver overlay --cgroup-manager cgroupfs "$@"
