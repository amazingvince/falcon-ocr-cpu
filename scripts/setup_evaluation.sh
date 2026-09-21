#!/usr/bin/env bash
# Keep evaluator Python and packages isolated from the GPU reference runtime.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
evaluation_root=${FOCR_EVALUATION_ENV:-/home/amazi/falcon-ocr-evaluation}
evaluator_revision=59b103c4b47d3a01fada83491585d6512a40c0bc
evaluator_source="$project_root/artifacts/OmniDocBench-eval"
if [[ ! -d "$evaluator_source/.git" ]]; then
    git clone --filter=blob:none --no-checkout --depth 1 --branch v1_5 \
        https://github.com/opendatalab/OmniDocBench.git "$evaluator_source"
    git -C "$evaluator_source" fetch --depth 1 origin "$evaluator_revision"
    git -C "$evaluator_source" sparse-checkout set configs dataset metrics registry task utils tools \
        /pdf_validation.py /requirements.txt /README.md /LICENSE
    git -C "$evaluator_source" checkout --detach "$evaluator_revision"
fi
uv_path=${UV:-uv}
if ! command -v "$uv_path" >/dev/null; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then uv_path="$HOME/.local/bin/uv";
    else printf 'Install uv or set UV to its path.\n' >&2; exit 1; fi
fi
[[ $(git -C "$evaluator_source" rev-parse HEAD) == "$evaluator_revision" ]] || {
    printf 'Evaluator revision mismatch; expected %s\n' "$evaluator_revision" >&2; exit 1;
}
mkdir -p "$evaluation_root"
export UV_CACHE_DIR="$evaluation_root/cache"
export UV_PYTHON_INSTALL_DIR="$evaluation_root/python"
if [[ ! -x "$evaluation_root/.venv/bin/python" ]]; then
    "$uv_path" venv --python 3.10.20 "$evaluation_root/.venv"
fi
"$uv_path" pip sync --python "$evaluation_root/.venv/bin/python" "$project_root/requirements/evaluation-resolved.txt"
"$uv_path" pip freeze --python "$evaluation_root/.venv/bin/python" > "$evaluation_root/packages.txt"
"$evaluation_root/.venv/bin/python" --version > "$evaluation_root/python-version.txt"
printf 'Evaluator environment: %s\n' "$evaluation_root"
