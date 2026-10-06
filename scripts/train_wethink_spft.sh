#!/usr/bin/env bash
set -euo pipefail

export DEBUG_SPFT_NAN="${DEBUG_SPFT_NAN:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/examples/extras/spft/qwen2_5vl_wethink_full_sft.yaml"

if ! command -v python >/dev/null 2>&1; then
  echo "Activate your Python/conda environment first." >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing training config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${ROOT_DIR}/data/wethink_sft.jsonl" || ! -d "${ROOT_DIR}/data/wethink_images" ]]; then
  echo "WeThink dataset is not prepared. Run: bash scripts/prepare_wethink.sh" >&2
  exit 1
fi

cd "${ROOT_DIR}"
python scripts/escape_wethink_media_tags.py data/wethink_sft.jsonl
python scripts/smoke_wethink.py
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m llamafactory.cli train "${CONFIG}" "$@"
