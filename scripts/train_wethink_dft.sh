#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/examples/extras/dft/qwen2_5vl_wethink_lora_sft.yaml"
DATASET="${ROOT_DIR}/data/wethink_sft.jsonl"
IMAGE_DIR="${ROOT_DIR}/data/wethink_images"

if ! command -v python >/dev/null 2>&1; then
  echo "Activate your Python/conda environment first." >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing training config: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${DATASET}" || ! -d "${IMAGE_DIR}" ]]; then
  echo "WeThink dataset is not prepared." >&2
  echo "Run: scripts/prepare_wethink.sh" >&2
  exit 1
fi

cd "${ROOT_DIR}"
python scripts/escape_wethink_media_tags.py data/wethink_sft.jsonl
python scripts/smoke_wethink.py
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m llamafactory.cli train "${CONFIG}" "$@"
