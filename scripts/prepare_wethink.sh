#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
RAW_DIR="${WETHINK_RAW_DIR:-${DATA_DIR}/wethink_raw}"
PARTS_DIR="${WETHINK_IMAGE_PARTS_DIR:-${DATA_DIR}/wethink_image_parts}"
IMAGE_ROOT="${WETHINK_IMAGE_ROOT:-${DATA_DIR}/wethink_images}"
RAW_JSONL="${WETHINK_RAW_JSONL:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

HF_BIN="${HF_BIN:-$(command -v hf || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"

if [[ ! -x "${HF_BIN}" ]]; then
  HF_BIN="$(command -v hf || true)"
fi
if [[ -z "${HF_BIN}" || ! -x "${HF_BIN}" ]]; then
  echo "Cannot find the hf CLI. Activate your environment or set HF_BIN=/path/to/hf." >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Cannot find Python at ${PYTHON_BIN}." >&2
  exit 1
fi

mkdir -p "${RAW_DIR}" "${PARTS_DIR}" "${IMAGE_ROOT}"

if [[ -z "${RAW_JSONL}" ]]; then
  "${HF_BIN}" download yangjie-cv/WeThink_Multimodal_Reasoning_120K \
    --repo-type dataset \
    --include "*.jsonl" \
    --local-dir "${RAW_DIR}"
  RAW_JSONL="$(find "${RAW_DIR}" -type f -name '*.jsonl' -print -quit)"
fi

if [[ -z "${RAW_JSONL}" || ! -f "${RAW_JSONL}" ]]; then
  echo "Cannot find the WeThink JSONL file." >&2
  exit 1
fi

if [[ ! -f "${IMAGE_ROOT}/.wethink_images_extracted" ]]; then
  ZIP_PATH="${PARTS_DIR}/image.zip"
  if [[ ! -f "${ZIP_PATH}" ]]; then
    "${HF_BIN}" download Xkev/LLaVA-CoT-100k \
      --repo-type dataset \
      --include "image.zip.part-*" \
      --local-dir "${PARTS_DIR}"

    mapfile -d '' IMAGE_PARTS < <(find "${PARTS_DIR}" -type f -name 'image.zip.part-*' -print0 | sort -z)
    if [[ "${#IMAGE_PARTS[@]}" -eq 0 ]]; then
      echo "Cannot find image.zip.part-* files." >&2
      exit 1
    fi
    : > "${ZIP_PATH}"
    for part in "${IMAGE_PARTS[@]}"; do
      cat "${part}" >> "${ZIP_PATH}"
    done
  fi

  unzip -q "${ZIP_PATH}" -d "${IMAGE_ROOT}"
  touch "${IMAGE_ROOT}/.wethink_images_extracted"
fi

CONVERTER_ARGS=(
  --raw-jsonl "${RAW_JSONL}"
  --image-root "${IMAGE_ROOT}"
  --output "${DATA_DIR}/wethink_sft.jsonl"
  --dataset-info "${DATA_DIR}/dataset_info.json"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  CONVERTER_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_wethink.py" "${CONVERTER_ARGS[@]}"
echo "Preparation completed. Use dataset=wethink_sft in your YAML config."
