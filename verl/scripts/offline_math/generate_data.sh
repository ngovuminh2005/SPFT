#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
verl_root="$(cd "${script_dir}/../.." && pwd)"
source "${verl_root}/dataset_profiles.sh"
dataset_source="${DATASET_SOURCE:-${DATASET:-numina}}"
configure_dataset_profile "${verl_root}" "${dataset_source}"

# Separate, resumable data construction. This script never starts training.
HF_MODEL="${HF_MODEL:-${DATASET_MODEL_NAME}}"
SOURCE_FILE="${SOURCE_FILE:-${DATASET_TRAIN_FILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${verl_root}/data/offline_math_${dataset_source}}"
NUM_QUESTIONS=100000
RESPONSES_PER_QUESTION=4
TEMPERATURE=1.0
TOP_P=1.0
MAX_NEW_TOKENS=2048
SEED=1
GENERATION_BATCH_SIZE=256
TENSOR_PARALLEL_SIZE=1
VERIFY_WORKERS=16

if [[ ! -f "${SOURCE_FILE}" ]]; then
    DATASET="${dataset_source}" bash "${verl_root}/prepare_dataset.sh"
fi

"${PYTHON_BIN:-python}" "${script_dir}/build_offline_dataset.py" \
    --stage all \
    --model "${HF_MODEL}" \
    --source-file "${SOURCE_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-questions "${NUM_QUESTIONS}" \
    --responses-per-question "${RESPONSES_PER_QUESTION}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --batch-size "${GENERATION_BATCH_SIZE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --verify-workers "${VERIFY_WORKERS}"
