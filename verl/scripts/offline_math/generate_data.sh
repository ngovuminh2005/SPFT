#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
verl_root="$(cd "${script_dir}/../.." && pwd)"

# Separate, resumable data construction. This script never starts training.
HF_MODEL="Qwen/Qwen2.5-Math-1.5B"
SOURCE_FILE="${verl_root}/data/numina_cot/train.parquet"
OUTPUT_DIR="${verl_root}/data/offline_math"
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
    mkdir -p "$(dirname "${SOURCE_FILE}")"
    "${PYTHON_BIN:-python}" "${verl_root}/examples/data_preprocess/numina_cot.py" \
        --local_dir "$(dirname "${SOURCE_FILE}")" --train_end "${NUM_QUESTIONS}"
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
