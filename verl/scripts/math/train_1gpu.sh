#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_common/run_text_task.sh"
verl_root="$(cd "${script_dir}/../.." && pwd)"
source "${verl_root}/dataset_profiles.sh"
DATASET="${DATASET:-numina}"
configure_dataset_profile "${verl_root}" "${DATASET}"

# Edit these arrays, then run: bash verl/scripts/math/train_1gpu.sh
TRAIN_MODES=(dft)  # Use (spft) or (dft spft) when needed.
HF_MODEL="${HF_MODEL:-${DATASET_MODEL_NAME}}"
LORA_RANK=0  # Full-parameter training, matching the paper's main setting.
LORA_ALPHA=16
LEARNING_RATES=(5e-5)
SPFT_LAMBDAS=(0.1)
SEEDS=(1)
WARMUP_RATIO=0.1
EPOCHS="${EPOCHS:-${TOTAL_EPOCHS:-1}}"
AUTO_PREPARE_DATA=true
TRAIN_DATA_FILE="${TRAIN_FILE:-${DATASET_TRAIN_FILE}}"
EVAL_DATA_FILE="${VAL_FILE:-${DATASET_VAL_FILE}}"

if [[ ! -f "${TRAIN_DATA_FILE}" || ! -f "${EVAL_DATA_FILE}" ]]; then
    if [[ "${AUTO_PREPARE_DATA}" != "true" ]]; then
        echo "Missing prepared math data." >&2
        exit 1
    fi
    DATASET="${DATASET}" bash "${DATASET_PREPARE_SCRIPT}"
fi

export EPOCHS

for seed in "${SEEDS[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
        for train_mode in "${TRAIN_MODES[@]}"; do
            if [[ "${train_mode}" == "spft" ]]; then
                for spft_lambda in "${SPFT_LAMBDAS[@]}"; do
                    DATASET="${DATASET}" run_text_task math "${HF_MODEL}" "${learning_rate}" "${WARMUP_RATIO}" "${train_mode}" "${spft_lambda}" "${seed}"
                done
            else
                DATASET="${DATASET}" run_text_task math "${HF_MODEL}" "${learning_rate}" "${WARMUP_RATIO}" "${train_mode}" 0.1 "${seed}"
            fi
        done
    done
done
