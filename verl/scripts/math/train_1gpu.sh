#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_common/run_text_task.sh"
verl_root="$(cd "${script_dir}/../.." && pwd)"

# Edit these arrays, then run: bash verl/scripts/math/train_1gpu.sh
TRAIN_MODES=(dft)  # Use (spft) or (dft spft) when needed.
HF_MODEL="Qwen/Qwen2.5-Math-1.5B"
LORA_RANK=0  # Full-parameter training, matching the paper's main setting.
LORA_ALPHA=16
LEARNING_RATES=(5e-5)
SPFT_LAMBDAS=(0.1)
SEEDS=(1)
WARMUP_RATIO=0.1
AUTO_PREPARE_DATA=true
TRAIN_DATA_DIR="${verl_root}/data/numina_cot"
EVAL_DATA_DIR="${verl_root}/data/math500"

if [[ ! -f "${TRAIN_DATA_DIR}/train.parquet" || ! -f "${EVAL_DATA_DIR}/test.parquet" ]]; then
    if [[ "${AUTO_PREPARE_DATA}" != "true" ]]; then
        echo "Missing prepared math data." >&2
        exit 1
    fi
    mkdir -p "${TRAIN_DATA_DIR}" "${EVAL_DATA_DIR}"
    if [[ ! -f "${TRAIN_DATA_DIR}/train.parquet" ]]; then
        "${PYTHON_BIN:-python}" "${verl_root}/examples/data_preprocess/numina_cot.py" \
            --local_dir "${TRAIN_DATA_DIR}" --train_end 100000
    fi
    if [[ ! -f "${EVAL_DATA_DIR}/test.parquet" ]]; then
        "${PYTHON_BIN:-python}" "${verl_root}/examples/data_preprocess/math_dataset.py" \
            --local_dir "${EVAL_DATA_DIR}"
    fi
fi

for seed in "${SEEDS[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
        for train_mode in "${TRAIN_MODES[@]}"; do
            if [[ "${train_mode}" == "spft" ]]; then
                for spft_lambda in "${SPFT_LAMBDAS[@]}"; do
                    run_text_task math "${HF_MODEL}" "${learning_rate}" "${WARMUP_RATIO}" "${train_mode}" "${spft_lambda}" "${seed}"
                done
            else
                run_text_task math "${HF_MODEL}" "${learning_rate}" "${WARMUP_RATIO}" "${train_mode}" 0.1 "${seed}"
            fi
        done
    done
done
