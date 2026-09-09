#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_common/run_text_task.sh"
verl_root="$(cd "${script_dir}/../.." && pwd)"

# Edit these arrays, then run: bash verl/scripts/offline_math/train_1gpu.sh
read -r -a TRAIN_MODES <<< "${OFFLINE_TRAIN_MODES:-dft}"
HF_MODEL="Qwen/Qwen2.5-Math-1.5B"
LORA_RANK=0
LORA_ALPHA=16
read -r -a LEARNING_RATES <<< "${OFFLINE_LEARNING_RATES:-5e-5}"
read -r -a SPFT_LAMBDAS <<< "${OFFLINE_SPFT_LAMBDAS:-0.1}"
SEEDS=(1)
WARMUP_RATIO=0.1
SAVE_FREQ=-1
TEST_FREQ=-1
# Default one-GPU DFT/SPFT accumulation:
# global batch 256 / micro-batch 8 = 32 gradient-accumulation micro-steps.
TRAIN_BATCH_SIZE=256
MICRO_BATCH_SIZE_PER_GPU=8
export TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU SAVE_FREQ TEST_FREQ
SOURCE_FILE="${verl_root}/data/numina_cot/train.parquet"
DATA_DIR="${verl_root}/data/offline_math"
NUM_QUESTIONS=100000
RESPONSES_PER_QUESTION=4
TEMPERATURE=1.0
TOP_P=1.0
MAX_NEW_TOKENS=2048
DATA_SEED=1

# Training never generates data. It also enforces that generator and training
# initialization use the exact same model identifier and source configuration.
"${PYTHON_BIN:-python}" "${script_dir}/build_offline_dataset.py" \
    --stage validate \
    --model "${HF_MODEL}" \
    --source-file "${SOURCE_FILE}" \
    --output-dir "${DATA_DIR}" \
    --num-questions "${NUM_QUESTIONS}" \
    --responses-per-question "${RESPONSES_PER_QUESTION}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${DATA_SEED}"

for seed in "${SEEDS[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
        for train_mode in "${TRAIN_MODES[@]}"; do
            if [[ "${train_mode}" == "spft" ]]; then
                for spft_lambda in "${SPFT_LAMBDAS[@]}"; do
                    run_text_task offline_math "${HF_MODEL}" "${learning_rate}" "${WARMUP_RATIO}" "${train_mode}" "${spft_lambda}" "${seed}"
                done
            else
                run_text_task offline_math "${HF_MODEL}" "${learning_rate}" "${WARMUP_RATIO}" "${train_mode}" 0.1 "${seed}"
            fi
        done
    done
done
