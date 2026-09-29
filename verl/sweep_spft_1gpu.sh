#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sigmoid_root="$(cd "${script_dir}/../.." && pwd)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${sigmoid_root}/Booster/cache}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
export SPFT_REFERENCE_CPU_OFFLOAD="${SPFT_REFERENCE_CPU_OFFLOAD:-true}"

# =========================
# SPFT sweep configuration
# Edit these values directly in this file, or override them from the shell.
# =========================
DATASET="${DATASET:-${SPFT_DATASET:-numina}}"       # numina or openr1
MODEL_NAME="${MODEL_NAME:-${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}}"
EPOCHS_LIST="${EPOCHS_LIST:-${EPOCHS:-${TOTAL_EPOCHS:-1}}}" # e.g. "1 3 5"
SPFT_LAMBDAS="${SPFT_LAMBDAS:-0.1}"                         # e.g. "0.05 0.1 0.2"
OPTIM_LRS="${OPTIM_LRS:-1e-4}"                              # e.g. "5e-5 1e-4"
OPTIM_WEIGHT_DECAYS="${OPTIM_WEIGHT_DECAYS:-0.01}"         # e.g. "0 0.01"

run_idx="${RUN_IDX_START:-0}"
spft_save_freq="${SPFT_SAVE_FREQ:-${SAVE_FREQ:--1}}"
spft_weight_threshold="${SPFT_WEIGHT_THRESHOLD:-0.01}"
source "${script_dir}/dataset_profiles.sh"
configure_dataset_profile "${script_dir}" "${DATASET}"
dataset="${DATASET}"
epochs_list="${EPOCHS_LIST}"
# MODEL_NAME selects the model independently of DATASET (Hub ID or local path).

# Baselines: SPFT=AdamW (lr=1e-4, lambda=0.1); DFT=SorenAuxAdam;
# PSFT=AdamW (lr=1e-6, wd=0.1, clip=0.2/0.28); MBO=projection, Muon lr=8e-4.
for epochs in ${epochs_list}; do
for spft_lambda in ${SPFT_LAMBDAS}; do
    for optim_lr in ${OPTIM_LRS}; do
        for weight_decay in ${OPTIM_WEIGHT_DECAYS}; do
            run_idx=$((run_idx + 1))
            experiment_name="${dataset}-spft_${run_idx}_${epochs}ep_adamw_lr${optim_lr}_lambda${spft_lambda}"

            echo "Starting ${experiment_name}:"
            echo "  DATASET=${dataset}"
            echo "  MODEL_NAME=${MODEL_NAME}"
            echo "  OPTIM_NAME=adamw"
            echo "  OPTIM_LR=${optim_lr}"
            echo "  OPTIM_WEIGHT_DECAY=${weight_decay}"
            echo "  SPFT_LAMBDA=${spft_lambda}"
            echo "  SPFT_WEIGHT_THRESHOLD=${spft_weight_threshold}"
            echo "  SAVE_FREQ=${spft_save_freq}"

            EXPERIMENT_NAME="${experiment_name}" \
            DATASET="${dataset}" \
            MODEL_NAME="${MODEL_NAME}" \
            LOSS_MODE=spft \
            OPTIM_NAME=adamw \
            OPTIM_LR="${optim_lr}" \
            OPTIM_WEIGHT_DECAY="${weight_decay}" \
            SPFT_LAMBDA="${spft_lambda}" \
            SPFT_WEIGHT_THRESHOLD="${spft_weight_threshold}" \
            EPOCHS="${epochs}" TOTAL_EPOCHS="${epochs}" \
            SAVE_FREQ="${spft_save_freq}" \
            bash "${script_dir}/train_spft_1gpu.sh" "$@"
done
done
done
done
