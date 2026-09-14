#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_idx="${RUN_IDX_START:-0}"
spft_save_freq="${SPFT_SAVE_FREQ:-${SAVE_FREQ:--1}}"
spft_weight_threshold="${SPFT_WEIGHT_THRESHOLD:-0.01}"
dataset="${DATASET:-numina}"
source "${script_dir}/dataset_profiles.sh"
configure_dataset_profile "${script_dir}" "${dataset}"
dataset="${DATASET}"
epochs_list="${EPOCHS_LIST:-${EPOCHS:-${TOTAL_EPOCHS:-1}}}"
MODEL_NAME="${MODEL_NAME:-${DATASET_MODEL_NAME}}"
# Models: numina=Qwen/Qwen2.5-Math-1.5B; openr1=Qwen/Qwen2.5-7B-Instruct.

# Baselines: SPFT=AdamW (lr=1e-4, lambda=0.1); DFT=SorenAuxAdam;
# PSFT=AdamW (lr=1e-6, wd=0.1, clip=0.2/0.28); MBO=projection, Muon lr=8e-4.
for epochs in ${epochs_list}; do
for spft_lambda in ${SPFT_LAMBDAS:-0.1}; do
    for optim_lr in ${OPTIM_LRS:-1e-4}; do
        for weight_decay in ${OPTIM_WEIGHT_DECAYS:-0.01}; do
            run_idx=$((run_idx + 1))
            experiment_name="${dataset}-spft_${run_idx}_${epochs}ep_adamw_lr${optim_lr}_lambda${spft_lambda}"

            echo "Starting ${experiment_name}:"
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
            bash "${script_dir}/train_spft_1gpu.sh"
done
done
done
done
