#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_idx="${RUN_IDX_START:-0}"
for clip_high in ${PSFT_CLIP_RATIO_HIGHS:-0.28}; do
    for optim_lr in ${OPTIM_LRS:-3e-6}; do
        for weight_decay in ${OPTIM_WEIGHT_DECAYS:-0.1}; do
            run_idx=$((run_idx + 1))
            EXPERIMENT_NAME="${TASK:-math}-psft_${run_idx}_adamw_lr${optim_lr}_clip${clip_high}" \
            LOSS_MODE=psft OPTIM_NAME=adamw \
            OPTIM_LR="${optim_lr}" OPTIM_WEIGHT_DECAY="${weight_decay}" \
            PSFT_CLIP_RATIO_HIGH="${clip_high}" SAVE_FREQ="${SAVE_FREQ:-100}" \
            bash "${script_dir}/train_dft_1gpu.sh"
        done
    done
done
