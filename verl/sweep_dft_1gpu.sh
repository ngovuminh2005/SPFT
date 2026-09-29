#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sigmoid_root="$(cd "${script_dir}/../.." && pwd)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${sigmoid_root}/Booster/cache}"
dataset="${DATASET:-numina}"
source "${script_dir}/dataset_profiles.sh"
configure_dataset_profile "${script_dir}" "${dataset}"
dataset="${DATASET}"
epochs_list="${EPOCHS_LIST:-${EPOCHS:-${TOTAL_EPOCHS:-1}}}"
# MODEL_NAME selects the model independently of DATASET (Hub ID or local path).

# Baselines: DFT=SorenAuxAdam (base=8e-4, aux=5e-5, muon=7e-4, lambda=1);
# SPFT=AdamW (lr=1e-4, lambda=0.1); PSFT=AdamW (lr=1e-6, clip=0.2/0.28).
# Edit sweep values here. DFT is the loss; Soren/HTMuon are the optimizers.
#OPTIMIZERS=(sorenauxadam htmuonauxadam)
OPTIMIZERS=(sorenauxadam)
MAIN_LRS=(7e-4)                 # OPTIM_MUON_LR, matching sweep_mbo_1gpu.sh.
SOREN_LAMBDAS=(1)
HTMUON_ALPHAS=(0.125)
SOREN_MODE=(polynomial)                  # polynomial or exact
BASE_LR=8e-4
AUX_LR=5e-5
SEEDS=(1)
SAVE_FREQ=100

run_idx=2
for epochs in ${epochs_list}; do
for seed in "${SEEDS[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        case "${optimizer}" in
            soren|sorenauxadam) sweep_values=("${SOREN_LAMBDAS[@]}") ;;
            htmuon|htmuonauxadam) sweep_values=("${HTMUON_ALPHAS[@]}") ;;
            *) echo "Unsupported optimizer: ${optimizer}" >&2; exit 2 ;;
        esac
        for main_lr in "${MAIN_LRS[@]}"; do
            for value in "${sweep_values[@]}"; do
                soren_lambda=0.01
                htmuon_alpha=0.125
                case "${optimizer}" in
                    soren*)
                        soren_lambda="${value}"
                        variant="${SOREN_MODE}-lambda${value}"
                        ;;
                    htmuon*)
                        htmuon_alpha="${value}"
                        variant="alpha${value}"
                        ;;
                esac
                run_idx=$((run_idx + 1))
                experiment_name="${dataset}-dft_${run_idx}-${optimizer}-${variant}-lr${main_lr}-${epochs}ep"
                echo "Starting ${experiment_name} (epochs=${epochs}, base_lr=${BASE_LR}, aux_lr=${AUX_LR})"
                EXPERIMENT_NAME="${experiment_name}" \
                DATASET="${dataset}" \
                MODEL_NAME="${MODEL_NAME}" \
                LOSS_MODE=dft \
                OPTIM_NAME="${optimizer}" \
                LORA_ALPHA=16 \
                LORA_RANK=8 \
                OPTIM_WEIGHT_DECAY=0 \
                OPTIM_LR="${BASE_LR}" \
                OPTIM_AUX_LR="${AUX_LR}" \
                OPTIM_MUON_LR="${main_lr}" \
                OPTIM_SOREN_LAMBDA="${soren_lambda}" \
                OPTIM_SOREN_MODE="${SOREN_MODE}" \
                OPTIM_HTMUON_ALPHA="${htmuon_alpha}" \
                SEED="${seed}" \
                EPOCHS="${epochs}" TOTAL_EPOCHS="${epochs}" \
                SAVE_FREQ="${SAVE_FREQ}" \
                bash "${script_dir}/train_dft_1gpu.sh"
            done
        done
    done
done
done
