#!/usr/bin/env bash

# Run exactly one text-task configuration. Task scripts own all sweep loops.
run_text_task() {
    if [[ $# -lt 7 ]]; then
        echo "run_text_task requires: TASK MODEL LR WARMUP MODE LAMBDA SEED" >&2
        return 2
    fi

    local task="$1"
    local model_name="$2"
    local learning_rate="$3"
    local warmup_ratio="$4"
    local mode="$5"
    local spft_lambda="$6"
    local seed="$7"
    local common_dir verl_root
    common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    verl_root="$(cd "${common_dir}/../.." && pwd)"

    case "${mode}" in
        dft|spft|psft) ;;
        *)
            echo "Unknown mode '${mode}'; expected dft, spft, or psft." >&2
            return 2
            ;;
    esac

    local model_tag experiment_name data_tag
    model_tag="${model_name##*/}"
    data_tag="${EXPERIMENT_DATA_TAG:-}"
    if [[ -n "${data_tag}" ]]; then
        data_tag="-${data_tag}"
    fi
    if [[ "${mode}" == "spft" ]]; then
        experiment_name="${task}-${model_tag}${data_tag}-spft-adamw-lr${learning_rate}-lambda${spft_lambda}-seed${seed}"
    else
        experiment_name="${task}-${model_tag}${data_tag}-${mode}-adamw-lr${learning_rate}-seed${seed}"
    fi

    echo "TASK=${task} MODEL=${model_name} MODE=${mode} LR=${learning_rate} LAMBDA=${spft_lambda} SEED=${seed}"
    TASK="${task}" \
    MODEL_NAME="${model_name}" \
    LORA_RANK="${LORA_RANK:-0}" \
    LORA_ALPHA="${LORA_ALPHA:-16}" \
    SEED="${seed}" \
    LOSS_MODE="${mode}" \
    EXPERIMENT_NAME="${experiment_name}" \
    SAVE_PATH="${verl_root}/checkpoints/${task}/${experiment_name}" \
    OPTIM_NAME="adamw" \
    OPTIM_LR="${learning_rate}" \
    OPTIM_BETA1="${OPTIM_BETA1:-0.9}" \
    OPTIM_BETA2="${OPTIM_BETA2:-0.95}" \
    OPTIM_EPS="${OPTIM_EPS:-1e-8}" \
    OPTIM_WEIGHT_DECAY="${OPTIM_WEIGHT_DECAY:-0.01}" \
    WARMUP_STEPS_RATIO="${warmup_ratio}" \
    LR_SCHEDULER="${LR_SCHEDULER:-cosine}" \
    SPFT_LAMBDA="${spft_lambda}" \
    SPFT_EPS="${SPFT_EPS:-1e-6}" \
    bash "${verl_root}/train_dft_1gpu.sh"
}
