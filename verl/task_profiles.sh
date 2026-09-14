#!/usr/bin/env bash

# Backward-compatible DFT/SPFT adapter around the shared dataset profiles.

_dft_set_default() {
    local name="$1"
    local value="$2"
    if [[ -z "${!name:-}" ]]; then
        printf -v "${name}" '%s' "${value}"
    fi
}

configure_dft_task_profile() {
    local script_dir="$1"
    local requested_dataset="${DATASET:-}"

    if [[ -z "${requested_dataset}" ]]; then
        case "${TASK:-}" in
            math) requested_dataset="numina" ;;
            offline_math) requested_dataset="offline_math" ;;
            *) requested_dataset="numina" ;;
        esac
    fi

    source "${script_dir}/dataset_profiles.sh"
    configure_dataset_profile "${script_dir}" "${requested_dataset}"

    _dft_set_default PROJECT_NAME "${DATASET_PROJECT_NAME}"
    _dft_set_default EXPERIMENT_NAME "${DATASET_EXPERIMENT_NAME}"
    _dft_set_default TRAIN_FILE "${DATASET_TRAIN_FILE}"
    _dft_set_default VAL_FILE "${DATASET_VAL_FILE}"
    _dft_set_default MODEL_NAME "${DATASET_MODEL_NAME}"
    _dft_set_default TRAIN_BATCH_SIZE "${DATASET_TRAIN_BATCH_SIZE}"
    _dft_set_default MICRO_BATCH_SIZE_PER_GPU "${DATASET_MICRO_BATCH_SIZE_PER_GPU}"
    _dft_set_default MAX_LENGTH "${DATASET_MAX_LENGTH}"
    _dft_set_default WARMUP_STEPS_RATIO "${DATASET_WARMUP_STEPS_RATIO}"
    _dft_set_default TASK_EVAL "${DATASET_TASK_EVAL}"
    _dft_set_default TRAIN_PROMPT_KEY "${DATASET_TRAIN_PROMPT_KEY}"
    _dft_set_default TRAIN_RESPONSE_KEY "${DATASET_TRAIN_RESPONSE_KEY}"
    _dft_set_default TRAIN_PROMPT_DICT_KEYS "${DATASET_TRAIN_PROMPT_DICT_KEYS}"
    _dft_set_default TRAIN_RESPONSE_DICT_KEYS "${DATASET_TRAIN_RESPONSE_DICT_KEYS}"
    export TASK TASK_EVAL DATASET
}
