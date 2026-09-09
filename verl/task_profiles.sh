#!/usr/bin/env bash

# Task defaults shared by DFT and SPFT launchers. Explicit environment values
# always win, so profiles remain easy to override for local datasets/models.

_dft_set_default() {
    local name="$1"
    local value="$2"
    if [[ -z "${!name:-}" ]]; then
        printf -v "${name}" '%s' "${value}"
    fi
}

configure_dft_task_profile() {
    local script_dir="$1"
    TASK="${TASK:-math}"

    case "${TASK}" in
        math)
            # Preserve the original one-GPU launcher's defaults.
            _dft_set_default PROJECT_NAME "DFT_log"
            _dft_set_default EXPERIMENT_NAME "muon-lora"
            _dft_set_default TRAIN_FILE "${script_dir}/data/numina_cot/train.parquet"
            _dft_set_default VAL_FILE "${script_dir}/data/math500/test.parquet"
            _dft_set_default MODEL_NAME "Qwen/Qwen2.5-Math-1.5B"
            _dft_set_default TRAIN_BATCH_SIZE "256"
            _dft_set_default MICRO_BATCH_SIZE_PER_GPU "8"
            _dft_set_default MAX_LENGTH "2048"
            _dft_set_default WARMUP_STEPS_RATIO "0.1"
            _dft_set_default TASK_EVAL "math"
            ;;
        offline_math)
            _dft_set_default PROJECT_NAME "DFT_offline_math"
            _dft_set_default EXPERIMENT_NAME "dft-offline-math-qwen2.5-math-1.5b"
            _dft_set_default TRAIN_FILE "${script_dir}/data/offline_math/train.parquet"
            _dft_set_default VAL_FILE "${script_dir}/data/math500/test.parquet"
            _dft_set_default MODEL_NAME "Qwen/Qwen2.5-Math-1.5B"
            _dft_set_default TRAIN_BATCH_SIZE "256"
            _dft_set_default MICRO_BATCH_SIZE_PER_GPU "8"
            _dft_set_default MAX_LENGTH "2048"
            _dft_set_default WARMUP_STEPS_RATIO "0.1"
            _dft_set_default TASK_EVAL "math"
            ;;
        *)
            echo "Unknown TASK=${TASK}; expected math or offline_math." >&2
            return 2
            ;;
    esac

    _dft_set_default TRAIN_PROMPT_KEY "extra_info"
    _dft_set_default TRAIN_RESPONSE_KEY "extra_info"
    _dft_set_default TRAIN_PROMPT_DICT_KEYS "['question']"
    _dft_set_default TRAIN_RESPONSE_DICT_KEYS "['answer']"
    export TASK TASK_EVAL
}
