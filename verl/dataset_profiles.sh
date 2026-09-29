#!/usr/bin/env bash

# Shared dataset contract for DFT, SPFT and upstream PSFT launchers.
# Both primary profiles expose prompt, extra_info.question/answer and
# reward_model. PSFT uses the normalized parquet paths; DFT/SPFT can use the
# same paths because extra_info is present in both profiles.

configure_dataset_profile() {
    local script_dir="$1"
    local requested="${2:-${DATASET:-}}"

    if [[ -z "${requested}" ]]; then
        case "${TASK:-}" in
            math) requested="numina" ;;
            offline_math) requested="offline_math" ;;
            *) requested="numina" ;;
        esac
    fi

    case "${requested}" in
        math|numina)
            DATASET="numina"
            TASK="math"
            DATASET_TRAIN_FILE="${script_dir}/data/numina_cot/train.parquet"
            DATASET_VAL_FILE="${script_dir}/data/math500/test.parquet"
            DATASET_PSFT_TRAIN_FILE="${script_dir}/data/numina_psft/train.parquet"
            DATASET_PSFT_VAL_FILE="${script_dir}/data/numina_psft/test.parquet"
            DATASET_EVAL_GROUPS="math500"
            DATASET_PREPARE_SCRIPT="${script_dir}/prepare_numina_psft.sh"
            DATASET_PROJECT_NAME="DFT_log"
            DATASET_EXPERIMENT_NAME="muon-lora"
            ;;
        openr1)
            DATASET="openr1"
            TASK="openr1"
            DATASET_TRAIN_FILE="${script_dir}/data/openr1_psft/train.parquet"
            DATASET_VAL_FILE="${script_dir}/data/openr1_psft/test.parquet"
            DATASET_PSFT_TRAIN_FILE="${script_dir}/data/openr1_psft/train.parquet"
            DATASET_PSFT_VAL_FILE="${script_dir}/data/openr1_psft/test.parquet"
            DATASET_EVAL_GROUPS="aime24"
            DATASET_PREPARE_SCRIPT="${script_dir}/prepare_openr1_psft.sh"
            DATASET_PROJECT_NAME="DFT_openr1"
            DATASET_EXPERIMENT_NAME="openr1"
            ;;
        offline_math)
            # Kept for the existing rejection-sampling workflow. Its source
            # dataset is selected with DATASET_SOURCE (numina/openr1).
            DATASET="offline_math"
            TASK="offline_math"
            DATASET_TRAIN_FILE="${script_dir}/data/offline_math/train.parquet"
            DATASET_VAL_FILE="${script_dir}/data/math500/test.parquet"
            DATASET_PSFT_TRAIN_FILE="${DATASET_TRAIN_FILE}"
            DATASET_PSFT_VAL_FILE="${DATASET_VAL_FILE}"
            DATASET_EVAL_GROUPS="math500"
            DATASET_PREPARE_SCRIPT="${script_dir}/scripts/offline_math/generate_data.sh"
            DATASET_PROJECT_NAME="DFT_offline_math"
            DATASET_EXPERIMENT_NAME="dft-offline-math-qwen2.5-math-1.5b"
            ;;
        *)
            echo "Unknown DATASET=${requested}; expected numina or openr1" >&2
            return 2
            ;;
    esac

    # Model selection is independent of the dataset. MODEL_PATH remains an alias.
    MODEL_NAME="${MODEL_NAME:-${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}}"
    export MODEL_NAME
    # Compatibility for older launchers that consume this variable.
    DATASET_MODEL_NAME="${MODEL_NAME}"

    DATASET_TRAIN_PROMPT_KEY="extra_info"
    DATASET_TRAIN_RESPONSE_KEY="extra_info"
    DATASET_TRAIN_PROMPT_DICT_KEYS="['question']"
    DATASET_TRAIN_RESPONSE_DICT_KEYS="['answer']"
    DATASET_TRAIN_BATCH_SIZE="256"
    DATASET_MICRO_BATCH_SIZE_PER_GPU="8"
    DATASET_MICRO_BATCH_SIZE_PER_GPU_8GPU="4"
    DATASET_MAX_LENGTH="2048"
    DATASET_WARMUP_STEPS_RATIO="0.1"
    DATASET_TASK_EVAL="math"
    export DATASET TASK DATASET_MODEL_NAME DATASET_TRAIN_FILE DATASET_VAL_FILE
    export DATASET_PSFT_TRAIN_FILE DATASET_PSFT_VAL_FILE DATASET_EVAL_GROUPS
    export DATASET_PREPARE_SCRIPT DATASET_PROJECT_NAME DATASET_EXPERIMENT_NAME
    export DATASET_TRAIN_PROMPT_KEY DATASET_TRAIN_RESPONSE_KEY
    export DATASET_TRAIN_PROMPT_DICT_KEYS DATASET_TRAIN_RESPONSE_DICT_KEYS
    export DATASET_TRAIN_BATCH_SIZE DATASET_MICRO_BATCH_SIZE_PER_GPU
    export DATASET_MICRO_BATCH_SIZE_PER_GPU_8GPU
    export DATASET_MAX_LENGTH DATASET_WARMUP_STEPS_RATIO DATASET_TASK_EVAL
}
