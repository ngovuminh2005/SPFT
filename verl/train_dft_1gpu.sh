#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHONPATH="${script_dir}:${PYTHONPATH:-}"

# Select dataset/model/evaluation defaults while preserving explicit overrides.
source "${script_dir}/task_profiles.sh"
configure_dft_task_profile "${script_dir}"

# Match the original 8-GPU script's effective global batch on 1 GPU.
# Original:
#   global batch = 256
#   8 GPUs -> per-GPU batch = 32
#   micro batch per GPU = 4
#   implicit grad accumulation = 32 / 4 = 8
#
# 1-GPU equivalent:
#   global batch = 256
#   1 GPU -> per-GPU batch = 256
#   micro batch per GPU = 8
#   implicit grad accumulation = 256 / 8 = 32

# train data generation
# python examples/data_preprocess/numina_cot.py --train_end 100000
# eval data generation
# python examples/data_preprocess/math_dataset.py

# ------------------------
# Train setup
# ------------------------
: "${SEED:=1}"
: "${WANDB_API_KEY:=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}"
: "${DIST_INIT_FILE:=${repo_root}/.tmp/verl_dft_1gpu_${USER:-user}_$$.dist}"
mkdir -p "$(dirname "${DIST_INIT_FILE}")"
: "${LORA_RANK:=0}"
: "${LORA_ALPHA:=16}"
: "${LORA_TARGET_MODULES:=all-linear}"
: "${EPOCHS:=${TOTAL_EPOCHS:-1}}"
: "${TOTAL_EPOCHS:=${EPOCHS}}"
: "${TEST_FREQ:=10}"
: "${SAVE_FREQ:=50}"
: "${USE_REMOVE_PADDING:=true}"
: "${LOSS_MODE:=dft}"
: "${PSFT_CLIP_RATIO_LOW:=0.2}"
: "${PSFT_CLIP_RATIO_HIGH:=0.28}"
: "${PSFT_MINI_BATCH_SIZE:=32}"
: "${PSFT_WARMUP_STEPS:=10}"
: "${SPFT_LAMBDA:=0.1}"
: "${SPFT_EPS:=1e-6}"
: "${SPFT_WEIGHT_THRESHOLD:=0.01}"
: "${SPFT_REFERENCE_CPU_OFFLOAD:=false}"
: "${WARMUP_STEPS_RATIO:=0.1}"
: "${LR_SCHEDULER:=cosine}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${EXPERIMENT_NAME}-seed${SEED}-${TIMESTAMP}"
SAVE_PATH="${SAVE_PATH:-${script_dir}/checkpoints/${EXPERIMENT_NAME}}"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export LOCAL_WORLD_SIZE=1
export DIST_INIT_METHOD="file://${DIST_INIT_FILE}?rank=${RANK}&world_size=${WORLD_SIZE}"
export WANDB_API_KEY
export DEBUG_MUON_NAN="${DEBUG_MUON_NAN:-0}"
export DEBUG_TRAINER_NAN="${DEBUG_TRAINER_NAN:-0}"
export DEBUG_MUON_PARAM_FILTER="${DEBUG_MUON_PARAM_FILTER:-model.layers.0.mlp.gate_proj.weight}"
export DEBUG_MUON_LOG_FILE="${DEBUG_MUON_LOG_FILE:-${script_dir}/muon_nan_debug.log}"
export DEBUG_MUON_SV="${DEBUG_MUON_SV:-0}"
: > "${DEBUG_MUON_LOG_FILE}"
trap 'rm -f "${DIST_INIT_FILE}"' EXIT

# ------------------------
# Optimizer setup
# ------------------------
# Available OPTIM_NAME values:
#   adamw
#   singledevicemuonwithauxadam
#   singledevicemuonwithauxsgd
#   singledevicembowithauxadam
#   singledevicembowindowwithauxadam
#   muonwithauxadam
#   muonwithauxsgd
#   mbowithauxadam
#
# Plain variants also exist, but usually should not be used here because
# embedding/lm_head fall back to aux optimizer:
#   singledevicemuon, muon, singledevicembo, mbo
#
# Copy/paste examples:
#   OPTIM_NAME=adamw bash train_dft_1gpu.sh
#   OPTIM_NAME=singledevicemuonwithauxadam bash train_dft_1gpu.sh
#   OPTIM_NAME=singledevicembowithauxadam bash train_dft_1gpu.sh
#   OPTIM_NAME=singledevicembowindowwithauxadam bash train_dft_1gpu.sh
: "${OPTIM_NAME:=adamw}"
: "${OPTIM_MODULE:=muon}"

# AdamW-compatible defaults
: "${OPTIM_LR:=5e-5}"
: "${OPTIM_BETA1:=0.9}"
: "${OPTIM_BETA2:=0.95}"
: "${OPTIM_EPS:=1e-8}"

# Standard AdamW weight decay.
: "${OPTIM_WEIGHT_DECAY:=0.01}"

# Muon/MBO defaults.
# Intentionally different from aux Adam LR.
: "${OPTIM_MUON_LR:=8e-4}"
: "${OPTIM_AUX_LR:=5e-5}"
: "${OPTIM_MOMENTUM:=0.95}"
: "${OPTIM_NESTEROV:=true}"
: "${OPTIM_NS_STEPS:=5}"
: "${OPTIM_HTMUON_ALPHA:=0.125}"
: "${OPTIM_SOREN_LAMBDA:=0.01}"
: "${OPTIM_SOREN_MODE:=polynomial}" # polynomial or exact
# Single-device names: soren, sorenauxadam, htmuon, htmuonauxadam.

# ------------------------
# MBO setup
# ------------------------
# Exposed projection-free LoRA MBO hyperparameters in verl/muon.py:
#   OPTIM_MBO_NUM_CENTROIDS      -> num_centroids
#   OPTIM_MBO_CENTROID_DIM       -> centroid_dim
#   OPTIM_MBO_GRADIENT_SOURCE    -> minibatch, projection, global, onlygrad, or greedy queue source
#   OPTIM_MBO_LAMBDA_MEMORY      -> lambda_memory
#   OPTIM_MBO_MEMORY_INIT_SEED   -> memory_init_seed
#   OPTIM_MBO_OT_WS_STEPS        -> OT warm-start sample count; projection history warm-up steps
#   OPTIM_MBO_OT_STEP_UPDATE     -> ot_step_update
#   OPTIM_MBO_KMEANS_STEPS       -> K-means refinement iterations after K-means++
#   OPTIM_MBO_OT_EPSILON         -> ot_epsilon
#   OPTIM_MBO_OT_G_LR            -> ot_g_lr
#   OPTIM_MBO_OT_Y_LR            -> ot_y_lr
#   OPTIM_MBO_OT_G_STEPS         -> ot_g_steps
#   OPTIM_MBO_OT_Y_STEPS         -> ot_y_steps
#   OPTIM_MBO_OFFSET             -> layer offset from final transformer block
#   OPTIM_MBO_WINDOW_SIZE        -> number of transformer blocks in the MBO sliding window
#
# Notes:
#   - "window" is selected by OPTIM_NAME=singledevicembowindowwithauxadam.
#   - OPTIM_MBO_OFFSET=0 ends at the final transformer block; 1 ends at the block before it.
#   - OPTIM_MBO_WINDOW_SIZE=0 selects all blocks up to that end block.
#   - OPTIM_MBO_WINDOW_SIZE=K selects K consecutive blocks ending at that end block.
: "${OPTIM_MBO_NUM_CENTROIDS:=32}"
: "${OPTIM_MBO_CENTROID_DIM:=1024}"
: "${OPTIM_MBO_GRADIENT_SOURCE:=minibatch}"
: "${OPTIM_MBO_LAMBDA_MEMORY:=0.08}"
: "${OPTIM_MBO_MEMORY_INIT_SEED:=null}"
: "${OPTIM_MBO_OT_WS_STEPS:=128}"
: "${OPTIM_MBO_OT_STEP_UPDATE:=128}"
: "${OPTIM_MBO_KMEANS_STEPS:=10}"
: "${OPTIM_MBO_OT_EPSILON:=0.1}"
: "${OPTIM_MBO_OT_G_LR:=0.08}"
: "${OPTIM_MBO_OT_Y_LR:=0.01}"
: "${OPTIM_MBO_OT_G_STEPS:=5}"
: "${OPTIM_MBO_OT_Y_STEPS:=1}"
: "${OPTIM_MBO_OFFSET:=0}"
: "${OPTIM_MBO_WINDOW_SIZE:=2}"

# ------------------------
# Eval setup
# ------------------------
: "${RUN_EVAL:=1}"
: "${EVAL_PROMPT_TYPE:=qwen-boxed}"
: "${EVAL_N_SAMPLING:=16}"
: "${EVAL_TEMPERATURE:=1}"
: "${EVAL_CUDA_VISIBLE_DEVICES:=0}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Missing DATASET=${DATASET} training data: ${TRAIN_FILE}" >&2
    echo "See ${script_dir}/DFT_TASKS.md for the preparation command." >&2
    exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
    echo "Missing DATASET=${DATASET} validation data: ${VAL_FILE}" >&2
    echo "See ${script_dir}/DFT_TASKS.md for the preparation command." >&2
    exit 1
fi

echo "Dataset profile: ${DATASET} (task=${TASK}, eval=${TASK_EVAL}, epochs=${TOTAL_EPOCHS})"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: DATASET=${DATASET} TRAIN_FILE=${TRAIN_FILE} VAL_FILE=${VAL_FILE} MODEL_NAME=${MODEL_NAME}"
    exit 0
fi

# MODEL_NAME is selected independently of DATASET.
${PYTHON_BIN} -m verl.trainer.fsdp_dft_trainer \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.prompt_key=${TRAIN_PROMPT_KEY} \
    data.response_key=${TRAIN_RESPONSE_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.max_length=${MAX_LENGTH} \
    "data.prompt_dict_keys=${TRAIN_PROMPT_DICT_KEYS}" \
    "data.response_dict_keys=${TRAIN_RESPONSE_DICT_KEYS}" \
    model.partial_pretrain=${MODEL_NAME} \
    model.lora_rank=${LORA_RANK} \
    model.lora_alpha=${LORA_ALPHA} \
    model.target_modules=${LORA_TARGET_MODULES} \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    optim.name=${OPTIM_NAME} \
    optim.module=${OPTIM_MODULE} \
    optim.loss_mode=${LOSS_MODE} \
    optim.psft.clip_ratio_low=${PSFT_CLIP_RATIO_LOW} \
    optim.psft.clip_ratio_high=${PSFT_CLIP_RATIO_HIGH} \
    optim.psft.mini_batch_size=${PSFT_MINI_BATCH_SIZE} \
    optim.psft.warmup_steps=${PSFT_WARMUP_STEPS} \
    optim.spft.lambda=${SPFT_LAMBDA} \
    optim.spft.eps=${SPFT_EPS} \
    optim.spft.weight_threshold=${SPFT_WEIGHT_THRESHOLD} \
    optim.spft.reference_cpu_offload=${SPFT_REFERENCE_CPU_OFFLOAD} \
    optim.lr=${OPTIM_LR} \
    optim.betas=[${OPTIM_BETA1},${OPTIM_BETA2}] \
    optim.eps=${OPTIM_EPS} \
    optim.weight_decay=${OPTIM_WEIGHT_DECAY} \
    optim.warmup_steps_ratio=${WARMUP_STEPS_RATIO} \
    optim.lr_scheduler=${LR_SCHEDULER} \
    optim.muon_lr=${OPTIM_MUON_LR} \
    optim.aux_lr=${OPTIM_AUX_LR} \
    optim.momentum=${OPTIM_MOMENTUM} \
    optim.nesterov=${OPTIM_NESTEROV} \
    optim.ns_steps=${OPTIM_NS_STEPS} \
    optim.htmuon.alpha=${OPTIM_HTMUON_ALPHA} \
    optim.soren.lamdba=${OPTIM_SOREN_LAMBDA} \
    optim.soren.mode=${OPTIM_SOREN_MODE} \
    optim.mbo.num_centroids=${OPTIM_MBO_NUM_CENTROIDS} \
    optim.mbo.centroid_dim=${OPTIM_MBO_CENTROID_DIM} \
    optim.mbo.gradient_source=${OPTIM_MBO_GRADIENT_SOURCE} \
    optim.mbo.lambda_memory=${OPTIM_MBO_LAMBDA_MEMORY} \
    optim.mbo.memory_init_seed=${OPTIM_MBO_MEMORY_INIT_SEED} \
    optim.mbo.ot_ws_steps=${OPTIM_MBO_OT_WS_STEPS} \
    optim.mbo.ot_step_update=${OPTIM_MBO_OT_STEP_UPDATE} \
    optim.mbo.kmeans_steps=${OPTIM_MBO_KMEANS_STEPS} \
    optim.mbo.ot_epsilon=${OPTIM_MBO_OT_EPSILON} \
    optim.mbo.ot_g_lr=${OPTIM_MBO_OT_G_LR} \
    optim.mbo.ot_y_lr=${OPTIM_MBO_OT_Y_LR} \
    optim.mbo.ot_g_steps=${OPTIM_MBO_OT_G_STEPS} \
    optim.mbo.ot_y_steps=${OPTIM_MBO_OT_Y_STEPS} \
    optim.mbo.offset=${OPTIM_MBO_OFFSET} \
    optim.mbo.window_size=${OPTIM_MBO_WINDOW_SIZE} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${RUN_NAME} \
    trainer.logger=['wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.seed=${SEED} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.n_gpus_per_node=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=${USE_REMOVE_PADDING}

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" ]]; then
    latest_ckpt="$(find "${SAVE_PATH}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [[ -z "${latest_ckpt:-}" ]]; then
        echo "No checkpoint found under ${SAVE_PATH}, skip eval." >&2
        exit 1
    fi

    echo "Running eval on checkpoint: $latest_ckpt"

    case "${TASK_EVAL}" in
        math)
            eval_output_dir="${latest_ckpt}/math_eval_${EVAL_PROMPT_TYPE}_n${EVAL_N_SAMPLING}_t${EVAL_TEMPERATURE}"
            MODEL_NAME_OR_PATH="$latest_ckpt" \
            DATASET="${DATASET}" \
            EVAL_DATA_GROUPS="${EVAL_DATA_GROUPS:-${DATASET_EVAL_GROUPS}}" \
            OUTPUT_DIR="$eval_output_dir" \
            EVAL_PROMPT_TYPE="${EVAL_PROMPT_TYPE}" \
            EVAL_N_SAMPLING="${EVAL_N_SAMPLING}" \
            EVAL_TEMPERATURE="${EVAL_TEMPERATURE}" \
            EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
            PYTHON_BIN="${PYTHON_BIN}" \
            bash "${script_dir}/eval_dft.sh"
            ;;
        *)
            echo "Unknown TASK_EVAL=${TASK_EVAL}" >&2
            exit 2
            ;;
    esac
fi
