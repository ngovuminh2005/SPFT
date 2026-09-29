#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the model cache on the data partition instead of the full root partition.
sigmoid_root="$(cd "${script_dir}/../.." && pwd)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${sigmoid_root}/Booster/cache}"

# Keep Ray's UNIX socket path short enough for AF_UNIX (107-byte limit).
export RAY_TMPDIR="${sigmoid_root}/r"
mkdir -p "${RAY_TMPDIR}"

# =========================
# PSFT sweep hyperparameters
# Edit these values and run: bash sweep_psft_1gpu.sh
# =========================
DATASET="${DATASET:-${PSFT_DATASET:-openr1}}"       # numina or openr1
# Change this default to select the model for this sweep; MODEL_NAME/MODEL_PATH
# supplied from the shell still take precedence.
MODEL_NAME="${MODEL_NAME:-${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}}"
EPOCHS_LIST="${EPOCHS_LIST:-${EPOCHS:-${TOTAL_EPOCHS:-10}}}" # e.g. "5 10"
source "${script_dir}/dataset_profiles.sh"
configure_dataset_profile "${script_dir}" "${DATASET}"
# MODEL_NAME selects the model independently of DATASET (Hub ID or local path).

# Baselines: PSFT=AdamW (lr=1e-6, wd=0.1, clip=0.2/0.28); DFT=SorenAuxAdam;
# SPFT=AdamW (lr=1e-4, lambda=0.1); MBO=projection, Muon lr=8e-4.
# Qwen2.5-Math-1.5B in this run uses a 4096-token context.
case "${DATASET}" in
    numina)
        prompt_length_default=2048
        response_length_default=2048
        sequence_length_default=4096
        ;;
    openr1)
        # Leave headroom for fixed demonstrations that can exceed response_length.
        prompt_length_default=384
        response_length_default=3072
        sequence_length_default=4096
        ;;
    *)
        echo "DATASET must be numina or openr1" >&2
        exit 2
        ;;
esac
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-${prompt_length_default}}"       # 512, 1024, 2048
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${response_length_default}}" # Qwen 1.5B: prompt + response <=4096
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-4096}"
# Dynamic micro-batch budgets are separate from the rollout context limit.
# Fixed demonstrations are not truncated to MAX_RESPONSE_LENGTH.
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-8192}"
INFER_MAX_TOKEN_LEN="${INFER_MAX_TOKEN_LEN:-8192}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-${sequence_length_default}}"

if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MODEL_CONTEXT_LENGTH )); then
    echo "MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must be <= MODEL_CONTEXT_LENGTH (${MODEL_CONTEXT_LENGTH}); got ${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH}" >&2
    exit 2
fi

# Sweep dimensions. Give space-separated values to test multiple settings.
PSFT_CLIP_RATIO_HIGHS="${PSFT_CLIP_RATIO_HIGHS:-0.28}" # 0.2, 0.24, 0.28
OPTIM_LRS="${OPTIM_LRS:-1e-6}"                            # 5e-7, 1e-6, 3e-6
OPTIM_WEIGHT_DECAYS="${OPTIM_WEIGHT_DECAYS:-0.1}"        # 0, 0.01, 0.1

# Fixed PSFT settings; 0.35 vLLM utilization is the 1-GPU-safe value.
PSFT_CLIP_RATIO_LOW="${PSFT_CLIP_RATIO_LOW:-0.2}"       # upstream lower clip
PSFT_WARMUP_STEPS="${PSFT_WARMUP_STEPS:-10}"             # optimizer steps
PSFT_MINI_BATCH_SIZE="${PSFT_MINI_BATCH_SIZE:-8}"      # upstream PPO mini-batch
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"             # upstream effective batch
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" # 0.80 upstream
TEST_FREQ="${TEST_FREQ:-100}"
PSFT_SAVE_FREQ="${PSFT_SAVE_FREQ:-${SAVE_FREQ:-1000000}}"
RUN_EVAL="${RUN_EVAL:-1}" # 1: run eval_dft after training; 0: disable
EVAL_PROMPT_TYPE="${EVAL_PROMPT_TYPE:-qwen-boxed}"
EVAL_N_SAMPLING="${EVAL_N_SAMPLING:-16}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

run_idx="${RUN_IDX_START:-0}"
dataset="${DATASET}"
case "${dataset}" in
    numina|openr1) ;;
    *) echo "DATASET must be numina or openr1" >&2; exit 2 ;;
esac
for epochs in ${EPOCHS_LIST}; do
for clip_high in ${PSFT_CLIP_RATIO_HIGHS}; do
    for optim_lr in ${OPTIM_LRS}; do
        for weight_decay in ${OPTIM_WEIGHT_DECAYS}; do
            run_idx=$((run_idx + 1))
            DATASET="${dataset}" \
            MODEL_NAME="${MODEL_NAME}" \
            EXPERIMENT_NAME="${EXPERIMENT_NAME:-${dataset}-psft_${run_idx}_${epochs}ep_adamw_lr${optim_lr}_clip${clip_high}}" \
            SAVE_PATH="${SAVE_PATH:-${script_dir}/checkpoints/${dataset}-psft_${run_idx}_lr${optim_lr}_clip${clip_high}_wd${weight_decay}}" \
            EPOCHS="${epochs}" TOTAL_EPOCHS="${epochs}" \
            MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
            MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
            PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN}" \
            INFER_MAX_TOKEN_LEN="${INFER_MAX_TOKEN_LEN}" \
            ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS}" \
            TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
            PSFT_CLIP_RATIO_LOW="${PSFT_CLIP_RATIO_LOW}" \
            PSFT_WARMUP_STEPS="${PSFT_WARMUP_STEPS}" \
            PSFT_MINI_BATCH_SIZE="${PSFT_MINI_BATCH_SIZE}" \
            ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
            OPTIM_LR="${optim_lr}" \
            OPTIM_WEIGHT_DECAY="${weight_decay}" \
            PSFT_CLIP_RATIO_HIGH="${clip_high}" \
            TEST_FREQ="${TEST_FREQ}" SAVE_FREQ="${PSFT_SAVE_FREQ}" \
            RUN_EVAL="${RUN_EVAL}" \
            EVAL_PROMPT_TYPE="${EVAL_PROMPT_TYPE}" \
            EVAL_N_SAMPLING="${EVAL_N_SAMPLING}" \
            EVAL_TEMPERATURE="${EVAL_TEMPERATURE}" \
            EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
            bash "${script_dir}/train_psft_1gpu.sh" "$@"
done
done
done
done
