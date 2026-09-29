#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="${script_dir}/third_party/psft"
python_bin="${PYTHON_BIN:-python3}"
source "${script_dir}/dataset_profiles.sh"
if [[ -z "${DATASET:-}" && -n "${PSFT_DATASET:-}" ]]; then
    DATASET="${PSFT_DATASET}"
fi
configure_dataset_profile "${script_dir}" "${DATASET:-numina}"
# Resolve local paths before changing cwd for upstream's Hydra searchpath.
train_file="$(realpath -m "${TRAIN_FILE:-${DATASET_PSFT_TRAIN_FILE}}")"
val_file="$(realpath -m "${VAL_FILE:-${TEST_FILE:-${DATASET_PSFT_VAL_FILE}}}")"
save_path="$(realpath -m "${SAVE_PATH:-${script_dir}/checkpoints/psft-${DATASET}}")"
EPOCHS="${EPOCHS:-${TOTAL_EPOCHS:-10}}"
epochs="${EPOCHS}"
export EPOCHS
: "${RUN_EVAL:=1}"
: "${EVAL_PROMPT_TYPE:=qwen-boxed}"
: "${EVAL_N_SAMPLING:=16}"
: "${EVAL_TEMPERATURE:=1}"
: "${EVAL_CUDA_VISIBLE_DEVICES:=0}"
: "${EVAL_SPLIT:=test}"
: "${EVAL_NUM_TEST_SAMPLE:=-1}"
: "${EVAL_TOP_P:=1}"
: "${EVAL_START:=0}"
: "${EVAL_END:=-1}"
: "${EVAL_ANTLR411_PATH:=${script_dir}/../.eval_deps/antlr4-python3-runtime-4.11.1}"
: "${EVAL_DATA_GROUPS:=${DATASET_EVAL_GROUPS}}"
ngpus="${NGPUS_PER_NODE:-8}"
gen_tp="${ROLLOUT_TP:-2}"
if (( ngpus < 1 || gen_tp < 1 || ngpus % gen_tp != 0 )); then
    echo "NGPUS_PER_NODE must be positive and divisible by ROLLOUT_TP" >&2
    exit 2
fi
export PYTHONPATH="${runtime_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
cd "${runtime_dir}"

# Same overrides as upstream recipe/psft/run_psft.sh, with local parquet paths.
args=(
    -m recipe.psft.main_psft
    "data.train_files=${train_file}"
    "data.val_files=${val_file}"
    data.prompt_key=prompt data.truncation=left
    "data.max_prompt_length=${MAX_PROMPT_LENGTH:-2048}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH:-6144}"
    "data.gen_batch_size=${TRAIN_BATCH_SIZE:-256}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-256}"
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=1
    algorithm.adv_estimator=psft algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0
    "actor_rollout_ref.actor.clip_ratio_low=${PSFT_CLIP_RATIO_LOW:-0.2}"
    "actor_rollout_ref.actor.clip_ratio_high=${PSFT_CLIP_RATIO_HIGH:-0.28}"
    actor_rollout_ref.actor.clip_ratio_c=10.0 actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN:-8192}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN:-8192}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN:-8192}"
    # MODEL_NAME is resolved independently of DATASET by the shared profile.
    "actor_rollout_ref.model.path=${MODEL_NAME}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "actor_rollout_ref.actor.optim.lr=${OPTIM_LR:-1e-6}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${PSFT_WARMUP_STEPS:-10}"
    "actor_rollout_ref.actor.optim.weight_decay=${OPTIM_WEIGHT_DECAY:-0.1}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PSFT_MINI_BATCH_SIZE:-32}"
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
    reward_model.reward_manager=dapo reward_model.overlong_buffer.enable=False
    reward_model.overlong_buffer.len=2048 reward_model.overlong_buffer.penalty_factor=1.0
    "trainer.logger=${TRAINER_LOGGER:-[console,wandb]}"
    "trainer.project_name=${PROJECT_NAME:-PSFT}"
    "trainer.experiment_name=${EXPERIMENT_NAME:-Qwen2.5-7B-Instruct-OpenR1-4k-PSFT}"
    "trainer.n_gpus_per_node=${ngpus}" "trainer.nnodes=${NNODES:-1}"
    trainer.val_before_train=False
    "trainer.test_freq=${TEST_FREQ:-100}" "trainer.save_freq=${SAVE_FREQ:-100}"
    "trainer.total_epochs=${epochs}"
    "trainer.default_local_dir=${save_path}" trainer.resume_mode=auto
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "runtime=${runtime_dir}" "python=${python_bin}" "${args[@]}" "$@"
    exit 0
fi
for data_file in "${train_file}" "${val_file}"; do
    if [[ ! -f "${data_file}" ]]; then
    echo "Missing parquet: ${data_file}. Run ${DATASET_PREPARE_SCRIPT} first." >&2
        exit 1
    fi
done

"${python_bin}" "${args[@]}" "$@"

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" ]]; then
    latest_ckpt="$(find "${save_path}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [[ -z "${latest_ckpt}" || ! -d "${latest_ckpt}/actor" ]]; then
        echo "No PSFT actor checkpoint found under ${save_path}; skip eval." >&2
        exit 1
    fi

    merged_hf="${PSFT_EVAL_MODEL_DIR:-${latest_ckpt}/merged_hf}"
    if [[ ! -f "${merged_hf}/config.json" ]] || ! find "${merged_hf}" -maxdepth 1 -type f \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
        echo "Merging PSFT FSDP checkpoint for evaluation: ${latest_ckpt}/actor"
        "${python_bin}" -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${latest_ckpt}/actor" \
            --target_dir "${merged_hf}"
    fi

    eval_output_dir="${latest_ckpt}/math_eval_${EVAL_PROMPT_TYPE}_n${EVAL_N_SAMPLING}_t${EVAL_TEMPERATURE}"
    echo "Running PSFT eval on checkpoint: ${merged_hf}"
    MODEL_NAME_OR_PATH="${merged_hf}" \
    DATASET="${DATASET}" \
    EVAL_DATA_GROUPS="${EVAL_DATA_GROUPS}" \
    OUTPUT_DIR="${eval_output_dir}" \
    EVAL_PROMPT_TYPE="${EVAL_PROMPT_TYPE}" \
    EVAL_N_SAMPLING="${EVAL_N_SAMPLING}" \
    EVAL_TEMPERATURE="${EVAL_TEMPERATURE}" \
    EVAL_SPLIT="${EVAL_SPLIT}" \
    EVAL_NUM_TEST_SAMPLE="${EVAL_NUM_TEST_SAMPLE}" \
    EVAL_TOP_P="${EVAL_TOP_P}" \
    EVAL_START="${EVAL_START}" \
    EVAL_END="${EVAL_END}" \
    EVAL_ANTLR411_PATH="${EVAL_ANTLR411_PATH}" \
    EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
    PYTHON_BIN="${python_bin}" \
    bash "${script_dir}/eval_dft.sh"
fi
