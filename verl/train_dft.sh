#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/task_profiles.sh"
configure_dft_task_profile "${script_dir}"

# train data generation
# python examples/data_preprocess/numina_cot.py --train_end 100000
# eval data generation
# python examples/data_preprocess/math_dataset.py


nproc_per_node="${NPROC_PER_NODE:-8}"
project_name="${PROJECT_NAME}"

experiment_name="${EXPERIMENT_NAME}"
save_path="$(realpath -m "${SAVE_PATH:-${script_dir}/checkpoints/${experiment_name}}")"
EPOCHS="${EPOCHS:-${TOTAL_EPOCHS:-1}}"
total_epochs="${EPOCHS}"
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
: "${PYTHON_BIN:=$(command -v python)}"

for data_file in "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [[ ! -f "${data_file}" ]]; then
        echo "Missing DATASET=${DATASET} data: ${data_file}" >&2
        echo "Run ${DATASET_PREPARE_SCRIPT} first." >&2
        exit 1
    fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: DATASET=${DATASET} TRAIN_FILE=${TRAIN_FILE} VAL_FILE=${VAL_FILE} MODEL_NAME=${MODEL_NAME} EPOCHS=${total_epochs}"
    exit 0
fi

optim_name=${OPTIM_NAME:-adamw}
optim_module=${OPTIM_MODULE:-muon}
optim_lr=${OPTIM_LR:-5e-5}
optim_beta1=${OPTIM_BETA1:-0.9}
optim_beta2=${OPTIM_BETA2:-0.95}
optim_eps=${OPTIM_EPS:-1e-8}
optim_weight_decay=${OPTIM_WEIGHT_DECAY:-0.01}
optim_muon_lr=${OPTIM_MUON_LR:-$optim_lr}
optim_aux_lr=${OPTIM_AUX_LR:-$optim_lr}
optim_momentum=${OPTIM_MOMENTUM:-0.95}
optim_nesterov=${OPTIM_NESTEROV:-true}
optim_ns_steps=${OPTIM_NS_STEPS:-5}

optim_mbo_num_centroids=${OPTIM_MBO_NUM_CENTROIDS:-16}
optim_mbo_centroid_dim=${OPTIM_MBO_CENTROID_DIM:-1024}
optim_mbo_lambda_memory=${OPTIM_MBO_LAMBDA_MEMORY:-0.01}
optim_mbo_memory_init_seed=${OPTIM_MBO_MEMORY_INIT_SEED:-null}
optim_mbo_ot_ws_steps=${OPTIM_MBO_OT_WS_STEPS:-512}
optim_mbo_ot_step_update=${OPTIM_MBO_OT_STEP_UPDATE:-64}
optim_mbo_ot_epsilon=${OPTIM_MBO_OT_EPSILON:-0.03}
optim_mbo_ot_g_lr=${OPTIM_MBO_OT_G_LR:-0.1}
optim_mbo_ot_y_lr=${OPTIM_MBO_OT_Y_LR:-0.01}
optim_mbo_ot_g_steps=${OPTIM_MBO_OT_G_STEPS:-5}
optim_mbo_ot_y_steps=${OPTIM_MBO_OT_Y_STEPS:-5}

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_dft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key="${TRAIN_PROMPT_KEY}" \
    data.response_key="${TRAIN_RESPONSE_KEY}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-256}" \
    data.max_length="${MAX_LENGTH:-2048}" \
    optim.name=$optim_name \
    optim.module=$optim_module \
    optim.lr=$optim_lr \
    optim.betas=[$optim_beta1,$optim_beta2] \
    optim.eps=$optim_eps \
    optim.weight_decay=$optim_weight_decay \
    optim.muon_lr=$optim_muon_lr \
    optim.aux_lr=$optim_aux_lr \
    optim.momentum=$optim_momentum \
    optim.nesterov=$optim_nesterov \
    optim.ns_steps=$optim_ns_steps \
    optim.mbo.num_centroids=$optim_mbo_num_centroids \
    optim.mbo.centroid_dim=$optim_mbo_centroid_dim \
    optim.mbo.lambda_memory=$optim_mbo_lambda_memory \
    optim.mbo.memory_init_seed=$optim_mbo_memory_init_seed \
    optim.mbo.ot_ws_steps=$optim_mbo_ot_ws_steps \
    optim.mbo.ot_step_update=$optim_mbo_ot_step_update \
    optim.mbo.ot_epsilon=$optim_mbo_ot_epsilon \
    optim.mbo.ot_g_lr=$optim_mbo_ot_g_lr \
    optim.mbo.ot_y_lr=$optim_mbo_ot_y_lr \
    optim.mbo.ot_g_steps=$optim_mbo_ot_g_steps \
    optim.mbo.ot_y_steps=$optim_mbo_ot_y_steps \
    data.prompt_dict_keys="${TRAIN_PROMPT_DICT_KEYS}" \
    data.response_dict_keys="${TRAIN_RESPONSE_DICT_KEYS}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU_8GPU:-4}" \
    model.partial_pretrain="${MODEL_NAME}" \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.total_epochs="${total_epochs}" \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" ]]; then
    latest_ckpt="$(find "${save_path}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [[ -z "${latest_ckpt}" || ! -f "${latest_ckpt}/config.json" ]]; then
        echo "No DFT Hugging Face checkpoint found under ${save_path}; cannot run eval." >&2
        exit 1
    fi

    eval_output_dir="${latest_ckpt}/math_eval_${EVAL_PROMPT_TYPE}_n${EVAL_N_SAMPLING}_t${EVAL_TEMPERATURE}"
    echo "Running DFT eval on checkpoint: ${latest_ckpt}"
    MODEL_NAME_OR_PATH="${latest_ckpt}" \
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
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${script_dir}/eval_dft.sh"
fi
