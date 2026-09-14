#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/dataset_profiles.sh"
configure_dataset_profile "${script_dir}" "${DATASET:-numina}"
# Models: numina=Qwen/Qwen2.5-Math-1.5B; openr1=Qwen/Qwen2.5-7B-Instruct.
# Set MODEL_NAME_OR_PATH to evaluate a trained checkpoint.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat >&2 <<EOF
Usage:
  MAX_ROLLOUT=N MODEL_NAME_OR_PATH=/path/to/global_step_390 \\
    bash ${script_dir}/eval_power_metrics_wandb.sh
  bash ${script_dir}/eval_power_metrics_wandb.sh N

N means that the evaluator samples N rollouts once, then reports
mean/maj/worst/best at every prefix size 1, 2, 3, ..., N.
All existing EVAL_* variables from eval_dft.sh are supported.
EOF
    exit 0
fi

: "${PYTHON_BIN:=$(command -v python)}"
: "${MODEL_NAME_OR_PATH:=${DATASET_MODEL_NAME}}"
: "${MAX_ROLLOUT:=${1:-}}"

if [[ -z "${MAX_ROLLOUT}" ]]; then
    echo "ERROR: provide MAX_ROLLOUT=N or pass N as the first argument" >&2
    exit 2
fi
if ! [[ "${MAX_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_ROLLOUT must be a positive integer" >&2
    exit 2
fi

max_samples="${MAX_ROLLOUT}"
if (( MAX_ROLLOUT > 64 )); then
    echo "WARNING: requesting ${max_samples} rollouts per question" >&2
fi

: "${EVAL_PROMPT_TYPE:=qwen-boxed}"
: "${EVAL_N_SAMPLING:=${max_samples}}"
: "${EVAL_TEMPERATURE:=1}"
: "${EVAL_CUDA_VISIBLE_DEVICES:=0}"
: "${EVAL_SPLIT:=test}"
: "${EVAL_NUM_TEST_SAMPLE:=-1}"
: "${EVAL_TOP_P:=1}"
: "${EVAL_START:=0}"
: "${EVAL_END:=-1}"
: "${EVAL_SEED:=0}"
: "${EVAL_ANTLR411_PATH:=${repo_root}/.eval_deps/antlr4-python3-runtime-4.11.1}"
: "${EVAL_DATA_GROUPS:=${DATASET_EVAL_GROUPS}}"

if (( EVAL_N_SAMPLING != max_samples )); then
    echo "ERROR: EVAL_N_SAMPLING must equal MAX_ROLLOUT (${max_samples})" >&2
    exit 2
fi

model_path="${MODEL_NAME_OR_PATH%/}"
: "${OUTPUT_DIR:=${script_dir}/eval_outputs/${DATASET}/math_eval_${EVAL_PROMPT_TYPE}_n${max_samples}_t${EVAL_TEMPERATURE}_prefix}"

echo "Running one evaluation with ${max_samples} rollouts per question"
echo "Raw eval output: ${OUTPUT_DIR}"

# Reuse the existing model merge and math_eval.py implementation, while
# exposing the eval seed without modifying the original eval_dft.sh.
: "${LORA_MERGED_MODEL_DIR:=${MODEL_NAME_OR_PATH}/merged_hf}"
eval_model_path="${MODEL_NAME_OR_PATH}"

if [[ -f "${MODEL_NAME_OR_PATH}/adapter_config.json" ]]; then
    if [[ ! -f "${LORA_MERGED_MODEL_DIR}/config.json" ]]; then
        echo "Merging LoRA adapter for evaluation: ${MODEL_NAME_OR_PATH}"
        MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
        LORA_MERGED_MODEL_DIR="${LORA_MERGED_MODEL_DIR}" \
        "${PYTHON_BIN}" - <<'PY'
import json
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_path = os.environ["MODEL_NAME_OR_PATH"]
output_path = os.environ["LORA_MERGED_MODEL_DIR"]

with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as f:
    base_model_path = json.load(f)["base_model_name_or_path"]

model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
model.save_pretrained(output_path, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter_path).save_pretrained(output_path)
PY
    fi
    eval_model_path="${LORA_MERGED_MODEL_DIR}"
fi

run_eval() {
    local data_name="$1"
    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
    PYTHONPATH="${EVAL_ANTLR411_PATH}:${PYTHONPATH:-}" \
    TOKENIZERS_PARALLELISM=false \
    "${PYTHON_BIN}" -u "${repo_root}/math_evaluation/math_eval.py" \
        --model_name_or_path "${eval_model_path}" \
        --data_names "${data_name}" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${EVAL_SPLIT}" \
        --prompt_type "${EVAL_PROMPT_TYPE}" \
        --num_test_sample "${EVAL_NUM_TEST_SAMPLE}" \
        --seed "${EVAL_SEED}" \
        --temperature "${EVAL_TEMPERATURE}" \
        --n_sampling "${max_samples}" \
        --top_p "${EVAL_TOP_P}" \
        --start "${EVAL_START}" \
        --end "${EVAL_END}" \
        --use_vllm
}

pushd "${repo_root}/math_evaluation" >/dev/null
IFS=';' read -ra data_groups <<< "${EVAL_DATA_GROUPS}"
for data_group in "${data_groups[@]}"; do
    run_eval "${data_group}"
done
popd >/dev/null

# Only now read the completed eval files, build every prefix metric, and
# create one W&B run in the shared eval_dft project with four tables.
"${PYTHON_BIN}" "${script_dir}/eval_power_metrics_wandb.py" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --max_rollout "${MAX_ROLLOUT}" \
    --repo_root "${repo_root}" \
    --prompt_type "${EVAL_PROMPT_TYPE}" \
    --antlr_path "${EVAL_ANTLR411_PATH}"
