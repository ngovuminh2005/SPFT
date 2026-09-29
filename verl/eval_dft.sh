#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/dataset_profiles.sh"
configure_dataset_profile "${script_dir}" "${DATASET:-numina}"
# MODEL_NAME selects the base model independently of DATASET.
# Training wrappers override this with the trained/merged checkpoint.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash ${script_dir}/eval_dft.sh" >&2
    exit 1
fi

: "${PYTHON_BIN:=$(command -v python)}"
: "${MODEL_NAME_OR_PATH:=${MODEL_NAME}}"
: "${LORA_MERGED_MODEL_DIR:=${MODEL_NAME_OR_PATH}/merged_hf}"
: "${EVAL_PROMPT_TYPE:=qwen-boxed}"
: "${EVAL_N_SAMPLING:=16}"
: "${EVAL_TEMPERATURE:=1}"
: "${EVAL_CUDA_VISIBLE_DEVICES:=0}"
: "${EVAL_SPLIT:=test}"
: "${EVAL_NUM_TEST_SAMPLE:=-1}"
: "${EVAL_TOP_P:=1}"
: "${EVAL_START:=0}"
: "${EVAL_END:=-1}"
: "${EVAL_ANTLR411_PATH:=${repo_root}/.eval_deps/antlr4-python3-runtime-4.11.1}"
: "${EVAL_DATA_GROUPS:=${DATASET_EVAL_GROUPS}}"

# vLLM requires a complete Hugging Face model. Merge adapter-only LoRA
# checkpoints with their base model before evaluation.
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

    MODEL_NAME_OR_PATH="${LORA_MERGED_MODEL_DIR}"
fi

: "${OUTPUT_DIR:=${script_dir}/eval_outputs/${DATASET}/math_eval_${EVAL_PROMPT_TYPE}_n${EVAL_N_SAMPLING}_t${EVAL_TEMPERATURE}}"

run_eval() {
    local data_name="$1"

    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
    PYTHONPATH="${EVAL_ANTLR411_PATH}:${PYTHONPATH:-}" \
    TOKENIZERS_PARALLELISM=false \
    "${PYTHON_BIN}" -u math_eval.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_names "${data_name}" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${EVAL_SPLIT}" \
        --prompt_type "${EVAL_PROMPT_TYPE}" \
        --num_test_sample "${EVAL_NUM_TEST_SAMPLE}" \
        --seed 0 \
        --temperature "${EVAL_TEMPERATURE}" \
        --n_sampling "${EVAL_N_SAMPLING}" \
        --top_p "${EVAL_TOP_P}" \
        --start "${EVAL_START}" \
        --end "${EVAL_END}" \
        --use_vllm
}

echo "Eval model: ${MODEL_NAME_OR_PATH}"
echo "Eval output dir: ${OUTPUT_DIR}"

pushd "${repo_root}/math_evaluation" >/dev/null
IFS=';' read -ra data_groups <<< "${EVAL_DATA_GROUPS}"
for data_group in "${data_groups[@]}"; do
    run_eval "${data_group}"
done
popd >/dev/null

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import glob
import json
import os
import sys

output_dir = sys.argv[1]
metrics = {}
for path in sorted(glob.glob(os.path.join(output_dir, "*_metrics.json"))):
    dataset = os.path.basename(path).removesuffix("_metrics.json")
    with open(path, "r", encoding="utf8") as f:
        metrics[dataset] = json.load(f)

summary = {
    dataset: {
        "acc": value.get("acc"),
        "mean_acc": value.get("mean_acc"),
        "all_acc": value.get("all_acc"),
        "num_samples": value.get("num_samples"),
        "num_scores": value.get("num_scores"),
        "timeout_samples": value.get("timeout_samples"),
        "empty_samples": value.get("empty_samples"),
    }
    for dataset, value in metrics.items()
}

if summary:
    summary["avg"] = {
        "acc": sum(v["acc"] for v in summary.values() if v["acc"] is not None) / len(summary),
        "mean_acc": sum(v["mean_acc"] for v in summary.values() if v["mean_acc"] is not None) / len(summary),
    }

summary_path = os.path.join(output_dir, "eval_metrics_summary.json")
with open(summary_path, "w", encoding="utf8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"Saved metric summary: {summary_path}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
