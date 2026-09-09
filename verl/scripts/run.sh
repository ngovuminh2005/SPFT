#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
eval_script="${repo_root}/verl/eval_power_metrics_wandb.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Runs the six requested baselines with 50 rollouts and eval seed 1337."
    exit 0
fi

max_rollout=50
eval_seed=1337

baselines=(
    "muon-lora"
    "numina-cot-sft-qwen-2.5-math-1.5b-1gpu-adamw"
    "mbo_minibatch_21_muon_lr_8e-4"
    "mbo_minibatch_56_muon_lr_8e-4"
    "mbo_minibatch_55_muon_lr_8e-4"d
)

for baseline in "${baselines[@]}"; do
    checkpoint="${repo_root}/verl/checkpoints/${baseline}/global_step_390"
    if [[ ! -d "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi

    echo "===== Evaluating ${baseline}: ${max_rollout} rollouts, seed=${eval_seed} ====="
    MODEL_NAME_OR_PATH="${checkpoint}" \
    EVAL_N_SAMPLING="${max_rollout}" \
    EVAL_SEED="${eval_seed}" \
    bash "${eval_script}" "${max_rollout}"
done

echo "Completed ${#baselines[@]} baselines"
