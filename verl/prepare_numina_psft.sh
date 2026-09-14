#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
export HF_HOME="${HF_HOME:-${script_dir}/.cache/huggingface}"
train_source="${NUMINA_SOURCE:-${script_dir}/data/numina_cot/train.parquet}"
val_source="${NUMINA_VAL_SOURCE:-${script_dir}/data/math500/test.parquet}"
output_dir="${NUMINA_PSFT_DIR:-${script_dir}/data/numina_psft}"
if [[ ( ! -f "${train_source}" || ! -f "${val_source}" ) && "${train_source}" == "${script_dir}"/* ]]; then
    DATASET=numina bash "${script_dir}/download_datasets.sh"
fi
if [[ -f "${output_dir}/train.parquet" && -f "${output_dir}/test.parquet" && -f "${output_dir}/manifest.json" ]]; then
    echo "Numina PSFT data already exists: ${output_dir}"
    exit 0
fi
exec "${python_bin}" "${script_dir}/examples/data_preprocess/openr1_psft.py" \
    --train-source "${train_source}" \
    --val-source "${val_source}" \
    --local-dir "${output_dir}" \
    --prompt-template numina "$@"
