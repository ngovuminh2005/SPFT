#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
export HF_HOME="${HF_HOME:-${script_dir}/.cache/huggingface}"
train_source="${OPENR1_SOURCE:-wh-zhu/train_openr1_4k}"
val_source="${OPENR1_VAL_SOURCE:-wh-zhu/aime-24}"
output_dir="${OPENR1_PSFT_DIR:-${script_dir}/data/openr1_psft}"

if [[ -f "${output_dir}/train.parquet" && -f "${output_dir}/test.parquet" && -f "${output_dir}/manifest.json" ]]; then
    echo "OpenR1 PSFT data already exists: ${output_dir}"
    exit 0
fi

exec "${python_bin}" "${script_dir}/examples/data_preprocess/openr1_psft.py" \
    --train-source "${train_source}" \
    --val-source "${val_source}" \
    --local-dir "${output_dir}" \
    --prompt-template numina "$@"
