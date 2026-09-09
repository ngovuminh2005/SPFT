#!/usr/bin/env bash
set -euo pipefail

# Download and preprocess the datasets required by the math task.
# Offline math is intentionally excluded; use scripts/offline_math/generate_data.sh.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
numina_dir="${script_dir}/data/numina_cot"
math500_dir="${script_dir}/data/math500"
numina_train_end="${NUMINA_TRAIN_END:-100000}"

mkdir -p "${numina_dir}" "${math500_dir}"

if [[ "${FORCE_DOWNLOAD:-false}" == "true" || ! -f "${numina_dir}/train.parquet" ]]; then
    "${python_bin}" "${script_dir}/examples/data_preprocess/numina_cot.py" \
        --local_dir "${numina_dir}" \
        --train_end "${numina_train_end}"
else
    echo "NuminaMath-CoT already exists: ${numina_dir}/train.parquet"
fi

if [[ "${FORCE_DOWNLOAD:-false}" == "true" || ! -f "${math500_dir}/test.parquet" ]]; then
    "${python_bin}" "${script_dir}/examples/data_preprocess/math_dataset.py" \
        --local_dir "${math500_dir}"
else
    echo "Math500 already exists: ${math500_dir}/test.parquet"
fi

echo "Math datasets are ready under ${script_dir}/data"
