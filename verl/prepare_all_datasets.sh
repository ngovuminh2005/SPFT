#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Space-separated profiles to prepare. Default: all training/eval datasets.
DATASETS="${DATASETS:-numina openr1}"  # numina, openr1

for dataset in ${DATASETS}; do
    case "${dataset}" in
        numina|openr1) ;;
        *)
            echo "Unknown dataset: ${dataset}; expected numina or openr1" >&2
            exit 2
            ;;
    esac

    echo "Preparing dataset=${dataset}"
    DATASET="${dataset}" bash "${script_dir}/prepare_dataset.sh"
done

echo "All requested datasets are ready."
