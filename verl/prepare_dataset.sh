#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/dataset_profiles.sh"
configure_dataset_profile "${script_dir}" "${DATASET:-numina}"

case "${DATASET}" in
    numina)
        "${script_dir}/download_datasets.sh"
        exec "${DATASET_PREPARE_SCRIPT}" "$@"
        ;;
    openr1)
        exec "${DATASET_PREPARE_SCRIPT}" "$@"
        ;;
    *)
        echo "DATASET=${DATASET} has no two-file PSFT preparation profile" >&2
        exit 2
        ;;
esac
