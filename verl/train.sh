#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset="${DATASET:-numina}"

DATASET="${dataset}" bash "${script_dir}/sweep_psft_1gpu.sh"
DATASET="${dataset}" bash "${script_dir}/sweep_spft_1gpu.sh"
