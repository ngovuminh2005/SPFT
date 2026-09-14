#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LOSS_MODE:=spft}"
: "${SAVE_FREQ:=-1}"
: "${EPOCHS:=${TOTAL_EPOCHS:-1}}"
export LOSS_MODE SAVE_FREQ EPOCHS

exec bash "${script_dir}/train_dft_1gpu.sh" "$@"
