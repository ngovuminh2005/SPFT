#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LOSS_MODE:=spft}"
export LOSS_MODE

exec bash "${script_dir}/train_dft_1gpu.sh" "$@"
