#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="${DATASET:-numina}"
exec bash "${script_dir}/train_psft.sh" "$@"
