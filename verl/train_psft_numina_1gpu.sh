#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NGPUS_PER_NODE=1
export NNODES=1
export ROLLOUT_TP=1
exec bash "${script_dir}/train_psft_numina.sh" "$@"
