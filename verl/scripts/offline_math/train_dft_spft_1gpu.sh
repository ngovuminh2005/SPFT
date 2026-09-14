#!/usr/bin/env bash
set -e
#OFFLINE_TRAIN_MODES="dft" bash "$(dirname "${BASH_SOURCE[0]}")/train_1gpu.sh"
OFFLINE_TRAIN_MODES="${OFFLINE_TRAIN_MODES:-spft}" \
OFFLINE_LEARNING_RATES="${OFFLINE_LEARNING_RATES:-1e-4}" \
OFFLINE_SPFT_LAMBDAS="${OFFLINE_SPFT_LAMBDAS:-0.025 0.01}" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_1gpu.sh" "$@"
