#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_idx=1
for mbo_gradient_source in projection; do
for optim_muon_lr in 8e-4; do
for mbo_num_centroids in 64; do
    for mbo_centroid_dim in 4096; do
        for mbo_singular_min_tau in 0.01; do
        for mbo_lambda_memory in 0.1; do
            for mbo_memory_init_seed in null; do
                for mbo_ot_ws_steps in 512; do
                    for mbo_ot_step_update in 512; do
                        for mbo_ot_epsilon in 0.07; do
                            for mbo_ot_g_lr in 0.05; do
                                for mbo_ot_y_lr in 0.05; do
                                    for mbo_ot_g_steps in 5; do
                                        for mbo_ot_y_steps in 1; do
                                            for mbo_offset in 3; do
                                                for mbo_window_size in 20; do
                                                    run_idx=$((run_idx + 1))
                                                    experiment_name="mbo_${run_idx}_projection"
                                                    echo "Starting ${experiment_name}:"
                                                    echo "  OPTIM_LR=5e-5"
                                                    echo "  OPTIM_MUON_LR=${optim_muon_lr}"
                                                    echo "  OPTIM_MBO_NUM_CENTROIDS=${mbo_num_centroids}"
                                                    echo "  OPTIM_MBO_CENTROID_DIM=${mbo_centroid_dim}"
                                                    echo "  OPTIM_MBO_SINGULAR_MIN_TAU=${mbo_singular_min_tau}"
                                                    echo "  OPTIM_MBO_GRADIENT_SOURCE=${mbo_gradient_source}"
                                                    echo "  OPTIM_MBO_LAMBDA_MEMORY=${mbo_lambda_memory}"
                                                    echo "  OPTIM_MBO_MEMORY_INIT_SEED=${mbo_memory_init_seed}"
                                                    echo "  OPTIM_MBO_OT_WS_STEPS=${mbo_ot_ws_steps}"
                                                    echo "  OPTIM_MBO_OT_STEP_UPDATE=${mbo_ot_step_update}"
                                                    echo "  OPTIM_MBO_OT_EPSILON=${mbo_ot_epsilon}"
                                                    echo "  OPTIM_MBO_OT_G_LR=${mbo_ot_g_lr}"
                                                    echo "  OPTIM_MBO_OT_Y_LR=${mbo_ot_y_lr}"
                                                    echo "  OPTIM_MBO_OT_G_STEPS=${mbo_ot_g_steps}"
                                                    echo "  OPTIM_MBO_OT_Y_STEPS=${mbo_ot_y_steps}"
                                                    echo "  OPTIM_MBO_OFFSET=${mbo_offset}"
                                                    echo "  OPTIM_MBO_WINDOW_SIZE=${mbo_window_size}"

                                                    EXPERIMENT_NAME="${experiment_name}" \
                                                    SAVE_FREQ="100" \
                                                    OPTIM_LR="5e-5" \
                                                    OPTIM_MUON_LR="${optim_muon_lr}" \
                                                    OPTIM_MBO_NUM_CENTROIDS="${mbo_num_centroids}" \
                                                    OPTIM_MBO_CENTROID_DIM="${mbo_centroid_dim}" \
                                                    OPTIM_MBO_SINGULAR_MIN_TAU="${mbo_singular_min_tau}" \
                                                    OPTIM_MBO_GRADIENT_SOURCE="${mbo_gradient_source}" \
                                                    OPTIM_MBO_LAMBDA_MEMORY="${mbo_lambda_memory}" \
                                                    OPTIM_MBO_MEMORY_INIT_SEED="${mbo_memory_init_seed}" \
                                                    OPTIM_MBO_OT_WS_STEPS="${mbo_ot_ws_steps}" \
                                                    OPTIM_MBO_OT_STEP_UPDATE="${mbo_ot_step_update}" \
                                                    OPTIM_MBO_OT_EPSILON="${mbo_ot_epsilon}" \
                                                    OPTIM_MBO_OT_G_LR="${mbo_ot_g_lr}" \
                                                    OPTIM_MBO_OT_Y_LR="${mbo_ot_y_lr}" \
                                                    OPTIM_MBO_OT_G_STEPS="${mbo_ot_g_steps}" \
                                                    OPTIM_MBO_OT_Y_STEPS="${mbo_ot_y_steps}" \
                                                    OPTIM_MBO_OFFSET="${mbo_offset}" \
                                                    OPTIM_MBO_WINDOW_SIZE="${mbo_window_size}" \
                                                    bash "${script_dir}/train_dft_1gpu.sh"
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
        done
    done
done
done
done
