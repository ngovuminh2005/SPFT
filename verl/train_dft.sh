# train data generation
# python examples/data_preprocess/numina_cot.py --train_end 100000
# eval data generation
# python examples/data_preprocess/math_dataset.py


nproc_per_node=8
project_name=numina-cot

experiment_name=numina-cot-sft-qwen-2.5-math-1.5b
save_path=checkpoints/$experiment_name

optim_name=${OPTIM_NAME:-adamw}
optim_module=${OPTIM_MODULE:-muon}
optim_lr=${OPTIM_LR:-5e-5}
optim_beta1=${OPTIM_BETA1:-0.9}
optim_beta2=${OPTIM_BETA2:-0.95}
optim_eps=${OPTIM_EPS:-1e-8}
optim_weight_decay=${OPTIM_WEIGHT_DECAY:-0.01}
optim_muon_lr=${OPTIM_MUON_LR:-$optim_lr}
optim_aux_lr=${OPTIM_AUX_LR:-$optim_lr}
optim_momentum=${OPTIM_MOMENTUM:-0.95}
optim_nesterov=${OPTIM_NESTEROV:-true}
optim_ns_steps=${OPTIM_NS_STEPS:-5}

optim_mbo_num_centroids=${OPTIM_MBO_NUM_CENTROIDS:-16}
optim_mbo_centroid_dim=${OPTIM_MBO_CENTROID_DIM:-1024}
optim_mbo_lambda_memory=${OPTIM_MBO_LAMBDA_MEMORY:-0.01}
optim_mbo_memory_init_seed=${OPTIM_MBO_MEMORY_INIT_SEED:-null}
optim_mbo_ot_ws_steps=${OPTIM_MBO_OT_WS_STEPS:-512}
optim_mbo_ot_step_update=${OPTIM_MBO_OT_STEP_UPDATE:-64}
optim_mbo_ot_epsilon=${OPTIM_MBO_OT_EPSILON:-0.03}
optim_mbo_ot_g_lr=${OPTIM_MBO_OT_G_LR:-0.1}
optim_mbo_ot_y_lr=${OPTIM_MBO_OT_Y_LR:-0.01}
optim_mbo_ot_g_steps=${OPTIM_MBO_OT_G_STEPS:-5}
optim_mbo_ot_y_steps=${OPTIM_MBO_OT_Y_STEPS:-5}

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_dft_trainer \
    data.train_files=data/numina_cot/train.parquet \
    data.val_files=data/math500/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=2048 \
    optim.name=$optim_name \
    optim.module=$optim_module \
    optim.lr=$optim_lr \
    optim.betas=[$optim_beta1,$optim_beta2] \
    optim.eps=$optim_eps \
    optim.weight_decay=$optim_weight_decay \
    optim.muon_lr=$optim_muon_lr \
    optim.aux_lr=$optim_aux_lr \
    optim.momentum=$optim_momentum \
    optim.nesterov=$optim_nesterov \
    optim.ns_steps=$optim_ns_steps \
    optim.mbo.num_centroids=$optim_mbo_num_centroids \
    optim.mbo.centroid_dim=$optim_mbo_centroid_dim \
    optim.mbo.lambda_memory=$optim_mbo_lambda_memory \
    optim.mbo.memory_init_seed=$optim_mbo_memory_init_seed \
    optim.mbo.ot_ws_steps=$optim_mbo_ot_ws_steps \
    optim.mbo.ot_step_update=$optim_mbo_ot_step_update \
    optim.mbo.ot_epsilon=$optim_mbo_ot_epsilon \
    optim.mbo.ot_g_lr=$optim_mbo_ot_g_lr \
    optim.mbo.ot_y_lr=$optim_mbo_ot_y_lr \
    optim.mbo.ot_g_steps=$optim_mbo_ot_g_steps \
    optim.mbo.ot_y_steps=$optim_mbo_ot_y_steps \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
