# DFT task profiles

Only mathematical-reasoning task profiles are supported. They share the same
DFT/SPFT/PSFT trainer, optimizer, checkpointing, logging, Math500 validation,
and post-training math evaluator.

## Math

```bash
bash scripts/math/train_1gpu.sh
```

This task trains on the first 100,000 NuminaMath-CoT examples and uses their
gold solutions as targets. The launcher prepares NuminaMath-CoT and Math500
data when they are missing.

## Offline math

First build the rejection-sampling dataset; this job is resumable and does not
start training:

```bash
bash scripts/offline_math/generate_data.sh
```

Qwen2.5-Math-1.5B generates four responses for each of 100,000 NuminaMath-CoT
questions. Math-Verify retains correct responses in
`data/offline_math/train.parquet`; `manifest.json` records the generator and
all source and generation settings.

Then train:

```bash
bash scripts/offline_math/train_1gpu.sh
```

Training never generates responses. It validates the dataset and requires the
training base model to exactly match the generator recorded in the manifest.

## Loss modes

`dft` is the default. `spft` and `psft` are available through the generic
launcher, for example:

```bash
TASK=math LOSS_MODE=psft OPTIM_NAME=adamw bash train_dft_1gpu.sh
```

The task-specific launchers default to full-parameter training; the generic
launcher and sweep default to LoRA rank 8 (set `LORA_RANK=0` for full tuning).
Task profiles use global batch 256,
micro-batch 8, maximum sequence length 2048, warmup ratio 0.1, and one epoch.

PSFT follows [zwhong714/PSFT](https://github.com/zwhong714/PSFT) at commit
`930e23980a723ecef5af138e6e32aa3798b1fd64`: PPO clipping with unit positive
advantages, clip bounds 0.2/0.28, and old-policy log probabilities cached before
each outer batch. Default mini-batch size 32 gives eight optimizer updates per
batch of 256. PSFT uses a constant schedule with 10 warmup outer batches,
overriding the generic cosine/warmup-ratio setting. All mini-batch updates in
an outer batch share one LR; the scheduler advances once afterward.
Logging and checkpoint steps count outer batches. Validation reports ordinary token NLL.
This integration currently supports one GPU without sequence parallelism.

Run `bash sweep_psft_1gpu.sh`, optionally with `TASK=offline_math`.
Sweep variables: `PSFT_CLIP_RATIO_HIGHS`, `OPTIM_LRS`,
`OPTIM_WEIGHT_DECAYS`. Other settings: `PSFT_CLIP_RATIO_LOW`,
`PSFT_MINI_BATCH_SIZE`, `PSFT_WARMUP_STEPS`. Prepare and validate offline data with the offline
dataset tools before using the generic launcher or sweep.
