# DFT task profiles

The `numina` and `openr1` profiles share the same parquet contract and can be
selected in every DFT/SPFT/PSFT launcher with `DATASET=numina` or
`DATASET=openr1`. Numina uses Qwen2.5-Math-1.5B + Math500; OpenR1 uses
Qwen2.5-7B-Instruct + AIME24.

`EPOCHS` selects the training epoch count in all launchers and overrides
`TOTAL_EPOCHS`.

Prepare all configured datasets with one command:

```bash
bash verl/prepare_all_datasets.sh
```

Use `DATASETS="numina" bash verl/prepare_all_datasets.sh` or
`DATASETS="openr1" bash verl/prepare_all_datasets.sh` to prepare only one profile.

Training launchers run the math evaluator automatically after the final
checkpoint; set `RUN_EVAL=0` to disable it.

Sweep one or more epoch values from the shell:

```bash
EPOCHS_LIST="1 3 5" DATASET=numina bash sweep_psft_1gpu.sh
```

## Math

```bash
bash scripts/math/train_1gpu.sh
```

Use `DATASET=openr1` with the same script to select the OpenR1 profile.

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

The legacy `LOSS_MODE=psft` SFT-trainer adaptation follows the loss in
[zwhong714/PSFT](https://github.com/zwhong714/PSFT) at commit
`930e23980a723ecef5af138e6e32aa3798b1fd64`: PPO clipping with unit positive
advantages, clip bounds 0.2/0.28, and old-policy log probabilities cached before
each outer batch. Default mini-batch size 32 gives eight optimizer updates per
batch of 256. PSFT uses a constant schedule with 10 warmup outer batches,
overriding the generic cosine/warmup-ratio setting. All mini-batch updates in
an outer batch share one LR; the scheduler advances once afterward.
Logging and checkpoint steps count outer batches. Validation reports ordinary token NLL.
This integration currently supports one GPU without sequence parallelism.

For the upstream runtime (including its exact demonstration tokenization,
response masks, actor updates and reward-based validation), see
[PSFT_UPSTREAM.md](PSFT_UPSTREAM.md). `sweep_psft_1gpu.sh` now launches this
runtime on OpenR1 converted to the Numina parquet schema. Its sweep variables
remain `PSFT_CLIP_RATIO_HIGHS`, `OPTIM_LRS`, and `OPTIM_WEIGHT_DECAYS`.
