# PSFT upstream integration

`third_party/psft` contains the upstream Python runtime, Hydra configs and PSFT
recipe from https://github.com/zwhong714/PSFT at commit
`930e23980a723ecef5af138e6e32aa3798b1fd64`, under its Apache-2.0 license.
Source text is unchanged except normalization of line endings/final newlines.
The separate runtime is necessary because this repository's older verl API
differs from upstream. The launchers set its working directory and PYTHONPATH
so both the driver and Ray workers use the same upstream package.

## Data preparation

Run from `verl/` using the Python environment containing datasets/pyarrow:

```bash
python examples/data_preprocess/openr1_psft.py
```

The converter accepts local parquet paths and hosted sources. For the OpenR1
profile, its defaults are the sources named in upstream's run script:
`wh-zhu/train_openr1_4k` and `wh-zhu/aime-24`, producing
`data/openr1_psft/train.parquet` and `test.parquet`. Use
`--train-source Elliott/Openr1-Math-46k-8192` for the full source corpus; that
changes the experiment's data size.

The hosted sources fetched for this integration contain 25,395 training rows
and 960 validation rows; the `4k` repository name is not a row-count guarantee.
The converter preserves all rows, including whitespace-only demonstrations,
and records counts/source fingerprints in `data/openr1_psft/manifest.json`.

The requested Numina schema contains `data_source`, `prompt`, `ability`,
`reward_model`, and `extra_info` (`question`, `answer`, `split`, `index`).
Training additionally exposes the unmodified answer as `demonstration`.
Validation has no `demonstration`, so it generates responses for reward evaluation.
Existing reward routing and ground truths are preserved.

## Dataset profiles

To keep the DFT repository's original model and dataset while retaining the
upstream PSFT trainer, convert the existing DFT parquet files:

```bash
PYTHON_BIN=/media/volume/tucnv/SDPO_sigmoid/dft_env/bin/python \
  bash prepare_numina_psft.sh
```

The Numina command creates `data/numina_psft/train.parquet` and `test.parquet`.
It removes an
already-present Numina instruction suffix before applying it once, adds the
upstream-required `demonstration` field from `extra_info.answer`, and preserves
the original Math reward ground truths. The resulting profile uses
`Qwen/Qwen2.5-Math-1.5B`:

```bash
PYTHON_BIN=/media/volume/tucnv/SDPO_sigmoid/dft_env/bin/python \
  bash train_psft_numina_1gpu.sh
```

The two profiles are selected consistently in all DFT/SPFT/PSFT launcher
scripts with `DATASET=numina` or `DATASET=openr1`:

```bash
DATASET=numina bash train_dft_1gpu.sh
DATASET=openr1 bash train_dft_1gpu.sh
DATASET=numina bash train_spft_1gpu.sh
DATASET=openr1 bash train_psft_1gpu.sh
```

Numina uses Qwen2.5-Math-1.5B and Math500. OpenR1 uses Qwen2.5-7B-Instruct
and AIME24. `sweep_psft_1gpu.sh` defaults to Numina; `PSFT_DATASET` remains a
backward-compatible alias.
The PSFT update remains the vendored upstream implementation in both cases;
only data paths, model, and topology are changed by the profile wrapper.

The intentional default formatting change is Numina's single user message:
raw question followed by "Let's think step by step and output the final answer
within \\boxed{}." To preserve upstream's system/user prompt instead, prepare
a separate directory with `--prompt-template upstream --local-dir data/openr1_original`.
No EOS is appended to demonstrations by the converter; upstream handles
tokenization, response padding, and masking itself.

## Training

```bash
# Upstream topology: 8 GPUs, vLLM tensor parallel 2.
bash train_psft.sh

# Same runtime/config, one GPU and tensor parallel 1.
bash train_psft_1gpu.sh

# One-GPU sweep, also using the upstream runtime.
bash sweep_psft_1gpu.sh

# Inspect arguments without loading data/models or starting Ray.
DRY_RUN=1 bash train_psft_1gpu.sh
```

Defaults match upstream: Qwen2.5-7B-Instruct, AdamW LR 1e-6, weight decay 0.1,
constant scheduler with 10 warmup steps, global batch 256, PPO mini-batch 32,
one PPO epoch per batch, clip 0.2/0.28, token-mean loss, dynamic batching,
gradient clip 1, no KL/entropy penalty, prompt/response limits 2048/6144,
10 epochs, validation/checkpoint frequency 100.

After training, `train_psft.sh` automatically merges the latest FSDP actor
checkpoint to Hugging Face format and calls `eval_dft.sh`. Set `RUN_EVAL=0` to
disable this post-training evaluation. The same post-training evaluator is
used by the DFT/SPFT launchers.

Training follows the upstream `demonstration` branch, which bypasses vLLM
generation and tokenizes fixed demonstrations. It then recomputes old log
probabilities, sets masked unit advantages, and runs upstream's PPO actor update.
vLLM is still initialized and used for validation. Claims that upstream PSFT
necessarily samples its training responses online are incorrect.

The one-GPU topology changes rank balancing and memory requirements; it does not
promise bitwise parity with eight GPUs. A full 7B model, optimizer states and
8192-token sequences require substantial GPU memory. `MODEL_NAME`, `OPTIM_LR`,
`TRAIN_FILE`, `VAL_FILE`, `SAVE_PATH`, `EPOCHS`, `TOTAL_EPOCHS`, `TEST_FREQ`, `SAVE_FREQ`,
`PSFT_MINI_BATCH_SIZE`, `PSFT_WARMUP_STEPS`, `PSFT_CLIP_RATIO_LOW/HIGH` and
other variables in `train_psft.sh` are configurable. Extra arguments are passed
to Hydra last. Changing model, context, offloading or batching changes the run.

Set `PYTHON_BIN` to the environment containing upstream dependencies. The
upstream installation instructions specify torch 2.6.0/CUDA 12.4 and vLLM 0.8.5;
the vendored requirements/setup files list the remaining dependencies. No
environment packages are installed or upgraded by these launchers.

`LOSS_MODE=psft bash train_dft_1gpu.sh` remains the older SFT-trainer adaptation;
it is not the entrypoint for this reproduction. `sweep_dft_1gpu.sh` remains DFT.

Every training launcher accepts `EPOCHS`; it takes precedence over
`TOTAL_EPOCHS`. For example: `DATASET=numina EPOCHS=3 bash sweep_dft_1gpu.sh`.

## CPU verification

```bash
python tests/trainer/test_psft_upstream_integration.py
# Requires the training environment's torch/Hydra dependencies, but no GPU:
python scripts/check_psft_upstream_runtime.py
```

These checks compare launcher arguments against upstream's actual shell script,
check one-GPU topology changes, and exercise raw/hosted OpenR1 plus validation
conversion without downloading data or models.
The runtime check exercises masked unit advantages, clipping gradients, EOS
masking, and full launcher/Hydra composition. `UPSTREAM_MANIFEST.json` records
source hashes so the integration tests also detect changes to vendored code.
