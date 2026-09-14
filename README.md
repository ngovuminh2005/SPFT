
<div align="center">

# *On the Generalization of SFT*: <br>A Reinforcement Learning Perspective with <br>Reward Rectification


<a href="http://arxiv.org/abs/2508.05629" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-DFT-red?logo=arxiv" height="25" />
</a>

<a href="https://huggingface.co/collections/Liang0223/dft-6892da5e421a56a8deb48c9f" target="_blank">
    <img alt="HF Model: Cambrian-1" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Huggingface-Models-ffc107?color=ffc107&logoColor=white" height="25" />
</a>

<div style="font-family: charter; text-align: center; margin: 0 auto;">
                    <a href="https://yongliang-wu.github.io/" class="author-link" target="_blank">Yongliang Wu*</a> &emsp;
                    <a href="https://scholar.google.com/citations?user=dHBNmSkAAAAJ" class="author-link" target="_blank">Yizhou Zhou*</a> &emsp;
                    <a href="https://scholar.google.com/citations?user=IH2wK1cAAAAJ" class="author-link" target="_blank">Zhou Ziheng</a> &emsp;
                    <a href="https://github.com/ForJadeForest" class="author-link" target="_blank">Yingzhe Peng</a> &emsp;
                    <br>
                    <a href="https://scholar.google.com/citations?user=fdwhd9gAAAAJ" class="author-link" target="_blank">Xinyu Ye</a> &emsp;
                    <a href="https://joyhuyy1412.github.io/" class="author-link" target="_blank">Xinting Hu</a> &emsp;
                    <a href="https://vitozhu04.github.io/" class="author-link" target="_blank">Wenbo Zhu</a> &emsp;
                    <a href="http://luqi.info/" class="author-link" target="_blank">Lu Qi</a> &emsp;
                    <a href="https://faculty.ucmerced.edu/mhyang/" class="author-link" target="_blank">Ming-Hsuan Yang</a> &emsp;
                    <a href="https://yxpalmweb.github.io/" class="author-link" target="_blank">Xu Yang</a> &emsp;
</div>

<br>
</div>

## 🌟 Thanks for the Feedback of Community

We are grateful for the many thoughtful comments and feedback from the community regarding DFT, ranging from discussions of related ideas to reports of its application in different scenarios. We have heard of both successes and failures when applying DFT, for instance in literary or financial tasks.

Here, we would like to clarify that we do not claim DFT can replace SFT in all cases, as noted in our limitations section:

> *“While our experiments demonstrate substantial gains from DFT on mathematical reasoning benchmarks, this evaluation is confined to math-focused and code-focused (will be released in next version) datasets and models up to 7 billion parameters.”*

---

Nonetheless, these less successful cases, as well as community discussions on platforms such as Zhihu or Xiao Hong Shu about the intuitive principles behind DFT, together with our own experimental experience, have prompted us to think more deeply about the conditions under which DFT works well, and why it may be less effective in other contexts.

All this feedback reminds us of a remark by computing pioneer Richard Hamming in *The Art of Doing Science and Engineering: Learning to Learn* (p.27), which we have slightly adapted:

> *“Almost everyone who opens up a new field does not really understand it the way the followers—or the critics—do.”*

---

We hope this work can contribute to renewed interest in exploring the interplay between SFT and RL, and in better understanding the factors that underlie both the successes and the limitations of methods like DFT. Looking ahead, we also welcome researchers who are interested in our work to improve DFT in some of the currently unsuccessful cases, or in leveraging the ideas to uncover other connections between RL algorithms and SFT, ultimately aiming to achieve RL-like benefits at the cost of SFT across a broader range of settings.

## 📰 News

* **\[2025.08.08]** We have released the training scripts, evaluation scripts, and model checkpoints.

## Abstract
We present a simple yet theoretically motivated improvement to Supervised Fine-Tuning (SFT) for the Large Language Model (LLM), addressing its limited generalization compared to reinforcement learning (RL). Through mathematical analysis, we reveal that standard SFT gradients implicitly encode a problematic reward structure that may severely restrict the generalization capabilities of model. To rectify this, we propose Dynamic Fine-Tuning (DFT), stabilizing gradient updates for each token by dynamically rescaling the objective function with the probability of this token. Remarkably, this single-line code change significantly outperforms standard SFT across multiple challenging benchmarks and base models, demonstrating greatly improved generalization. Additionally, our approach shows competitive results in offline RL settings, offering an effective yet simpler alternative. This work bridges theoretical insight and practical solutions, substantially advancing SFT performance.

## Code Implementation
DFT is a **one-line change** to standard SFT: scale each token’s loss by its predicted probability (detached to avoid gradient flow).

```python
loss = loss * torch.softmax(shift_logits, dim=-1).gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
```

## ⚙️ Installation

Our codebase has been tested on H100 servers with the following environment:

* `python 3.10.0`
* `torch 2.6.0+cu124`

```bash
git clone https://github.com/yongliang-wu/DFT.git
cd DFT
```

### 🔧 Set Up Training Environment

```bash
conda create -n DFT python=3.10 -y
conda activate DFT
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

## 🚀 Getting Started

### Step 1: Prepare Datasets

Run the following commands from the repository root (`DFT/`):

```bash
# Download and preprocess both configured datasets
bash verl/prepare_all_datasets.sh
```

The profiles and base models are:

| Profile | Base model | Training/evaluation data |
|---|---|---|
| `numina` | `Qwen/Qwen2.5-Math-1.5B` | NuminaMath-CoT / Math500 |
| `openr1` | `Qwen/Qwen2.5-7B-Instruct` | OpenR1 / AIME24 |

Prepare only one profile when needed:

```bash
DATASETS=numina bash verl/prepare_all_datasets.sh
DATASETS=openr1 bash verl/prepare_all_datasets.sh
```

The script is safe to rerun and only downloads or converts missing files. The
Qwen model weights are downloaded automatically by Hugging Face on first use.

For offline math, generate and verify the rejection-sampling data separately:

```bash
bash verl/scripts/offline_math/generate_data.sh
```

The optional `NUMINA_TRAIN_END` variable controls the number of NuminaMath-CoT
training examples:

```bash
NUMINA_TRAIN_END=100000 bash verl/prepare_all_datasets.sh
```

### Step 2: Launch Training and Evaluation

```bash
# DFT sweep
DATASET=numina bash verl/sweep_dft_1gpu.sh

# SPFT sweep
DATASET=numina bash verl/sweep_spft_1gpu.sh

# PSFT sweep
DATASET=numina bash verl/sweep_psft_1gpu.sh

# MBO sweep
DATASET=numina bash verl/sweep_mbo_1gpu.sh
```

Use `DATASET=openr1` for the OpenR1 profile. Set `EPOCHS_LIST` to sweep
multiple epoch values:

```bash
DATASET=numina EPOCHS_LIST="1 3 5" bash verl/sweep_psft_1gpu.sh
```

Each sweep uses the model associated with its dataset profile, saves
checkpoints under `verl/checkpoints/`, and evaluates the latest checkpoint
automatically. Set `RUN_EVAL=0` to disable automatic evaluation.

### Sweep Options and Recommended Configurations

Recommended epoch settings:

| Dataset | Recommended epochs | Model |
|---|---:|---|
| `numina` | `1` | `Qwen/Qwen2.5-Math-1.5B` |
| `openr1` | `10` | `Qwen/Qwen2.5-7B-Instruct` |

Run a recommended configuration with:

```bash
# Numina: one epoch
DATASET=numina EPOCHS_LIST=1 bash verl/sweep_psft_1gpu.sh

# OpenR1: ten epochs
DATASET=openr1 EPOCHS_LIST=10 bash verl/sweep_psft_1gpu.sh
```

Common options for all sweeps:

| Option | Purpose |
|---|---|
| `DATASET` | `numina` or `openr1` |
| `EPOCHS_LIST` | Space-separated epoch values, for example `"1 3 5"` |
| `RUN_EVAL` | `1` to evaluate automatically, `0` to disable |
| `SAVE_FREQ` | Checkpoint frequency |
| `TEST_FREQ` | Validation frequency |

Method-specific options and current priorities:

| Sweep | Main options | Current priority |
|---|---|---|
| DFT | `OPTIMIZERS`, `MAIN_LRS`, `SOREN_LAMBDAS`, `SOREN_MODE`, `BASE_LR`, `AUX_LR` | SorenAuxAdam, base LR `8e-4`, aux LR `5e-5`, Muon LR `7e-4`, lambda `1` |
| SPFT | `SPFT_LAMBDAS`, `OPTIM_LRS`, `OPTIM_WEIGHT_DECAYS`, `SPFT_WEIGHT_THRESHOLD` | AdamW, LR `1e-4`, lambda `0.1`, weight decay `0.01` |
| PSFT | `PSFT_CLIP_RATIO_HIGHS`, `OPTIM_LRS`, `OPTIM_WEIGHT_DECAYS`, sequence-length and rollout settings | AdamW, LR `1e-6`, weight decay `0.1`, clip `0.2/0.28`, warmup `10` steps |
| MBO | `OPTIM_MUON_LR` and the `OPTIM_MBO_*` parameters in the sweep loops | `singledevicembowindowwithauxadam`, projection source, Muon LR `8e-4`, 64 centroids, dimension `4096` |

DFT, SPFT, and PSFT accept space-separated environment values directly. MBO
currently defines its sweep values in nested loops inside
`verl/sweep_mbo_1gpu.sh`; edit those loop values to expand the MBO grid.

Values can be overridden inline without editing the scripts. For example:

```bash
DATASET=numina EPOCHS_LIST="1 3" OPTIM_LRS="5e-7 1e-6" \
  bash verl/sweep_psft_1gpu.sh
```

The lower-level math launchers remain available:

```bash
# DFT on math (also prepares missing math data)
bash verl/scripts/math/train_1gpu.sh

# DFT on offline math (after verl/scripts/offline_math/generate_data.sh)
bash verl/scripts/offline_math/train_1gpu.sh
```

### Step 3: Evaluation Only

To evaluate the trained model separately, please first follow the
[Qwen2.5-Math repository](https://github.com/QwenLM/Qwen2.5-Math) to set up the
evaluation environment.

```bash
MODEL_NAME_OR_PATH=checkpoints/<task>/<experiment>/global_step_<step> \
OUTPUT_DIR=outputs/eval_results \
bash verl/eval_dft.sh
```

Set `DATASET=numina` or `DATASET=openr1` so the evaluator selects the matching
evaluation group (`math500` or `aime24`).

## Limitations
Based on our evaluations and community feedback, DFT performs strongly on mathematical tasks with non-deterministic solution trajectories—i.e., those that admit multiple valid reasoning paths. By contrast, its performance is weaker on tasks with a single, well-specified ground-truth answer, particularly when the associated CoT (if exists) is highly constrained and near-deterministic (low-entropy).

## Citation
If you find this paper valuable for your research or applications, we would appreciate it if you could cite our work:
```latex
@article{wu2025generalization,
  title={On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification},
  author={Wu, Yongliang and Zhou, Yizhou and Ziheng, Zhou and Peng, Yingzhe and Ye, Xinyu and Hu, Xinting and Zhu, Wenbo and Qi, Lu and Yang, Ming-Hsuan and Yang, Xu},
  journal={arXiv preprint arXiv:2508.05629},
  year={2025}
}
```

## Related Repositories
* [https://github.com/huggingface/trl](https://github.com/huggingface/trl): TRL supports DFT now, check [this script](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md).
* [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): LLaMA-Factory supports DFT now, check [this script](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/extras/dft/qwen2_full_sft.yaml).
* [https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift): ms-swift supports DFT now, check [this script](https://github.com/modelscope/ms-swift/blob/main/examples/train/full/dft.sh).
* [https://github.com/Lauorie/DFT](https://github.com/Lauorie/DFT): Reproduced the DFT method without using Verl.
* [https://github.com/volcengine/verl](https://github.com/volcengine/verl): Codebase used for training.
* [https://github.com/QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math): Codebase used for evaluation.
