# WeThink dataset preparation

Upstream: `third_party/WeThink`, commit `54df32321a3464d84528a1ca1d2976f36cc1728d`.
Follow its README's Dataset Download and Supervised Fine-Tuning sections.
The upstream repository provides a ShareGPT example, not a conversion script.

The original JSONL comes from `yangjie-cv/WeThink_Multimodal_Reasoning_120K`.
Companion images come from `Xkev/LLaVA-CoT-100k`; existing extracted images are reused.

Our converter adds the real `<image>` placeholder, uses the upstream system prompt,
removes existing answer blocks and reasoning wrappers from `refined_cot`, and appends
the original `answer` field once. Literal media tags are escaped. Empty normalized
reasoning/answers are rejected and listed in the conversion report. This normalization
is our adaptation, not an upstream-provided algorithm; plain-text answer mentions inside
reasoning are preserved. Raw data is not changed.

Run from the repository root, in the training environment:

```bash
env/bin/python scripts/prepare_wethink.py \
  --raw-jsonl data/wethink_raw_verified/WeThink_Multimodal_Reasoning_120K.jsonl \
  --image-root data/wethink_images \
  --output data/wethink_sft.jsonl \
  --dataset-info data/dataset_info.json

env/bin/python scripts/smoke_wethink.py --preprocess --samples 3 \
  --raw-jsonl data/wethink_raw_verified/WeThink_Multimodal_Reasoning_120K.jsonl
```

The smoke test audits all rows and their exact mapping from raw data. It runs the real
LLaMA-Factory converter, Qwen-VL tokenizer/processor and multimodal collator on selected
rows, including escaped-media edge cases. It checks finite image tensors, grid/token
counts, masked image labels and complete supervised targets. It does not train a model
or guarantee absence of NaN during optimization. Training configs use `overwrite_cache: true`.
