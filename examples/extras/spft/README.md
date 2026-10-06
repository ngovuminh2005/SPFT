# SPFT

Run `bash scripts/train_wethink_spft.sh` after preparing WeThink.
The script uses `python` from the activated environment, escapes literal media
tags, and audits the full JSONL before training. `overwrite_cache: true` rebuilds
the raw local JSON cache as well as format-conversion and tokenization caches.

Run `bash scripts/smoke_wethink.sh` to check every row and image path. For real
Qwen-VL tokenization without training or loading model weights, run
`bash scripts/smoke_wethink.sh --preprocess`. This requires the training Python
dependencies and cached tokenizer/processor artifacts. It tests the first two
rows and up to ten rows with escaped media tags, bypassing all dataset caches.

SPFT uses the same token averaging and gradient accumulation denominator as DFT.
It multiplies each DFT token loss by a detached reference-odds sigmoid.
`spft_lambda` controls only the sigmoid, not an outer scale.
Prompt and padding labels must be masked with -100. Packing is unsupported.

The exact initial policy (including any adapters) is cloned and frozen. Training
writes `spft_reference.pt` in the output directory. Keep this file alongside the
checkpoint directories when copying a run. Resume with:

```bash
bash scripts/train_wethink_spft.sh resume_from_checkpoint=saves/qwen2_5vl-3b/wethink_spft_full/checkpoint-500
```

Resume requires this snapshot and fails if it is missing or incompatible, rather
than replacing the reference with the updated policy. A fresh run overwrites the
snapshot; use a separate output directory for a new experiment. Evaluation also
requires the snapshot. Expect an additional full model in memory, a CPU state
dictionary during snapshot creation, disk space for the snapshot, and a reference
forward per batch. The initial implementation supports unquantized standard
models on one device or DDP; DeepSpeed, FSDP, FP8 and alternative trainers are
rejected explicitly.
