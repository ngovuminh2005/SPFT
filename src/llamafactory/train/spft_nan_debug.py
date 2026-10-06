"""Opt-in, fail-fast SPFT diagnostics. Enable with DEBUG_SPFT_NAN=1.

Reports metadata only, never saves a full batch/model. Remove the debug hooks
in trainer.py/trainer_utils.py and this file to uninstall. Disabled by default.
"""

import json
import os
import sys
from pathlib import Path

import torch


class SPFTNaNDebug:
    def __init__(self, trainer):
        self.trainer = trainer
        self.microbatch = 0
        self.context = {}
        self.handles = []
        self.report_path = Path(trainer.args.output_dir) / "spft_nan_debug.jsonl"
        # Hooks execute during backward, before gradients can reach an optimizer step.
        for name, parameter in trainer.model.named_parameters():
            if parameter.requires_grad:
                self.handles.append(parameter.register_hook(self._gradient_hook(name)))
        if os.environ.get("DEBUG_SPFT_ANOMALY") == "1":
            torch.autograd.set_detect_anomaly(True)
        print(f"[SPFT DEBUG] enabled; report={self.report_path}", file=sys.stderr, flush=True)

    def _gradient_hook(self, name):
        def hook(gradient):
            self.check(f"parameter.grad:{name}", gradient)
            return gradient
        return hook

    def watch(self, name, tensor):
        if tensor.requires_grad:
            tensor.register_hook(lambda gradient: self.check(name, gradient))

    def begin(self, inputs, labels):
        self.microbatch += 1
        self.context = {
            "optimizer_step_completed": self.trainer.state.global_step,
            "microbatch": self.microbatch,
            "rank": getattr(self.trainer.args, "process_index", 0),
            "input_shape": list(inputs["input_ids"].shape),
            "supervised_tokens_per_sample": (labels[..., 1:] != -100).sum(-1).tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist() if inputs.get("image_grid_thw") is not None else None,
        }
        for name, value in inputs.items():
            if torch.is_tensor(value):
                self.check(f"input:{name}", value)
        # Find an already-corrupt parameter, e.g. after the preceding optimizer update.
        for name, parameter in self.trainer.model.named_parameters():
            self.check(f"parameter:{name}", parameter)
            if parameter.grad is not None:
                self.check(f"accumulated.grad:{name}", parameter.grad)
        for name, parameter in self.trainer.ref_model.named_parameters():
            self.check(f"reference.parameter:{name}", parameter)

    def check(self, name, tensor):
        if not (tensor.is_floating_point() or tensor.is_complex()):
            return
        # Scan chunks to avoid allocating a logits-sized boolean tensor on GPU.
        flat = tensor.detach().reshape(-1)
        chunk_size = 4194304
        for start in range(0, flat.numel(), chunk_size):
            chunk = flat[start:start + chunk_size]
            finite = torch.isfinite(chunk)
            if bool(finite.all()):
                continue
            bad_offset = start + int((~finite).nonzero()[0].item())
            index = []
            offset = bad_offset
            for dimension in reversed(tensor.shape):
                index.append(offset % dimension)
                offset //= dimension
            valid_values = chunk[finite]
            record = dict(self.context, tensor=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                          first_bad_index=list(reversed(index)), value=str(flat[bad_offset].item()),
                          bad_count_in_chunk=int((~finite).sum().item()),
                          finite_min_in_chunk=valid_values.min().item() if valid_values.numel() else None,
                          finite_max_in_chunk=valid_values.max().item() if valid_values.numel() else None)
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with self.report_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            message = "[SPFT FIRST NONFINITE] " + json.dumps(record)
            print(message, file=sys.stderr, flush=True)
            raise FloatingPointError(message)
