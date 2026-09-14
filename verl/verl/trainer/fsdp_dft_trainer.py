# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext
import importlib

import hydra
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn, optim
from torch.nn import functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.trainer.mbo_gradient_sources import build_mbo_gradient_source
from verl.trainer.psft import psft_token_loss
from verl.trainer.spft import spft_token_weights
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def _is_muon_compatible_param(p: nn.Parameter) -> bool:
    return p.requires_grad and p.ndim in (2, 4)


def _set_param_debug_name(param: nn.Parameter, name: str):
    param._debug_name = name


def _matches_any_name_pattern(name: str, patterns) -> bool:
    name = name.lower()
    return any(str(pattern).lower() in name for pattern in patterns)


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        self.device_name = get_device_name()
        self.loss_mode = str(self.config.optim.get("loss_mode", "dft")).lower()
        if self.loss_mode not in {"dft", "spft", "psft"}:
            raise ValueError(f"Unknown optim.loss_mode={self.loss_mode!r}; expected 'dft', 'spft', or 'psft'.")
        spft_config = self.config.optim.get("spft", {})
        self.spft_lambda = float(spft_config.get("lambda", 0.1))
        if not torch.isfinite(torch.as_tensor(self.spft_lambda)):
            raise ValueError("optim.spft.lambda must be finite.")
        self.spft_eps = float(spft_config.get("eps", 1e-6))
        if not torch.isfinite(torch.as_tensor(self.spft_eps)) or not 0.0 < self.spft_eps < 0.5:
            raise ValueError("optim.spft.eps must be finite and in (0, 0.5).")
        self.spft_weight_threshold = float(spft_config.get("weight_threshold", 0.0))
        if not torch.isfinite(torch.as_tensor(self.spft_weight_threshold)):
            raise ValueError("optim.spft.weight_threshold must be finite.")
        self.spft_reference_cpu_offload = str(spft_config.get("reference_cpu_offload", False)).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        psft_config = self.config.optim.get("psft", {})
        self.psft_clip_low = float(psft_config.get("clip_ratio_low", 0.2))
        self.psft_clip_high = float(psft_config.get("clip_ratio_high", 0.28))
        self.psft_mini_batch_size = int(psft_config.get("mini_batch_size", 32))
        if not 0 <= self.psft_clip_low < 1 or not 0 <= self.psft_clip_high < float("inf"):
            raise ValueError("Invalid PSFT clip ratios")
        if self.psft_mini_batch_size <= 0:
            raise ValueError("PSFT mini_batch_size must be positive")
        if self.loss_mode == "psft" and (self.config.ulysses_sequence_parallel_size != 1 or torch.distributed.get_world_size() != 1):
            raise ValueError("PSFT currently supports one GPU without sequence parallelism")
        if self.loss_mode == "psft" and (
            self.config.data.train_batch_size % self.psft_mini_batch_size
            or self.psft_mini_batch_size % self.config.data.micro_batch_size_per_gpu
        ):
            raise ValueError("PSFT requires batch divisible by mini-batch, and mini-batch by micro-batch")
        self.reference_model = None
        self._uses_policy_reference = self.loss_mode == "spft" and self.config.model.get("lora_rank", 0) > 0
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self._debug_step = 0
        self._debug_micro_step = 0

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                from peft import LoraConfig, TaskType, get_peft_model

                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(
                    self.model,
                    LoraConfig(**lora_config),
                    autocast_adapter_dtype=False,
                )

        self._muon_aux_param_ids, self._muon_aux_param_names = self._collect_muon_aux_params(self.model)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        if self.loss_mode == "spft" and not self._uses_policy_reference:
            self._build_reference_model(local_model_path, torch_dtype)

        self.optimizer = self._build_optimizer()
        mbo_config = self.config.optim.get("mbo", {})
        gradient_source_name = mbo_config.get("gradient_source", "minibatch")
        self.mbo_gradient_source = build_mbo_gradient_source(self.optimizer, gradient_source_name)
        self.mbo_gradient_source.attach()

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if self.loss_mode == "psft":
            from transformers import get_constant_schedule_with_warmup

            warmup_steps = int(self.config.optim.psft.get("warmup_steps", 10))
            if warmup_steps < 0:
                raise ValueError("PSFT warmup_steps must be non-negative")
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps
            )
        elif not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    @property
    def _reference_cpu_offload(self):
        return self.spft_reference_cpu_offload

    def _build_reference_model(self, local_model_path, torch_dtype):
        """Load the immutable pre-tuning policy required by full fine-tuning SPFT."""
        reference_config = AutoConfig.from_pretrained(
            local_model_path, trust_remote_code=self.config.model.trust_remote_code
        )
        if hasattr(reference_config, "max_position_embeddings"):
            reference_config.max_position_embeddings = max(
                reference_config.max_position_embeddings, self.config.data.max_length
            )
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            config=reference_config,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()
        if not self._reference_cpu_offload:
            self.reference_model.to(self.device_name)

    def _forward_reference(self, **model_inputs):
        """Run pi_0 before the trainable-model forward and return its logits."""
        restore_training = None
        adapter_context = nullcontext()
        if self._uses_policy_reference:
            disable_adapter = getattr(self.model, "disable_adapter", None)
            if disable_adapter is None:
                raise RuntimeError("SPFT LoRA reference requires a PEFT model with disable_adapter().")
            reference_model = self.fsdp_model
            restore_training = reference_model.training
            reference_model.eval()
            adapter_context = disable_adapter()
        else:
            reference_model = self.reference_model
            if self._reference_cpu_offload:
                reference_model.to(self.device_name)

        try:
            with torch.no_grad(), adapter_context:
                logits = reference_model(**model_inputs).logits
        finally:
            if restore_training is not None:
                self.fsdp_model.train(restore_training)
            if self._reference_cpu_offload:
                reference_model.to("cpu")
        return logits

    @staticmethod
    def _target_log_probs_from_logits(logits, labels):
        """Get selected-token log probabilities without materializing a full softmax."""
        token_ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1).to(logits.device), reduction="none"
        )
        return -token_ce

    def _standard_reference_log_probs(self, input_ids, attention_mask, position_ids):
        reference_logits = self._forward_reference(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        )
        reference_log_probs = self._target_log_probs_from_logits(reference_logits[:, :-1], input_ids[:, 1:])
        del reference_logits
        return reference_log_probs

    @staticmethod
    def _spft_metrics(weights, reference_log_probs, loss_mask, weight_threshold):
        valid = loss_mask.bool()
        valid_weights = weights[valid]
        valid_reference_log_probs = reference_log_probs[valid]
        metrics = {
            "spft/weight_mean": valid_weights.mean().item(),
            "spft/weight_min": valid_weights.min().item(),
            "spft/weight_max": valid_weights.max().item(),
            "spft/w_ratio": (valid_weights > weight_threshold).float().mean().item(),
            "spft/ref_target_logprob_mean": valid_reference_log_probs.mean().item(),
            "spft/_valid_token_count": int(valid.sum().item()),
        }
        return metrics

    def _build_optimizer(self):
        optim_name = str(self.config.optim.get("name", "adamw")).lower()

        if optim_name == "adamw":
            return optim.AdamW(
                self.fsdp_model.parameters(),
                lr=self.config.optim.lr,
                betas=tuple(self.config.optim.betas),
                eps=self.config.optim.get("eps", 1e-8),
                weight_decay=self.config.optim.weight_decay,
            )

        if optim_name in {'soren', 'sorenauxadam', 'htmuon', 'htmuonauxadam'}:
            if self.device_mesh.size() != 1:
                raise ValueError(f'{optim_name} only supports one device')
            is_soren = optim_name.startswith('soren')
            module = importlib.import_module('soren_lamdba' if is_soren else 'htmuon')
            class_name = 'SingleDeviceSoren' if is_soren else 'SingleDeviceHTMuon'
            spectral_kwargs = (
                dict(lamdba=self.config.optim.soren.lamdba, mode=self.config.optim.soren.mode)
                if is_soren else dict(alpha=self.config.optim.htmuon.alpha)
            )
            if optim_name.endswith('auxadam'):
                return getattr(module, class_name + 'WithAuxAdam')(
                    self._build_muon_aux_param_groups(False), **spectral_kwargs
                )
            main, aux = self._split_muon_param_groups()
            if aux:
                raise ValueError(f'{optim_name} has auxiliary parameters; use {optim_name}auxadam')
            return getattr(module, class_name)(
                [p for _, p in main], **self._build_plain_optimizer_kwargs(), **spectral_kwargs
            )

        optim_module = importlib.import_module(self.config.optim.get("module", "muon"))
        world_size = self.device_mesh.size()
        if optim_name.startswith("singledevice") and world_size != 1:
            raise ValueError(
                f"Optimizer {self.config.optim.name} only supports world_size=1, but current world_size={world_size}"
            )

        class_name_map = {
            "singledevicemuon": "SingleDeviceMuon",
            "singledevicemuonwithauxadam": "SingleDeviceMuonWithAuxAdam",
            "singledevicemuonwithauxsgd": "SingleDeviceMuonWithAuxSGD",
            "muon": "Muon",
            "muonwithauxadam": "MuonWithAuxAdam",
            "muonwithauxsgd": "MuonWithAuxSGD",
            "singledevicembo": "SingleDeviceMBO",
            "singledevicembowithauxadam": "SingleDeviceMBOWithAuxAdam",
            "singledevicembowindowwithauxadam": "SingleDeviceMBOWindowWithAuxAdam",
            "mbo": "MBO",
            "mbowithauxadam": "MBOWithAuxAdam",
        }
        if optim_name not in class_name_map:
            raise ValueError(f"Unknown optimizer {self.config.optim.name}")

        optimizer_cls = getattr(optim_module, class_name_map[optim_name])

        if optim_name in {"singledevicemuon", "muon", "singledevicembo", "mbo"}:
            muon_named_params, aux_params = self._split_muon_param_groups()
            muon_params = [p for _, p in muon_named_params]
            if aux_params:
                raise ValueError(
                    f"Optimizer {self.config.optim.name} has no aux AdamW fallback, but "
                    f"{len(aux_params)} parameters are excluded from Muon/MBO. Use a *withauxadam optimizer."
                )
            if not muon_params:
                raise ValueError(f"Optimizer {self.config.optim.name} requires at least one 2D/4D parameter.")
            kwargs = self._build_plain_optimizer_kwargs()
            if "mbo" in optim_name:
                kwargs.update(self._build_mbo_kwargs())
            return optimizer_cls(muon_params, **kwargs)

        param_groups = self._build_muon_aux_param_groups(optim_name == "singledevicembowindowwithauxadam")
        kwargs = {}
        if "mbo" in optim_name:
            kwargs.update(self._build_mbo_kwargs())
        return optimizer_cls(param_groups, **kwargs)

    def _collect_muon_aux_params(self, model: nn.Module):
        patterns = self.config.optim.get(
            "muon_exclude_name_patterns",
            ["embed_tokens", "embeddings", "word_embeddings", "lm_head", "output_layer"],
        )
        aux_param_ids = set()
        aux_param_names = []

        for module_name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    aux_param_ids.add(id(param))
                    aux_param_names.append(full_name)

        for name, param in model.named_parameters():
            if _matches_any_name_pattern(name, patterns):
                aux_param_ids.add(id(param))
                aux_param_names.append(name)

        return aux_param_ids, sorted(set(aux_param_names))

    def _split_muon_param_groups(self):
        muon_named_params = []
        aux_params = []
        muon_numel = 0
        aux_numel = 0
        forced_aux_numel = 0
        for name, p in self.fsdp_model.named_parameters():
            if not p.requires_grad:
                continue
            _set_param_debug_name(p, name)
            force_aux = id(p) in self._muon_aux_param_ids or _matches_any_name_pattern(
                name, self.config.optim.get("muon_exclude_name_patterns", [])
            )
            if _is_muon_compatible_param(p) and not force_aux:
                muon_named_params.append((name, p))
                muon_numel += p.numel()
            else:
                aux_params.append(p)
                aux_numel += p.numel()
                if force_aux:
                    forced_aux_numel += p.numel()
        if self.device_mesh.get_rank() == 0:
            print(
                "Optimizer param split: "
                f"muon_params={len(muon_named_params)} ({muon_numel} params), "
                f"aux_adam_params={len(aux_params)} ({aux_numel} params), "
                f"forced_aux={forced_aux_numel} params from {self._muon_aux_param_names}"
            )
        return muon_named_params, aux_params

    @staticmethod
    def _extract_transformer_layer_idx(param_name: str):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", param_name)
        return None if match is None else int(match.group(1))

    def _split_mbo_window_params(self, muon_named_params):
        layer_indices = [
            layer_idx
            for name, _ in muon_named_params
            for layer_idx in [self._extract_transformer_layer_idx(name)]
            if layer_idx is not None
        ]
        if not layer_indices:
            return [p for _, p in muon_named_params], []

        last_layer = max(layer_indices)
        offset = int(self.config.optim.mbo.get("offset", 0))
        window_size = int(self.config.optim.mbo.get("window_size", 0))
        end_layer = last_layer - offset
        start_layer = 0 if window_size <= 0 else max(0, end_layer - window_size + 1)

        mbo_params = []
        plain_muon_params = []
        for name, p in muon_named_params:
            layer_idx = self._extract_transformer_layer_idx(name)
            if layer_idx is not None and start_layer <= layer_idx <= end_layer:
                mbo_params.append(p)
            else:
                plain_muon_params.append(p)

        if self.device_mesh.get_rank() == 0:
            print(
                "MBO layer window: "
                f"last_layer={last_layer}, offset={offset}, window_size={window_size}, "
                f"selected_layers=[{start_layer},{end_layer}], "
                f"mbo_params={len(mbo_params)}, plain_muon_params={len(plain_muon_params)}"
            )
        return mbo_params, plain_muon_params

    def _build_plain_optimizer_kwargs(self):
        return {
            "lr": self.config.optim.get("muon_lr", None) or self.config.optim.lr,
            "weight_decay": self.config.optim.weight_decay,
            "momentum": self.config.optim.get("momentum", 0.95),
            "nesterov": self.config.optim.get("nesterov", True),
            "ns_steps": self.config.optim.get("ns_steps", 5),
        }

    def _build_muon_aux_param_groups(self, include_use_mbo: bool):
        muon_named_params, aux_params = self._split_muon_param_groups()
        if not muon_named_params:
            raise ValueError(f"Optimizer {self.config.optim.name} requires at least one 2D/4D parameter.")

        base_muon_group = {
            "lr": self.config.optim.get("muon_lr", None) or self.config.optim.lr,
            "momentum": self.config.optim.get("momentum", 0.95),
            "weight_decay": self.config.optim.weight_decay,
            "nesterov": self.config.optim.get("nesterov", True),
            "ns_steps": self.config.optim.get("ns_steps", 5),
            "use_muon": True,
        }
        if include_use_mbo:
            mbo_params, plain_muon_params = self._split_mbo_window_params(muon_named_params)
            param_groups = []
            if mbo_params:
                param_groups.append({**base_muon_group, "params": mbo_params, "use_mbo": True})
            if plain_muon_params:
                param_groups.append({**base_muon_group, "params": plain_muon_params, "use_mbo": False})
        else:
            param_groups = [{**base_muon_group, "params": [p for _, p in muon_named_params]}]

        if aux_params:
            param_groups.append(
                {
                    "params": aux_params,
                    "lr": self.config.optim.get("aux_lr", None) or self.config.optim.lr,
                    "betas": tuple(self.config.optim.betas),
                    "eps": self.config.optim.get("eps", 1e-8),
                    "weight_decay": self.config.optim.weight_decay,
                    "use_muon": False,
                }
            )
        return param_groups

    def _build_mbo_kwargs(self):
        mbo = self.config.optim.mbo
        return {
            "num_centroids": mbo.num_centroids,
            "centroid_dim": mbo.centroid_dim,
            "gradient_source": mbo.get("gradient_source", "minibatch"),
            "lambda_memory": mbo.lambda_memory,
            "memory_init_seed": mbo.memory_init_seed,
            "ot_ws_steps": mbo.ot_ws_steps,
            "ot_step_update": mbo.ot_step_update,
            "ot_epsilon": mbo.ot_epsilon,
            "ot_g_lr": mbo.ot_g_lr,
            "ot_y_lr": mbo.ot_y_lr,
            "ot_g_steps": mbo.ot_g_steps,
            "ot_y_steps": mbo.ot_y_steps,
            "kmeans_steps": mbo.get("kmeans_steps", 10),
            "offset": mbo.offset,
            "window_size": mbo.window_size,
        }

    def _tensor_stats(self, tensor: torch.Tensor):
        t = tensor.detach()
        finite = torch.isfinite(t)
        finite_count = int(finite.sum().item())
        total_count = t.numel()
        if finite_count == 0:
            return f"shape={tuple(t.shape)} dtype={t.dtype} finite=0/{total_count}"
        finite_t = t[finite].float()
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} finite={finite_count}/{total_count} "
            f"min={finite_t.min().item():.6g} max={finite_t.max().item():.6g} "
            f"mean={finite_t.mean().item():.6g}"
        )

    def _raise_if_nonfinite(self, name: str, tensor: torch.Tensor):
        if not bool(int(os.environ.get("DEBUG_TRAINER_NAN", "0"))):
            return
        if torch.isfinite(tensor).all():
            return
        if self.device_mesh.get_rank() == 0:
            print(
                f"[NAN_DEBUG] first non-finite tensor at step={self._debug_step} "
                f"micro={self._debug_micro_step}: {name}: {self._tensor_stats(tensor)}",
                flush=True,
            )
        raise FloatingPointError(f"Non-finite tensor detected at {name}")

    def _check_named_tensors_nonfinite(self, stage: str, named_tensors):
        for name, tensor in named_tensors:
            if tensor is not None:
                self._raise_if_nonfinite(f"{stage}:{name}", tensor)

    def _check_model_nonfinite(self, stage: str, check_grad: bool = False):
        for name, p in self.fsdp_model.named_parameters():
            self._raise_if_nonfinite(f"{stage}:param:{name}", p)
            if check_grad and p.grad is not None:
                self._raise_if_nonfinite(f"{stage}:grad:{name}", p.grad)

    def _compute_loss_and_backward(self, batch, do_backward=True, backward_loss_scale=1.0):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        spft_metrics = {}

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask_2d = batch.pop("loss_mask")[:, :-1].to(self.device_name)
        loss_mask = loss_mask_2d.reshape(-1)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                reference_log_probs = None
                reference_logits = None
                if self.loss_mode == "spft":
                    reference_log_probs = self._standard_reference_log_probs(
                        input_ids, attention_mask, position_ids
                    )

                output = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = output.logits
                self._raise_if_nonfinite("forward:logits", logits)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                original_loss = loss.clone()
                self._raise_if_nonfinite("loss:token_ce", loss)

                if self.loss_mode == "spft":
                    weights = spft_token_weights(-original_loss, reference_log_probs, self.spft_lambda, self.spft_eps)
                    spft_metrics = self._spft_metrics(
                        weights, reference_log_probs, loss_mask, self.spft_weight_threshold
                    )
                    loss = loss * weights.to(dtype=loss.dtype)
                elif self.loss_mode == "dft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    self._raise_if_nonfinite("loss:probs", probs)
                    prob_coefficients = probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    self._raise_if_nonfinite("loss:prob_coefficients", prob_coefficients)
                    loss = loss * prob_coefficients.detach()
                elif self.loss_mode == "psft" and do_backward:
                    loss = psft_token_loss(
                        -original_loss, batch["psft_old_log_probs"].reshape(-1).to(self.device_name),
                        self.psft_clip_low, self.psft_clip_high,
                    )
                loss = loss * loss_mask.to(loss.device)
                original_loss = original_loss * loss_mask.to(original_loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                reference_log_probs = None
                reference_logits = None
                if self.loss_mode == "spft":
                    if self._uses_policy_reference:
                        reference_logits = self._forward_reference(
                            input_ids=input_ids_rmpad_sliced,
                            attention_mask=None,
                            position_ids=position_ids_rmpad_padded,
                            use_cache=False,
                        ).squeeze(0)
                        reference_token_ce = loss_fct(reference_logits, input_ids_rmpad_rolled.to(reference_logits.device))
                        reference_token_ce = gather_outpus_and_unpad(
                            reference_token_ce, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                        reference_full_loss = pad_input(
                            hidden_states=reference_token_ce.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        ).squeeze(-1)[:, :-1]
                        reference_log_probs = -reference_full_loss.reshape(-1)
                        del reference_logits
                    else:
                        reference_log_probs = self._standard_reference_log_probs(
                            input_ids, attention_mask, position_ids
                        )

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                logits_rmpad = output.logits.squeeze(0)
                self._raise_if_nonfinite("forward:logits_rmpad", logits_rmpad)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                self._raise_if_nonfinite("loss:token_ce_sp", loss)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                original_loss = full_loss * loss_mask
                if self.loss_mode == "spft":
                    weights = spft_token_weights(-full_loss, reference_log_probs, self.spft_lambda, self.spft_eps)
                    spft_metrics = self._spft_metrics(
                        weights, reference_log_probs, loss_mask, self.spft_weight_threshold
                    )
                    loss = full_loss * weights.to(dtype=full_loss.dtype) * loss_mask
                elif self.loss_mode == "dft":
                    loss = original_loss

            if self.loss_mode == "spft":
                spft_metrics["spft/lambda"] = self.spft_lambda
                spft_metrics["spft/w_thr"] = self.spft_weight_threshold

            valid_token_this_rank = torch.sum(loss_mask)
            normalization_token_count = valid_token_this_rank

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(normalization_token_count)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (normalization_token_count + 1e-8) * dp_size

            original_loss = torch.sum(original_loss) / (normalization_token_count + 1e-8) * dp_size
            self._check_named_tensors_nonfinite(
                "loss:scalar",
                [
                    ("valid_token_this_rank", valid_token_this_rank),
                    ("normalization_token_count", normalization_token_count),
                    ("loss", loss),
                    ("original_loss", original_loss),
                ],
            )

            if do_backward:
                (loss * backward_loss_scale).backward()
            return loss, original_loss, spft_metrics

    def psft_training_step(self, batch: TensorDict):
        # Freeze scalar old-policy log probabilities across all updates in this batch.
        self.fsdp_model.eval()
        old_log_probs = []
        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            for micro in batch.split(self.config.data.micro_batch_size_per_gpu):
                logits = self.fsdp_model(
                    input_ids=micro["input_ids"], attention_mask=micro["attention_mask"],
                    position_ids=micro["position_ids"], use_cache=False,
                ).logits[:, :-1]
                labels = micro["input_ids"][:, 1:]
                log_probs = -torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none",
                ).reshape_as(labels)
                old_log_probs.append(log_probs.detach())
                del logits
        batch["psft_old_log_probs"] = torch.cat(old_log_probs)
        metrics = [
            self.training_step(mini, step_scheduler=False)
            for mini in batch.split(self.psft_mini_batch_size)
        ]
        self.lr_scheduler.step()
        result = {key: sum(m[key] for m in metrics) / len(metrics) for key in metrics[0]}
        result["train/lr(1e-3)"] = self.lr_scheduler.get_last_lr()[0] * 1e3
        return result

    def training_step(self, batch: TensorDict, step_scheduler=True):
        self._debug_step += 1
        self._debug_micro_step = 0
        self.fsdp_model.train()
        self._check_model_nonfinite("before_zero_grad")

        if hasattr(self.optimizer, "clear_mbo_metrics"):
            self.optimizer.clear_mbo_metrics()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        self.mbo_gradient_source.begin_step(n_micro_batches)
        step_loss = 0
        step_original_loss = 0
        spft_metric_values = {}
        for micro_batch in micro_batches:
            self._debug_micro_step += 1
            loss, original_loss, micro_spft_metrics = self._compute_loss_and_backward(
                batch=micro_batch,
                backward_loss_scale=1.0 / n_micro_batches,
            )
            self._check_model_nonfinite("after_micro_backward", check_grad=True)
            loss = loss / n_micro_batches
            original_loss = original_loss / n_micro_batches
            step_loss += loss.item()
            step_original_loss += original_loss.item()
            for key, value in micro_spft_metrics.items():
                spft_metric_values.setdefault(key, []).append(value)

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")
        self._raise_if_nonfinite("grad_norm_after_clip", grad_norm)
        self._check_model_nonfinite("after_clip_grad", check_grad=True)

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.mbo_gradient_source.discard_step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.mbo_gradient_source.commit_step()
            self._check_model_nonfinite("after_optimizer_step")

        log_gpu_memory_usage("After optimizer step", logger=logger)

        if step_scheduler:
            self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        metric = {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/original_loss": step_original_loss,
        }
        if hasattr(self.optimizer, "get_mbo_metrics"):
            metric.update(self.optimizer.get_mbo_metrics())
        if spft_metric_values:
            valid_count_key = next(key for key in spft_metric_values if key.endswith("/_valid_token_count"))
            valid_counts = spft_metric_values.pop(valid_count_key)
            total_valid_tokens = sum(valid_counts)
            for key, values in spft_metric_values.items():
                if key.endswith("_min"):
                    metric[key] = min(values)
                elif key.endswith("_max"):
                    metric[key] = max(values)
                elif key.endswith(("_mean", "_ratio")):
                    metric[key] = (
                        sum(value * count for value, count in zip(values, valid_counts)) / total_valid_tokens
                        if total_valid_tokens
                        else 0.0
                    )
                else:
                    metric[key] = sum(values) / len(values)
        return metric

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, original_loss, _ = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(original_loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
                original_loss /= self.device_mesh.size(0)
        return loss, original_loss

    def save_checkpoint(self, step):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.psft_training_step(data) if self.loss_mode == "psft" else self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    val_original_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss, val_original_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                        val_original_losses.append(val_original_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        val_original_loss = torch.mean(torch.stack(val_original_losses))
                        metric = {"val/loss": val_loss.detach().item(), "val/original_loss": val_original_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
