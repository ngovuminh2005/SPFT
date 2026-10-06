"""Offline SPFT formula, trainer, image-context and resume checks.

Run with python -m unittest discover -s tests/train -p test_spft.py.
"""

import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig, PreTrainedModel, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutput

from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.trainer_utils import (
    create_spft_ref_model,
    dft_loss_func,
    spft_loss_func,
    spft_target_logps,
)


class TinyImageModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self):
        super().__init__(PretrainedConfig(vocab_size=7))
        self.embeddings = torch.nn.Embedding(7, 8)
        self.projection = torch.nn.Linear(8, 7)
        self.contexts = []

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, position_ids=None):
        self.contexts.append((pixel_values, image_grid_thw, position_ids, attention_mask))
        hidden = self.embeddings(input_ids)
        if pixel_values is not None:
            hidden = hidden + pixel_values.mean()
        return CausalLMOutput(logits=self.projection(hidden))


class SPFTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        self.labels = torch.tensor([[-100, 1, 2, 3], [-100, 2, -100, -100]])

    def test_formula_and_detached_gradient(self):
        logits = torch.randn(2, 4, 7, requires_grad=True)
        ref_logits = torch.randn(2, 4, 7, requires_grad=True)
        ref_logps = spft_target_logps(ref_logits, self.labels)
        loss = spft_loss_func({"logits": logits}, self.labels, ref_logps, 1.7)
        lp = logits[:, :-1].log_softmax(-1)
        targets = self.labels[:, 1:].clamp_min(0)
        lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        q = ref_logps.detach().exp().clamp(1e-7, 1 - 1e-7)
        p = lp.detach().exp().clamp(1e-7, 1 - 1e-7)
        weight = lp.detach().exp() * torch.sigmoid(1.7 * (q.log() - p.log() + (1 - p).log() - (1 - q).log()))
        mask = self.labels[:, 1:] != -100
        expected = -(weight * lp)[mask].mean()
        torch.testing.assert_close(loss, expected)
        actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        loss.backward()
        assert ref_logits.grad is None
        assert torch.equal(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1]))

    def test_equal_reference_is_half_dft(self):
        logits = torch.randn(2, 4, 7, requires_grad=True)
        lp = spft_target_logps(logits, self.labels)
        loss = spft_loss_func({"logits": logits}, self.labels, lp.detach())
        dft = dft_loss_func({"logits": logits}, self.labels)
        torch.testing.assert_close(loss, dft / 2)
        torch.testing.assert_close(
            torch.autograd.grad(loss, logits, retain_graph=True)[0],
            torch.autograd.grad(dft / 2, logits)[0],
        )

    def test_extreme_probabilities_and_empty_samples(self):
        logits = torch.tensor([[[1000.0, -1000.0], [-1000.0, 1000.0], [0.0, 0.0]]], requires_grad=True)
        labels = torch.tensor([[-100, 0, 0]])
        ref_lp = spft_target_logps(-logits.detach(), labels)
        loss = spft_loss_func({"logits": logits}, labels, ref_lp)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        with self.assertRaisesRegex(ValueError, "no supervised"):
            spft_loss_func({"logits": logits}, torch.full_like(labels, -100), ref_lp)

    def make_trainer(self, directory, model=None, accumulation=1, batch_size=1):
        model = model or TinyImageModel()
        reference = copy.deepcopy(model)
        args = TrainingArguments(
            output_dir=directory,
            use_cpu=True,
            report_to="none",
            gradient_accumulation_steps=accumulation,
            per_device_train_batch_size=batch_size,
            num_train_epochs=1,
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            learning_rate=0.01,
            max_grad_norm=0.0,
            lr_scheduler_type="constant",
            remove_unused_columns=False,
        )
        return CustomSeq2SeqTrainer(
            model=model,
            ref_model=reference,
            args=args,
            finetuning_args=FinetuningArguments(use_spft_loss=True, finetuning_type="full"),
            tokenizer=None,
            processor=None,
        )

    def test_image_context_update_and_reference_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            inputs = {
                "input_ids": torch.tensor([[0, 1, 2, 3]]),
                "labels": self.labels[:1],
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "pixel_values": torch.rand(1, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "position_ids": torch.arange(4).view(1, 1, 4).expand(3, 1, 4),
            }
            old_policy = trainer.model.projection.weight.detach().clone()
            old_reference = trainer.ref_model.projection.weight.detach().clone()
            loss, outputs = trainer.compute_loss(trainer.model, inputs, return_outputs=True)
            assert torch.isfinite(loss)
            assert outputs.logits.shape == (1, 4, 7)
            loss.backward()
            torch.optim.SGD(trainer.model.parameters(), lr=0.1).step()
            assert not torch.equal(old_policy, trainer.model.projection.weight)
            torch.testing.assert_close(old_reference, trainer.ref_model.projection.weight)
            assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())
            assert not trainer.ref_model.training
            for policy_input, reference_input in zip(trainer.model.contexts[-1], trainer.ref_model.contexts[-1]):
                assert policy_input is reference_input

    def test_accumulation_counts_tokens_and_partial_group(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, accumulation=8)
            batches = [{"labels": self.labels[:1]}, {"labels": self.labels[1:]}]
            _, denominator = trainer.get_batch_samples(iter(batches), 2, trainer.args.device)
            assert denominator.item() == 4
            logits = torch.randn(2, 4, 7, requires_grad=True)
            ref_lp = spft_target_logps(torch.randn_like(logits), self.labels)
            full = spft_loss_func({"logits": logits}, self.labels, ref_lp)
            split = sum(
                spft_loss_func(
                    {"logits": logits[i : i + 1]},
                    self.labels[i : i + 1],
                    ref_lp[i : i + 1],
                    num_items_in_batch=denominator,
                )
                for i in range(2)
            )
            torch.testing.assert_close(full, split)
            torch.testing.assert_close(
                torch.autograd.grad(full, logits, retain_graph=True)[0], torch.autograd.grad(split, logits)[0]
            )

    def test_tiny_qwen_vl_image_step(self):
        config = Qwen2_5_VLConfig(
            text_config={
                "vocab_size": 16,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 2]},
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 16,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 2,
                "window_size": 4,
                "fullatt_block_indexes": [0],
            },
            image_token_id=4,
            video_token_id=3,
            vision_start_token_id=5,
            vision_end_token_id=6,
        )
        config._attn_implementation = "eager"
        with tempfile.TemporaryDirectory() as directory:
            model = Qwen2_5_VLForConditionalGeneration(config)
            trainer = self.make_trainer(directory, model=model)
            inputs = {
                "input_ids": torch.tensor([[5, 4, 6, 7, 8, 9]]),
                "labels": torch.tensor([[-100, -100, -100, -100, 8, 9]]),
                "attention_mask": torch.ones(1, 6, dtype=torch.long),
                "pixel_values": torch.rand(4, 12),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "position_ids": torch.arange(6).view(1, 1, 6).expand(3, 1, 6),
            }
            old_policy = model.lm_head.weight.detach().clone()
            old_reference = trainer.ref_model.lm_head.weight.detach().clone()
            loss = trainer.compute_loss(model, inputs)
            loss.backward()
            torch.optim.SGD(model.parameters(), lr=0.1).step()
            assert torch.isfinite(loss)
            assert not torch.equal(old_policy, model.lm_head.weight)
            torch.testing.assert_close(old_reference, trainer.ref_model.lm_head.weight)
            assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())

    def test_training_loop_partial_accumulation_matches_full_batch(self):
        examples = [{"input_ids": torch.tensor([0, 1, 2, 3]), "labels": row} for row in self.labels]
        with tempfile.TemporaryDirectory() as directory:
            accumulated = self.make_trainer(str(Path(directory) / "accumulated"), accumulation=8)
            full_batch = self.make_trainer(
                str(Path(directory) / "full"), model=copy.deepcopy(accumulated.model), batch_size=2
            )
            for trainer in (accumulated, full_batch):
                trainer.train_dataset = examples
                trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
            accumulated.train()
            full_batch.train()
            for actual, expected in zip(accumulated.model.parameters(), full_batch.model.parameters()):
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)

    def test_reference_snapshot_resume_and_missing_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_dir=directory,
                resume_from_checkpoint=None,
                do_train=True,
                process_index=0,
                main_process_first=lambda **kwargs: nullcontext(),
            )
            model = TinyImageModel()
            reference = create_spft_ref_model(model, args)
            with torch.no_grad():
                model.projection.weight.add_(1)
            args.resume_from_checkpoint = str(Path(directory) / "checkpoint-1")
            restored = create_spft_ref_model(model, args)
            torch.testing.assert_close(restored.projection.weight, reference.projection.weight)
            assert not restored.training
            assert all(not parameter.requires_grad for parameter in restored.parameters())
            args.resume_from_checkpoint = str(Path(directory) / "elsewhere" / "checkpoint-1")
            with self.assertRaisesRegex(ValueError, "original frozen reference"):
                create_spft_ref_model(model, args)

    def test_argument_validation(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FinetuningArguments(use_spft_loss=True, spft_lambda=value)
        for flag in ("use_dft_loss", "use_eaft_loss", "use_asft_loss"):
            with self.assertRaises(ValueError):
                FinetuningArguments(use_spft_loss=True, **{flag: True})
        with self.assertRaises(ValueError):
            FinetuningArguments(use_spft_loss=True, stage="pt")


if __name__ == "__main__":
    unittest.main()
