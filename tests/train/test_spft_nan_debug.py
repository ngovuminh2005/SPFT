"""CPU tests for opt-in SPFT fail-fast diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from llamafactory.train.spft_nan_debug import SPFTNaNDebug
from llamafactory.train.trainer_utils import spft_loss_func, spft_target_logps


class DebugTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.trainer = SimpleNamespace(
            model=torch.nn.Linear(2, 3), ref_model=torch.nn.Linear(2, 3),
            state=SimpleNamespace(global_step=0),
            args=SimpleNamespace(output_dir=self.directory.name, process_index=0),
        )
        self.debug = SPFTNaNDebug(self.trainer)
        self.labels = torch.tensor([[-100, 1, 2]])
        self.inputs = {"input_ids": torch.tensor([[0, 1, 2]]), "pixel_values": torch.ones(2, 2)}
        self.debug.begin(self.inputs, self.labels)

    def tearDown(self):
        for handle in self.debug.handles:
            handle.remove()
        self.directory.cleanup()

    def test_finite_loss_and_gradient_unchanged(self):
        logits = torch.randn(1, 3, 3, requires_grad=True)
        ref = spft_target_logps(logits.detach(), self.labels)
        ordinary = spft_loss_func({"logits": logits}, self.labels, ref)
        traced = spft_loss_func({"logits": logits}, self.labels, ref, debug=self.debug)
        torch.testing.assert_close(ordinary, traced)
        torch.testing.assert_close(torch.autograd.grad(ordinary, logits, retain_graph=True)[0],
                                   torch.autograd.grad(traced, logits)[0])

    def test_input_nan_is_reported(self):
        self.inputs["pixel_values"][0, 1] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "input:pixel_values"):
            self.debug.begin(self.inputs, self.labels)
        record = json.loads((Path(self.directory.name) / "spft_nan_debug.jsonl").read_text())
        self.assertEqual(record["first_bad_index"], [0, 1])

    def test_backward_nonfinite_stops_before_update(self):
        loss = self.trainer.model(torch.ones(1, 2)).sum() * float("inf")
        with self.assertRaisesRegex(FloatingPointError, "parameter.grad"):
            loss.backward()

    def test_sigmoid_inf_is_reported(self):
        with self.assertRaisesRegex(FloatingPointError, "sigmoid.input"):
            self.debug.check("sigmoid.input", torch.tensor([float("inf")]))


if __name__ == "__main__":
    unittest.main()
