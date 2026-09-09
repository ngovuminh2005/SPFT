import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "psft", Path(__file__).resolve().parents[2] / "verl/trainer/psft.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PSFTLossTest(unittest.TestCase):
    def test_clipping_and_frozen_old_policy(self):
        old = torch.full((3,), -3.0, requires_grad=True)
        current = (old.detach() + torch.tensor([0.5, 1.0, 1.5]).log()).requires_grad_()
        loss = module.psft_token_loss(current, old)
        torch.testing.assert_close(loss, torch.tensor([-0.5, -1.0, -1.28]))
        loss.sum().backward()
        torch.testing.assert_close(current.grad, torch.tensor([-0.5, -1.0, 0.0]))
        self.assertIsNone(old.grad)

    def test_matches_upstream_unit_advantage_objective(self):
        torch.manual_seed(1)
        current = torch.randn(32, requires_grad=True)
        old = torch.randn(32)
        ratio = (current - old).clamp(-20, 20).exp()
        expected = torch.maximum(-ratio, -ratio.clamp(0.8, 1.28))
        actual = module.psft_token_loss(current, old)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), current, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), current)[0],
        )


if __name__ == "__main__":
    unittest.main()
