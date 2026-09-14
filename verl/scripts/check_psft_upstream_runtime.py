"""CPU smoke tests using the vendored runtime, not the local SFT loss port."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "third_party/psft"
sys.path.insert(0, str(RUNTIME))

import torch
import yaml
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_policy_loss_vanilla
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils.torch_functional import get_response_mask


class UpstreamRuntime(unittest.TestCase):
    def test_masked_unit_advantages_ignore_reward_values(self):
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        data = DataProto.from_dict({"response_mask": mask, "token_level_rewards": torch.randn(2, 3)})
        result = compute_advantage(data, AdvantageEstimator.PSFT)
        torch.testing.assert_close(result.batch["advantages"], mask.float())
        torch.testing.assert_close(result.batch["returns"], mask.float())

    def test_clipping_and_masked_gradient(self):
        old = torch.full((1, 4), -2.0)
        log_prob = (old + torch.tensor([[0.5, 1.1, 1.5, 1.0]]).log()).requires_grad_()
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        config = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "clip_ratio_c": 10.0})
        loss, *_ = compute_policy_loss_vanilla(old, log_prob, mask, mask, config=config)
        torch.testing.assert_close(loss, torch.tensor(-(0.5 + 1.1 + 1.28) / 3))
        loss.backward()
        torch.testing.assert_close(log_prob.grad, torch.tensor([[-0.5 / 3, -1.1 / 3, 0.0, 0.0]]))

    def test_response_mask_includes_first_eos_only(self):
        response = torch.tensor([[12, 99, 99, 99], [12, 13, 14, 99]])
        expected = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        torch.testing.assert_close(get_response_mask(response, eos_token=99), expected)

    def test_full_launcher_hydra_config_composes(self):
        env = {"PATH": os.environ["PATH"], "DRY_RUN": "1"}
        output = subprocess.check_output(["bash", str(ROOT / "train_psft_1gpu.sh")], env=env, text=True)
        args = output.splitlines()[2:]
        env.pop("DRY_RUN")
        env["PYTHONPATH"] = str(RUNTIME)
        result = subprocess.run(
            [sys.executable, *args, "--cfg", "job"], cwd=RUNTIME, env=env,
            text=True, capture_output=True, check=True,
        )
        config = yaml.safe_load(result.stdout)
        actor = config["actor_rollout_ref"]["actor"]
        self.assertEqual(config["algorithm"]["adv_estimator"], "psft")
        self.assertEqual(actor["optim"]["warmup_style"], "constant")
        self.assertEqual(actor["optim"]["lr_warmup_steps"], 10)
        self.assertEqual(actor["optim"]["lr"], 1e-6)
        self.assertEqual(actor["ppo_mini_batch_size"], 32)
        self.assertEqual(actor["ppo_epochs"], 1)
        self.assertEqual(actor["policy_loss"]["loss_mode"], "vanilla")
        self.assertEqual(config["trainer"]["total_epochs"], 10)


if __name__ == "__main__":
    unittest.main()
