"""Proximal SFT: PPO clipped objective with unit positive advantages.

Adapted from zwhong714/PSFT commit 930e23980a723ecef5af138e6e32aa3798b1fd64,
verl/trainer/ppo/core_algos.py (Apache-2.0).
Old log probabilities come from the policy before the current outer batch.
"""

import math

import torch


def psft_token_loss(log_probs, old_log_probs, clip_ratio_low=0.2, clip_ratio_high=0.28):
    """Return unreduced loss; callers apply the response/padding mask."""
    if log_probs.shape != old_log_probs.shape:
        raise ValueError("Current and old log probabilities must have identical shapes")
    if not 0 <= clip_ratio_low < 1 or not math.isfinite(clip_ratio_high) or clip_ratio_high < 0:
        raise ValueError("Invalid PSFT clip ratios")
    ratio = (log_probs.float() - old_log_probs.detach().float()).clamp(-20, 20).exp()
    return torch.maximum(-ratio, -ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high))
