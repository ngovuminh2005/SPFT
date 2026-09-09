"""Numerically stable token weights for sigmoid preference fine-tuning."""

import torch


def spft_token_weights(
    target_log_probs: torch.Tensor,
    reference_target_log_probs: torch.Tensor,
    lambda_: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return detached SPFT weights for aligned target-token log probabilities."""
    if target_log_probs.shape != reference_target_log_probs.shape:
        raise ValueError("target and reference log probabilities must have the same shape")
    if not torch.isfinite(torch.as_tensor(lambda_)):
        raise ValueError("SPFT lambda must be finite")
    if not 0.0 < eps < 0.5:
        raise ValueError("SPFT eps must be in (0, 0.5)")
    with torch.no_grad():
        target_probs = target_log_probs.float().exp().clamp(min=eps, max=1.0 - eps)
        reference_probs = reference_target_log_probs.float().exp().clamp(min=eps, max=1.0 - eps)
        target_log_odds = target_probs.log() - torch.log1p(-target_probs)
        reference_log_odds = reference_probs.log() - torch.log1p(-reference_probs)
        preference = torch.sigmoid(-float(lambda_) * (target_log_odds - reference_log_odds))
        weights = target_probs * preference
        return weights.detach()
