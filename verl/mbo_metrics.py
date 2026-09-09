"""Optional W&B metrics for MBO optimizers.

Keeping metric collection here makes it possible to remove or extend MBO
observability without changing the optimizer's update logic.
"""

import torch


class MBOMetricTracker:
    """Accumulates per-parameter MBO metrics for one optimizer step."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._vendi_scores = []
        self._correction_gradient_cosines = []

    def record_vendi_score(self, value):
        self._vendi_scores.append(value.detach())

    def record_correction_gradient_cosine(self, correction, gradient):
        """Record cosine(correction, pre-correction gradient/update)."""
        correction = correction.detach().float().reshape(-1)
        gradient = gradient.detach().float().reshape(-1)
        denominator = correction.norm() * gradient.norm()
        if denominator > 0:
            self._correction_gradient_cosines.append(torch.dot(correction, gradient) / denominator)

    def get_metrics(self):
        metrics = {}
        if self._vendi_scores:
            metrics["mbo/vendi_score"] = torch.stack(self._vendi_scores).mean().item()
        if self._correction_gradient_cosines:
            metrics["mbo/correction_gradient_cosine"] = torch.stack(
                self._correction_gradient_cosines
            ).mean().item()
        return metrics
