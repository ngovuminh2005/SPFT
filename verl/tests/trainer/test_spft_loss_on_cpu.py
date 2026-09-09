import pytest
import torch

from verl.trainer.spft import spft_token_weights


def test_spft_weights_match_weighted_sft_formula_and_are_detached():
    target_log_probs = torch.tensor([-2.0, -0.7], requires_grad=True)
    reference_log_probs = torch.tensor([-1.5, -1.2], requires_grad=True)
    lambda_ = 0.8

    weights = spft_token_weights(target_log_probs, reference_log_probs, lambda_)

    target_probs = target_log_probs.detach().exp().clamp(min=1e-6, max=1.0 - 1e-6)
    reference_probs = reference_log_probs.detach().exp().clamp(min=1e-6, max=1.0 - 1e-6)
    expected = target_probs * torch.sigmoid(
        -lambda_
        * (
            torch.log(target_probs / (1.0 - target_probs))
            - torch.log(reference_probs / (1.0 - reference_probs))
        )
    )
    torch.testing.assert_close(weights, expected)
    assert not weights.requires_grad

    (-target_log_probs * weights).sum().backward()
    torch.testing.assert_close(target_log_probs.grad, -weights)
    assert reference_log_probs.grad is None


def test_spft_lambda_zero_is_half_target_probability():
    target_log_probs = torch.tensor([-6.0, -1.0, -0.01])
    reference_log_probs = torch.tensor([-0.2, -2.0, -4.0])

    weights = spft_token_weights(target_log_probs, reference_log_probs, lambda_=0.0)

    torch.testing.assert_close(weights, target_log_probs.exp().clamp(min=1e-6, max=1.0 - 1e-6) / 2.0)


@pytest.mark.parametrize("lambda_", [float("inf"), float("nan")])
def test_spft_rejects_nonfinite_lambda(lambda_):
    with pytest.raises(ValueError, match="lambda"):
        spft_token_weights(torch.tensor([-1.0]), torch.tensor([-1.0]), lambda_)


def test_spft_weights_are_finite_at_probability_extremes():
    target_log_probs = torch.tensor([-1000.0, -1e-9])
    reference_log_probs = torch.tensor([-1e-9, -1000.0])

    weights = spft_token_weights(target_log_probs, reference_log_probs, lambda_=1.0)

    assert torch.isfinite(weights).all()
