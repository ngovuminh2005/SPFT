"""Soren-lamdba optimizer.

This module implements the matrix map

    F_lamdba(M) = U f_lamdba(Sigma) V^T,

without an SVD or an inverse-square-root recurrence. The polar factor is
computed directly, and sqrt(C), C = M^T M + lamdba I, is approximated by a
bounded degree-5 polynomial.

The spelling ``lamdba`` is kept for API compatibility with the requested
optimizer/file name.  ``lamdba`` must be non-negative.
"""

import math

import torch
import torch.distributed as dist


DEFAULT_LAMBDA = 0.01
DIRECT_POLAR_EXPRESS_COEFFICIENTS = (
    (8.2872120181, -23.5958865191, 17.3003873125),
    (4.1070591115, -2.9478499167, 0.5448431083),
    (3.9486908535, -2.9089021160, 0.5518191394),
    (3.3184196574, -2.4884880243, 0.5100489401),
    (2.3006520200, -1.6689039846, 0.4188073120),
)

'''#lamdba = 0.2
SQRT_C_POLYNOMIAL_COEFFICIENTS = (
    0.1714558153,
    1.7285168497,
    -2.1355981315,
    2.1968040967,
    -1.2554722078,
    0.2943679349,
)'''

'''#lamdba = 0.3
SQRT_C_POLYNOMIAL_COEFFICIENTS = (
    0.1938685770721983,
    1.540096103731852,
    -1.543010720095479,
    1.319708220595351,
    -0.6399749445660882,
    0.1293776362881557,
)'''

#lamdba = 0.4
SQRT_C_POLYNOMIAL_COEFFICIENTS = (
    0.2123979434544122,
    1.411278985719221,
    -1.200976738359324,
    0.8847951978115703,
    -0.3741056435985010,
    0.06661243922330126,
)

'''lamdba = 0.5
SQRT_C_POLYNOMIAL_COEFFICIENTS = (
    0.2286050232878405,
    1.314383920732008,
    -0.9770813804227896,
    0.6344183505610992,
    -0.2382927035535882,
    0.03795316479510489,
)'''

'''lamdba = 0.6
SQRT_C_POLYNOMIAL_COEFFICIENTS = (
    0.2432501272152470,
    1.237233272530939,
    -0.8187968718217352,
    0.4761646849625242,
    -0.1610763761108560,
    0.02321962939895216,
)'''

'''lamdba = 0.7
SQRT_C_POLYNOMIAL_COEFFICIENTS = (
    0.2567311565983472,
    1.173595076750519,
    -0.7011876178568792,
    0.3697260961537200,
    -0.1138587421365026,
    0.01499658012196278,
)'''

'''lamdba = 0.8
SQRT_C_POLYNOMIAL_COEFFICIENTS = (
    0.2693194073949569,
    1.119676854238469,
    -0.6104318573077383,
    0.2946515695795659,
    -0.08331812713364857,
    0.01010483017140051,
)'''





def _validate_lamdba(lamdba):
    lamdba = float(lamdba)
    if not math.isfinite(lamdba) or lamdba < 0.0:
        raise ValueError(f"lamdba must be a finite non-negative number, got {lamdba}")
    return lamdba


def _symmetrize(matrix):
    """Project a square matrix/batch back onto the symmetric subspace."""
    return 0.5 * (matrix + matrix.transpose(-2, -1))


def _direct_polar_factor(matrix, steps, identity, symmetrize=True):
    """Approximate a rectangular polar factor with staged quintic maps."""
    polar = matrix
    for a, b, c in DIRECT_POLAR_EXPRESS_COEFFICIENTS[:steps]:
        gram = polar.transpose(-2, -1) @ polar
        if symmetrize:
            gram = _symmetrize(gram)
        polar = polar @ (a * identity + b * gram + c * (gram @ gram))
    return polar


def _sqrt_c_polynomial(matrix, identity, symmetrize=True):
    """Evaluate the supplied q_5(C) approximation using Horner's rule."""
    result = SQRT_C_POLYNOMIAL_COEFFICIENTS[-1] * identity
    for coefficient in reversed(SQRT_C_POLYNOMIAL_COEFFICIENTS[:-1]):
        result = result @ matrix + coefficient * identity
    return _symmetrize(result) if symmetrize else result


def soren_lamdba_transform(
    matrix,
    lamdba=DEFAULT_LAMBDA,
    steps=5,
    eps=1e-7,
    normalize=True,
    stable_accumulation=True,
    symmetrize=True,
    check_finite=False,
):
    """Approximate F_lamdba(matrix) with direct polar and sqrt polynomials.

    ``matrix`` may be a 2-D matrix or a batch of matrices.

    float16/bfloat16 inputs are accumulated in float32 by default. The
    transform never forms an inverse or inverse-square-root state.

    The result is cast back to the input dtype before returning.
    """
    if matrix.ndim < 2:
        raise ValueError(f"matrix must have at least 2 dimensions, got {matrix.shape}")
    if steps < 1 or steps > len(DIRECT_POLAR_EXPRESS_COEFFICIENTS):
        raise ValueError(
            "steps must be between 1 and "
            f"{len(DIRECT_POLAR_EXPRESS_COEFFICIENTS)}, got {steps}"
        )

    lamdba = _validate_lamdba(lamdba)
    input_dtype = matrix.dtype
    work_matrix = matrix

    # BF16/FP16 are kept for storage, while matrix-polynomial accumulation is FP32.
    if stable_accumulation and input_dtype in (torch.float16, torch.bfloat16):
        work_matrix = work_matrix.float()

    if normalize:
        matrix_norm = work_matrix.norm(dim=(-2, -1), keepdim=True)
        work_matrix = work_matrix / (matrix_norm + eps)

    n = work_matrix.size(-1)
    identity = torch.eye(n, device=work_matrix.device, dtype=work_matrix.dtype)

    gram_a = work_matrix.transpose(-2, -1) @ work_matrix
    if symmetrize:
        gram_a = _symmetrize(gram_a)

    identity = identity.expand(gram_a.shape)
    gram_c = gram_a + lamdba * identity
    if symmetrize:
        gram_c = _symmetrize(gram_c)

    polar = _direct_polar_factor(
        work_matrix, steps=steps, identity=identity, symmetrize=symmetrize
    )
    sqrt_c = _sqrt_c_polynomial(gram_c, identity=identity, symmetrize=symmetrize)

    denominator = 1.0 + math.sqrt(1.0 + lamdba)
    result = (work_matrix + polar @ sqrt_c) / denominator

    if check_finite and not torch.isfinite(result).all():
        raise FloatingPointError("non-finite output from soren_lamdba_transform")

    return result.to(input_dtype)



def soren_lamdba_update(
    grad,
    momentum,
    lamdba=DEFAULT_LAMBDA,
    beta=0.95,
    ns_steps=5,
    nesterov=True,
):
    """Apply momentum followed by the Soren-lamdba matrix map."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:
        update = update.view(len(update), -1)
    if update.ndim < 2:
        raise ValueError(f"Soren-lamdba requires matrix parameters, got {update.shape}")

    update = soren_lamdba_transform(
        update,
        lamdba=lamdba,
        steps=ns_steps,
    )
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Bias-corrected Adam update used for auxiliary parameters."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon_lamdba(torch.optim.Optimizer):
    """Distributed Soren-lamdba optimizer for matrix parameters."""

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        lamdba=DEFAULT_LAMBDA,
        ns_steps=5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            lamdba=_validate_lamdba(lamdba),
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            pad = (-len(params)) % dist.get_world_size()
            params_pad = params + [torch.empty_like(params[-1])] * pad
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = soren_lamdba_update(
                        p.grad,
                        state["momentum_buffer"],
                        lamdba=group["lamdba"],
                        ns_steps=group["ns_steps"],
                        beta=group["momentum"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


class SingleDeviceMuon_lamdba(torch.optim.Optimizer):
    """Single-device Soren-lamdba optimizer for matrix parameters."""

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        lamdba=DEFAULT_LAMBDA,
        ns_steps=5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            lamdba=_validate_lamdba(lamdba),
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = soren_lamdba_update(
                    p.grad,
                    state["momentum_buffer"],
                    lamdba=group["lamdba"],
                    ns_steps=group["ns_steps"],
                    beta=group["momentum"],
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class MuonWithAuxAdam_lamdba(torch.optim.Optimizer):
    """Soren-lamdba for matrix parameters and Adam for auxiliary parameters."""

    def __init__(
        self,
        param_groups,
        lamdba=DEFAULT_LAMBDA,
        ns_steps=5,
    ):
        lamdba = _validate_lamdba(lamdba)
        if ns_steps < 1 or ns_steps > len(DIRECT_POLAR_EXPRESS_COEFFICIENTS):
            raise ValueError(
                "ns_steps must be between 1 and "
                f"{len(DIRECT_POLAR_EXPRESS_COEFFICIENTS)}, got {ns_steps}"
            )

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )

        # Keeping these in optimizer defaults makes them part of state_dict().
        super().__init__(
            param_groups,
            {
                "lamdba": lamdba,
                "ns_steps": int(ns_steps),
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                pad = (-len(params)) % dist.get_world_size()
                params_pad = params + [torch.empty_like(params[-1])] * pad
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = soren_lamdba_update(
                            p.grad,
                            state["momentum_buffer"],
                            lamdba=group["lamdba"],
                            ns_steps=group["ns_steps"],
                            beta=group["momentum"],
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam_lamdba(torch.optim.Optimizer):
    """Single-device Soren-lamdba with an auxiliary Adam branch."""

    def __init__(
        self,
        param_groups,
        lamdba=DEFAULT_LAMBDA,
        ns_steps=5,
    ):
        lamdba = _validate_lamdba(lamdba)
        if ns_steps < 1 or ns_steps > len(DIRECT_POLAR_EXPRESS_COEFFICIENTS):
            raise ValueError(
                "ns_steps must be between 1 and "
                f"{len(DIRECT_POLAR_EXPRESS_COEFFICIENTS)}, got {ns_steps}"
            )

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )

        super().__init__(
            param_groups,
            {
                "lamdba": lamdba,
                "ns_steps": int(ns_steps),
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = soren_lamdba_update(
                        p.grad,
                        state["momentum_buffer"],
                        lamdba=group["lamdba"],
                        ns_steps=group["ns_steps"],
                        beta=group["momentum"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# Correctly-spelled aliases are useful for Python callers; the requested
# ``lamdba`` names remain the canonical CLI/module names.
soren_lambda_transform = soren_lamdba_transform
soren_lambda_update = soren_lamdba_update
MuonWithAuxAdam_lambda = MuonWithAuxAdam_lamdba
