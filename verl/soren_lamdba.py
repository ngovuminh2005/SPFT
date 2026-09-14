"""Parameterized Soren from the workspace's soren_lamda.py reference.

polynomial/exact: (x + sqrt(x*x + lamdba)) / (1 + sqrt(1 + lamdba)).
Null singular directions are excluded by the compact Gram transform.
"""
import math
from functools import lru_cache
import torch
from htmuon import SpectralOptimizer, spectral_transform

POLAR_COEFFICIENTS = (
    (8.2872120181, -23.5958865191, 17.3003873125),
    (4.1070591115, -2.9478499167, 0.5448431083),
    (3.9486908535, -2.9089021160, 0.5518191394),
    (3.3184196574, -2.4884880243, 0.5100489401),
    (2.3006520200, -1.6689039846, 0.4188073120),
)


@lru_cache(maxsize=128)
def sqrt_coefficients(lamdba, degree=5, iterations=8, grid_size=16385):
    """Return degree-5 Remez minimax coefficients for sqrt(t + lambda), t in [0, 1]."""
    n = int(degree)
    grid = torch.linspace(0.0, 1.0, int(grid_size), dtype=torch.float64)
    target = (grid + float(lamdba)).sqrt()
    # Chebyshev-Lobatto starting alternation points, ordered from left to right.
    points = ((1.0 + torch.cos(torch.arange(n + 2, dtype=torch.float64) * math.pi / (n + 1))) / 2.0).flip(0)
    coeff = None
    for _ in range(int(iterations)):
        vandermonde = torch.stack([points**power for power in range(n + 1)] +
                                  [(-torch.ones(n + 2, dtype=torch.float64))**torch.arange(n + 2)], dim=1)
        solution = torch.linalg.solve(vandermonde, (points + float(lamdba)).sqrt())
        coeff, error_level = solution[:-1], solution[-1]
        approximation = torch.zeros_like(grid)
        for value in reversed(coeff.tolist()):
            approximation = approximation * grid + value
        error = approximation - target
        magnitude = error.abs()
        local = torch.where((magnitude[1:-1] >= magnitude[:-2]) &
                            (magnitude[1:-1] >= magnitude[2:]))[0] + 1
        candidates = torch.cat((torch.zeros(1, dtype=torch.long), local,
                                torch.full((1,), grid.numel() - 1, dtype=torch.long)))
        # Keep the strongest point in each same-sign run.
        extrema = []
        for index in candidates.tolist():
            if not extrema or torch.sign(error[index]) != torch.sign(error[extrema[-1]]):
                extrema.append(index)
            elif magnitude[index] > magnitude[extrema[-1]]:
                extrema[-1] = index
        if len(extrema) >= n + 2:
            windows = [extrema[start:start + n + 2] for start in range(len(extrema) - n - 1)]
            extrema = max(windows, key=lambda window: min(float(magnitude[index]) for index in window))
        if len(extrema) != n + 2:
            raise RuntimeError(f"Remez failed to find {n + 2} alternating extrema")
        new_points = grid[torch.tensor(extrema, dtype=torch.long)]
        if torch.max((new_points - points).abs()) < 2.0 / grid_size:
            break
        points = new_points
    return tuple(float(value) for value in coeff)


def soren_lamdba_transform(matrix, lamdba=0.01, mode='polynomial', steps=5):
    if not math.isfinite(lamdba) or lamdba < 0:
        raise ValueError('lamdba must be finite and non-negative')
    denominator = 1 + math.sqrt(1 + lamdba)
    if mode == 'exact':
        return spectral_transform(matrix, lambda s: (s + (s.square() + lamdba).sqrt()) / denominator)
    if mode not in ('polynomial', 'exact'):
        raise ValueError("mode must be 'polynomial' or 'exact'")
    if mode != 'polynomial' or not 1 <= steps <= 5:
        raise ValueError('polynomial mode requires 1 <= steps <= 5')
    dtype = matrix.dtype
    work = matrix if dtype == torch.float64 else matrix.float()
    transpose = work.shape[-2] > work.shape[-1]
    p = work.mT if transpose else work
    p = p / (p.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    gram = p @ p.mT
    identity = torch.eye(gram.shape[-1], device=p.device, dtype=p.dtype)
    polar = p
    for a, b, c in POLAR_COEFFICIENTS[:steps]:
        pp = polar @ polar.mT
        polar = (a * identity + b * pp + c * (pp @ pp)) @ polar
    coefficients = sqrt_coefficients(lamdba)
    root = coefficients[-1] * identity
    for coefficient in reversed(coefficients[:-1]):
        root = root @ gram + coefficient * identity
    result = (p + root @ polar) / denominator
    return (result.mT if transpose else result).to(dtype)


class SingleDeviceSoren(SpectralOptimizer):
    def transform(self, matrix, group):
        return soren_lamdba_transform(matrix, group['lamdba'], group['mode'], group['ns_steps'])


class SingleDeviceSorenWithAuxAdam(SingleDeviceSoren):
    pass


Soren = SingleDeviceSoren
SorenWithAuxAdam = SingleDeviceSorenWithAuxAdam
