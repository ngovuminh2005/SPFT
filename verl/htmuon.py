"""Single-device spectral optimizers; exact transforms use the smaller Gram matrix."""
import math
import torch
from muon_ot import _maybe_to_local_tensor, _maybe_from_local_like


def spectral_transform(matrix, fn):
    dtype = matrix.dtype
    work = matrix if dtype == torch.float64 else matrix.float()
    transpose = work.shape[-2] > work.shape[-1]
    p = work.mT if transpose else work
    p = p / p.norm(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    values, u = torch.linalg.eigh(p @ p.mT)
    cutoff = values.amax(dim=-1, keepdim=True) * max(p.shape[-2:]) * torch.finfo(p.dtype).eps
    active = values > cutoff
    s = values.clamp_min(0).sqrt()
    factors = torch.where(active, fn(s) / s.clamp_min(torch.finfo(s.dtype).tiny), 0)
    result = (u * factors.unsqueeze(-2)) @ (u.mT @ p)
    return (result.mT if transpose else result).to(dtype)


def htmuon_transform(matrix, alpha=0.125):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and non-negative')
    return spectral_transform(matrix, lambda s: s.pow(alpha))


class SpectralOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0., momentum=0.95,
                 nesterov=True, ns_steps=5, alpha=0.125, lamdba=0.01, mode='polynomial'):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, ns_steps=ns_steps, alpha=alpha, lamdba=lamdba, mode=mode,
            use_muon=True, betas=(0.9, 0.95), eps=1e-8))
        for group in self.param_groups:
            for key in ('lr', 'weight_decay', 'alpha', 'lamdba'):
                if not math.isfinite(group[key]) or group[key] < 0:
                    raise ValueError(f'{key} must be finite and non-negative')
            if not 0 <= group['momentum'] < 1:
                raise ValueError('momentum must be in [0, 1)')
            if group['mode'] not in ('polynomial', 'exact', 'direct'):
                raise ValueError('mode must be polynomial, exact, or direct')
            for p in group['params']:
                if hasattr(p, 'device_mesh') and p.device_mesh.size() != 1:
                    raise ValueError('Only single-device DTensors are supported')
                if group['use_muon'] and p.ndim not in (2, 4):
                    raise ValueError('Main parameters must be 2D or 4D')

    def transform(self, matrix, group):
        return htmuon_transform(matrix, group['alpha'])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = _maybe_to_local_tensor(param.grad)
                if grad.is_sparse:
                    raise ValueError('Sparse gradients are unsupported')
                state = self.state[param]
                if group['use_muon']:
                    state.setdefault('momentum_buffer', torch.zeros_like(param))
                    buf = _maybe_to_local_tensor(state['momentum_buffer'])
                    buf.lerp_(grad, 1 - group['momentum'])
                    update = torch.lerp(grad, buf, group['momentum']) if group['nesterov'] else buf
                    matrix = update.reshape(update.shape[0], -1)
                    update = self.transform(matrix, group)
                    update = update * max(1, matrix.shape[0] / matrix.shape[1]) ** 0.5
                else:
                    state.setdefault('exp_avg', torch.zeros_like(param))
                    state.setdefault('exp_avg_sq', torch.zeros_like(param))
                    state['step'] = state.get('step', 0) + 1
                    b1, b2 = group['betas']
                    avg = _maybe_to_local_tensor(state['exp_avg'])
                    sq = _maybe_to_local_tensor(state['exp_avg_sq'])
                    avg.lerp_(grad, 1 - b1)
                    sq.lerp_(grad.square(), 1 - b2)
                    update = (avg / (1-b1**state['step'])) / ((sq / (1-b2**state['step'])).sqrt() + group['eps'])
                param.mul_(1 - group['lr'] * group['weight_decay'])
                param.add_(_maybe_from_local_like(update.reshape(param.shape), param), alpha=-group['lr'])
        return loss


class SingleDeviceHTMuon(SpectralOptimizer):
    pass


class SingleDeviceHTMuonWithAuxAdam(SpectralOptimizer):
    pass


HTMuon = SingleDeviceHTMuon
HTMuonWithAuxAdam = SingleDeviceHTMuonWithAuxAdam
