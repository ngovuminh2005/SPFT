import math
import torch
import torch.distributed as dist


# -----------------------------------------------------------------------------
# Muon core utilities kept because MBO still uses Muon-style momentum and
# Newton-Schulz orthogonalization for the final optimizer update.


def _is_single_rank_dtensor(tensor):
    if not all(hasattr(tensor, attr) for attr in ("to_local", "device_mesh", "placements")):
        return False
    try:
        return tensor.device_mesh.size() == 1
    except Exception:
        return False


def _maybe_to_local_tensor(tensor):
    return tensor.to_local() if _is_single_rank_dtensor(tensor) else tensor


def _maybe_from_local_like(local_tensor, like_tensor):
    if not _is_single_rank_dtensor(like_tensor):
        return local_tensor
    from torch.distributed._tensor import DTensor

    return DTensor.from_local(
        local_tensor,
        device_mesh=like_tensor.device_mesh,
        placements=like_tensor.placements,
        shape=like_tensor.shape,
        stride=local_tensor.stride(),
    )


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Uses the standard quintic Muon coefficients.
    """
    assert G.ndim >= 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    use_local_dtensor = _is_single_rank_dtensor(G)
    if use_local_dtensor:
        from torch.distributed._tensor import DTensor

        device_mesh = G.device_mesh
        placements = G.placements
        global_shape = G.shape
        G_work = G.to_local()
    else:
        DTensor = None
        device_mesh = None
        placements = None
        global_shape = None
        G_work = G

    out_dtype = G_work.dtype
    X = G_work.float()
    if G_work.size(-2) > G_work.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for idx in range(1, steps + 1):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G_work.size(-2) > G_work.size(-1):
        X = X.mT

    X = X.to(dtype=out_dtype)
    if use_local_dtensor:
        X = DTensor.from_local(
            X,
            device_mesh=device_mesh,
            placements=placements,
            shape=global_shape,
            stride=X.stride(),
        )
    return X


def _ensure_grad(p):
    if p.grad is None:
        p.grad = torch.zeros_like(p)
    return p.grad


def _ensure_state_tensor(state, key, like_tensor):
    if key not in state:
        state[key] = torch.zeros_like(like_tensor)
    else:
        state[key] = state[key].to(device=like_tensor.device, dtype=like_tensor.dtype)
    return state[key]


def next_power_of_two(n: int):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def fast_walsh_hadamard_transform(X):
    n = X.size(-1)
    if n & (n - 1):
        raise ValueError(f"Hadamard transform requires power-of-two width, got {n}")
    Y = X.contiguous().reshape(-1, n)
    h = 1
    while h < n:
        Y = Y.view(-1, n // (2 * h), 2, h)
        a = Y[:, :, 0, :]
        b = Y[:, :, 1, :]
        Y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (Y / math.sqrt(n)).reshape_as(X)


def orthogonalize_update_like_muon(G, steps: int):
    assert G.ndim == 2, f"expected a 2D matrix, got {tuple(G.shape)}"
    out = zeropower_via_newtonschulz5(G, steps=steps)
    dim_scale = max(1, G.size(-2) / G.size(-1)) ** 0.5
    return out, dim_scale


def orthogonalize_rows_direct(G):
    assert G.ndim == 2, f"expected a 2D matrix, got {tuple(G.shape)}"
    Q, _ = torch.linalg.qr(G.float().mT, mode="reduced")
    return Q.mT.to(dtype=G.dtype)


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _muon_update_param(p, state, group):
    if p.grad is None:
        p.grad = torch.zeros_like(p)

    if len(state) == 0:
        state["momentum_buffer"] = torch.zeros_like(p)

    beta = group["momentum"]
    momentum = state["momentum_buffer"]
    momentum.lerp_(p.grad, 1 - beta)

    update = torch.lerp(p.grad, momentum, beta) if group.get("nesterov", True) else momentum

    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=group.get("ns_steps", 5))

    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

    p.mul_(1 - group["lr"] * group["weight_decay"])
    p.add_(update.reshape(p.shape), alpha=-group["lr"])


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    _muon_update_param(p, self.state[p], group)
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                _muon_update_param(p, self.state[p], group)

        return loss


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        _muon_update_param(p, self.state[p], group)
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    _muon_update_param(p, self.state[p], group)

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class MuonWithAuxSGD(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        _muon_update_param(p, self.state[p], group)
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class SingleDeviceMuonWithAuxSGD(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    _muon_update_param(p, self.state[p], group)
        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


# -----------------------------------------------------------------------------
# MBO: Memory-Based Optimizer with OT-updated memory centroids.
#
# This replaces the old EMA memory update with an online semi-dual OT update:
#   min_Y max_g L(g; Y)
# For every memory batch, update g by ascent, then update Y by descent.


class _MBOBase(torch.optim.Optimizer):
    memory_state_dtype = torch.bfloat16

    @staticmethod
    def _validate_main_params(params):
        if not params:
            raise ValueError("MBO requires at least one main parameter.")
        for p in params:
            if not isinstance(p, torch.nn.Parameter):
                raise TypeError("MBO expects torch.nn.Parameter objects.")
            if p.ndim not in (2, 4):
                raise ValueError(f"MBO only supports 2D matrices or 4D conv kernels, got shape {tuple(p.shape)}.")

    @staticmethod
    def _matrix_view(G):
        if G.ndim == 4:
            return G.contiguous().reshape(len(G), -1)
        if G.ndim == 2:
            return G.contiguous()
        raise ValueError(f"MBO only supports 2D/4D params, got gradient with shape {tuple(G.shape)}")

    @staticmethod
    def _vectorize_gradient(G):
        if G.ndim not in (2, 4):
            raise ValueError(f"MBO only supports 2D/4D params, got gradient with shape {tuple(G.shape)}")
        return G.contiguous().reshape(-1)

    @staticmethod
    def _randn(shape, device, dtype, seed):
        if seed is None:
            return torch.randn(*shape, device=device, dtype=dtype)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        out = torch.randn(*shape, generator=gen, device="cpu", dtype=torch.float32)
        return out.to(device=device, dtype=dtype)

    def _build_srht_state(self, input_dim, target_dim, device, dtype, param_offset):
        if target_dim >= input_dim:
            return {
                "input_dim": input_dim,
                "padded_dim": input_dim,
                "target_dim": input_dim,
                "scale": 1.0,
                "signs": None,
                "sample_idx": None,
                "identity": True,
            }

        padded_dim = next_power_of_two(input_dim)
        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        seed = base_seed + int(input_dim) + int(param_offset)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (padded_dim,), generator=gen, device="cpu", dtype=torch.int8).to(device=device)
        sample_idx = torch.randperm(padded_dim, generator=gen, device="cpu")[:target_dim].to(device=device)
        signs = signs.to(dtype=dtype).mul_(2).sub_(1)
        sample_idx, _ = torch.sort(sample_idx)
        return {
            "input_dim": input_dim,
            "padded_dim": padded_dim,
            "target_dim": target_dim,
            "scale": math.sqrt(padded_dim / target_dim),
            "signs": signs,
            "sample_idx": sample_idx,
            "identity": False,
        }

    @staticmethod
    def _move_srht_state(srht_state, device, dtype):
        if srht_state is None or srht_state.get("identity", False):
            return srht_state
        srht_state["signs"] = srht_state["signs"].to(device=device, dtype=dtype)
        srht_state["sample_idx"] = srht_state["sample_idx"].to(device=device)
        return srht_state

    @staticmethod
    def _apply_srht(x, srht_state):
        if srht_state.get("identity", False):
            return x
        input_dim = srht_state["input_dim"]
        padded_dim = srht_state["padded_dim"]
        Y = x.unsqueeze(0)
        if padded_dim != input_dim:
            Y_pad = torch.zeros(1, padded_dim, device=x.device, dtype=x.dtype)
            Y_pad[:, :input_dim] = Y
            Y = Y_pad
        Y = Y * srht_state["signs"].unsqueeze(0)
        Y = fast_walsh_hadamard_transform(Y)
        Y = Y.index_select(1, srht_state["sample_idx"])
        return (Y * srht_state["scale"]).squeeze(0)

    @staticmethod
    def _apply_srht_transpose(x_proj, srht_state):
        if srht_state.get("identity", False):
            return x_proj
        padded_dim = srht_state["padded_dim"]
        Y = torch.zeros(1, padded_dim, device=x_proj.device, dtype=x_proj.dtype)
        Y[:, srht_state["sample_idx"]] = x_proj.unsqueeze(0)
        Y = Y * srht_state["scale"]
        Y = fast_walsh_hadamard_transform(Y)
        Y = Y * srht_state["signs"].unsqueeze(0)
        return Y[:, :srht_state["input_dim"]].squeeze(0)

    def _init_centroids(self, memory_dim, device, dtype, param_offset):
        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        seed = base_seed + int(param_offset)
        return self._randn((self.num_centroids, memory_dim), device=device, dtype=dtype, seed=seed)

    def _kmeanspp_init_centroids(self, samples, device, dtype, param_offset):
        """k-means++ warm-start seeding only; no Lloyd iterations."""
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        samples = samples.to(device=device, dtype=dtype)
        n, d = samples.shape
        if n == 0:
            return self._init_centroids(d, device, dtype, param_offset)

        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        seed = base_seed + int(param_offset) + 17
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        centers = torch.empty(self.num_centroids, d, device=device, dtype=dtype)
        first_idx = torch.randint(n, (1,), generator=gen, device="cpu").item()
        centers[0].copy_(samples[first_idx])

        closest_dist_sq = torch.cdist(samples.float(), centers[:1].float(), p=2).squeeze(1).pow_(2)
        for k in range(1, self.num_centroids):
            total = closest_dist_sq.sum()
            if not torch.isfinite(total) or total <= 0:
                idx = torch.randint(n, (1,), generator=gen, device="cpu").item()
            else:
                probs = (closest_dist_sq / total).detach().float().cpu()
                idx = torch.multinomial(probs, 1, replacement=True, generator=gen).item()
            centers[k].copy_(samples[idx])
            new_dist_sq = torch.cdist(samples.float(), centers[k:k + 1].float(), p=2).squeeze(1).pow_(2)
            closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)

        return centers

    def _finish_kmeanspp_warmstart(self, state, samples, param_offset):
        centroids = self._kmeanspp_init_centroids(
            samples,
            device=samples.device,
            dtype=samples.dtype,
            param_offset=param_offset,
        )
        state["memory_centroids"].copy_(centroids)
        state["memory_step"] = self.num_centroids
        state["memory_ws_initialized"] = True
        state["memory_ws_buffer"] = []
        queue_len = min(self.ot_step_update, samples.size(0))
        state["memory_pending_z"] = [z.detach().clone() for z in samples[-queue_len:]]
        state["memory_ot_g"].zero_()
        state["memory_ot_step"] = 1
        state["memory_ot_y_step"] = 1

    def _ensure_param_memory_state(self, p, state, grad_vec):
        device = grad_vec.device
        dtype = grad_vec.dtype
        memory_dtype = self.memory_state_dtype
        input_dim = grad_vec.numel()
        memory_dim = min(self.centroid_dim, input_dim)

        _ensure_state_tensor(state, "momentum_buffer", p)
        if "momentum_buffer_proj" not in state:
            state["momentum_buffer_proj"] = torch.zeros(memory_dim, device=device, dtype=memory_dtype)
        else:
            state["momentum_buffer_proj"] = state["momentum_buffer_proj"].to(device=device, dtype=memory_dtype)

        needs_reinit = (
            "memory_centroids" not in state
            or state.get("memory_input_dim") != input_dim
            or state.get("memory_dim") != memory_dim
            or state.get("memory_dtype") != memory_dtype
        )
        if needs_reinit:
            state["memory_input_dim"] = input_dim
            state["memory_dim"] = memory_dim
            state["memory_dtype"] = memory_dtype
            state["memory_pi"] = self._build_srht_state(input_dim, memory_dim, device, memory_dtype, id(p))
            state["memory_centroids"] = self._init_centroids(memory_dim, device, memory_dtype, id(p))
            state["memory_step"] = 0
            state["memory_m_sum"] = torch.zeros(memory_dim, device=device, dtype=memory_dtype)
            state["memory_ot_g"] = torch.zeros(self.num_centroids, device=device, dtype=torch.float32)
            state["memory_ot_step"] = 1
            state["memory_ot_y_step"] = 1
            state["memory_pending_z"] = []
            state["memory_ws_buffer"] = []
            state["memory_ws_initialized"] = self.ot_ws_steps <= 0
            return

        state["memory_dtype"] = memory_dtype
        state["memory_centroids"] = state["memory_centroids"].to(device=device, dtype=memory_dtype)
        state["memory_m_sum"] = state["memory_m_sum"].to(device=device, dtype=memory_dtype)
        state["memory_ot_g"] = state.get("memory_ot_g", torch.zeros(self.num_centroids, device=device, dtype=torch.float32))
        state["memory_ot_g"] = state["memory_ot_g"].to(device=device, dtype=torch.float32)
        state["memory_ot_step"] = int(state.get("memory_ot_step", 1))
        state["memory_ot_y_step"] = int(state.get("memory_ot_y_step", 1))
        state["memory_pi"] = self._move_srht_state(state["memory_pi"], device, memory_dtype)
        state["memory_pending_z"] = [z.to(device=device, dtype=memory_dtype) for z in state.get("memory_pending_z", [])]
        state["memory_ws_buffer"] = [z.to(device=device, dtype=memory_dtype) for z in state.get("memory_ws_buffer", [])]
        state["memory_ws_initialized"] = state.get("memory_ws_initialized", self.ot_ws_steps <= 0)

    def _project_vector(self, vec, state):
        return self._apply_srht(vec.to(dtype=state["memory_dtype"]), state["memory_pi"])

    @staticmethod
    def _normalize_vector(vec, eps=1e-8):
        return vec / (vec.norm() + eps)

    @staticmethod
    def _warmstart_memory(z, state, num_centroids):
        step = state["memory_step"]
        state["memory_centroids"][step].copy_(z)
        state["memory_m_sum"].add_(z)
        mean = state["memory_m_sum"] / (step + 1)
        if step + 1 < num_centroids:
            state["memory_centroids"][step + 1:].copy_(mean.unsqueeze(0))
        state["memory_step"] = step + 1

    def _ot_soft_assignment(self, samples_f, centroids_f, g_f):
        cost = torch.cdist(samples_f, centroids_f, p=2).pow(2)
        log_a = -math.log(float(self.num_centroids))
        logits = (g_f.unsqueeze(0) - cost) / self.ot_epsilon + log_a
        p = torch.softmax(logits, dim=1)
        return p

    def _ot_update_centroids(self, samples, state):
        if samples.numel() == 0:
            return
        samples_f = samples.float()
        centroids_f = state["memory_centroids"].float()
        g_f = state["memory_ot_g"].float()
        a = torch.full((self.num_centroids,), 1.0 / self.num_centroids, device=samples.device, dtype=torch.float32)
        batch_size = max(1, samples_f.size(0))

        # Dual ascent: update g first, holding centroids fixed.
        for _ in range(self.ot_g_steps):
            p = self._ot_soft_assignment(samples_f, centroids_f, g_f)
            grad_g = a - p.mean(dim=0)
            eta_g = self.ot_g_lr / math.log2(int(state["memory_ot_step"]) + 1)
            g_f = g_f + eta_g * grad_g
            g_f = g_f - g_f.mean()
            state["memory_ot_step"] = int(state["memory_ot_step"]) + 1

        # Primal descent: update y/centroids after g is updated.
        for _ in range(self.ot_y_steps):
            p = self._ot_soft_assignment(samples_f, centroids_f, g_f)
            weight_sum = p.sum(dim=0).unsqueeze(1)
            weighted_samples = p.transpose(0, 1) @ samples_f
            grad_y = (2.0 / batch_size) * (weight_sum * centroids_f - weighted_samples)
            eta_y = self.ot_y_lr / math.log2(int(state["memory_ot_y_step"]) + 1)
            centroids_f = centroids_f - eta_y * grad_y
            state["memory_ot_y_step"] = int(state["memory_ot_y_step"]) + 1

        state["memory_ot_g"].copy_(g_f)
        state["memory_centroids"].copy_(centroids_f.to(dtype=state["memory_dtype"]))

    def _update_memory_batch(self, samples, state, param_offset=0):
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        samples = samples.to(device=state["memory_centroids"].device, dtype=state["memory_dtype"])

        if self.ot_ws_steps > 0 and not state.get("memory_ws_initialized", False):
            state["memory_ws_buffer"].extend([z.detach().clone() for z in samples])
            if len(state["memory_ws_buffer"]) < self.ot_ws_steps:
                return
            ws_samples = torch.stack(state["memory_ws_buffer"], dim=0)
            self._finish_kmeanspp_warmstart(state, ws_samples, param_offset)
            return

        if state["memory_step"] < self.num_centroids:
            remaining = self.num_centroids - state["memory_step"]
            warm_count = min(remaining, samples.size(0))
            for i in range(warm_count):
                self._warmstart_memory(samples[i], state, self.num_centroids)
            if state["memory_step"] >= self.num_centroids:
                state["memory_ws_initialized"] = True
                state["memory_ot_g"].zero_()
                state["memory_ot_step"] = 1
                state["memory_ot_y_step"] = 1
            if warm_count == samples.size(0):
                return
            samples = samples[warm_count:]

        if not state.get("memory_ws_initialized", False):
            return

        self._ot_update_centroids(samples, state)

    @staticmethod
    def _queue_memory_point(z, state, max_queue_len):
        state["memory_pending_z"].append(z.detach().clone())
        overflow = len(state["memory_pending_z"]) - max_queue_len
        if overflow > 0:
            del state["memory_pending_z"][:overflow]

    def _flush_pending_memory(self, state, force=False, param_offset=0):
        pending = state.get("memory_pending_z", [])
        if not pending:
            return False

        if not state.get("memory_ws_initialized", False):
            self._update_memory_batch(pending[-1], state, param_offset=param_offset)
            return state.get("memory_ws_initialized", False)

        samples = torch.stack(pending, dim=0)
        self._update_memory_batch(samples, state, param_offset=param_offset)
        return True

    def _memory_correction(self, z_ns, state, ns_steps, out_dtype):
        basis = torch.cat([state["memory_centroids"], z_ns.unsqueeze(0)], dim=0)
        basis_ortho = orthogonalize_rows_direct(basis)
        basis_f = basis.float()
        basis_ortho_f = basis_ortho.float()
        alpha = (basis_f * basis_ortho_f).sum() / (basis_f.pow(2).sum() + 1e-8)
        correction_proj = (basis_ortho[-1] - alpha * basis[-1]).to(state["memory_dtype"])
        return self._apply_srht_transpose(correction_proj, state["memory_pi"]).to(out_dtype)

    def _step_muon_param(self, p, state, group):
        grad = _ensure_grad(p)
        grad_local = _maybe_to_local_tensor(grad)
        grad_vec = self._vectorize_gradient(grad_local)
        self._ensure_param_memory_state(p, state, grad_vec)

        beta = group["momentum"]
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - beta)

        if group["nesterov"]:
            update_pre_ns = torch.lerp(grad, buf, beta)
        else:
            update_pre_ns = buf

        update_pre_ns_local = _maybe_to_local_tensor(update_pre_ns)
        update_pre_ns_vec = self._vectorize_gradient(update_pre_ns_local)
        update_pre_ns_vec = self._normalize_vector(update_pre_ns_vec)
        z_for_correction = self._project_vector(update_pre_ns_vec, state)

        if state.get("memory_ws_initialized", False):
            correction_vec = self._memory_correction(z_for_correction, state, group["ns_steps"], grad_vec.dtype)
            correction = _maybe_from_local_like(correction_vec.view_as(grad_local), grad)
            update_in = update_pre_ns + self.lambda_memory * correction
        else:
            update_in = update_pre_ns

        O_t, dim_scale = orthogonalize_update_like_muon(self._matrix_view(update_in), steps=group["ns_steps"])

        p.mul_(1 - group["lr"] * group.get("weight_decay", 0.0))
        p.add_(O_t.reshape_as(p), alpha=-group["lr"] * dim_scale)

        self._queue_memory_point(z_for_correction, state, self.ot_step_update)
        self._flush_pending_memory(state, param_offset=id(p))

    def _step_plain_muon_param(self, p, state, group):
        grad = _ensure_grad(p)
        buf = _ensure_state_tensor(state, "momentum_buffer", p)

        beta = group["momentum"]
        buf.lerp_(grad, 1 - beta)

        if group["nesterov"]:
            update_in = torch.lerp(grad, buf, beta)
        else:
            update_in = buf

        O_t, dim_scale = orthogonalize_update_like_muon(self._matrix_view(update_in), steps=group["ns_steps"])

        p.mul_(1 - group["lr"] * group.get("weight_decay", 0.0))
        p.add_(O_t.reshape_as(p), alpha=-group["lr"] * dim_scale)

    def _flush_all_pending_memory(self):
        for state in self.state.values():
            if isinstance(state, dict) and "memory_pending_z" in state:
                self._flush_pending_memory(state, force=True)

    def _set_mbo_hparams(
        self,
        *,
        num_centroids,
        centroid_dim,
        lambda_memory,
        memory_init_seed,
        ot_ws_steps,
        ot_step_update,
        ot_epsilon,
        ot_g_lr,
        ot_y_lr,
        ot_g_steps,
        ot_y_steps,
        offset,
        window_size,
    ):
        self.num_centroids = int(num_centroids)
        self.centroid_dim = int(centroid_dim)
        self.lambda_memory = float(lambda_memory)
        self.memory_init_seed = memory_init_seed
        self.ot_ws_steps = int(ot_ws_steps)
        self.ot_step_update = int(ot_step_update)
        self.ot_epsilon = float(ot_epsilon)
        self.ot_g_lr = float(ot_g_lr)
        self.ot_y_lr = float(ot_y_lr)
        self.ot_g_steps = int(ot_g_steps)
        self.ot_y_steps = int(ot_y_steps)
        self.mbo_offset = int(offset)
        self.mbo_window_size = int(window_size)

        if self.num_centroids < 1:
            raise ValueError("num_centroids must be >= 1")
        if self.centroid_dim < 1:
            raise ValueError("centroid_dim must be >= 1")
        if self.ot_ws_steps < 0:
            raise ValueError("ot_ws_steps must be >= 0")
        if self.ot_step_update < 1:
            raise ValueError("ot_step_update must be >= 1")
        if self.ot_epsilon <= 0:
            raise ValueError("ot_epsilon must be > 0")
        if self.ot_g_lr <= 0:
            raise ValueError("ot_g_lr must be > 0")
        if self.ot_y_lr <= 0:
            raise ValueError("ot_y_lr must be > 0")
        if self.ot_g_steps < 1:
            raise ValueError("ot_g_steps must be >= 1")
        if self.ot_y_steps < 1:
            raise ValueError("ot_y_steps must be >= 1")
        if self.mbo_window_size < 0:
            raise ValueError("window_size must be >= 0")


class MBO(_MBOBase):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ot_ws_steps=512,
        ot_step_update=64,
        ot_epsilon=0.03,
        ot_g_lr=0.1,
        ot_y_lr=0.01,
        ot_g_steps=5,
        ot_y_steps=5,
        offset=0,
        window_size=0,
    ):
        params = list(params)
        self._validate_main_params(params)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self._set_mbo_hparams(
            num_centroids=num_centroids,
            centroid_dim=centroid_dim,
            lambda_memory=lambda_memory,
            memory_init_seed=memory_init_seed,
            ot_ws_steps=ot_ws_steps,
            ot_step_update=ot_step_update,
            ot_epsilon=ot_epsilon,
            ot_g_lr=ot_g_lr,
            ot_y_lr=ot_y_lr,
            ot_g_steps=ot_g_steps,
            ot_y_steps=ot_y_steps,
            offset=offset,
            window_size=window_size,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    self._step_muon_param(p, state, group)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

        return loss


class SingleDeviceMBO(_MBOBase):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ot_ws_steps=512,
        ot_step_update=64,
        ot_epsilon=0.03,
        ot_g_lr=0.1,
        ot_y_lr=0.01,
        ot_g_steps=5,
        ot_y_steps=5,
        offset=0,
        window_size=0,
    ):
        params = list(params)
        self._validate_main_params(params)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self._set_mbo_hparams(
            num_centroids=num_centroids,
            centroid_dim=centroid_dim,
            lambda_memory=lambda_memory,
            memory_init_seed=memory_init_seed,
            ot_ws_steps=ot_ws_steps,
            ot_step_update=ot_step_update,
            ot_epsilon=ot_epsilon,
            ot_g_lr=ot_g_lr,
            ot_y_lr=ot_y_lr,
            ot_g_steps=ot_g_steps,
            ot_y_steps=ot_y_steps,
            offset=offset,
            window_size=window_size,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                self._step_muon_param(p, state, group)

        return loss


def _make_aux_adam(param_groups):
    aux_groups = []
    for group in param_groups:
        if not group["use_muon"]:
            aux_groups.append({
                "params": group["params"],
                "lr": group.get("lr", 1e-3),
                "betas": group.get("betas", (0.9, 0.999)),
                "eps": group.get("eps", 1e-8),
                "weight_decay": group.get("weight_decay", 0),
            })
    return None if len(aux_groups) == 0 else torch.optim.AdamW(aux_groups)


class MBOWithAuxAdam(_MBOBase):
    def __init__(
        self,
        param_groups,
        *,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ot_ws_steps=512,
        ot_step_update=64,
        ot_epsilon=0.03,
        ot_g_lr=0.1,
        ot_y_lr=0.01,
        ot_g_steps=5,
        ot_y_steps=5,
        offset=0,
        window_size=0,
    ):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                self._validate_main_params(group["params"])
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)
        self._set_mbo_hparams(
            num_centroids=num_centroids,
            centroid_dim=centroid_dim,
            lambda_memory=lambda_memory,
            memory_init_seed=memory_init_seed,
            ot_ws_steps=ot_ws_steps,
            ot_step_update=ot_step_update,
            ot_epsilon=ot_epsilon,
            ot_g_lr=ot_g_lr,
            ot_y_lr=ot_y_lr,
            ot_g_steps=ot_g_steps,
            ot_y_steps=ot_y_steps,
            offset=offset,
            window_size=window_size,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
                for base_i in range(len(params))[::world_size]:
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        state = self.state[p]
                        self._step_muon_param(p, state, group)
                    dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class SingleDeviceMBOWithAuxAdam(_MBOBase):
    def __init__(
        self,
        param_groups,
        *,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ot_ws_steps=512,
        ot_step_update=64,
        ot_epsilon=0.03,
        ot_g_lr=0.1,
        ot_y_lr=0.01,
        ot_g_steps=5,
        ot_y_steps=5,
        offset=0,
        window_size=0,
    ):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                self._validate_main_params(group["params"])
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)
        self._set_mbo_hparams(
            num_centroids=num_centroids,
            centroid_dim=centroid_dim,
            lambda_memory=lambda_memory,
            memory_init_seed=memory_init_seed,
            ot_ws_steps=ot_ws_steps,
            ot_step_update=ot_step_update,
            ot_epsilon=ot_epsilon,
            ot_g_lr=ot_g_lr,
            ot_y_lr=ot_y_lr,
            ot_g_steps=ot_g_steps,
            ot_y_steps=ot_y_steps,
            offset=offset,
            window_size=window_size,
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
                    state = self.state[p]
                    self._step_muon_param(p, state, group)

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class SingleDeviceMBOWindowWithAuxAdam(_MBOBase):
    def __init__(
        self,
        param_groups,
        *,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ot_ws_steps=512,
        ot_step_update=64,
        ot_epsilon=0.03,
        ot_g_lr=0.1,
        ot_y_lr=0.01,
        ot_g_steps=5,
        ot_y_steps=5,
        offset=0,
        window_size=0,
    ):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                self._validate_main_params(group["params"])
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["use_mbo"] = bool(group.get("use_mbo", False))
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon", "use_mbo"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)
        self._set_mbo_hparams(
            num_centroids=num_centroids,
            centroid_dim=centroid_dim,
            lambda_memory=lambda_memory,
            memory_init_seed=memory_init_seed,
            ot_ws_steps=ot_ws_steps,
            ot_step_update=ot_step_update,
            ot_epsilon=ot_epsilon,
            ot_g_lr=ot_g_lr,
            ot_y_lr=ot_y_lr,
            ot_g_steps=ot_g_steps,
            ot_y_steps=ot_y_steps,
            offset=offset,
            window_size=window_size,
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
                    state = self.state[p]
                    if group.get("use_mbo", False):
                        self._step_muon_param(p, state, group)
                    else:
                        self._step_plain_muon_param(p, state, group)

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss
