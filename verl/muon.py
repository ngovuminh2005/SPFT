"""Projection-free, 2D eRank MBO variants for LoRA training on one device."""

import math
import os

import torch

from mbo_metrics import MBOMetricTracker

from muon_ot import (
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
    SingleDeviceMuonWithAuxSGD,
    _ensure_grad,
    _ensure_state_tensor,
    _make_aux_adam,
    _maybe_from_local_like,
    _maybe_to_local_tensor,
)


# MBO objective selection is intentionally local to this module.
# Supported values: "erank", "entropy", and "singular_min".
DEFAULT_MBO_OBJECTIVE = "singular_min"
# Soft-min temperature for the smallest-eigenvalue objective.
DEFAULT_MBO_SINGULAR_MIN_TAU = float(os.environ.get("OPTIM_MBO_SINGULAR_MIN_TAU", "0.01"))


class _LoRAMBOBase(torch.optim.Optimizer):
    """MBO with vectorized LoRA memory and an eRank-derived update direction."""

    memory_state_dtype = torch.bfloat16

    @staticmethod
    def _validate_main_params(params):
        if not params:
            raise ValueError("MBO requires at least one main parameter.")
        for p in params:
            if not isinstance(p, torch.nn.Parameter) or p.ndim not in (2, 4):
                raise ValueError("MBO requires 2D parameters or 4D convolution kernels.")

    @staticmethod
    def _matrix_view(tensor):
        if tensor.ndim == 4:
            return tensor.contiguous().reshape(len(tensor), -1)
        if tensor.ndim == 2:
            return tensor.contiguous()
        raise ValueError(f"MBO only supports 2D/4D parameters, got {tuple(tensor.shape)}")

    @staticmethod
    def _orthogonalize_via_gram(matrix, eps=1e-8):
        """Compute the Muon polar factor from a small Gram eigendecomposition."""
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got {tuple(matrix.shape)}")

        rows, cols = matrix.shape
        transposed = rows > cols
        work = matrix.mT.contiguous() if transposed else matrix
        work_float = work.float()

        gram = work_float @ work_float.mT
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        inv_sqrt = torch.rsqrt(eigenvalues.clamp_min(0.0) + eps)
        gram_inv_sqrt = (eigenvectors * inv_sqrt.unsqueeze(0)) @ eigenvectors.mT
        orthogonal = gram_inv_sqrt @ work_float

        if transposed:
            orthogonal = orthogonal.mT.contiguous()

        dim_scale = max(1, rows / cols) ** 0.5
        return orthogonal.to(dtype=matrix.dtype), dim_scale

    def _orthogonalize_update(self, update_input):
        """Orthogonalize a possibly single-rank DTensor and restore its wrapper."""
        update_local = _maybe_to_local_tensor(update_input)
        update_matrix = self._matrix_view(update_local)
        update_orthogonal, scale = self._orthogonalize_via_gram(update_matrix)
        update_local = update_orthogonal.reshape_as(update_local)
        return _maybe_from_local_like(update_local, update_input), scale

    @staticmethod
    def _orient_lora_matrix(tensor):
        """Use rank as rows: transpose a tall m x r LoRA matrix to r x m."""
        matrix = _LoRAMBOBase._matrix_view(tensor)
        is_transposed = matrix.size(0) > matrix.size(1)
        return (matrix.mT.contiguous() if is_transposed else matrix), is_transposed

    @staticmethod
    def _normalize_vector(vector, eps=1e-8):
        return vector / (vector.float().norm() + eps)

    def _kmeans_refine(self, samples, centers):
        """Run Lloyd refinement on vector warm-start samples."""
        samples_float = samples.float()
        centers_float = centers.float()
        for _ in range(self.kmeans_steps):
            distances = (samples_float[:, None] - centers_float[None]).square().sum(dim=-1)
            labels = distances.argmin(dim=1)
            new_centers = torch.zeros_like(centers_float)
            new_centers.index_add_(0, labels, samples_float)
            counts = torch.bincount(labels, minlength=self.num_centroids).to(dtype=centers_float.dtype)
            nonempty = counts > 0
            new_centers[nonempty] /= counts[nonempty].view(-1, 1)
            new_centers[~nonempty] = centers_float[~nonempty]
            centers_float = new_centers
        return centers_float.to(dtype=centers.dtype)

    @staticmethod
    def _randn(shape, device, dtype, seed):
        if seed is None:
            return torch.randn(*shape, device=device, dtype=dtype)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randn(*shape, generator=generator, device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)

    def _init_centroids(self, memory_dim, device, dtype, param_offset):
        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        return self._randn(
            (self.num_centroids, memory_dim),
            device,
            dtype,
            base_seed + int(param_offset),
        )

    def _kmeanspp_init_centroids(self, samples, param_offset):
        samples = samples.to(dtype=self.memory_state_dtype)
        n = samples.size(0)
        if n == 0:
            return self._init_centroids(samples.size(1), samples.device, samples.dtype, param_offset)

        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(base_seed + int(param_offset) + 17)
        centers = torch.empty(
            (self.num_centroids, samples.size(1)),
            device=samples.device,
            dtype=samples.dtype,
        )
        first = torch.randint(n, (1,), generator=generator, device="cpu").item()
        centers[0].copy_(samples[first])
        closest = (samples.float() - centers[:1].float()).square().sum(dim=-1)
        for index in range(1, self.num_centroids):
            total = closest.sum()
            if not torch.isfinite(total) or total <= 0:
                choice = torch.randint(n, (1,), generator=generator, device="cpu").item()
            else:
                choice = torch.multinomial((closest / total).float().cpu(), 1, generator=generator).item()
            centers[index].copy_(samples[choice])
            distance = (samples.float() - centers[index : index + 1].float()).square().sum(dim=-1)
            closest = torch.minimum(closest, distance)
        return centers

    def _ensure_memory_state(self, param, state, vector):
        memory_dim = vector.numel()
        dtype = self.memory_state_dtype
        _ensure_state_tensor(state, "momentum_buffer", param)
        reinitialize = (
            "memory_centroids" not in state
            or state.get("memory_dim") != memory_dim
            or state.get("memory_dtype") != dtype
        )
        if reinitialize:
            state["memory_mode"] = "centroid"
            state["memory_dim"] = memory_dim
            state["memory_dtype"] = dtype
            state["memory_centroids"] = self._init_centroids(
                memory_dim,
                vector.device,
                dtype,
                id(param),
            )
            state["memory_step"] = 0
            state["memory_m_sum"] = torch.zeros(memory_dim, device=vector.device, dtype=dtype)
            state["memory_ot_g"] = torch.zeros(self.num_centroids, device=vector.device, dtype=torch.float32)
            state["memory_ot_step"] = 1
            state["memory_ot_y_step"] = 1
            state["memory_pending_z"] = []
            state["memory_ws_buffer"] = []
            state["memory_ws_initialized"] = self.ot_ws_steps <= 0
            return
        state["memory_mode"] = "centroid"
        state["memory_centroids"] = _maybe_to_local_tensor(state["memory_centroids"]).to(
            device=vector.device, dtype=dtype
        )
        state["memory_m_sum"] = _maybe_to_local_tensor(state["memory_m_sum"]).to(
            device=vector.device, dtype=dtype
        )
        state["memory_ot_g"] = _maybe_to_local_tensor(state["memory_ot_g"]).to(
            device=vector.device, dtype=torch.float32
        )
        state["memory_pending_z"] = [
            _maybe_to_local_tensor(z).to(device=vector.device, dtype=dtype) for z in state["memory_pending_z"]
        ]
        state["memory_ws_buffer"] = [
            _maybe_to_local_tensor(z).to(device=vector.device, dtype=dtype) for z in state["memory_ws_buffer"]
        ]

    def _ensure_onlygrad_state(self, param, state, vector):
        """Initialize onlygrad's rolling final-update queue without centroid state."""
        memory_dim = vector.numel()
        dtype = self.memory_state_dtype
        _ensure_state_tensor(state, "momentum_buffer", param)
        reinitialize = (
            state.get("memory_mode") != "onlygrad"
            or state.get("memory_dim") != memory_dim
            or state.get("memory_dtype") != dtype
        )
        if reinitialize:
            for key in (
                "memory_centroids",
                "memory_step",
                "memory_m_sum",
                "memory_ot_g",
                "memory_ot_step",
                "memory_ot_y_step",
                "memory_ws_buffer",
            ):
                state.pop(key, None)
            state["memory_mode"] = "onlygrad"
            state["memory_dim"] = memory_dim
            state["memory_dtype"] = dtype
            state["memory_pending_z"] = []
            state["memory_ws_initialized"] = self.ot_ws_steps <= 0
            state["memory_ws_count"] = 0
            return

        state["memory_pending_z"] = [
            _maybe_to_local_tensor(z).to(device=vector.device, dtype=dtype) for z in state["memory_pending_z"]
        ]
        state.setdefault("memory_ws_count", min(self.ot_ws_steps, len(state["memory_pending_z"])))
        state["memory_ws_initialized"] = state["memory_ws_count"] >= self.ot_ws_steps

    def _ensure_projection_state(self, param, state, vector):
        """Initialize projection's rolling normalized pre-split-update queue."""
        memory_dim = vector.numel()
        dtype = self.memory_state_dtype
        _ensure_state_tensor(state, "momentum_buffer", param)
        reinitialize = (
            state.get("memory_mode") != "projection"
            or state.get("memory_dim") != memory_dim
            or state.get("memory_dtype") != dtype
        )
        if reinitialize:
            for key in (
                "memory_centroids",
                "memory_step",
                "memory_m_sum",
                "memory_ot_g",
                "memory_ot_step",
                "memory_ot_y_step",
                "memory_ws_buffer",
            ):
                state.pop(key, None)
            state["memory_mode"] = "projection"
            state["memory_dim"] = memory_dim
            state["memory_dtype"] = dtype
            state["memory_pending_z"] = []
            state["memory_ws_initialized"] = self.ot_ws_steps <= 0
            state["memory_ws_count"] = 0
            return

        state["memory_pending_z"] = [
            _maybe_to_local_tensor(z).to(device=vector.device, dtype=dtype) for z in state["memory_pending_z"]
        ]
        state.setdefault("memory_ws_count", min(self.ot_ws_steps, len(state["memory_pending_z"])))
        state["memory_ws_initialized"] = state["memory_ws_count"] >= self.ot_ws_steps

    def _queue_projection_update(self, sample, state):
        """Append normalized O_t; P_{t-1} therefore excludes the current update."""
        self._queue_onlygrad_final_update(sample, state)

    def _queue_onlygrad_final_update(self, sample, state):
        """Append a normalized final update and keep at most ot_step_update entries."""
        pending = state["memory_pending_z"]
        pending.append(sample.detach().clone())
        if len(pending) > self.ot_step_update:
            del pending[:-self.ot_step_update]
        state["memory_ws_count"] = min(self.ot_ws_steps, state["memory_ws_count"] + 1)
        state["memory_ws_initialized"] = state["memory_ws_count"] >= self.ot_ws_steps

    def _ensure_greedy_state(self, param, state, vector):
        """Initialize greedy's rolling queue of normalized orthogonalized gradients."""
        memory_dim = vector.numel()
        dtype = self.memory_state_dtype
        _ensure_state_tensor(state, "momentum_buffer", param)
        reinitialize = (
            state.get("memory_mode") != "greedy"
            or state.get("memory_dim") != memory_dim
            or state.get("memory_dtype") != dtype
        )
        if reinitialize:
            for key in (
                "memory_centroids",
                "memory_step",
                "memory_m_sum",
                "memory_ot_g",
                "memory_ot_step",
                "memory_ot_y_step",
                "memory_ws_buffer",
            ):
                state.pop(key, None)
            state["memory_mode"] = "greedy"
            state["memory_dim"] = memory_dim
            state["memory_dtype"] = dtype
            state["memory_pending_z"] = []
            state["memory_ws_initialized"] = self.ot_ws_steps <= 0
            state["memory_ws_count"] = 0
            return

        state["memory_pending_z"] = [
            _maybe_to_local_tensor(z).to(device=vector.device, dtype=dtype) for z in state["memory_pending_z"]
        ]
        state.setdefault("memory_ws_count", min(self.ot_ws_steps, len(state["memory_pending_z"])))
        state["memory_ws_initialized"] = state["memory_ws_count"] >= self.ot_ws_steps

    def _queue_greedy_gradient(self, gradient, state):
        """Append a normalized orthogonalized gradient."""
        pending = state["memory_pending_z"]
        pending.append(gradient.detach().clone().to(dtype=self.memory_state_dtype))
        if len(pending) > self.ot_step_update:
            del pending[:-self.ot_step_update]
        state["memory_ws_count"] = min(self.ot_ws_steps, state["memory_ws_count"] + 1)
        state["memory_ws_initialized"] = state["memory_ws_count"] >= self.ot_ws_steps

    def _warmstart(self, sample, state):
        step = state["memory_step"]
        state["memory_centroids"][step].copy_(sample)
        state["memory_m_sum"].add_(sample)
        mean = state["memory_m_sum"] / (step + 1)
        if step + 1 < self.num_centroids:
            state["memory_centroids"][step + 1 :].copy_(mean)
        state["memory_step"] = step + 1

    def _finish_kmeanspp_warmstart(self, state, samples, param_offset, reset_queue=True):
        centroids = self._kmeanspp_init_centroids(samples, param_offset)
        state["memory_centroids"].copy_(self._kmeans_refine(samples, centroids))
        state["memory_step"] = self.num_centroids
        state["memory_ws_initialized"] = True
        state["memory_ws_buffer"] = []
        if reset_queue:
            state["memory_pending_z"] = [sample.detach().clone() for sample in samples[-self.ot_step_update :]]
        state["memory_ot_g"].zero_()
        state["memory_ot_step"] = state["memory_ot_y_step"] = 1

    def _ot_soft_assignment(self, samples, centroids, dual):
        cost = (samples[:, None].float() - centroids[None].float()).square().sum(dim=-1)
        logits = (dual.unsqueeze(0) - cost) / self.ot_epsilon - math.log(float(self.num_centroids))
        return torch.softmax(logits, dim=1)

    def _ot_update_centroids(self, samples, state):
        samples = samples.float()
        centroids = state["memory_centroids"].float()
        dual = state["memory_ot_g"].float()
        uniform = torch.full((self.num_centroids,), 1 / self.num_centroids, device=samples.device)
        batch_size = max(1, samples.size(0))
        for _ in range(self.ot_g_steps):
            assignment = self._ot_soft_assignment(samples, centroids, dual)
            dual += self.ot_g_lr / math.log2(state["memory_ot_step"] + 1) * (uniform - assignment.mean(0))
            dual -= dual.mean()
            state["memory_ot_step"] += 1
        for _ in range(self.ot_y_steps):
            assignment = self._ot_soft_assignment(samples, centroids, dual)
            weights = assignment.sum(0).view(-1, 1)
            weighted_samples = assignment.T @ samples.float()
            gradient = 2 / batch_size * (weights * centroids - weighted_samples)
            centroids -= self.ot_y_lr / math.log2(state["memory_ot_y_step"] + 1) * gradient
            state["memory_ot_y_step"] += 1
        state["memory_ot_g"].copy_(dual)
        state["memory_centroids"].copy_(centroids.to(dtype=self.memory_state_dtype))

    def _update_memory(self, samples, state, param_offset):
        samples = _maybe_to_local_tensor(samples)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        elif samples.ndim != 2:
            samples = samples.reshape(samples.shape[0], -1)
        samples = samples.to(device=state["memory_centroids"].device, dtype=self.memory_state_dtype)
        if self.ot_ws_steps > 0 and not state["memory_ws_initialized"]:
            state["memory_ws_buffer"].extend(sample.detach().clone() for sample in samples)
            if len(state["memory_ws_buffer"]) < self.ot_ws_steps:
                return
            self._finish_kmeanspp_warmstart(state, torch.stack(state["memory_ws_buffer"]), param_offset)
            return
        if state["memory_step"] < self.num_centroids:
            count = min(self.num_centroids - state["memory_step"], samples.size(0))
            for sample in samples[:count]:
                self._warmstart(sample, state)
            if state["memory_step"] >= self.num_centroids:
                state["memory_ws_initialized"] = True
                state["memory_ot_g"].zero_()
                state["memory_ot_step"] = state["memory_ot_y_step"] = 1
            if count == samples.size(0):
                return
            samples = samples[count:]
        if state["memory_ws_initialized"]:
            self._ot_update_centroids(samples, state)

    def _queue_and_update_memory(self, sample, state, param_offset):
        pending = state["memory_pending_z"]
        pending.append(sample.detach().clone())
        if len(pending) > self.ot_step_update:
            del pending[:-self.ot_step_update]
        if not state["memory_ws_initialized"]:
            self._update_memory(pending[-1], state, param_offset)
        else:
            self._update_memory(torch.stack(pending), state, param_offset)

    def _mbo_param_groups(self):
        for group in self.param_groups:
            if not group.get("use_muon", True):
                continue
            if group.get("use_mbo", True):
                yield group

    def install_microbatch_hooks(self):
        if getattr(self, "_microbatch_hook_handles", None):
            return
        self._microbatch_hook_handles = []
        for group in self._mbo_param_groups():
            for param in group["params"]:
                if not param.requires_grad:
                    continue

                def capture_hook(gradient, param=param):
                    if getattr(self, "_microbatch_capture_enabled", False):
                        self._capture_microbatch_gradient(param, gradient)
                    return gradient

                self._microbatch_hook_handles.append(param.register_hook(capture_hook))

    def begin_microbatch_capture(self, gradient_scale):
        self.install_microbatch_hooks()
        self._microbatch_capture_enabled = True
        self._microbatch_gradient_scale = float(gradient_scale)
        for group in self._mbo_param_groups():
            for param in group["params"]:
                self.state[param].pop("_microbatch_memory_snapshot", None)

    def _capture_microbatch_gradient(self, param, gradient):
        state = self.state[param]
        if "_microbatch_memory_snapshot" not in state:
            state["_microbatch_memory_snapshot"] = (
                "memory_mode" in state,
                list(state.get("memory_pending_z", [])),
                list(state.get("memory_ws_buffer", [])),
            )

        raw_gradient = gradient.detach().clone()
        if self._microbatch_gradient_scale != 1.0:
            raw_gradient.mul_(self._microbatch_gradient_scale)
        gradient_local = _maybe_to_local_tensor(raw_gradient)
        oriented_gradient, _ = self._orient_lora_matrix(gradient_local)
        sample = self._normalize_vector(oriented_gradient.reshape(-1)).to(dtype=self.memory_state_dtype)
        self._ensure_memory_state(param, state, sample)

        pending = state["memory_pending_z"]
        pending.append(sample.detach().clone())
        if len(pending) > self.ot_step_update:
            del pending[:-self.ot_step_update]
        if not state["memory_ws_initialized"]:
            state["memory_ws_buffer"].append(sample.detach().clone())

    @torch.no_grad()
    def flush_microbatch_memory(self):
        """Finalize minibatch-derived memory after the global optimizer step."""
        for group in self._mbo_param_groups():
            for param in group["params"]:
                state = self.state[param]
                if "_microbatch_memory_snapshot" not in state:
                    continue
                just_initialized = False
                if not state["memory_ws_initialized"] and len(state["memory_ws_buffer"]) >= self.ot_ws_steps:
                    warm_samples = torch.stack(state["memory_ws_buffer"][: self.ot_ws_steps])
                    self._finish_kmeanspp_warmstart(
                        state,
                        warm_samples,
                        id(param),
                        reset_queue=False,
                    )
                    just_initialized = True
                if state["memory_ws_initialized"] and not just_initialized and state["memory_pending_z"]:
                    self._ot_update_centroids(torch.stack(state["memory_pending_z"]), state)
                state.pop("_microbatch_memory_snapshot", None)
        self._microbatch_capture_enabled = False

    @torch.no_grad()
    def discard_microbatch_memory(self):
        for group in self._mbo_param_groups():
            for param in group["params"]:
                state = self.state[param]
                snapshot = state.pop("_microbatch_memory_snapshot", None)
                if snapshot is None:
                    continue
                had_memory, pending, ws_buffer = snapshot
                if not had_memory:
                    for key in (
                        "memory_dim",
                        "memory_dtype",
                        "memory_centroids",
                        "memory_step",
                        "memory_m_sum",
                        "memory_ot_g",
                        "memory_ot_step",
                        "memory_ot_y_step",
                        "memory_pending_z",
                        "memory_ws_buffer",
                        "memory_ws_initialized",
                    ):
                        state.pop(key, None)
                else:
                    state["memory_pending_z"] = pending
                    state["memory_ws_buffer"] = ws_buffer
        self._microbatch_capture_enabled = False

    @staticmethod
    def _make_spectral_rows(matrix, history, spectral_matrix=None):
        current = matrix.float().reshape(-1)
        spectral_current = current if spectral_matrix is None else spectral_matrix.float().reshape(-1)
        if history is None or history.numel() == 0:
            return spectral_current.unsqueeze(0), current
        history = history.float().reshape(-1, current.numel())
        return torch.cat([history, spectral_current.unsqueeze(0)], dim=0), current

    def _erank_direction_from_history(self, matrix, history, out_dtype, spectral_matrix=None):
        """Apply the eRank correction using history rows plus the current vector."""
        P, vector = self._make_spectral_rows(matrix, history, spectral_matrix=spectral_matrix)

        gram = P @ P.mT
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min_(0)

        inv_sqrt = torch.rsqrt(eigenvalues + 1e-8)
        polar_left = (eigenvectors[-1] * inv_sqrt) @ eigenvectors.mT
        polar_vector = polar_left @ P

        frobenius_norm = torch.sqrt(eigenvalues.sum())
        nuclear_norm = torch.sqrt(eigenvalues).sum()
        erank = nuclear_norm / (frobenius_norm + 1e-8)
        alpha = 1.0 - self.lambda_memory * erank / (frobenius_norm + 1e-8)
        beta = self.lambda_memory / (frobenius_norm + 1e-8)
        direction = alpha * vector + beta * polar_vector
        self._record_mbo_correction_gradient_cosine(direction - vector, vector)
        return direction.reshape_as(matrix).to(dtype=out_dtype)

    def _erank_direction(self, matrix, state, ns_steps, out_dtype):
        """Apply the vector eRank correction using centroid memory."""
        del ns_steps  # The final Muon orthogonalization still owns ns_steps.
        centroids = state["memory_centroids"].float()
        return self._erank_direction_from_history(matrix, centroids, out_dtype)

    def _entropy_direction_from_history(
        self,
        matrix,
        history,
        out_dtype,
        apply_correction=True,
        spectral_matrix=None,
    ):
        """Apply entropy MBO using arbitrary history rows plus the current vector."""
        P, vector = self._make_spectral_rows(matrix, history, spectral_matrix=spectral_matrix)

        gram = P @ P.mT
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min_(0)
        frobenius_sq = eigenvalues.sum()

        probabilities = eigenvalues / (frobenius_sq + 1e-8)
        log_probabilities = torch.log(probabilities.clamp_min(1e-8))
        entropy = -(probabilities * log_probabilities).sum()
        self._record_mbo_vendi_score(entropy.exp())

        # U_G @ \hat{Sigma} @ V^T without materializing V^T:
        # \hat{sigma}_i / sigma_i = -log(p_i) - H(p).
        spectral_weight = -log_probabilities - entropy
        spectral_weight = torch.where(
            eigenvalues > 1e-8,
            spectral_weight,
            torch.zeros_like(spectral_weight),
        )
        correction = (eigenvectors[-1] * spectral_weight) @ eigenvectors.mT @ P

        if not apply_correction:
            return vector.reshape_as(matrix).to(dtype=out_dtype)

        direction = vector + self.lambda_memory / (frobenius_sq + 1e-8) * correction
        self._record_mbo_correction_gradient_cosine(correction, vector)
        return direction.reshape_as(matrix).to(dtype=out_dtype)

    def _entropy_direction(self, matrix, state, out_dtype):
        """Apply the Shannon-entropy spectral correction to centroid memory."""
        centroids = state["memory_centroids"].float()
        return self._entropy_direction_from_history(matrix, centroids, out_dtype)

    def _singular_min_direction_from_history(
        self,
        matrix,
        history,
        out_dtype,
        apply_correction=True,
        spectral_matrix=None,
    ):
        """Increase the soft minimum nonzero eigenvalue of K = P P^T / n."""
        P, vector = self._make_spectral_rows(matrix, history, spectral_matrix=spectral_matrix)
        n_rows = P.size(0)

        gram = P @ P.mT
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min_(0)

        # Log the same spectrum-diversity metric as entropy mode.  It is
        # observational only: singular_min still uses its soft-min objective.
        probabilities = eigenvalues / (eigenvalues.sum() + 1e-8)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum()
        self._record_mbo_vendi_score(entropy.exp())

        # Compact SVD: retain only the q = rank(P) positive singular values.
        active = eigenvalues > 1e-8
        if not apply_correction or not torch.any(active):
            return vector.reshape_as(matrix).to(dtype=out_dtype)
        singular_values_sq = eigenvalues[active]
        left_vectors = eigenvectors[:, active]

        # w_i = softmax(-sigma_i^2 / (n tau)); lambda_i(K) = sigma_i^2 / n.
        weights = torch.softmax(
            -singular_values_sq / (n_rows * DEFAULT_MBO_SINGULAR_MIN_TAU), dim=0
        )
        correction = (left_vectors[-1] * weights) @ left_vectors.mT @ P
        direction = vector + self.lambda_memory / n_rows * correction
        self._record_mbo_correction_gradient_cosine(correction, vector)
        return direction.reshape_as(matrix).to(dtype=out_dtype)

    def _projection_direction_from_history(self, matrix, history, out_dtype):
        """Return the component of ``matrix`` in the normalized-history span."""
        vector = matrix.float().reshape(-1)
        if history is None or (not isinstance(history, torch.Tensor) and not history):
            return matrix.to(dtype=out_dtype)
        P = history.float() if isinstance(history, torch.Tensor) else torch.stack(history).float()
        P = P.reshape(-1, vector.numel())
        gram = P @ P.mT
        projection = P.mT @ (torch.linalg.pinv(gram, hermitian=True) @ (P @ vector))
        if not torch.isfinite(projection).all() or projection.norm() <= 1e-8:
            return matrix.to(dtype=out_dtype)
        self._record_mbo_correction_gradient_cosine(projection - vector, vector)
        return projection.reshape_as(matrix).to(dtype=out_dtype)

    def _singular_min_direction(self, matrix, state, out_dtype):
        """Apply the soft smallest-eigenvalue correction to centroid memory."""
        centroids = state["memory_centroids"].float()
        return self._singular_min_direction_from_history(matrix, centroids, out_dtype)

    def _begin_mbo_metrics(self):
        self._mbo_metric_tracker = MBOMetricTracker()

    def _record_mbo_vendi_score(self, vendi_score):
        if hasattr(self, "_mbo_metric_tracker"):
            self._mbo_metric_tracker.record_vendi_score(vendi_score)

    def _record_mbo_correction_gradient_cosine(self, correction, gradient):
        if hasattr(self, "_mbo_metric_tracker"):
            self._mbo_metric_tracker.record_correction_gradient_cosine(correction, gradient)

    def _finish_mbo_metrics(self):
        return None

    def clear_mbo_metrics(self):
        self._mbo_metric_tracker = MBOMetricTracker()

    def get_mbo_metrics(self):
        if not hasattr(self, "_mbo_metric_tracker"):
            return {}
        return self._mbo_metric_tracker.get_metrics()

    def _objective_direction(
        self,
        matrix,
        state,
        group,
        out_dtype,
        apply_correction=True,
        spectral_matrix=None,
    ):
        if self.gradient_source in {"onlygrad", "greedy"}:
            history = torch.stack(state["memory_pending_z"]) if state["memory_pending_z"] else None
            if self.objective == "erank":
                if not apply_correction:
                    return matrix.to(dtype=out_dtype)
                return self._erank_direction_from_history(
                    matrix,
                    history,
                    out_dtype,
                    spectral_matrix=spectral_matrix,
                )
            if self.objective == "singular_min":
                return self._singular_min_direction_from_history(
                    matrix,
                    history,
                    out_dtype,
                    apply_correction=apply_correction,
                    spectral_matrix=spectral_matrix,
                )
            return self._entropy_direction_from_history(
                matrix,
                history,
                out_dtype,
                apply_correction=apply_correction,
                spectral_matrix=spectral_matrix,
            )
        if self.objective == "erank":
            return self._erank_direction(matrix, state, group["ns_steps"], out_dtype)
        if self.objective == "singular_min":
            return self._singular_min_direction(matrix, state, out_dtype)
        return self._entropy_direction(matrix, state, out_dtype)

    def _step_mbo_param(self, param, state, group):
        gradient = _ensure_grad(param)
        gradient_local = _maybe_to_local_tensor(gradient)
        momentum = _ensure_state_tensor(state, "momentum_buffer", param)
        momentum.lerp_(gradient, 1 - group["momentum"])
        update_pre_ns = torch.lerp(gradient, momentum, group["momentum"]) if group["nesterov"] else momentum
        update_local = _maybe_to_local_tensor(update_pre_ns)
        oriented_update, was_transposed = self._orient_lora_matrix(update_local)
        # Keep the normalized sample used by centroid/onlygrad memory initialization.
        sample = self._normalize_vector(oriented_update.reshape(-1)).to(dtype=self.memory_state_dtype)
        greedy_history_sample = None
        projection_history_sample = None
        if self.gradient_source == "projection":
            # First construct O_t with the normal Muon Polar update.  P_{t-1}
            # contains normalized prior O's, never the current O_t.
            full_update, scale = self._orthogonalize_update(update_pre_ns)
            full_update_local = _maybe_to_local_tensor(full_update)
            oriented_full_update, was_full_transposed = self._orient_lora_matrix(full_update_local)
            projection_history_sample = self._normalize_vector(
                oriented_full_update.reshape(-1)
            ).to(dtype=self.memory_state_dtype)
            self._ensure_projection_state(param, state, projection_history_sample)
            if state["memory_ws_initialized"]:
                main_local = self._projection_direction_from_history(
                    oriented_full_update, state["memory_pending_z"], gradient_local.dtype
                )
                aux_local = oriented_full_update - main_local
                direction_local = main_local + self.lambda_memory * aux_local
            else:
                direction_local = oriented_full_update
            if was_full_transposed:
                direction_local = direction_local.mT.contiguous()
            update_input = _maybe_from_local_like(direction_local.view_as(gradient_local), gradient)
            update = update_input
        elif self.gradient_source == "onlygrad":
            self._ensure_onlygrad_state(param, state, sample)
            direction_local = self._objective_direction(
                oriented_update,
                state,
                group,
                gradient_local.dtype,
                apply_correction=state["memory_ws_initialized"],
            )
            if was_transposed:
                direction_local = direction_local.mT.contiguous()
            update_input = _maybe_from_local_like(direction_local.view_as(gradient_local), gradient)
            update, scale = self._orthogonalize_update(update_input)
        elif self.gradient_source == "greedy":
            original_matrix = self._matrix_view(update_local)
            scale = max(1, original_matrix.size(0) / original_matrix.size(1)) ** 0.5
            orthogonalized_update, _ = self._orthogonalize_update(oriented_update)
            orthogonalized_update = _maybe_to_local_tensor(orthogonalized_update)
            normalized_orthogonalized_update = self._normalize_vector(
                orthogonalized_update.reshape(-1)
            ).reshape_as(orthogonalized_update)
            greedy_history_sample = normalized_orthogonalized_update.reshape(-1).to(
                dtype=self.memory_state_dtype
            )
            self._ensure_greedy_state(param, state, greedy_history_sample)
            direction_local = self._objective_direction(
                orthogonalized_update,
                state,
                group,
                gradient_local.dtype,
                apply_correction=state["memory_ws_initialized"],
                spectral_matrix=normalized_orthogonalized_update,
            )
            if was_transposed:
                direction_local = direction_local.mT.contiguous()
            update = _maybe_from_local_like(direction_local.view_as(gradient_local), gradient)
        else:
            self._ensure_memory_state(param, state, sample)
            if state["memory_ws_initialized"]:
                direction_local = self._objective_direction(
                    oriented_update,
                    state,
                    group,
                    gradient_local.dtype,
                )
                if was_transposed:
                    direction_local = direction_local.mT.contiguous()
                update_input = _maybe_from_local_like(direction_local.view_as(gradient_local), gradient)
            else:
                update_input = update_pre_ns
            update, scale = self._orthogonalize_update(update_input)
        param.mul_(1 - group["lr"] * group["weight_decay"])
        param.add_(update.reshape_as(param), alpha=-group["lr"] * scale)
        if not getattr(self, "_microbatch_capture_enabled", False):
            if self.gradient_source == "projection":
                self._queue_projection_update(projection_history_sample, state)
            elif self.gradient_source == "onlygrad":
                final_update, _ = self._orient_lora_matrix(_maybe_to_local_tensor(update))
                final_sample = self._normalize_vector(final_update.reshape(-1)).to(dtype=self.memory_state_dtype)
                self._queue_onlygrad_final_update(final_sample, state)
            elif self.gradient_source == "greedy":
                self._queue_greedy_gradient(greedy_history_sample, state)
            else:
                # Global mode stores the direction actually applied to the model,
                # after Nesterov, eRank/MBO, and Muon orthogonalization.
                final_update, _ = self._orient_lora_matrix(_maybe_to_local_tensor(update))
                final_sample = self._normalize_vector(final_update.reshape(-1)).to(dtype=self.memory_state_dtype)
                self._queue_and_update_memory(final_sample, state, id(param))

    def _step_plain_muon_param(self, param, state, group):
        gradient = _ensure_grad(param)
        momentum = _ensure_state_tensor(state, "momentum_buffer", param)
        momentum.lerp_(gradient, 1 - group["momentum"])
        update_input = torch.lerp(gradient, momentum, group["momentum"]) if group["nesterov"] else momentum
        update, scale = self._orthogonalize_update(update_input)
        param.mul_(1 - group["lr"] * group["weight_decay"])
        param.add_(update.reshape_as(param), alpha=-group["lr"] * scale)

    def _set_mbo_hparams(self, **kwargs):
        gradient_source = kwargs.pop("gradient_source", "global")
        self.set_mbo_gradient_source(gradient_source)

        if self.gradient_source == "projection":
            # Projection has no spectral MBO objective or centroid/OT memory.
            # lambda_memory is alpha, the residual-direction attenuation.
            self.objective = None
            kwargs.pop("objective", None)
        else:
            objective = kwargs.pop("objective", None)
            if objective is None:
                objective = os.environ.get("OPTIM_MBO_OBJECTIVE", DEFAULT_MBO_OBJECTIVE)
            objective = str(objective).strip().lower()
            if objective not in {"erank", "entropy", "singular_min"}:
                raise ValueError("objective must be 'erank', 'entropy', or 'singular_min'.")
            self.objective = objective

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.lambda_memory = float(self.lambda_memory)
        self.ot_ws_steps = int(self.ot_ws_steps)
        self.ot_step_update = int(self.ot_step_update)
        if self.ot_ws_steps < 0 or self.ot_step_update < 1:
            raise ValueError("Invalid MBO memory configuration.")
        if self.gradient_source == "projection":
            if not 0.0 <= self.lambda_memory <= 1.0:
                raise ValueError("projection requires lambda_memory (alpha) in [0, 1].")
        else:
            self.num_centroids = int(self.num_centroids)
            self.centroid_dim = int(self.centroid_dim)  # Kept for CLI compatibility; no projection uses it.
            self.ot_epsilon = float(self.ot_epsilon)
            self.ot_g_lr = float(self.ot_g_lr)
            self.ot_y_lr = float(self.ot_y_lr)
            self.ot_g_steps = int(self.ot_g_steps)
            self.ot_y_steps = int(self.ot_y_steps)
            self.kmeans_steps = int(kwargs.get("kmeans_steps", 10))
            if self.num_centroids < 1:
                raise ValueError("Invalid MBO memory configuration.")
            if self.ot_epsilon <= 0 or self.ot_g_lr <= 0 or self.ot_y_lr <= 0 or self.ot_g_steps < 1 or self.ot_y_steps < 1:
                raise ValueError("Invalid MBO OT configuration.")
            if self.kmeans_steps < 0:
                raise ValueError("kmeans_steps must be >= 0")

    def set_mbo_gradient_source(self, source_name):
        gradient_source = str(source_name).strip().lower()
        if gradient_source in {"microbatch"}:
            gradient_source = "minibatch"
        elif gradient_source in {"global_batch", "batch"}:
            gradient_source = "global"
        elif gradient_source in {"only_grad", "gradient_queue"}:
            gradient_source = "onlygrad"
        elif gradient_source in {"greedy_grad", "greedy_gradient"}:
            gradient_source = "greedy"
        elif gradient_source in {"project", "history_projection"}:
            gradient_source = "projection"
        if gradient_source not in {"minibatch", "projection", "global", "onlygrad", "greedy"}:
            raise ValueError("gradient_source must be 'minibatch', 'projection', 'global', 'onlygrad', or 'greedy'.")
        self.gradient_source = gradient_source


def _set_group_defaults(param_groups, windowed=False):
    for group in param_groups:
        if group["use_muon"]:
            _LoRAMBOBase._validate_main_params(group["params"])
            group.setdefault("lr", 0.02)
            group.setdefault("momentum", 0.95)
            group.setdefault("weight_decay", 0.0)
            group.setdefault("nesterov", True)
            group.setdefault("ns_steps", 5)
            if windowed:
                group["use_mbo"] = bool(group.get("use_mbo", False))
        else:
            group.setdefault("lr", 1e-3)
            group.setdefault("betas", (0.9, 0.999))
            group.setdefault("eps", 1e-8)
            group.setdefault("weight_decay", 0.0)


class _SingleDeviceMBOWithAuxBase(_LoRAMBOBase):
    def __init__(self, param_groups, *, windowed=False, **mbo_kwargs):
        _set_group_defaults(param_groups, windowed=windowed)
        torch.optim.Optimizer.__init__(self, param_groups, {})
        self.aux_adam = _make_aux_adam(self.param_groups)
        self._windowed = windowed
        self._set_mbo_hparams(**mbo_kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        self._begin_mbo_metrics()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for param in group["params"]:
                if self._windowed and not group["use_mbo"]:
                    self._step_plain_muon_param(param, self.state[param], group)
                else:
                    self._step_mbo_param(param, self.state[param], group)
        if self.aux_adam is not None:
            self.aux_adam.step()
        self._finish_mbo_metrics()
        return loss


class SingleDeviceMBOWithAuxAdam(_SingleDeviceMBOWithAuxBase):
    def __init__(self, param_groups, **kwargs):
        super().__init__(param_groups, windowed=False, **kwargs)


class SingleDeviceMBOWindowWithAuxAdam(_SingleDeviceMBOWithAuxBase):
    def __init__(self, param_groups, **kwargs):
        super().__init__(param_groups, windowed=True, **kwargs)


class SingleDeviceMBO(_LoRAMBOBase):
    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5, **mbo_kwargs):
        params = list(params)
        self._validate_main_params(params)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))
        self._set_mbo_hparams(**mbo_kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        self._begin_mbo_metrics()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                self._step_mbo_param(param, self.state[param], group)
        self._finish_mbo_metrics()
        return loss


# This project uses the single-device trainer.  Keep the public class names so
# the existing optimizer selector remains compatible with OPTIM_MODULE=muon_lora.
MBO = SingleDeviceMBO
MBOWithAuxAdam = SingleDeviceMBOWithAuxAdam
