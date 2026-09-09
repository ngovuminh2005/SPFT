"""Gradient sources used to feed MBO memory.

The optimizer update remains global-batch based. These classes only select
which gradient samples are sent to the MBO memory queue.
"""


class _BaseGradientSource:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def attach(self):
        return None

    def begin_step(self, num_micro_batches):
        return None

    def commit_step(self):
        return None

    def discard_step(self):
        return None


class GlobalBatchGradientSource(_BaseGradientSource):
    """Use the optimizer's existing one-sample-per-global-step path."""


class OnlyGradGradientSource(_BaseGradientSource):
    """Use a rolling queue of normalized final update directions, without centroids or OT."""


class GreedyGradientSource(_BaseGradientSource):
    """Use a rolling queue of orthogonalized gradients, without a second orthogonalization after MBO."""


class MinibatchGradientSource(_BaseGradientSource):
    """Capture one raw gradient sample after every micro-batch backward."""

    def attach(self):
        self.optimizer.install_microbatch_hooks()

    def begin_step(self, num_micro_batches):
        self.optimizer.begin_microbatch_capture(gradient_scale=num_micro_batches)

    def commit_step(self):
        self.optimizer.flush_microbatch_memory()

    def discard_step(self):
        self.optimizer.discard_microbatch_memory()


class ProjectionGradientSource(_BaseGradientSource):
    """Use a rolling queue of normalized pre-split Muon updates."""


class NoopGradientSource(_BaseGradientSource):
    """Used for optimizers that do not implement MBO memory."""


def build_mbo_gradient_source(optimizer, source_name):
    if not hasattr(optimizer, "begin_microbatch_capture"):
        return NoopGradientSource(optimizer)

    normalized_name = str(source_name).lower()
    if hasattr(optimizer, "set_mbo_gradient_source"):
        optimizer.set_mbo_gradient_source(normalized_name)
    if normalized_name in {"minibatch", "microbatch"}:
        return MinibatchGradientSource(optimizer)
    if normalized_name in {"projection", "project", "history_projection"}:
        return ProjectionGradientSource(optimizer)
    if normalized_name in {"global", "global_batch", "batch"}:
        return GlobalBatchGradientSource(optimizer)
    if normalized_name in {"onlygrad", "only_grad", "gradient_queue"}:
        return OnlyGradGradientSource(optimizer)
    if normalized_name in {"greedy", "greedy_grad", "greedy_gradient"}:
        return GreedyGradientSource(optimizer)
    raise ValueError(
        f"Unknown optim.mbo.gradient_source={source_name!r}; "
        "expected 'minibatch', 'projection', 'global', 'onlygrad', or 'greedy'."
    )
