# MBO Logic Backup

This file records the MBO changes made for minibatch-gradient queueing. It is
intended as a reconstruction guide for returning to the previous behavior.

## New behavior

- `verl/verl/trainer/fsdp_dft_trainer.py` installs MBO gradient hooks after the
  optimizer is built.
- During one global batch, each micro-batch backward sends its raw gradient to
  the MBO queue through the hook. The hook multiplies by the number of
  micro-batches because the trainer calls backward with a `1 / n_micro_batches`
  scale.
- The raw gradient is still oriented and Frobenius-normalized before entering
  the queue. It does not use momentum or Nesterov for the queue sample.
- `optim.mbo.gradient_source=minibatch` selects this behavior. The separate
  `MinibatchGradientSource` is implemented in
  `verl/verl/trainer/mbo_gradient_sources.py`.
- `param.grad` continues to accumulate normally. Gradient clipping and the
  model optimizer update still happen once per global batch.
- The queue is flushed once after a successful global optimizer step. The OT
  update therefore runs once per global batch, using the last
  `ot_step_update` minibatch samples.
- If gradient norm is non-finite, the captured queue changes are discarded and
  the previous queue is restored.
- After `ot_ws_steps` queue samples are collected, MBO initializes centroids
  with K-means++, then runs `kmeans_steps` Lloyd iterations. The default is 10.
- `kmeans_steps` is configurable through `optim.mbo.kmeans_steps` and
  `OPTIM_MBO_KMEANS_STEPS`.

## Previous behavior

- No minibatch hooks were installed.
- Each micro-batch backward only contributed to the accumulated `param.grad`.
- `optimizer.step()` computed one momentum/Nesterov gradient from the global
  batch gradient.
- That one normalized Nesterov sample was appended to the MBO queue from
  `_step_mbo_param()`.
- OT was updated immediately during each optimizer step.
- Warm-start used K-means++ initialization without Lloyd refinement.
- `optim.mbo.gradient_source=global` selects this behavior through the
  separate `GlobalBatchGradientSource` module. It leaves the optimizer's
  global-step queue path active and does not install gradient hooks.
- In global mode, the queue sample is now the final normalized update direction
  actually applied to the parameter: after global-batch Nesterov, eRank/MBO,
  and Muon orthogonalization. The learning-rate and weight-decay magnitudes
  are not stored because the queue stores directions.

## Restoring the previous behavior

1. In `verl/verl/trainer/fsdp_dft_trainer.py`, remove the
   `install_microbatch_hooks()` call after `_build_optimizer()`.
2. Remove `begin_microbatch_capture()` before the micro-batch loop.
3. Remove the `discard_microbatch_memory()` call in the non-finite-gradient
   branch.
4. Remove the `flush_microbatch_memory()` call after `optimizer.step()`.
5. In `verl/muon.py`, restore the final line of `_step_mbo_param()` to:

   ```python
   self._queue_and_update_memory(sample, state, id(param))
   ```

   This restores the previous global-mode behavior where the queue receives
   the normalized Nesterov sample before eRank/Muon orthogonalization.

6. Remove the micro-batch hook, queue transaction, and flush methods from
   `_LoRAMBOBase`:
   `install_microbatch_hooks`, `begin_microbatch_capture`,
   `_capture_microbatch_gradient`, `flush_microbatch_memory`, and
   `discard_microbatch_memory`.
7. To restore the old warm-start exactly, remove `_kmeans_refine()` from
   `_finish_kmeanspp_warmstart()` and copy K-means++ centroids directly:

   ```python
   state["memory_centroids"].copy_(self._kmeanspp_init_centroids(samples, param_offset))
   ```

8. Remove `kmeans_steps` from the trainer MBO kwargs, the YAML config, and
   `verl/train_dft_1gpu.sh`, or leave it unused with the default value.

The practical switch between the two current implementations is:

```yaml
optim:
  mbo:
    gradient_source: minibatch  # or global
```

Use `gradient_source: global` to reproduce the old gradient granularity while
keeping the new ten-step K-means refinement. To reproduce the old warm-start
as well, set `optim.mbo.kmeans_steps: 0`.

The ordinary backward accumulation and global-batch model update are shared by
both versions. The main behavioral difference is the granularity and timing
of samples entering the MBO queue.
