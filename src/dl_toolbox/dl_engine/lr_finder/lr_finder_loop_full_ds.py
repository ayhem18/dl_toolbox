"""Subset-mode range test: several loader passes per LR, weights reset between LRs.

Smith's finder takes one optimizer step per candidate LR. That estimates
instantaneous gradient sensitivity, which is noisy when a batch is not
representative of the task (small batch, high-resolution images, high
intra-class variance).

This loop asks a different question: if the model trains for
``num_iterations_per_lr`` full passes over the given loader at a fixed LR,
what is the resulting mean loss on that same loader? Each candidate starts
from the same snapshot, so the curve is empirical training progress, not a
walk along a changing weight trajectory. A diverging LR is recorded as
non-finite and does not abort later candidates.

smooth_f and diverge_th on the spec are unused.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.lr_finder.batches import prepare_on_device, set_optimizer_lr
from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError
from dl_toolbox.dl_engine.lr_finder.loop import RangeTestInput
from dl_toolbox.dl_engine.lr_finder.restore import restore_trainable, snapshot_trainable
from dl_toolbox.dl_engine.lr_finder.runtime import resolve_device
from dl_toolbox.dl_engine.lr_finder.spec import (
    log_spaced_lrs,
    validate_lr_finder_spec,
    validate_num_iterations_per_lr,
)
from dl_toolbox.dl_engine.lr_finder.step import LrFinderStep
from dl_toolbox.dl_engine_utils.precision import AmpSession
from dl_toolbox.dl_engine_utils.progress import range_progress


def run_full_ds_range_test(
    inputs: RangeTestInput,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    spec = inputs.config.spec
    validate_lr_finder_spec(spec)
    n_passes = inputs.config.num_iterations_per_lr
    validate_num_iterations_per_lr(n_passes)
    device = resolve_device(inputs.config.runtime)
    model = inputs.model
    optimizer = inputs.optimizer
    model.to(device)
    snapshot = snapshot_trainable(model, optimizer)
    candidate_lrs = log_spaced_lrs(spec)
    num_iter = spec.num_iter
    lrs: list[float] = [0.0] * num_iter
    losses: list[float] = [0.0] * num_iter
    steps = range_progress(
        num_iter,
        enabled=inputs.config.runtime.enable_progress_bar,
        desc="full_ds_lr_finder",
        unit="lr",
    )
    try:
        for i in steps:
            lr = candidate_lrs[i]
            restore_trainable(model, optimizer, snapshot, device)
            set_optimizer_lr(optimizer, lr)
            amp = AmpSession.from_precision(
                inputs.config.runtime.precision, device
            )
            train_ok = True
            for _epoch in range(n_passes):
                train_ok = _train_one_pass(
                    model=model,
                    optimizer=optimizer,
                    train_loader=inputs.train_loader,
                    step=inputs.step,
                    amp=amp,
                    device=device,
                )
                if not train_ok:
                    break
            if train_ok:
                eval_loss = _eval_mean_loss(
                    model=model,
                    train_loader=inputs.train_loader,
                    step=inputs.step,
                    amp=amp,
                    device=device,
                )
            else:
                eval_loss = float("inf")
            lrs[i] = lr
            losses[i] = eval_loss
            steps.set_postfix(lr=f"{lr:.2e}", loss=_loss_postfix(eval_loss))
    finally:
        steps.close()
        restore_trainable(model, optimizer, snapshot, device)
    return tuple(lrs), tuple(losses)


def _train_one_pass(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    step: LrFinderStep,
    amp: AmpSession,
    device: torch.device,
) -> bool:
    """One epoch at the current LR. False if any batch loss is non-finite."""
    model.train()
    n_batches = 0
    for raw_batch in train_loader:
        prepared = prepare_on_device(step, raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _forward_loss(model, step, prepared, amp)
        raw = float(loss.detach().cpu())
        if not math.isfinite(raw):
            return False
        amp.optimizer_step(loss, optimizer)
        n_batches += 1
    if n_batches == 0:
        raise LrFinderError("train loader is empty")
    return True


def _eval_mean_loss(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    step: LrFinderStep,
    amp: AmpSession,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for raw_batch in train_loader:
            prepared = prepare_on_device(step, raw_batch, device)
            loss = _forward_loss(model, step, prepared, amp)
            raw = float(loss.detach().cpu())
            if not math.isfinite(raw):
                return float("inf")
            total += raw
            n_batches += 1
    if n_batches == 0:
        raise LrFinderError("train loader is empty")
    return total / float(n_batches)


def _forward_loss(
    model: nn.Module,
    step: LrFinderStep,
    prepared: dict,
    amp: AmpSession,
) -> Tensor:
    with amp.autocast():
        output = step.forward_fn(model, prepared)
        return step.loss_fn(output, prepared)


def _loss_postfix(loss: float) -> str:
    if math.isfinite(loss):
        return f"{loss:.4f}"
    return "inf"
