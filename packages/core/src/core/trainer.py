"""
Pure-function Trainer for `EncoderDecoderModel`
====================================================================

Training is expressed as a single FUNCTION, not a class. A class
would tempt us to bake domain knowledge (TS tokenizer, validators,
checkpoint paths) into trainer state — instead we keep the trainer
totally domain-agnostic and let the caller wire in:

    * a factory that produces `(src, tgt_in, tgt_target)` tensor batches
    * a single int `pad_id` (the only token-level fact the trainer needs)
    * an optional `eval_fn(model) -> float` callback for metrics
    * an optional `on_epoch_end(stats)` callback for logging / checkpoints

Everything else (dataset construction, learning-rate schedules, eval
strategies, persistence) is the caller's responsibility. The trainer
only owns:  forward → cross-entropy loss → backward → grad-clip → step.

Notes on `train_batches`:
    It MUST be a callable that returns a fresh iterable each time it is
    invoked, because we need to re-shuffle / re-iterate at every epoch.
    Passing a one-shot generator would silently train epoch 1 only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .encoder_decoder_model import EncoderDecoderModel


# ══════════════════════════════════════════════════════════════════════
# Public dataclasses
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Hyper-parameters that govern the optimisation loop (not the model)."""

    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_every: int = 10          # epochs between eval_fn invocations (0 disables)
    log_every_batches: int = 0    # per-batch progress logs (0 disables)
    seed: int = 42


@dataclass
class EpochStats:
    """One row of per-epoch metrics, passed to the `on_epoch_end` callback."""

    epoch: int
    train_loss: float            # mean cross-entropy loss across all batches
    train_tf_acc: float          # teacher-forced token-level accuracy
    val_metric: Optional[float]  # whatever `eval_fn` returns this epoch (or None)
    elapsed_s: float


# Type aliases for clarity at call-sites.
Batch       = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]   # (src, tgt_in, tgt_target)
BatchSource = Callable[[], Iterable[Batch]]
EvalFn      = Callable[[EncoderDecoderModel], float]
EpochHook   = Callable[[EpochStats], None]


# ══════════════════════════════════════════════════════════════════════
# Train function
# ══════════════════════════════════════════════════════════════════════

def train(
    model: EncoderDecoderModel,
    train_batches: BatchSource,
    pad_id: int,
    cfg: TrainConfig,
    eval_fn: Optional[EvalFn] = None,
    on_epoch_end: Optional[EpochHook] = None,
) -> None:
    """
    Optimise `model` in place over `cfg.epochs` epochs of teacher-forced
    cross-entropy training. Returns nothing — all observable effects go
    through `on_epoch_end` and the model parameters themselves.

    Parameters
    ----------
    model
        An `EncoderDecoderModel`. Must already be on the desired device.
    train_batches
        Zero-arg callable. Each invocation must yield an iterable of
        `(src_ids, tgt_in_ids, tgt_target_ids)` tensor triples already
        sitting on the model's device. Called ONCE PER EPOCH so the
        underlying dataset can re-shuffle.
    pad_id
        Token id to ignore in the loss (CrossEntropyLoss `ignore_index`)
        and to mask out of the teacher-forced accuracy metric.
    cfg
        Hyper-parameters for the optimisation loop.
    eval_fn
        Optional callback `eval_fn(model) -> float`. Invoked every
        `cfg.eval_every` epochs (and on the final epoch). The numeric
        return value is reported via `EpochStats.val_metric`. The
        trainer never looks at the value — semantics (e.g. "exact
        match accuracy") are entirely up to the caller.
    on_epoch_end
        Optional callback invoked AFTER each epoch with an
        `EpochStats` row. Useful for logging, early-stopping book-
        keeping, periodic checkpoints — none of which belong inside
        the trainer itself.
    """

    # Determinism: same seed reproduces the same shuffle order *given*
    # that `train_batches` honours its own seed. We seed torch globally
    # so that the model's dropout layers (if any are added later) and
    # multinomial sampling at eval time are also reproducible.
    torch.manual_seed(cfg.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0
        num_batches   = 0
        t_epoch       = time.time()

        # `train_batches()` produces a fresh, possibly-shuffled iterable.
        for src, tgt_in, tgt_target in train_batches():
            optimizer.zero_grad()

            logits = model(src, tgt_in, verbose=False)
            # logits: (batch, tgt_len, vocab_size)

            b, t, v = logits.shape
            loss = loss_fn(
                logits.reshape(b * t, v),
                tgt_target.reshape(b * t),
            )
            loss.backward()

            # Standard transformer hygiene — keeps gradients from blowing
            # up early in training when softmax outputs are still flat.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Teacher-forced token accuracy (only on non-pad positions).
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask  = tgt_target != pad_id
                epoch_correct += ((preds == tgt_target) & mask).sum().item()
                epoch_total   += mask.sum().item()

            epoch_loss += loss.item()
            num_batches += 1

            # Heartbeat — on slow GPUs an epoch can take minutes; without
            # this the user has no idea whether the run is alive. We print
            # the running mean so the line is meaningful by itself rather
            # than just a single batch's noisy loss.
            if cfg.log_every_batches > 0 and num_batches % cfg.log_every_batches == 0:
                running_loss = epoch_loss / num_batches
                running_acc  = epoch_correct / epoch_total if epoch_total else 0.0
                rate         = num_batches / max(time.time() - t_epoch, 1e-6)
                print(f"    epoch {epoch:3d}  batch {num_batches:5d}  "
                      f"loss={running_loss:.4f}  tf_acc={running_acc:.0%}  "
                      f"({rate:.1f} batch/s)", flush=True)

        avg_loss = epoch_loss / num_batches if num_batches else 0.0
        tf_acc   = epoch_correct / epoch_total if epoch_total else 0.0

        # Decide whether to run `eval_fn` this epoch.
        run_eval = (
            eval_fn is not None
            and cfg.eval_every > 0
            and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs)
        )
        val_metric = eval_fn(model) if run_eval else None

        if on_epoch_end is not None:
            on_epoch_end(EpochStats(
                epoch        = epoch,
                train_loss   = avg_loss,
                train_tf_acc = tf_acc,
                val_metric   = val_metric,
                elapsed_s    = time.time() - t_epoch,
            ))
