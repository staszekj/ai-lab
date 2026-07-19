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

from ts_type_refiner.encoder_decoder_model import EncoderDecoderModel


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
    lr_schedule: str = "none"          # "none" | "cosine" | "plateau"
    lr_schedule_patience: int = 5      # for "plateau": epochs without loss improvement
    lr_schedule_factor: float = 0.5    # for "plateau": multiplicative LR reduction


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
EpochHook   = Callable[[EpochStats], Optional[bool]]


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

    # Baseline intuition (plain gradient descent):
    #   w = w - lr * dL/dw
    # where:
    #   - w      is a model weight
    #   - lr     is learning rate
    #   - dL/dw  is gradient of loss wrt that weight
    #
    # AdamW is a stronger variant applied to EVERY weight separately
    # (per-parameter adaptive updates), not one shared scalar step.
    # For each parameter w_i at step t:
    #   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    #   v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t^2)
    #   m_hat = m_t / (1 - beta1^t)
    #   v_hat = v_t / (1 - beta2^t)
    #   w_i <- w_i - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * w_i
    #
    # `optimizer` is still one global object over all model parameters.
    # Internally it performs the per-parameter math for each w_i.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Scheduler controls the GLOBAL learning-rate trajectory over epochs.
    # It does not replace optimizer math; it modulates lr used by optimizer.
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )
    elif cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "min",
            patience = cfg.lr_schedule_patience,
            factor   = cfg.lr_schedule_factor,
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0
        num_batches   = 0
        t_epoch       = time.time()

        # ════════════════════════════════════════════════════════════════════════
        # TRAINING LOOP — one epoch over all mini-batches
        # ════════════════════════════════════════════════════════════════════════
        # Each mini-batch is one "step" of the optimization. We cycle through
        # train_batches() which yields `(src, tgt_in, tgt_target)` tensor triples.
        #
        # The canonical training loop for seq2seq transformers:
        #   1. Forward pass: compute logits via teacher forcing
        #   2. Loss computation: cross-entropy between logits and target labels
        #   3. Backward pass: compute gradients via autograd
        #   4. Gradient clipping: prevent vanishing/exploding gradients
        #   5. Optimizer step: update all model parameters
        #
        # Example (batch=2, src_len=4, tgt_len=3, vocab_size=5):
        #
        #   Step 1: Forward pass
        #     src       = [[2, 5, 6, 7], [3, 4, 5, 8]]       shape (2, 4)
        #     tgt_in    = [[1, 9, 10], [1, 11, 12]]          shape (2, 3)
        #     tgt_target= [[9, 10, 11], [11, 12, 13]]        shape (2, 3)
        #     logits    = model(src, tgt_in)                 shape (2, 3, 5)
        #                 Each logits[b, t, :] is a vec of 5 raw scores (one per vocab token).
        #
        #   Step 2: Loss
        #     We reshape logits to (batch*tgt_len, vocab_size) = (6, 5)
        #     and tgt_target to (batch*tgt_len,) = (6,)
        #     Then cross_entropy computes: for each position, -log(softmax(logits)[target_id])
        #     Loss is MINIMIZED when logits peak at the true target tokens.
        #
        #   Step 3-5: Backward + step
        #     loss.backward() computes dL/dw for every model weight w
        #     optimizer.step() applies: w <- w - lr * (dL/dw + weight_decay * w)
        #     scheduler adjusts the LEARNING RATE for the next step
        # ════════════════════════════════════════════════════════════════════════

        for src, tgt_in, tgt_target in train_batches():
            optimizer.zero_grad()

            # ──────────────────────────────────────────────────────────────────────
            # STEP 1: Forward pass — teacher forcing
            # ──────────────────────────────────────────────────────────────────────
            # Teacher forcing: the decoder sees the TRUE previous tokens (tgt_in),
            # not the model's own predictions. This is standard in training — the
            # model learns to predict the NEXT token given the correct context.
            #
            # src     : (batch, src_len) encoder input token ids
            # tgt_in  : (batch, tgt_len-1) decoder input (shifted target, no <EOS>)
            # returns : (batch, tgt_len, vocab_size) logits
            logits = model(src, tgt_in, verbose=False)

            # ──────────────────────────────────────────────────────────────────────
            # STEP 2: Compute loss — compare logits to true tokens
            # ──────────────────────────────────────────────────────────────────────
            # Logits (model output):                 Targets (ground truth):
            #   logits shape: (batch, tgt_len,       tgt_target shape: (batch, tgt_len)
            #                   vocab_size)
            #
            #   Example (batch=3, tgt_len=5, vocab=10):
            #
            #   Batch 0:  [[ 1.2, -0.5, 0.8, ...],    Batch 0:  [2, 5, 8, 3, 0]
            #             [ 0.3,  2.1, 1.5, ...],                  ↑pad_id (token 0 = padding)
            #             [ 1.0, -1.2, 0.4, ...],    
            #             [ 0.1,  0.2, 0.3, ...],     Mask (ignore_index=pad_id):
            #             [ 2.5,  1.1, 0.0, ...]]     [1, 1, 1, 1, 0]  ← position 4 is padding!
            #                                                  ↑
            #   Batch 1, 2: ...similar...
            #
            # Why masking? Sequences have different true lengths; we pad them to the
            # maximum sequence length in the batch for efficient batching. But the loss
            # should ONLY accumulate over real tokens, not padding! Without masking:
            #   - pad_ids get random logits (because decoder never sees them at test time)
            #   - loss includes these random -log(softmax) terms → noise in gradient
            #   - model wastes gradient budget trying to predict pad tokens → slower convergence
            #
            # With ignore_index=pad_id, CrossEntropyLoss:
            #   1. Applies softmax to logits → (batch, tgt_len, vocab_size) probabilities
            #   2. Takes -log(prob[target_id]) at each position
            #   3. Multiplies by mask: if target==pad_id, contribution is 0
            #   4. Returns mean loss ONLY over real (non-pad) positions
            # Loss is minimized when the model assigns high probability to true tokens.
            b, t, v = logits.shape
            loss = loss_fn(
                logits.reshape(b * t, v),
                tgt_target.reshape(b * t),
            )

            # ──────────────────────────────────────────────────────────────────────
            # STEP 3: Backward pass — compute gradients via autograd
            # ──────────────────────────────────────────────────────────────────────
            # For each model weight w:
            #   dL/dw is computed by backpropagating through the entire graph
            #   (encoder → decoder → logits → cross-entropy → loss)
            # Gradients accumulate in w.grad for all parameters in the model.
            loss.backward()

            # Gradient clipping: cap the norm of the gradient vector to prevent
            # "exploding gradients" — when early in training softmax outputs are
            # flat and can produce enormous gradients. max_grad_norm typically 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            # ──────────────────────────────────────────────────────────────────────
            # STEP 4: Optimizer step — update all model weights
            # ──────────────────────────────────────────────────────────────────────
            # Core intuition: ALL optimizers follow the same basic pattern:
            #   w_new = w_old - (learning_rate) × (gradient_influence)
            #
            # Plain SGD:    w_new = w_old - lr * dL/dw
            #               Simple: just move down the steepest slope by lr amount
            #
            # AdamW:        w_new = w_old - lr * m_hat / (√v_hat + eps) - lr * weight_decay * w_old
            #               More complex: m_hat and v_hat are adaptive moments that smooth the
            #               gradient signal based on recent history, so:
            #               - fast-changing weights get smaller steps
            #               - consistent directions get larger steps
            #               - weight_decay adds L2 regularization (penalize large weights)
            #
            # The formula looks intimidating, but the INTENT is identical:
            # move weights in the direction opposite to the gradient, with adaptive step size.
            # (m_hat / √v_hat is just an adaptive scaling factor instead of plain "1")
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

        if scheduler is not None:
            # Update the scheduler once per epoch.
            # SCHEDULER PURPOSE: Modulate the GLOBAL learning rate over the training trajectory.
            #
            # Why scheduler? Early in training, loss decreases quickly and we want aggressive
            # steps (high lr). Later, we approach a minimum and start oscillating around it—
            # smaller lr helps us settle into the valley instead of bouncing around.
            #
            # IMPORTANT DISTINCTION from AdamW:
            #   - AdamW: per-parameter adaptive updates (each weight gets its own step size)
            #   - Scheduler: modulates ONE GLOBAL lr that affects ALL parameters together
            #   - Think of AdamW as "fine-tuning each instrument" and scheduler as
            #     "turning down the amplifier dial for everyone"
            #
            # Two strategies in this trainer:
            # - plateau: monitors average train loss; if no improvement for N epochs, cut lr
            # - cosine: smooth cosine decay from lr to near-zero over cfg.epochs
            #
            if cfg.lr_schedule == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # ────────────────────────────────────────────────────────────────────────
        # Evaluation — optional callback to compute validation metrics
        # ────────────────────────────────────────────────────────────────────────
        # eval_fn is a caller-provided function that runs the model on a held-out
        # validation set and returns a scalar metric (e.g., exact-match accuracy).
        # We only invoke it every eval_every epochs (+ final epoch) to save time.
        # The validation metric is NOT used by the trainer itself — the caller
        # decides (via on_epoch_end callback) whether to early-stop or checkpoint.
        run_eval = (
            eval_fn is not None
            and cfg.eval_every > 0
            and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs)
        )
        val_metric = eval_fn(model) if run_eval else None

        if on_epoch_end is not None:
            stop = on_epoch_end(EpochStats(
                epoch        = epoch,
                train_loss   = avg_loss,
                train_tf_acc = tf_acc,
                val_metric   = val_metric,
                elapsed_s    = time.time() - t_epoch,
            ))
            if stop:
                break
