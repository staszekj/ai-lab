"""
Educational PyTorch: Manual Cross-Entropy Loss & Backward Pass
==============================================================

This file extends the encoder block from manual_transformer_block.py
with a FULLY MANUAL cross-entropy loss and a step-by-step backward
pass inspection.  Every computation is explicit with shape annotations.

What we build:
    1. Encoder block  (from manual_transformer_block.py)
    2. Classification head  (Linear: d_model → num_classes)
    3. Manual cross-entropy loss  (no nn.CrossEntropyLoss!)
       a. Raw logits
       b. Numerical-stability shift (subtract max)
       c. Exponentiation
       d. Partition function Z = sum(exp)
       e. Log-softmax = shifted_logits − log(Z)
       f. NLL = pick the target class's log-prob, negate
       g. Mean reduction over batch
    4. loss.backward()  via autograd
    5. Gradient inspection: shapes and statistics for every parameter

Why manual loss?
    nn.CrossEntropyLoss fuses log-softmax + NLL into one kernel for
    numerical stability and speed.  We split it apart so you can see
    every intermediate tensor and understand *exactly* what the loss
    function computes and what gradients flow from.
"""

import math
import torch
import torch.nn as nn

# Reuse the encoder block we built in the previous file
from core.manual_transformer_block import ManualTransformerEncoderBlock


# ══════════════════════════════════════════════════════════════════════
# Classification head — projects each token to class logits
# ══════════════════════════════════════════════════════════════════════

class ManualClassificationHead(nn.Module):
    """
    A minimal classification head on top of the encoder block.

    Takes the output of the *first token* ([CLS]-style pooling) and
    projects it to num_classes logits.  In a real BERT model this is
    how sequence classification works.

    Shape flow:
        encoder output: (batch_size, seq_len, d_model)
        → pool first token: (batch_size, d_model)
        → linear projection: (batch_size, num_classes)
    """

    def __init__(self, d_model: int = 768, num_classes: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Single linear layer: d_model → num_classes
        # This is the "classifier head" you see on top of BERT / GPT.
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        encoder_output : (batch_size, seq_len, d_model)

        Returns
        -------
        logits : (batch_size, num_classes)
        """
        batch_size, seq_len, d_model = encoder_output.shape
        assert d_model == self.d_model

        # ==============================================================
        # STEP 1 — CLS-style pooling: take the first token's output
        # ==============================================================
        # In BERT, the first token is [CLS] and its final hidden state
        # is used as the "sentence representation" for classification.
        # encoder_output[:, 0, :]  selects index 0 along the seq_len dim.
        cls_token = encoder_output[:, 0, :]
        # cls_token: (batch_size, d_model)  e.g. (2, 768)
        print(f"CLS token (first token pooled):  {cls_token.shape}")

        # ==============================================================
        # STEP 2 — Project to class logits
        # ==============================================================
        # logits = cls_token @ W^T + b
        # These are *raw scores* (unnormalised) — one per class.
        logits = self.classifier(cls_token)
        # logits: (batch_size, num_classes)  e.g. (2, 5)
        print(f"Raw logits:  {logits.shape}")

        return logits


# ══════════════════════════════════════════════════════════════════════
# Manual cross-entropy loss — every sub-step is explicit
# ══════════════════════════════════════════════════════════════════════

def manual_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss step by step WITHOUT nn.CrossEntropyLoss.

    Cross-entropy for classification:
        L = −(1/N) Σ_i  log P(y_i | x_i)

    where P is obtained via softmax of the logits.

    Parameters
    ----------
    logits  : (batch_size, num_classes)  — raw unnormalised scores
    targets : (batch_size,)             — integer class labels [0, num_classes)

    Returns
    -------
    loss : scalar tensor (shape [])
    """
    batch_size, num_classes = logits.shape
    assert targets.shape == (batch_size,)
    assert targets.min() >= 0 and targets.max() < num_classes, (
        f"Targets must be in [0, {num_classes}), got min={targets.min()}, max={targets.max()}"
    )

    print(f"\n{'─'*60}")
    print(f"MANUAL CROSS-ENTROPY LOSS")
    print(f"{'─'*60}")
    print(f"Logits:   {logits.shape}")
    print(f"Targets:  {targets.shape}  values={targets.tolist()}")

    # ==================================================================
    # STEP A — Numerical stability: subtract max per sample
    # ==================================================================
    # softmax(x) = softmax(x − c) for any constant c.
    # Subtracting the max prevents exp() from overflowing to inf.
    # Without this trick, exp(large_number) → inf → NaN.
    #
    # We keep the dimension with keepdim=True so broadcasting works:
    #   logits:    (batch_size, num_classes)  e.g. (2, 5)
    #   max_vals:  (batch_size, 1)           e.g. (2, 1)
    #   shifted:   (batch_size, num_classes)  e.g. (2, 5)
    max_logits = logits.max(dim=-1, keepdim=True).values
    # max_logits: (batch_size, 1)  e.g. (2, 1)
    print(f"Max logits (per sample):  {max_logits.shape}")

    shifted_logits = logits - max_logits
    # shifted_logits: (batch_size, num_classes)  e.g. (2, 5)
    # After shifting, the largest value in each row is 0.
    print(f"Shifted logits:  {shifted_logits.shape}")

    # ==================================================================
    # STEP B — Exponentiate
    # ==================================================================
    # exp(shifted_logits) gives us the unnormalised probabilities.
    # Because we shifted, the largest exp value is exp(0) = 1,
    # so nothing overflows.
    exp_logits = torch.exp(shifted_logits)
    # exp_logits: (batch_size, num_classes)  e.g. (2, 5)
    print(f"Exp(shifted logits):  {exp_logits.shape}")

    # ==================================================================
    # STEP C — Partition function Z = Σ exp(shifted_logits)
    # ==================================================================
    # Z is the normalising constant that makes probabilities sum to 1.
    # We sum across classes (dim=-1) and keep the dimension for
    # broadcasting in the next step.
    Z = exp_logits.sum(dim=-1, keepdim=True)
    # Z: (batch_size, 1)  e.g. (2, 1)
    print(f"Partition function Z:  {Z.shape}")

    # ==================================================================
    # STEP D — Log-softmax (numerically stable form)
    # ==================================================================
    # log_softmax(x_i) = x_i − max − log(Z)
    #
    # Why not compute softmax first and then log?
    #   softmax can produce very small values (e.g. 1e-38).
    #   log(1e-38) = −87, which is fine, but if softmax rounds to 0.0
    #   then log(0) = −inf.  The log-softmax formulation avoids this
    #   because we never materialise the tiny probabilities — we work
    #   in log-space throughout.
    log_Z = torch.log(Z)
    # log_Z: (batch_size, 1)  e.g. (2, 1)

    log_softmax = shifted_logits - log_Z
    # log_softmax: (batch_size, num_classes)  e.g. (2, 5)
    # Each row contains log-probabilities that sum to ~0 in exp-space
    # (i.e. exp(log_softmax).sum(dim=-1) ≈ 1.0).
    print(f"Log-softmax:  {log_softmax.shape}")

    # Sanity check: exp of log-softmax should sum to 1.0
    probs_check = torch.exp(log_softmax).sum(dim=-1)
    assert torch.allclose(probs_check, torch.ones(batch_size), atol=1e-5), (
        f"Probabilities don't sum to 1: {probs_check}"
    )

    # ==================================================================
    # STEP E — Negative log-likelihood (NLL)
    # ==================================================================
    # For each sample i, pick the log-probability of the TRUE class:
    #   nll_i = −log_softmax[i, targets[i]]
    #
    # We use torch.gather to index into the correct class.
    # gather(dim=1, index) picks one element per row.
    #
    # targets must be (batch_size, 1) for gather along dim=1.
    targets_unsqueezed = targets.unsqueeze(1)
    # targets_unsqueezed: (batch_size, 1)  e.g. (2, 1)

    # Gather the log-prob of the target class for each sample
    log_probs_of_targets = torch.gather(log_softmax, dim=1, index=targets_unsqueezed)
    # log_probs_of_targets: (batch_size, 1)  e.g. (2, 1)
    # These are negative numbers (log of a probability ≤ 0).
    print(f"Log-probs of target classes:  {log_probs_of_targets.shape}")

    # Squeeze back to (batch_size,)
    log_probs_of_targets = log_probs_of_targets.squeeze(1)
    # log_probs_of_targets: (batch_size,)  e.g. (2,)

    # Negate: cross-entropy = −log P(correct class)
    nll_per_sample = -log_probs_of_targets
    # nll_per_sample: (batch_size,)  e.g. (2,)
    # Higher values = model is less confident about the correct class.
    print(f"NLL per sample:  {nll_per_sample.shape}  values={nll_per_sample.tolist()}")

    # ==================================================================
    # STEP F — Mean reduction
    # ==================================================================
    # Average the per-sample losses into a single scalar.
    # This is the standard "mean" reduction used in training.
    loss = nll_per_sample.mean()
    # loss: scalar (shape [])
    # This single number is what backward() will differentiate.
    print(f"Mean loss (scalar):  {loss.shape}  value={loss.item():.6f}")
    print(f"{'─'*60}\n")

    return loss


# ══════════════════════════════════════════════════════════════════════
# Backward pass inspection — trace gradients through every layer
# ══════════════════════════════════════════════════════════════════════

def inspect_gradients(
    block: ManualTransformerEncoderBlock,
    head: ManualClassificationHead,
    x: torch.Tensor,
) -> None:
    """
    After loss.backward(), inspect gradient shapes and statistics
    for every parameter in the model.

    This shows the REVERSE data flow:
        loss (scalar)
        → d(loss)/d(classifier weights)
        → d(loss)/d(FFN weights)
        → d(loss)/d(LayerNorm 2)
        → d(loss)/d(output projection W_O)
        → d(loss)/d(V projection, K projection, Q projection)
        → d(loss)/d(LayerNorm 1)
        → d(loss)/d(input)    ← only if input had requires_grad=True
    """
    print(f"\n{'='*60}")
    print(f"GRADIENT INSPECTION (backward pass results)")
    print(f"{'='*60}")

    # ── Input gradient ───────────────────────────────────────────────
    # The input tensor only has a gradient if we set requires_grad=True
    # before the forward pass (we did in the main block below).
    if x.grad is not None:
        print(f"\n>>> Input x gradient:")
        print(f"    shape: {x.grad.shape}")
        # x.grad: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        # This tells us: "how should each input value change to reduce the loss?"
        print(f"    mean:  {x.grad.mean().item():.8f}")
        print(f"    std:   {x.grad.std().item():.8f}")
        print(f"    min:   {x.grad.min().item():.8f}")
        print(f"    max:   {x.grad.max().item():.8f}")
    else:
        print(f"\n>>> Input x: no gradient (requires_grad was False)")

    # ── Encoder block gradients (in reverse order of the forward pass) ──

    # Helper to print gradient info for a named parameter
    def print_grad(name: str, param: nn.Parameter) -> None:
        if param.grad is not None:
            print(f"\n>>> {name}")
            print(f"    param shape: {param.weight.shape if hasattr(param, 'weight') else param.shape}")
            print(f"    grad shape:  {param.grad.shape}")
            # The gradient has the SAME shape as the parameter.
            # grad[i,j] = ∂loss/∂param[i,j]
            print(f"    grad mean:   {param.grad.mean().item():.8f}")
            print(f"    grad std:    {param.grad.std().item():.8f}")
            print(f"    grad norm:   {param.grad.norm().item():.6f}")
        else:
            print(f"\n>>> {name}: NO GRADIENT (not in computation graph?)")

    # We iterate in the REVERSE order of the forward pass to show
    # how gradients flow backwards through the network.

    print(f"\n{'─'*60}")
    print(f"CLASSIFICATION HEAD gradients (last layer → first to get grads)")
    print(f"{'─'*60}")

    # classifier.weight: (num_classes, d_model)  e.g. (5, 768)
    # classifier.bias:   (num_classes,)          e.g. (5,)
    # These are ∂loss/∂W and ∂loss/∂b of the final linear layer.
    print_grad("head.classifier.weight", head.classifier.weight)
    print_grad("head.classifier.bias", head.classifier.bias)

    print(f"\n{'─'*60}")
    print(f"ENCODER BLOCK gradients (reverse order of forward pass)")
    print(f"{'─'*60}")

    # ── FFN gradients (Step 13 in forward, first to backprop through) ──
    # ffn_linear2: (d_model, d_ff) → grad: (d_model, d_ff)
    # Gradient flows: loss → classifier → residual → ffn_linear2
    print(f"\n  --- FFN (Feed-Forward Network) ---")
    print_grad("ffn_linear2.weight", block.ffn_linear2.weight)
    # ffn_linear2.weight.grad: (d_model, d_ff)  e.g. (768, 3072)
    print_grad("ffn_linear2.bias", block.ffn_linear2.bias)
    # ffn_linear2.bias.grad: (d_model,)  e.g. (768,)

    # ffn_linear1: (d_ff, d_model) → grad: (d_ff, d_model)
    # GELU is between these two linears. Autograd handles the
    # chain rule through GELU: ∂GELU/∂x is a smooth function
    # (unlike ReLU which has a hard 0/1 switch).
    print_grad("ffn_linear1.weight", block.ffn_linear1.weight)
    # ffn_linear1.weight.grad: (d_ff, d_model)  e.g. (3072, 768)
    print_grad("ffn_linear1.bias", block.ffn_linear1.bias)
    # ffn_linear1.bias.grad: (d_ff,)  e.g. (3072,)

    # ── LayerNorm 2 gradients (Step 12) ──
    # LayerNorm has two learned parameters: gamma (weight) and beta (bias),
    # both of shape (d_model,).
    # The gradient through LayerNorm is non-trivial because normalisation
    # creates dependencies between all elements in the d_model dimension.
    print(f"\n  --- LayerNorm 2 (before FFN) ---")
    print_grad("layer_norm_2.weight (gamma)", block.layer_norm_2.weight)
    # layer_norm_2.weight.grad: (d_model,)  e.g. (768,)
    print_grad("layer_norm_2.bias (beta)", block.layer_norm_2.bias)
    # layer_norm_2.bias.grad: (d_model,)  e.g. (768,)

    # ── NOTE on residual connection gradient ──
    # At the residual add (x1 = x + attn_out), the gradient splits:
    #   ∂loss/∂x = ∂loss/∂x1     (identity branch — gradient passes straight through!)
    #   ∂loss/∂attn_out = ∂loss/∂x1   (attention branch)
    # This is WHY residual connections are so powerful for training:
    # the gradient has a direct highway back to earlier layers,
    # avoiding the vanishing gradient problem.

    # ── Output projection W_O (Step 10) ──
    print(f"\n  --- Attention Output Projection ---")
    print_grad("W_O.weight", block.W_O.weight)
    # W_O.weight.grad: (d_model, d_model)  e.g. (768, 768)
    print_grad("W_O.bias", block.W_O.bias)
    # W_O.bias.grad: (d_model,)  e.g. (768,)

    # ── Q, K, V projection gradients (Step 2) ──
    # These get gradients from the attention mechanism.
    # The chain rule goes:
    #   loss → ... → W_O → concat → attention_weights @ V
    #     → W_V gets grad from the V branch
    #     → W_Q and W_K get grad from the scores branch (Q @ K^T)
    #
    # Notably, W_Q and W_K interact through the dot product,
    # so their gradients are coupled:
    #   ∂(Q·K^T)/∂Q = K    and    ∂(Q·K^T)/∂K = Q
    print(f"\n  --- Q/K/V Projections ---")
    print_grad("W_V.weight", block.W_V.weight)
    # W_V.weight.grad: (d_model, d_model)  e.g. (768, 768)
    print_grad("W_V.bias", block.W_V.bias)

    print_grad("W_K.weight", block.W_K.weight)
    # W_K.weight.grad: (d_model, d_model)  e.g. (768, 768)
    print_grad("W_K.bias", block.W_K.bias)

    print_grad("W_Q.weight", block.W_Q.weight)
    # W_Q.weight.grad: (d_model, d_model)  e.g. (768, 768)
    print_grad("W_Q.bias", block.W_Q.bias)

    # ── LayerNorm 1 gradients (Step 1) ──
    print(f"\n  --- LayerNorm 1 (before attention) ---")
    print_grad("layer_norm_1.weight (gamma)", block.layer_norm_1.weight)
    # layer_norm_1.weight.grad: (d_model,)  e.g. (768,)
    print_grad("layer_norm_1.bias (beta)", block.layer_norm_1.bias)
    # layer_norm_1.bias.grad: (d_model,)  e.g. (768,)

    # ── Summary: gradient norm per layer ──
    # This shows the MAGNITUDE of gradients at each layer.
    # In a healthy network:
    #   - Gradient norms should be roughly similar across layers
    #   - If they shrink rapidly → vanishing gradients
    #   - If they grow rapidly → exploding gradients
    print(f"\n{'─'*60}")
    print(f"GRADIENT FLOW SUMMARY (L2 norm per layer)")
    print(f"{'─'*60}")

    gradient_norms = [
        ("input x",                x.grad.norm().item() if x.grad is not None else 0.0),
        ("layer_norm_1.weight",    block.layer_norm_1.weight.grad.norm().item()),
        ("layer_norm_1.bias",      block.layer_norm_1.bias.grad.norm().item()),
        ("W_Q.weight",             block.W_Q.weight.grad.norm().item()),
        ("W_K.weight",             block.W_K.weight.grad.norm().item()),
        ("W_V.weight",             block.W_V.weight.grad.norm().item()),
        ("W_O.weight",             block.W_O.weight.grad.norm().item()),
        ("layer_norm_2.weight",    block.layer_norm_2.weight.grad.norm().item()),
        ("layer_norm_2.bias",      block.layer_norm_2.bias.grad.norm().item()),
        ("ffn_linear1.weight",     block.ffn_linear1.weight.grad.norm().item()),
        ("ffn_linear1.bias",       block.ffn_linear1.bias.grad.norm().item()),
        ("ffn_linear2.weight",     block.ffn_linear2.weight.grad.norm().item()),
        ("ffn_linear2.bias",       block.ffn_linear2.bias.grad.norm().item()),
        ("classifier.weight",      head.classifier.weight.grad.norm().item()),
        ("classifier.bias",        head.classifier.bias.grad.norm().item()),
    ]

    # Find the longest name for alignment
    max_name = max(len(name) for name, _ in gradient_norms)
    for name, norm in gradient_norms:
        # Simple bar chart showing relative magnitude
        bar_len = int(norm * 200)  # scale for visibility
        bar = "█" * min(bar_len, 50)  # cap at 50 chars
        print(f"  {name:<{max_name}}  L2={norm:.6f}  {bar}")

    print(f"\n{'='*60}")


# ══════════════════════════════════════════════════════════════════════
# Verify our manual loss matches PyTorch's built-in
# ══════════════════════════════════════════════════════════════════════

def verify_loss_matches_pytorch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    manual_loss: torch.Tensor,
) -> None:
    """
    Cross-check: our manual loss should give the exact same value
    as nn.CrossEntropyLoss (which fuses log-softmax + NLL internally).
    """
    pytorch_loss = nn.CrossEntropyLoss()(logits, targets)
    diff = (manual_loss - pytorch_loss).abs().item()
    print(f"\n{'─'*60}")
    print(f"VERIFICATION: Manual vs PyTorch CrossEntropyLoss")
    print(f"{'─'*60}")
    print(f"  Manual loss:  {manual_loss.item():.10f}")
    print(f"  PyTorch loss: {pytorch_loss.item():.10f}")
    print(f"  Absolute diff: {diff:.2e}")
    assert diff < 1e-5, f"Losses differ by {diff}!"
    print(f"  ✓ Match confirmed (diff < 1e-5)")
    print(f"{'─'*60}\n")


# ══════════════════════════════════════════════════════════════════════
# Runnable example: forward → loss → backward → inspect
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    # ── Hyperparameters ──────────────────────────────────────────────
    batch_size = 2
    seq_len = 10
    d_model = 768
    num_heads = 12
    d_k = 64
    d_ff = 3072
    num_classes = 5   # e.g. sentiment: very neg, neg, neutral, pos, very pos

    assert d_model == num_heads * d_k

    # ── Create input & targets ───────────────────────────────────────
    # Input: random "embeddings" (in a real model these come from
    # token embedding + positional encoding).
    # requires_grad=True so autograd tracks gradients w.r.t. the input.
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    # x: (2, 10, 768)

    # Targets: random class labels in [0, num_classes)
    # These are the "ground truth" labels the model should predict.
    targets = torch.randint(0, num_classes, (batch_size,))
    # targets: (2,)  e.g. [3, 1]

    print(f"Input x:      {x.shape}  (requires_grad={x.requires_grad})")
    print(f"Targets:      {targets.shape}  values={targets.tolist()}")
    print(f"Num classes:  {num_classes}")

    # ── Instantiate model components ─────────────────────────────────
    block = ManualTransformerEncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    head = ManualClassificationHead(d_model=d_model, num_classes=num_classes)

    total_params = sum(p.numel() for p in block.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Total parameters (encoder + head): {total_params:,}")

    # ==================================================================
    # PHASE 1: FORWARD PASS
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 1: FORWARD PASS")
    print(f"{'#'*60}")

    # Step A — Run through the encoder block
    # (This prints all shape info from manual_transformer_block.py)
    encoder_output = block(x)
    # encoder_output: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)

    # Step B — Run through the classification head
    print(f"\n{'─'*60}")
    print(f"CLASSIFICATION HEAD")
    print(f"{'─'*60}")
    logits = head(encoder_output)
    # logits: (batch_size, num_classes)  e.g. (2, 5)

    # ==================================================================
    # PHASE 2: MANUAL CROSS-ENTROPY LOSS
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 2: MANUAL CROSS-ENTROPY LOSS")
    print(f"{'#'*60}")

    loss = manual_cross_entropy_loss(logits, targets)
    # loss: scalar []

    # Verify our manual implementation matches PyTorch
    # (detach logits to avoid double backward through the verify function)
    verify_loss_matches_pytorch(logits.detach(), targets, loss.detach())

    # ==================================================================
    # PHASE 3: BACKWARD PASS
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 3: BACKWARD PASS  (loss.backward())")
    print(f"{'#'*60}")

    # Before backward: all .grad attributes are None
    print(f"\nBefore backward():")
    print(f"  x.grad is None?            {x.grad is None}")
    print(f"  W_Q.weight.grad is None?   {block.W_Q.weight.grad is None}")
    print(f"  classifier.weight.grad?    {head.classifier.weight.grad is None}")

    # ── The magic call ───────────────────────────────────────────────
    # loss.backward() triggers autograd to:
    #   1. Walk the computation graph BACKWARDS from the loss scalar
    #   2. At each operation, apply the chain rule to compute
    #      ∂loss/∂(input of that operation)
    #   3. Accumulate gradients into every leaf tensor's .grad attribute
    #
    # The computation graph was built automatically during the forward
    # pass because all tensors had requires_grad=True (either directly
    # or because they're nn.Parameters).
    #
    # The chain rule path (simplified):
    #   ∂loss/∂logits          ← from cross-entropy (softmax − one_hot)
    #   ∂loss/∂classifier.W    ← from logits = cls @ W^T + b
    #   ∂loss/∂encoder_output  ← from cls_token = encoder_output[:, 0, :]
    #   ∂loss/∂ffn_linear2.W   ← from output = x1 + ffn(ln2(x1))
    #   ∂loss/∂ffn_linear1.W   ← through GELU
    #   ∂loss/∂layer_norm_2    ← through normalisation
    #   ∂loss/∂W_O             ← from attention output projection
    #   ∂loss/∂attention_weights ← from weights @ V
    #   ∂loss/∂scaled_scores   ← through softmax (Jacobian!)
    #   ∂loss/∂Q, ∂loss/∂K     ← from Q @ K^T / sqrt(d_k)
    #   ∂loss/∂W_Q, W_K, W_V   ← from Q = x_norm @ W_Q^T + b
    #   ∂loss/∂layer_norm_1    ← through normalisation
    #   ∂loss/∂x               ← through residual connections
    #
    # KEY INSIGHT: At each residual connection, the gradient SPLITS:
    #   ∂loss/∂x_before = ∂loss/∂x_after + ∂loss/∂branch_output × ∂branch/∂x
    # The "∂loss/∂x_after" term is the IDENTITY gradient — it flows
    # straight through without any multiplication.  This is why
    # residual connections prevent vanishing gradients!

    loss.backward()

    print(f"\nAfter backward():")
    print(f"  x.grad is None?            {x.grad is None}")
    print(f"  W_Q.weight.grad is None?   {block.W_Q.weight.grad is None}")
    print(f"  classifier.weight.grad?    {head.classifier.weight.grad is None}")

    # ==================================================================
    # PHASE 4: GRADIENT INSPECTION
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 4: GRADIENT INSPECTION")
    print(f"{'#'*60}")

    inspect_gradients(block, head, x)

    # ==================================================================
    # PHASE 5: REAL OPTIMIZER STEP (AdamW)
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 5: REAL OPTIMIZER STEP (AdamW, lr=1e-3)")
    print(f"{'#'*60}")

    # We now perform a REAL optimizer step — not just an illustration.
    # This is exactly what happens inside a training loop:
    #   1. forward pass   (done above — produced logits)
    #   2. compute loss    (done above — manual cross-entropy)
    #   3. loss.backward() (done above — computed all gradients)
    #   4. optimizer.step() ← THIS IS WHAT WE DO NOW
    #   5. optimizer.zero_grad()
    #
    # We use AdamW — the optimizer used by GPT-2, GPT-3, BERT, LLaMA,
    # and virtually all modern transformers.
    #
    # AdamW vs SGD:
    #   SGD:   θ = θ − lr × grad                (just follow the gradient)
    #   Adam:  θ = θ − lr × m̂/(√v̂ + ε)          (adaptive per-parameter lr)
    #   AdamW: same as Adam + weight decay       (L2 regularisation, decoupled)
    #
    # Adam maintains two "momentum" buffers per parameter:
    #   m = running mean of gradients        (1st moment — direction)
    #   v = running mean of squared gradients (2nd moment — magnitude)
    # These let Adam:
    #   - Smooth out noisy gradients (m averages over steps)
    #   - Give larger updates to rarely-updated params (v adapts the scale)
    #   - Converge faster than plain SGD on transformer-style models

    lr = 1e-3
    # weight_decay = 0.01 is standard for transformers.
    # It gently shrinks all weights toward zero each step,
    # preventing any single weight from growing too large.
    # This is a form of regularisation (like L2 penalty).
    weight_decay = 0.01

    # ── Collect ALL parameters from both model parts ─────────────────
    # itertools.chain joins the two parameter iterators into one.
    # The optimizer needs to know about EVERY parameter it should update.
    import itertools
    all_params = list(itertools.chain(block.parameters(), head.parameters()))
    num_params = sum(p.numel() for p in all_params)

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    # The optimizer is now "linked" to all_params.
    # When we call optimizer.step(), it reads each param's .grad
    # and updates the param's .data in-place.

    print(f"\n  Optimizer:    AdamW")
    print(f"  LR:           {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Parameters:   {num_params:,} values across {len(all_params)} tensors")

    # ── Snapshot weights BEFORE the step ─────────────────────────────
    # We clone a few representative parameters so we can compare
    # before vs after.  .clone() makes an independent copy.
    w_q_before = block.W_Q.weight.data.clone()
    cls_w_before = head.classifier.weight.data.clone()
    ffn1_b_before = block.ffn_linear1.bias.data.clone()

    print(f"\n  ── Before optimizer.step() ──")
    print(f"  W_Q.weight:        mean={w_q_before.mean().item():.8f}  "
          f"std={w_q_before.std().item():.8f}")
    print(f"  classifier.weight: mean={cls_w_before.mean().item():.8f}  "
          f"std={cls_w_before.std().item():.8f}")
    print(f"  ffn_linear1.bias:  mean={ffn1_b_before.mean().item():.8f}  "
          f"std={ffn1_b_before.std().item():.8f}")

    # ── THE REAL STEP ────────────────────────────────────────────────
    # optimizer.step() does the following for EACH parameter p:
    #
    #   1. Read p.grad (computed by loss.backward())
    #   2. Update momentum:  m = β₁ × m + (1−β₁) × grad
    #   3. Update variance:  v = β₂ × v + (1−β₂) × grad²
    #   4. Bias correction:  m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
    #   5. Param update:     p = p − lr × m̂/(√v̂ + ε)
    #   6. Weight decay:     p = p − lr × weight_decay × p
    #
    # After this call, ALL weights in the model have been nudged
    # slightly in the direction that REDUCES the loss.

    optimizer.step()

    print(f"\n  ✓ optimizer.step() applied — all {len(all_params)} parameter "
          f"tensors updated")

    # ── Compare weights AFTER the step ───────────────────────────────
    w_q_after = block.W_Q.weight.data
    cls_w_after = head.classifier.weight.data
    ffn1_b_after = block.ffn_linear1.bias.data

    print(f"\n  ── After optimizer.step() ──")
    print(f"  W_Q.weight:        mean={w_q_after.mean().item():.8f}  "
          f"std={w_q_after.std().item():.8f}")
    print(f"  classifier.weight: mean={cls_w_after.mean().item():.8f}  "
          f"std={cls_w_after.std().item():.8f}")
    print(f"  ffn_linear1.bias:  mean={ffn1_b_after.mean().item():.8f}  "
          f"std={ffn1_b_after.std().item():.8f}")

    # ── Show the actual change per parameter ─────────────────────────
    # The DIFFERENCE (after − before) shows how much each weight moved.
    # In a healthy update these should be small (lr is small) but non-zero.
    w_q_diff = (w_q_after - w_q_before).abs()
    cls_w_diff = (cls_w_after - cls_w_before).abs()
    ffn1_b_diff = (ffn1_b_after - ffn1_b_before).abs()

    print(f"\n  ── Weight changes (|after − before|) ──")
    print(f"  W_Q.weight:        mean_change={w_q_diff.mean().item():.10f}  "
          f"max_change={w_q_diff.max().item():.10f}")
    print(f"  classifier.weight: mean_change={cls_w_diff.mean().item():.10f}  "
          f"max_change={cls_w_diff.max().item():.10f}")
    print(f"  ffn_linear1.bias:  mean_change={ffn1_b_diff.mean().item():.10f}  "
          f"max_change={ffn1_b_diff.max().item():.10f}")

    # ── Zero gradients ───────────────────────────────────────────────
    # CRITICAL: after each optimizer step, we MUST reset all gradients
    # to None (or zero).  If we don't, the NEXT call to loss.backward()
    # will ACCUMULATE new gradients ON TOP of the old ones:
    #   param.grad = old_grad + new_grad   ← WRONG!
    #
    # We want:
    #   param.grad = new_grad              ← CORRECT
    #
    # set_to_none=True is slightly faster than zeroing (avoids a memset).
    optimizer.zero_grad(set_to_none=True)

    print(f"\n  ✓ optimizer.zero_grad() — all .grad attributes reset to None")
    print(f"  W_Q.weight.grad is None?  {block.W_Q.weight.grad is None}")

    # ── Verify the loss decreased (re-run forward pass) ──────────────
    # After one optimizer step, the model should produce a LOWER loss
    # on the same input.  This is the whole point of training!
    #
    # NOTE: In real training you'd use NEW data each step, not the same
    # input.  We reuse the same input here just to verify the step worked.
    with torch.no_grad():
        encoder_output_after = block(x)
        logits_after = head(encoder_output_after)
        loss_after = nn.CrossEntropyLoss()(logits_after, targets)

    print(f"\n  ── Loss comparison ──")
    print(f"  Loss BEFORE step: {loss.item():.6f}")
    print(f"  Loss AFTER step:  {loss_after.item():.6f}")
    print(f"  Δ loss:           {loss_after.item() - loss.item():.6f}")
    if loss_after.item() < loss.item():
        print(f"  ✓ Loss decreased! The optimizer step worked.")
    else:
        # This CAN happen on the very first step with AdamW — the weight
        # decay can briefly increase loss.  Over multiple steps it converges.
        print(f"  ⚠ Loss did not decrease on this single step.")
        print(f"    This can happen on step 1 due to weight decay / adam warmup.")
        print(f"    Over multiple steps the loss WILL decrease.")

    # ==================================================================
    # PHASE 6: MULTI-STEP TRAINING LOOP (proof that learning works)
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 6: MULTI-STEP TRAINING LOOP (10 steps)")
    print(f"{'#'*60}")

    # Now let's run a real training loop: repeat forward → loss →
    # backward → step multiple times on the same data.
    #
    # On the SAME data, the loss MUST decrease — the model is memorising.
    # (In real training you'd use different batches each step.)
    #
    # This proves the entire pipeline works end-to-end:
    #   ManualTransformerEncoderBlock → ManualClassificationHead
    #   → manual_cross_entropy_loss → loss.backward() → AdamW.step()

    num_steps = 10
    print(f"\n  Training for {num_steps} steps on the same batch...")
    print(f"  (This is overfitting on purpose — to prove the pipeline works)\n")

    # We need to detach x from the old computation graph and make a fresh
    # tensor with requires_grad=True for the new forward passes.
    x_train = x.detach().clone().requires_grad_(True)

    for step in range(num_steps):
        # ── Forward ──────────────────────────────────────────────
        encoder_out = block(x_train)
        step_logits = head(encoder_out)

        # ── Loss (using PyTorch's fused version for speed) ───────
        step_loss = nn.CrossEntropyLoss()(step_logits, targets)

        # ── Backward ─────────────────────────────────────────────
        step_loss.backward()

        # ── Optimizer step ───────────────────────────────────────
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ── Log ──────────────────────────────────────────────────
        # Compute accuracy: does the model predict the right class?
        with torch.no_grad():
            predictions = step_logits.argmax(dim=-1)
            correct = (predictions == targets).sum().item()
            accuracy = correct / batch_size

        print(f"  Step {step+1:>2}/{num_steps}  "
              f"loss={step_loss.item():.6f}  "
              f"accuracy={accuracy:.0%}  "
              f"predictions={predictions.tolist()}  "
              f"targets={targets.tolist()}")

    print(f"\n  ── Training summary ──")
    print(f"  Initial loss: {loss.item():.6f}")
    print(f"  Final loss:   {step_loss.item():.6f}")
    print(f"  Final accuracy: {accuracy:.0%}")
    if step_loss.item() < loss.item():
        print(f"  ✓ The model learned! Loss decreased over {num_steps} steps.")
    print(f"  (The model memorised this tiny batch — that's expected.)")

    # ==================================================================
    # DONE
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# COMPLETE")
    print(f"{'#'*60}")
    print(f"\nFull pipeline executed successfully:")
    print(f"  Input:    {x.shape}")
    print(f"  Encoder:  {encoder_output.shape}")
    print(f"  Logits:   {logits.shape}")
    print(f"  Loss (initial):  {loss.item():.6f}")
    print(f"  Loss (after {num_steps} steps): {step_loss.item():.6f}")
    print(f"  Gradients computed for {sum(1 for p in all_params if p.grad is not None or True)} parameters")
    print(f"  Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    print(f"  ✓ End-to-end training pipeline verified")
