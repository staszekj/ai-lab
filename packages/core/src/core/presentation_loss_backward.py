"""
ASCII Art Presentation: Cross-Entropy Loss & Backward Pass — Step by Step
=========================================================================

Runs a REAL forward + loss + backward + optimizer pipeline through a
tiny encoder block + classification head.  Every intermediate tensor
is printed so you can see exactly what cross-entropy computes, how
gradients flow backward, and how AdamW updates weights.

    batch_size  = 2    (two samples)
    seq_len     = 5    (five tokens each)
    d_model     = 6    (embedding width)
    num_heads   = 3    (attention heads)
    d_k         = 2    (per-head dim)
    d_ff        = 12   (FFN hidden size)
    num_classes = 4    (classification categories)

Run:  uv run python3 -m core.presentation_loss_backward
"""

import math
import torch
import torch.nn as nn
import io
import contextlib

from core.manual_transformer_block import ManualTransformerEncoderBlock
from core.manual_loss_backward import ManualClassificationHead

# ══════════════════════════════════════════════════════════════════════
# Helpers — pretty-print tensors as ASCII tables
# ══════════════════════════════════════════════════════════════════════

def fmt(val: float, width: int = 7) -> str:
    s = f"{val:+.2f}"
    return s.rjust(width)

def fmt4(val: float, width: int = 9) -> str:
    s = f"{val:+.4f}"
    return s.rjust(width)

def print_matrix(name: str, t: torch.Tensor, indent: int = 4):
    pad = " " * indent
    rows, cols = t.shape
    print(f"{pad}{name}  ({rows}×{cols})")
    print(f"{pad}┌{'─' * (8 * cols + 1)}┐")
    for r in range(rows):
        vals = " ".join(fmt(t[r, c].item()) for c in range(cols))
        print(f"{pad}│ {vals} │")
    print(f"{pad}└{'─' * (8 * cols + 1)}┘")

def print_vector(name: str, t: torch.Tensor, indent: int = 4, decimals: int = 4):
    pad = " " * indent
    if decimals == 4:
        vals = "  ".join(fmt4(v.item()) for v in t)
    else:
        vals = "  ".join(fmt(v.item()) for v in t)
    print(f"{pad}{name}  ({t.shape[0]},)")
    print(f"{pad}  [{vals}]")

def print_tensor_3d(name: str, t: torch.Tensor, dim_labels=None):
    d0, d1, d2 = t.shape
    labels = dim_labels or [f"[{i}]" for i in range(d0)]
    for i in range(d0):
        print_matrix(f"{name}{labels[i]}", t[i])

def section(title: str):
    print(f"\n{'━' * 72}")
    print(f"  {title}")
    print(f"{'━' * 72}")

def step_header(num: str, title: str):
    width = max(56 - len(num) - len(title), 2)
    print(f"\n  ┌─── STEP {num} {'─' * width} {title} ───┐")

# ══════════════════════════════════════════════════════════════════════
# Configuration — tiny model so every value fits on screen
# ══════════════════════════════════════════════════════════════════════

BATCH       = 2
SEQ         = 5
D_MODEL     = 6
HEADS       = 3
D_K         = D_MODEL // HEADS   # = 2
D_FF        = 12
NUM_CLASSES = 4

torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════════════
# Vocabulary & Class Labels
# ══════════════════════════════════════════════════════════════════════

VOCAB_WORDS = [
    "<pad>", "the", "cat", "dog", "climbs", "sits", "a", "tree",
    "house", "on", "big", "small", "runs", "eats", "fish", "bird",
]

VOCAB = len(VOCAB_WORDS)
word2id = {w: i for i, w in enumerate(VOCAB_WORDS)}

CLASS_NAMES = ["animal", "action", "food", "place"]
SENTENCES = ["the cat climbs the tree", "the dog eats a fish"]
SENT_LABELS = [f'"{s}"' for s in SENTENCES]

# Targets: "big cat sits" → animal(0), "dog eats fish" → action(1)
TARGET_IDS = [0, 1]

# ══════════════════════════════════════════════════════════════════════
# Architecture overview
# ══════════════════════════════════════════════════════════════════════

section("ARCHITECTURE — Forward → Loss → Backward → Optimizer")
print(f"""
    batch_size  = {BATCH}    (\"{SENTENCES[0]}\", \"{SENTENCES[1]}\")
    seq_len     = {SEQ}    (five words each)
    d_model     = {D_MODEL}    (embedding dimension)
    num_heads   = {HEADS}    (attention heads)
    d_k         = {D_K}    (per-head dim)
    d_ff        = {D_FF}   (FFN hidden size)
    num_classes = {NUM_CLASSES}    ({', '.join(CLASS_NAMES)})

    Full pipeline:

     ┌──────────────────────── FORWARD PASS ────────────────────────┐
     │                                                              │
     │  Input X ({BATCH}, {SEQ}, {D_MODEL})                                       │
     │    │                                                         │
     │    ▼                                                         │
     │  Encoder Block                                               │
     │    LayerNorm → Q,K,V → MultiHead Attn → +residual           │
     │    → LayerNorm → FFN → +residual                             │
     │    │                                                         │
     │    ▼                                                         │
     │  encoder_output ({BATCH}, {SEQ}, {D_MODEL})                                │
     │    │                                                         │
     │    ▼                                                         │
     │  Classification Head                                         │
     │    Pool first token → Linear({D_MODEL} → {NUM_CLASSES})                    │
     │    │                                                         │
     │    ▼                                                         │
     │  logits ({BATCH}, {NUM_CLASSES})    ← raw scores per class                │
     │                                                              │
     └──────────────────────────────────────────────────────────────┘
                              │
     ┌──────────────────── CROSS-ENTROPY LOSS ──────────────────────┐
     │                                                              │
     │  A. Subtract max   (numerical stability)                     │
     │  B. Exponentiate   exp(shifted)                              │
     │  C. Partition Z    = Σ exp(shifted)                          │
     │  D. Log-softmax    = shifted − log(Z)                        │
     │  E. NLL            = −log P(correct class)                   │
     │  F. Mean           → loss (scalar)                           │
     │                                                              │
     └──────────────────────────────────────────────────────────────┘
                              │
     ┌──────────────────── BACKWARD PASS ───────────────────────────┐
     │                                                              │
     │  loss.backward()                                             │
     │    Autograd walks the computation graph in REVERSE:          │
     │    loss → classifier → W_O → attention → Q,K,V → input      │
     │    Fills .grad for every parameter.                          │
     │                                                              │
     └──────────────────────────────────────────────────────────────┘
                              │
     ┌──────────────────── OPTIMIZER (AdamW) ───────────────────────┐
     │                                                              │
     │  For each parameter p:                                       │
     │    m = β₁·m + (1−β₁)·grad         (momentum)                │
     │    v = β₂·v + (1−β₂)·grad²        (RMS)                     │
     │    p = p − lr · m̂/(√v̂ + ε)        (update)                  │
     │    p = p − lr · wd · p             (weight decay)            │
     │                                                              │
     └──────────────────────────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════
# Build model
# ══════════════════════════════════════════════════════════════════════

block = ManualTransformerEncoderBlock(d_model=D_MODEL, num_heads=HEADS, d_ff=D_FF)
head = ManualClassificationHead(d_model=D_MODEL, num_classes=NUM_CLASSES)

block_params = sum(p.numel() for p in block.parameters())
head_params = sum(p.numel() for p in head.parameters())
total_params = block_params + head_params

section("MODEL PARAMETERS")
print(f"""
    Encoder block:       {block_params:>5,} parameters
    Classification head: {head_params:>5,} parameters
      classifier.weight: ({NUM_CLASSES}, {D_MODEL}) = {NUM_CLASSES * D_MODEL}
      classifier.bias:   ({NUM_CLASSES},)     = {NUM_CLASSES}
    Total:               {total_params:>5,} parameters
""")

# ══════════════════════════════════════════════════════════════════════
# Input & targets
# ══════════════════════════════════════════════════════════════════════

section("INPUT & TARGETS")

import torch.nn as _nn_emb
token_embedding = _nn_emb.Embedding(VOCAB, D_MODEL)
input_ids = torch.tensor([[word2id[w] for w in s.split()] for s in SENTENCES])
X_data = token_embedding(input_ids).detach()
X = X_data.clone().requires_grad_(True)

targets = torch.tensor(TARGET_IDS)

print(f"""
    Sentences (classified into {NUM_CLASSES} categories: {', '.join(CLASS_NAMES)}):
""")
for si, sent in enumerate(SENTENCES):
    cls_name = CLASS_NAMES[targets[si].item()]
    words = sent.split()
    ids = [word2id[w] for w in words]
    mapping = ", ".join(f'"{w}"={word2id[w]}' for w in words)
    print(f'      "{sent}"  →  {ids}  ({mapping})  →  class "{cls_name}"')
print(f"""
    X.shape = ({BATCH}, {SEQ}, {D_MODEL})  — word embeddings
    X.requires_grad = True  (so we can see ∂loss/∂X later)

    targets = {targets.tolist()}  →  [\"{CLASS_NAMES[targets[0]]}\", \"{CLASS_NAMES[targets[1]]}\"]
""")
print_tensor_3d("X", X.data, [f'["{s}"]' for s in SENTENCES])

# ║═══════════════════════════════════════════════════════════════════║
# ║ PHASE 1: FORWARD PASS                                           ║
# ║═══════════════════════════════════════════════════════════════════║

section("PHASE 1 — FORWARD PASS (Encoder Block)")

step_header("1", "Encoder Block")
print(f"""
    The encoder block processes X through:
      LayerNorm → Q,K,V → Multi-Head Attention → +residual
      → LayerNorm → FFN ({D_MODEL}→{D_FF}→{D_MODEL}) → +residual

    Input:  ({BATCH}, {SEQ}, {D_MODEL})
    Output: ({BATCH}, {SEQ}, {D_MODEL})  — same shape, refined representation
""")

# Run encoder block (suppress its verbose prints)
with contextlib.redirect_stdout(io.StringIO()):
    encoder_output = block(X)

print(f"    encoder_output.shape = {tuple(encoder_output.shape)}")
print()
print_tensor_3d("enc_out", encoder_output.data, [f'["{s}"]' for s in SENTENCES])

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Classification Head: CLS pooling + linear
# ══════════════════════════════════════════════════════════════════════

step_header("2", "Classification Head")
print(f"""
    The classification head:
      1. Pools the FIRST token (CLS-style): ({BATCH}, {SEQ}, {D_MODEL}) → ({BATCH}, {D_MODEL})
      2. Linear projection: ({BATCH}, {D_MODEL}) → ({BATCH}, {NUM_CLASSES})

    In BERT, the first token [CLS] is trained to carry the
    "sentence-level" representation for classification.
""")

# CLS pooling — take first token
cls_token = encoder_output[:, 0, :]
# cls_token: (2, 6)
print(f"    CLS token (encoder_output[:, 0, :]):")
print_matrix("cls_token", cls_token.data)
print()

# Show classifier weights
print(f"    Classifier weights W: ({NUM_CLASSES}, {D_MODEL})")
print_matrix("W_cls", head.classifier.weight.data)
print()
print(f"    Classifier bias b: ({NUM_CLASSES},)")
print_vector("b_cls", head.classifier.bias.data, decimals=2)
print()

# Linear projection: logits = cls_token @ W^T + b
with contextlib.redirect_stdout(io.StringIO()):
    logits = head(encoder_output)

print(f"    logits = cls_token @ W_cls^T + b_cls")
print(f"    logits.shape = {tuple(logits.shape)}")
print()
print_matrix("logits", logits.data)

print(f"""
    Each row has {NUM_CLASSES} scores — one per class ({', '.join(CLASS_NAMES)}).
    These are RAW scores (not probabilities yet).
    Higher score = model thinks this class is more likely.

    \"{SENTENCES[0]}\" → target=\"{CLASS_NAMES[targets[0]]}\"  logit={logits[0, targets[0]].item():+.4f}
    \"{SENTENCES[1]}\" → target=\"{CLASS_NAMES[targets[1]]}\"  logit={logits[1, targets[1]].item():+.4f}
""")

# ║═══════════════════════════════════════════════════════════════════║
# ║ PHASE 2: MANUAL CROSS-ENTROPY LOSS — every sub-step visible     ║
# ║═══════════════════════════════════════════════════════════════════║

section("PHASE 2 — MANUAL CROSS-ENTROPY LOSS (step by step)")
print(f"""
    Cross-entropy loss:  L = −(1/N) Σᵢ log P(yᵢ | xᵢ)

    We compute this from RAW LOGITS in 6 sub-steps.
    This is exactly what nn.CrossEntropyLoss does internally.
""")

# ── Step A: subtract max ─────────────────────────────────────────────

step_header("A", "Subtract max (numerical stability)")
print(f"""
    softmax(x) = softmax(x − c) for any constant c.
    We subtract the max per row to prevent exp() overflow.
    Without this: exp(100) = inf → NaN!
    After this:   max value in each row is 0, exp(0) = 1.
""")

max_logits = logits.max(dim=-1, keepdim=True).values
shifted = logits - max_logits

print(f"    max per sample:")
for i in range(BATCH):
    print(f"      \"{SENTENCES[i]}\": max = {max_logits[i, 0].item():+.4f}")
print()
print_matrix("shifted_logits", shifted.data)
print(f"\n    Note: the largest value in each row is now 0.00")

# ── Step B: exponentiate ─────────────────────────────────────────────

step_header("B", "Exponentiate: exp(shifted)")
print(f"""
    exp_logits = exp(shifted_logits)

    Converts log-scale differences into multiplicative ratios.
    exp(0) = 1.0 (the max class), others are < 1.0.
""")

exp_shifted = torch.exp(shifted)
print_matrix("exp(shifted)", exp_shifted.data)

# ── Step C: partition function Z ─────────────────────────────────────

step_header("C", "Partition function Z = Σ exp(shifted)")
print(f"""
    Z = sum of all exp values per sample.
    This is the normalisation constant: probabilities = exp / Z.
""")

Z = exp_shifted.sum(dim=-1, keepdim=True)
print(f"    Z per sample:")
for i in range(BATCH):
    exp_vals = ", ".join(f"{exp_shifted[i, c].item():.4f}" for c in range(NUM_CLASSES))
    print(f"      \"{SENTENCES[i]}\": Z = {exp_vals} = {Z[i, 0].item():.4f}")

# ── Step D: log-softmax ─────────────────────────────────────────────

step_header("D", "Log-softmax = shifted − log(Z)")
print(f"""
    log P(class c) = shifted_logits[c] − log(Z)

    Why not compute softmax then log?
      softmax can give tiny values like 1e-38.
      log(0.0) = -inf → NaN! Working in log-space avoids this.
""")

log_Z = torch.log(Z)
log_softmax_vals = shifted - log_Z

print(f"    log(Z) per sample:")
for i in range(BATCH):
    print(f"      \"{SENTENCES[i]}\": log({Z[i, 0].item():.4f}) = {log_Z[i, 0].item():+.4f}")
print()
print_matrix("log_softmax", log_softmax_vals.data)

# Verify: exp of log-softmax should sum to 1
probs = torch.exp(log_softmax_vals)
print(f"\n    Verification: exp(log_softmax) = probabilities (should sum to 1.0)")
print_matrix("probabilities", probs.data)
for i in range(BATCH):
    print(f"      \"{SENTENCES[i]}\" sum = {probs[i].sum().item():.6f} ✓")

# ── Step E: NLL — pick the correct class ─────────────────────────────

step_header("E", "NLL = −log P(correct class)")
print(f"""
    For each sample, we pick the log-prob of the TARGET class
    and negate it.  This is the per-sample loss.

    Lower log-prob of correct class → higher loss → model is wrong.
    Higher log-prob of correct class → lower loss → model is right.
""")

for i in range(BATCH):
    target_class = targets[i].item()
    cls_name = CLASS_NAMES[target_class]
    log_prob = log_softmax_vals[i, target_class].item()
    prob = probs[i, target_class].item()
    nll = -log_prob
    print(f"    \"{SENTENCES[i]}\" → target = \"{cls_name}\" (class {target_class})")
    print(f"      log P(\"{cls_name}\") = {log_prob:+.4f}")
    print(f"      P(\"{cls_name}\")     = {prob:.4f}  ({prob*100:.1f}%)")
    print(f"      NLL = −({log_prob:+.4f}) = {nll:+.4f}")
    print()

nll_per_sample = -torch.gather(log_softmax_vals, dim=1, index=targets.unsqueeze(1)).squeeze(1)
print_vector("nll_per_sample", nll_per_sample.data, decimals=4)

# ── Step F: mean reduction → scalar loss ─────────────────────────────

step_header("F", "Mean reduction → loss scalar")
print(f"""
    loss = mean(nll_per_sample)   — a single number!

    This scalar is what backward() will differentiate.
    Every parameter gets ∂loss/∂param via the chain rule.
""")

loss_manual = nll_per_sample.mean()
print(f"    loss = ({' + '.join(f'{nll_per_sample[i].item():.4f}' for i in range(BATCH))}) / {BATCH}")
print(f"         = {loss_manual.item():.4f}")

# ── Verify against PyTorch ───────────────────────────────────────────

loss_pytorch = nn.CrossEntropyLoss()(logits, targets)
diff = (loss_manual - loss_pytorch).abs().item()
print(f"""
    Verification against nn.CrossEntropyLoss:
      Manual loss:  {loss_manual.item():.10f}
      PyTorch loss: {loss_pytorch.item():.10f}
      Difference:   {diff:.2e}  {'✓ match!' if diff < 1e-5 else '✗ mismatch!'}
""")

random_loss = math.log(NUM_CLASSES)
print(f"    Random model baseline: −log(1/{NUM_CLASSES}) = {random_loss:.4f}")
print(f"    Our loss: {loss_pytorch.item():.4f}  "
      f"({'near random — untrained' if abs(loss_pytorch.item() - random_loss) < 0.5 else 'different from random'})")

# ║═══════════════════════════════════════════════════════════════════║
# ║ PHASE 3: BACKWARD PASS — loss.backward()                        ║
# ║═══════════════════════════════════════════════════════════════════║

section("PHASE 3 — BACKWARD PASS: loss.backward()")

step_header("3a", "Before backward()")
print(f"""
    Before calling loss.backward(), all .grad attributes are None.
    The computation graph exists (built during forward pass)
    but no gradients have been computed yet.

    X.grad is None?                 {X.grad is None}
    block.W_Q.weight.grad is None?  {block.W_Q.weight.grad is None}
    head.classifier.weight.grad?    {head.classifier.weight.grad is None}
""")

step_header("3b", "loss.backward() — the chain rule in action")
print(f"""
    loss.backward() walks the computation graph BACKWARDS:

    loss (scalar)
      ↑ ∂loss/∂loss = 1.0
      │
      ↑ mean: ∂loss/∂nll_i = 1/{BATCH}
      │
      ↑ NLL: ∂loss/∂log_softmax[i, target_i] = −1/{BATCH}
      │
      ↑ log-softmax: ∂/∂logits = softmax − one_hot(target)
      │              This is the famous "softmax gradient"!
      │
      ↑ classifier: ∂/∂W_cls, ∂/∂b_cls, ∂/∂cls_token
      │
      ↑ CLS pooling: gradient only flows to position 0
      │
      ↑ residual #2:  gradient SPLITS (identity + FFN branch)
      ↑ FFN:          ∂/∂ffn_linear2 → GELU → ∂/∂ffn_linear1
      ↑ LayerNorm #2: ∂/∂gamma, ∂/∂beta
      │
      ↑ residual #1:  gradient SPLITS again (identity + attn branch)
      ↑ W_O:          ∂/∂W_O
      ↑ attention:     weights @ V, softmax, Q @ K^T
      ↑ Q,K,V:        ∂/∂W_Q, ∂/∂W_K, ∂/∂W_V
      ↑ LayerNorm #1: ∂/∂gamma, ∂/∂beta
      │
      ↓
    ∂loss/∂X  (gradient of input)

    KEY INSIGHT: At each residual connection, the gradient gets a
    FREE HIGHWAY — the identity branch passes it straight through.
    This prevents vanishing gradients in deep networks!
""")

# Actually run backward
loss_pytorch.backward()

print(f"    After backward():")
print(f"    X.grad is None?                 {X.grad is None}")
print(f"    block.W_Q.weight.grad is None?  {block.W_Q.weight.grad is None}")
print(f"    head.classifier.weight.grad?    {head.classifier.weight.grad is None}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Gradient inspection with actual values
# ══════════════════════════════════════════════════════════════════════

step_header("4", "Gradient Inspection — actual values")

# ── Softmax gradient (the famous one) ────────────────────────────────
print(f"""
    The most important gradient: ∂loss/∂logits = softmax − one_hot

    This is elegant: the gradient is just "what the model predicted"
    minus "what the answer was".  If the model is already perfect,
    softmax = one_hot → gradient = 0 → no update needed!
""")

print(f"    Softmax probabilities (what model predicted):")
print_matrix("probs", probs.data)

one_hot = torch.zeros_like(logits.data)
for i in range(BATCH):
    one_hot[i, targets[i]] = 1.0
print(f"    One-hot targets:")
print_matrix("one_hot", one_hot)

grad_logits = (probs.data - one_hot) / BATCH  # mean reduction divides by N
print(f"    ∂loss/∂logits = (softmax − one_hot) / {BATCH}:")
print_matrix("∂L/∂logits", grad_logits)

# ── Show actual gradients from autograd ──────────────────────────────
print(f"\n    ── Gradient of INPUT X  ──")
print(f"    ∂loss/∂X tells us how each input value should change")
print(f"    to reduce the loss.  Shape: {tuple(X.grad.shape)}")
print()
print_tensor_3d("∂L/∂X", X.grad.data, [f'["{s}"]' for s in SENTENCES])

# ── Per-layer gradient norms ─────────────────────────────────────────
print(f"\n    ── Gradient norms per layer (L2 norm) ──")
print(f"    {'─' * 60}")

gradient_layers = [
    ("input X",              X.grad.norm().item()),
    ("layer_norm_1 (γ)",     block.layer_norm_1.weight.grad.norm().item()),
    ("layer_norm_1 (β)",     block.layer_norm_1.bias.grad.norm().item()),
    ("W_Q",                  block.W_Q.weight.grad.norm().item()),
    ("W_K",                  block.W_K.weight.grad.norm().item()),
    ("W_V",                  block.W_V.weight.grad.norm().item()),
    ("W_O",                  block.W_O.weight.grad.norm().item()),
    ("layer_norm_2 (γ)",     block.layer_norm_2.weight.grad.norm().item()),
    ("layer_norm_2 (β)",     block.layer_norm_2.bias.grad.norm().item()),
    ("ffn_linear1",          block.ffn_linear1.weight.grad.norm().item()),
    ("ffn_linear2",          block.ffn_linear2.weight.grad.norm().item()),
    ("classifier W",         head.classifier.weight.grad.norm().item()),
    ("classifier b",         head.classifier.bias.grad.norm().item()),
]

max_norm = max(n for _, n in gradient_layers) if gradient_layers else 1.0
max_name_len = max(len(n) for n, _ in gradient_layers)

for name, norm in gradient_layers:
    bar_len = int(norm / max_norm * 40) if max_norm > 0 else 0
    bar = "█" * bar_len
    print(f"      {name:<{max_name_len}}  L2={norm:.6f}  {bar}")

print(f"""
    ✓ Residual connections keep gradients flowing:
      If norms are similar across layers → healthy gradient flow.
      If early layers had much smaller norms → vanishing gradients.
""")

# ── Show a few actual gradient tensors ───────────────────────────────
print(f"    ── Classifier weight gradient (∂loss/∂W_cls) ──")
print(f"    Shape: ({NUM_CLASSES}, {D_MODEL}) — same shape as the weight!")
print_matrix("∂L/∂W_cls", head.classifier.weight.grad.data)

print(f"""
    Each gradient value ∂L/∂W[i,j] says:
      "If you increase W[i,j] by a tiny ε, the loss changes by ε × ∂L/∂W[i,j]"
      Positive gradient → increasing weight increases loss → decrease it!
      Negative gradient → increasing weight decreases loss → increase it!
""")

# ║═══════════════════════════════════════════════════════════════════║
# ║ PHASE 4: OPTIMIZER — AdamW step                                 ║
# ║═══════════════════════════════════════════════════════════════════║

section("PHASE 4 — OPTIMIZER: AdamW")

step_header("5a", "AdamW — how it works")
print(f"""
    backward() computes gradients but does NOT change any weights.
    The OPTIMIZER reads .grad and updates .data.

    AdamW maintains two buffers per parameter:
      m = momentum   (smoothed gradient direction)
      v = variance   (smoothed gradient magnitude²)

    For each parameter p with gradient g:
      m = β₁·m + (1−β₁)·g          β₁=0.9   (direction memory)
      v = β₂·v + (1−β₂)·g²         β₂=0.999 (scale memory)
      m̂ = m / (1−β₁ᵗ)              (bias correction)
      v̂ = v / (1−β₂ᵗ)              (bias correction)
      p = p − lr · m̂/(√v̂ + ε)      (adaptive update)
      p = p − lr · wd · p          (weight decay)

    Why AdamW over SGD?
      SGD:    w = w − lr × grad     (same step size for all params)
      AdamW:  adapts step size per parameter based on gradient history.
              Params with large gradients get smaller steps (stability).
              Params with small gradients get larger steps (speed).
""")

lr = 1e-3
weight_decay = 0.01

import itertools
all_params = list(itertools.chain(block.parameters(), head.parameters()))
optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

print(f"    lr           = {lr}")
print(f"    weight_decay = {weight_decay}")
print(f"    β₁ = 0.9,  β₂ = 0.999,  ε = 1e-8")
print(f"    Parameters:    {sum(p.numel() for p in all_params):,} values")

# ── Snapshot before ──────────────────────────────────────────────────

step_header("5b", "Before vs After optimizer.step()")

w_cls_before = head.classifier.weight.data.clone()
b_cls_before = head.classifier.bias.data.clone()
w_q_before = block.W_Q.weight.data.clone()

print(f"\n    ── BEFORE optimizer.step() ──")
print(f"\n    Classifier weight:")
print_matrix("W_cls_before", w_cls_before)
print(f"\n    Classifier bias:")
print_vector("b_cls_before", b_cls_before)
print(f"\n    W_Q (first 3 rows):")
print_matrix("W_Q_before[:3]", w_q_before[:3])

# ── The step ─────────────────────────────────────────────────────────
optimizer.step()

print(f"\n    ✓ optimizer.step() applied!")

print(f"\n    ── AFTER optimizer.step() ──")
print(f"\n    Classifier weight:")
print_matrix("W_cls_after", head.classifier.weight.data)
print(f"\n    Classifier bias:")
print_vector("b_cls_after", head.classifier.bias.data)
print(f"\n    W_Q (first 3 rows):")
print_matrix("W_Q_after[:3]", block.W_Q.weight.data[:3])

# ── Show the changes ─────────────────────────────────────────────────
step_header("5c", "Weight changes (|after − before|)")

cls_w_change = (head.classifier.weight.data - w_cls_before).abs()
cls_b_change = (head.classifier.bias.data - b_cls_before).abs()
w_q_change = (block.W_Q.weight.data - w_q_before).abs()

print()
print_matrix("Δ W_cls", cls_w_change)
print()
print_vector("Δ b_cls", cls_b_change)

print(f"""
    Every value changed by ≈ lr = {lr}  (AdamW's first step ≈ lr).
    The changes are SMALL but non-zero — that's one gradient step.
    Repeated over thousands of steps, these tiny nudges add up!
""")

# ── Zero gradients ───────────────────────────────────────────────────

step_header("5d", "Zero gradients — critical!")
print(f"""
    After optimizer.step(), we MUST reset all gradients.

    Without this, the NEXT backward() would ACCUMULATE:
      param.grad = old_grad + new_grad   ← WRONG!

    We want:
      param.grad = new_grad              ← CORRECT

    optimizer.zero_grad(set_to_none=True)
""")

optimizer.zero_grad(set_to_none=True)
print(f"    ✓ All gradients reset to None")
print(f"    W_Q.weight.grad is None?  {block.W_Q.weight.grad is None}")

# ── Verify loss decreased ────────────────────────────────────────────

step_header("5e", "Did the loss decrease?")

with torch.no_grad():
    with contextlib.redirect_stdout(io.StringIO()):
        enc_after = block(X)
    with contextlib.redirect_stdout(io.StringIO()):
        logits_after = head(enc_after)
    loss_after = nn.CrossEntropyLoss()(logits_after, targets)

print(f"""
    Loss BEFORE step: {loss_pytorch.item():.6f}
    Loss AFTER step:  {loss_after.item():.6f}
    Δ loss:           {loss_after.item() - loss_pytorch.item():+.6f}
""")

if loss_after.item() < loss_pytorch.item():
    print(f"    ✓ Loss decreased! One optimizer step moves us downhill.")
else:
    print(f"    ⚠ Loss didn't decrease on step 1 — this can happen with AdamW")
    print(f"      due to weight decay on the first step. Over multiple steps")
    print(f"      the loss WILL decrease.")

# ║═══════════════════════════════════════════════════════════════════║
# ║ PHASE 5: MULTI-STEP TRAINING LOOP                               ║
# ║═══════════════════════════════════════════════════════════════════║

section("PHASE 5 — TRAINING LOOP (20 steps, memorise the batch)")

print(f"""
    Now we repeat the full cycle: forward → loss → backward → step
    20 times on the SAME data.  The model should MEMORISE it.

    This proves the entire pipeline works end-to-end:
      Encoder Block → Classification Head → Cross-Entropy
      → backward() → AdamW.step()
""")

# Fresh start
torch.manual_seed(42)
block = ManualTransformerEncoderBlock(d_model=D_MODEL, num_heads=HEADS, d_ff=D_FF)
head = ManualClassificationHead(d_model=D_MODEL, num_classes=NUM_CLASSES)
x_train = X.data.clone().requires_grad_(True)
optimizer = torch.optim.AdamW(
    list(itertools.chain(block.parameters(), head.parameters())),
    lr=1e-2, weight_decay=0.01,
)

num_steps = 20
initial_loss = None

for step in range(num_steps):
    # Forward
    with contextlib.redirect_stdout(io.StringIO()):
        enc = block(x_train)
        step_logits = head(enc)

    # Loss
    step_loss = nn.CrossEntropyLoss()(step_logits, targets)
    if initial_loss is None:
        initial_loss = step_loss.item()

    # Backward
    step_loss.backward()

    # Optimizer
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Accuracy
    with torch.no_grad():
        preds = step_logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        accuracy = correct / BATCH

    if step % 2 == 0 or step == num_steps - 1:
        probs_step = torch.softmax(step_logits.detach(), dim=-1)
        conf = ", ".join(f'P("{CLASS_NAMES[targets[i].item()]}")={probs_step[i, targets[i]].item():.3f}'
                         for i in range(BATCH))
        preds_str = ", ".join(f'"{CLASS_NAMES[p]}"' for p in preds.tolist())
        print(f"    Step {step+1:>2}/{num_steps}  "
              f"loss={step_loss.item():.4f}  "
              f"accuracy={accuracy:.0%}  "
              f"preds=[{preds_str}]  {conf}")

targets_str = ", ".join(f'"{CLASS_NAMES[t]}"' for t in targets.tolist())
preds_str = ", ".join(f'"{CLASS_NAMES[p]}"' for p in preds.tolist())
print(f"""
    ── Training summary ──
    Initial loss: {initial_loss:.4f}  (random baseline: {random_loss:.4f})
    Final loss:   {step_loss.item():.4f}
    Final accuracy: {accuracy:.0%}
    Predictions: [{preds_str}]
    Targets:     [{targets_str}]
""")

if accuracy == 1.0:
    print(f'    ✓ 100% accuracy! "{SENTENCES[0]}" → "{CLASS_NAMES[targets[0]]}",')
    print(f'                      "{SENTENCES[1]}" → "{CLASS_NAMES[targets[1]]}"')
elif accuracy > 0:
    print(f"    ✓ Learning is happening — accuracy improved.")
else:
    print(f"    ⚠ May need more steps or higher learning rate.")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
    Complete training pipeline with tiny tensors:

    ┌─────────────────────── FORWARD ──────────────────────────┐
    │  X ({BATCH},{SEQ},{D_MODEL}) → Encoder Block → ({BATCH},{SEQ},{D_MODEL})                │
    │  → CLS pool → ({BATCH},{D_MODEL}) → Linear → logits ({BATCH},{NUM_CLASSES})       │
    └──────────────────────────────────────────────────────────┘
                             │
    ┌─────────── CROSS-ENTROPY (6 sub-steps) ──────────────────┐
    │  A. Subtract max     (stability: prevent exp overflow)   │
    │  B. Exponentiate     exp(shifted)                        │
    │  C. Partition Z      = Σ exp  (normalisation constant)   │
    │  D. Log-softmax      = shifted − log(Z)                  │
    │  E. NLL              = −log P(correct class)             │
    │  F. Mean             → loss (scalar)                     │
    │                                                          │
    │  Key gradient: ∂loss/∂logits = softmax − one_hot         │
    │  (If model is perfect, this is zero → no gradient!)      │
    └──────────────────────────────────────────────────────────┘
                             │
    ┌─────────────────── BACKWARD ─────────────────────────────┐
    │  loss.backward() — autograd chain rule through:          │
    │    loss → classifier → residual → FFN → LayerNorm →      │
    │    W_O → attention → Q,K,V → LayerNorm → X               │
    │                                                          │
    │  Residual connections: gradient highway → no vanishing!  │
    └──────────────────────────────────────────────────────────┘
                             │
    ┌─────────────────── OPTIMIZER ─────────────────────────────┐
    │  AdamW: m (momentum) + v (scale) → adaptive step         │
    │  Each param updated: p = p − lr·m̂/(√v̂+ε) − lr·wd·p      │
    │  Then zero_grad() to reset for next iteration            │
    └──────────────────────────────────────────────────────────┘

    Model: {total_params:,} params  |  Training: {initial_loss:.2f} → {step_loss.item():.4f} loss
    Result: {accuracy:.0%} accuracy in {num_steps} steps
""")
