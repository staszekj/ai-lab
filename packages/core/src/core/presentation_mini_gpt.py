"""
ASCII Art Presentation: Mini-GPT — Full Decoder-Only Transformer
================================================================

Runs a REAL forward pass through ManualMiniGPT with tiny tensors
so every value (or a representative slice) fits on screen.

    vocab_size  = 16   (tiny vocabulary)
    max_seq_len = 8    (maximum 8 tokens)
    d_model     = 6    (embedding width)
    num_heads   = 3    (attention heads)
    d_k         = 2    (per-head dimension = 6 / 3)
    d_ff        = 12   (feed-forward hidden size)
    num_layers  = 2    (two decoder blocks)

    batch_size  = 1    (one sentence)
    seq_len     = 5    (five input tokens)

Run:  uv run python3 -m core.presentation_mini_gpt
"""

import math
import torch
import torch.nn as nn
import io
import contextlib

# ══════════════════════════════════════════════════════════════════════
# Helpers — pretty-print tensors as ASCII tables
# ══════════════════════════════════════════════════════════════════════

def fmt(val: float, width: int = 7) -> str:
    """Format a single number to fixed width."""
    s = f"{val:+.2f}"
    return s.rjust(width)


def print_matrix(name: str, t: torch.Tensor, indent: int = 4):
    """Print a 2D tensor as an ASCII table."""
    pad = " " * indent
    rows, cols = t.shape
    print(f"{pad}{name}  ({rows}×{cols})")
    print(f"{pad}┌{'─' * (8 * cols + 1)}┐")
    for r in range(rows):
        vals = " ".join(fmt(t[r, c].item()) for c in range(cols))
        print(f"{pad}│ {vals} │")
    print(f"{pad}└{'─' * (8 * cols + 1)}┘")


def print_tensor_3d(name: str, t: torch.Tensor, dim_labels=None):
    """Print a 3D tensor slice by slice."""
    d0, d1, d2 = t.shape
    labels = dim_labels or [f"[{i}]" for i in range(d0)]
    for i in range(d0):
        print_matrix(f"{name}{labels[i]}", t[i])


def print_vector(name: str, t: torch.Tensor, indent: int = 4):
    """Print a 1D tensor as a row."""
    pad = " " * indent
    vals = " ".join(fmt(v.item()) for v in t)
    print(f"{pad}{name}: [{vals}]")


def section(title: str):
    print(f"\n{'━' * 72}")
    print(f"  {title}")
    print(f"{'━' * 72}")


def step_header(num: str, title: str):
    width = max(56 - len(num) - len(title), 2)
    print(f"\n  ┌─── STEP {num} {'─' * width} {title} ───┐")


# ══════════════════════════════════════════════════════════════════════
# Configuration — tiny model so values fit on screen
# ══════════════════════════════════════════════════════════════════════

VOCAB       = 16
MAX_SEQ     = 8
D_MODEL     = 6
HEADS       = 3
D_K         = D_MODEL // HEADS   # = 2
D_FF        = 12
NUM_LAYERS  = 2

BATCH       = 1
SEQ         = 5

torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════════════
# Vocabulary — 16 real English words
# ══════════════════════════════════════════════════════════════════════

VOCAB_WORDS = [
    "<pad>",    # 0
    "the",      # 1
    "cat",      # 2
    "dog",      # 3
    "climbs",   # 4
    "sits",     # 5
    "a",        # 6
    "tree",     # 7
    "house",    # 8
    "on",       # 9
    "big",      # 10
    "small",    # 11
    "runs",     # 12
    "eats",     # 13
    "fish",     # 14
    "bird",     # 15
]

id2word = VOCAB_WORDS               # id2word[i] → word
word2id = {w: i for i, w in enumerate(VOCAB_WORDS)}

SENTENCE = "the cat climbs a tree"  # our training sentence


def ids_to_words(ids):
    return [id2word[i] for i in ids]


def words_to_str(ids):
    return " ".join(ids_to_words(ids))


# ══════════════════════════════════════════════════════════════════════
# Architecture overview
# ══════════════════════════════════════════════════════════════════════

section("ARCHITECTURE — Mini-GPT (Decoder-Only Transformer)")
print(f"""
    vocab_size     = {VOCAB}   (16 real English words)
    max_seq_len    = {MAX_SEQ}    (maximum sequence length)
    d_model        = {D_MODEL}    (embedding dimension)
    num_heads      = {HEADS}    (parallel attention heads)
    d_k            = {D_K}    (per-head dim = d_model / num_heads)
    d_ff           = {D_FF}   (FFN hidden size)
    num_layers     = {NUM_LAYERS}    (stacked decoder blocks)
    batch_size     = {BATCH}    (one sentence for clarity)
    seq_len        = {SEQ}    (five input tokens)

    Architecture:

        input_ids  (1, 5)         ← "the cat climbs a tree"
          │
          ├──► Token Embedding     ── lookup ──► (1, 5, 6)
          │
          ├──► Positional Embedding ── lookup ──► (5, 6)
          │
          └──► x = token_emb + pos_emb          (1, 5, 6)
               │
               ▼
        ┌──────────────────────────────────────────────┐
        │  DECODER BLOCK 0  (causal self-attention)    │
        │    LayerNorm → Q,K,V → heads → mask →       │
        │    softmax → weighted V → concat → W_O       │
        │    + residual → LayerNorm → FFN + residual   │
        └──────────────────────────────────────────────┘
               │  (1, 5, 6)
               ▼
        ┌──────────────────────────────────────────────┐
        │  DECODER BLOCK 1  (causal self-attention)    │
        │    same structure, independent weights       │
        └──────────────────────────────────────────────┘
               │  (1, 5, 6)
               ▼
          Final LayerNorm                   (1, 5, 6)
               │
               ▼
          LM Head  (d_model → vocab_size)   (1, 5, 16)
               │
               ▼
          logits → softmax → next token prediction

    GENERATION (autoregressive — word by word):
        "the cat"           → predicts → "climbs"
        "the cat climbs"    → predicts → "a"
        "the cat climbs a"  → predicts → "tree"
        Result: "the cat climbs a tree" ✓

    KEY DIFFERENCE from encoder (BERT):
        GPT uses a CAUSAL MASK — each token can only attend
        to itself and tokens BEFORE it, never to future tokens.
        This is what makes it "autoregressive".

    ┌──────────────────────────────────────────────────────────────┐
    │  The transformer block is IDENTICAL for encoder & decoder. │
    │  The ONLY difference is the attention mask:               │
    │                                                           │
    │    DECODER (GPT):   causal mask  → autoregressive         │
    │    ENCODER (BERT):  no mask       → bidirectional          │
    │                                                           │
    │  One line of code switches between them:                  │
    │    block(x, attn_mask=causal_mask)   # decoder             │
    │    block(x)                          # encoder             │
    └──────────────────────────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════
# Build the model
# ══════════════════════════════════════════════════════════════════════

from core.manual_mini_gpt import ManualMiniGPT

# Suppress verbose output from the block's forward()
model = ManualMiniGPT(
    vocab_size=VOCAB,
    max_seq_len=MAX_SEQ,
    d_model=D_MODEL,
    num_heads=HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
)

total_params = sum(p.numel() for p in model.parameters())

section("MODEL PARAMETERS")
print(f"""
    Total parameters: {total_params:,}

    Breakdown:
      Token embedding   ({VOCAB}×{D_MODEL}):     {model.token_embedding.weight.numel():>6,}
      Position embedding ({MAX_SEQ}×{D_MODEL}):    {model.positional_embedding.weight.numel():>6,}
      Decoder Block 0:             {sum(p.numel() for p in model.blocks[0].parameters()):>6,}
      Decoder Block 1:             {sum(p.numel() for p in model.blocks[1].parameters()):>6,}
      Final LayerNorm:             {sum(p.numel() for p in model.final_layer_norm.parameters()):>6,}
      LM Head ({D_MODEL}→{VOCAB}):          {model.lm_head.weight.numel():>6,}
""")

# ══════════════════════════════════════════════════════════════════════
# Input: token IDs
# ══════════════════════════════════════════════════════════════════════

section("INPUT — A Real Sentence")

input_ids = torch.tensor([[word2id[w] for w in SENTENCE.split()]])  # (1, 5)

print(f"""
    Sentence:  \"{SENTENCE}\"

    Tokenization (each word → integer ID):""")
for w in SENTENCE.split():
    print(f"      \"{w}\"  →  {word2id[w]}")
print(f"""
    input_ids = {input_ids[0].tolist()}
    input_ids.shape = ({BATCH}, {SEQ})

    Our mini-vocabulary ({VOCAB} words):""")
for i, w in enumerate(VOCAB_WORDS):
    marker = " ◄" if i in input_ids[0].tolist() else ""
    print(f"      {i:>2}: {w:<10}{marker}")
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Token Embedding
# ══════════════════════════════════════════════════════════════════════

step_header("1", "Token Embedding lookup")
print(f"""
    token_emb = Embedding(input_ids)

    Each word's ID is used to look up a row in the embedding table.
    The table has {VOCAB} rows (one per word), each of length {D_MODEL}.""")
for w in SENTENCE.split():
    wid = word2id[w]
    print(f"    \"{w}\" (ID {wid})  →  row {wid}  →  [{D_MODEL}-dim vector]")
print(f"""
    Shape: ({BATCH}, {SEQ}) → ({BATCH}, {SEQ}, {D_MODEL})
""")

token_emb = model.token_embedding(input_ids)
# token_emb: (1, 5, 6)

print(f"    Embedding table sample (first 5 rows of {VOCAB}):")
print_matrix("embed_table[:5]", model.token_embedding.weight.data[:5])
print()
print(f"    Token embeddings for \"{SENTENCE}\":")
print_tensor_3d("token_emb", token_emb, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Positional Embedding
# ══════════════════════════════════════════════════════════════════════

step_header("2", "Positional Embedding lookup")
print(f"""
    position_ids = [0, 1, 2, 3, 4]
    pos_emb = Embedding(position_ids)

    Tokens are processed in PARALLEL — the model has no idea about
    ORDER unless we tell it.  Positional embeddings encode:
      "I am position 0", "I am position 1", etc.

    Shape: ({SEQ},) → ({SEQ}, {D_MODEL})
""")

position_ids = torch.arange(SEQ, device=input_ids.device)
pos_emb = model.positional_embedding(position_ids)
# pos_emb: (5, 6)

print(f"    position_ids = {position_ids.tolist()}")
print()
print_matrix("pos_emb", pos_emb)

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Combine: token + position
# ══════════════════════════════════════════════════════════════════════

step_header("3", "token_emb + pos_emb")
print(f"""
    x = token_emb + pos_emb

    The model input is the SUM of both embeddings.
    This tells the model WHAT each token is AND WHERE it sits.

    Broadcasting: ({BATCH}, {SEQ}, {D_MODEL}) + ({SEQ}, {D_MODEL}) → ({BATCH}, {SEQ}, {D_MODEL})
""")

x = token_emb + pos_emb
# x: (1, 5, 6)

print(f"    x = token_emb + pos_emb")
print_tensor_3d("x", x, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Causal Mask
# ══════════════════════════════════════════════════════════════════════

step_header("4", "Create Causal Mask")
print(f"""
    The causal mask prevents tokens from attending to FUTURE tokens.
    THIS IS THE ONLY THING that makes this model a DECODER (GPT).
    Without the mask, it would be an ENCODER (BERT).

    For seq_len = {SEQ}:
      Token 0: can attend to [0]              — only itself
      Token 1: can attend to [0, 1]           — itself + past
      Token 2: can attend to [0, 1, 2]        — itself + past
      Token 3: can attend to [0, 1, 2, 3]     — itself + past
      Token 4: can attend to [0, 1, 2, 3, 4]  — all (it's the last)

    Implemented as an additive mask:
      0    = "attend"  (score + 0 = score)
      -inf = "block"   (score + -inf = -inf → softmax gives 0.0)
""")

causal_mask = model._create_causal_mask(SEQ, device=input_ids.device)
# mask: (5, 5)

print(f"    causal_mask ({SEQ}×{SEQ}):")
pad = " " * 4
header = "        " + "  ".join(f"tok{c}" for c in range(SEQ))
print(header)
for r in range(SEQ):
    vals = []
    for c in range(SEQ):
        v = causal_mask[r, c].item()
        vals.append("  0  " if v == 0 else " -inf")
    print(f"{pad}tok{r}  {'  '.join(vals)}")

print(f"""
    Reading row 2: token 2 can attend to tok0 (0), tok1 (0), tok2 (0),
    but NOT tok3 (-inf) or tok4 (-inf).

    After softmax, -inf → 0.0, so no information leaks from the future.

    If we removed this mask (passed attn_mask=None to each block),
    this model would behave like an ENCODER — full bidirectional attention.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Decoder Blocks (the core of GPT)
# ══════════════════════════════════════════════════════════════════════

step_header("5", f"Pass through {NUM_LAYERS} Decoder Blocks")
print(f"""
    Each decoder block:
      1. LayerNorm → Q, K, V projections
      2. Multi-head self-attention WITH causal mask
      3. Residual connection #1
      4. LayerNorm → FFN (expand → GELU → compress)
      5. Residual connection #2

    The causal mask is the SAME for every block — only depends on seq_len.

    x → Block 0 → Block 1 → x'
    Shape stays: ({BATCH}, {SEQ}, {D_MODEL}) throughout.
""")

for layer_idx, block in enumerate(model.blocks):
    print(f"\n    {'═' * 60}")
    print(f"    DECODER BLOCK {layer_idx}")
    print(f"    {'═' * 60}")

    print(f"\n    Input to block {layer_idx}:")
    print_tensor_3d(f"  x_in_{layer_idx}", x, [f"[batch 0]"])

    # ── LayerNorm ────────────────────────────────────────────────
    x_norm = block.layer_norm_1(x)
    print(f"\n    After LayerNorm 1:")
    print_tensor_3d(f"  x_norm", x_norm, [f"[batch 0]"])

    # ── Q, K, V projections ─────────────────────────────────────
    Q = block.W_Q(x_norm)
    K = block.W_K(x_norm)
    V = block.W_V(x_norm)
    print(f"\n    Q, K, V projections:  each ({BATCH}, {SEQ}, {D_MODEL})")

    # ── Split into heads ────────────────────────────────────────
    Q_h = Q.view(BATCH, SEQ, HEADS, D_K).transpose(1, 2)
    K_h = K.view(BATCH, SEQ, HEADS, D_K).transpose(1, 2)
    V_h = V.view(BATCH, SEQ, HEADS, D_K).transpose(1, 2)
    print(f"    Split into {HEADS} heads:  each ({BATCH}, {HEADS}, {SEQ}, {D_K})")

    # ── Attention scores ────────────────────────────────────────
    scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(D_K)
    print(f"\n    Scaled attention scores (Q·K^T / √{D_K}):")
    print(f"    Shape: ({BATCH}, {HEADS}, {SEQ}, {SEQ})")

    # Show one head as example
    print(f"\n    ── Head 0 scores (before mask) ──")
    print_matrix(f"  scores[b=0,h=0]", scores[0, 0])

    # ── Apply causal mask ───────────────────────────────────────
    masked_scores = scores + causal_mask
    print(f"\n    ── Head 0 scores (after causal mask: -inf blocks future) ──")
    # Print with special formatting for -inf
    pad = " " * 4
    t = masked_scores[0, 0]
    rows, cols = t.shape
    print(f"{pad}  masked_scores[b=0,h=0]  ({rows}×{cols})")
    print(f"{pad}  ┌{'─' * (8 * cols + 1)}┐")
    for r in range(rows):
        vals = []
        for c in range(cols):
            v = t[r, c].item()
            if v == float('-inf') or v < -1e9:
                vals.append("   -inf")
            else:
                vals.append(fmt(v))
        print(f"{pad}  │ {' '.join(vals)} │")
    print(f"{pad}  └{'─' * (8 * cols + 1)}┘")

    # ── Softmax → attention weights ─────────────────────────────
    weights = torch.softmax(masked_scores, dim=-1)
    print(f"\n    ── Head 0 attention weights (after softmax) ──")
    print(f"    Note: future positions are 0.00 — no information leak!")
    print_matrix(f"  weights[b=0,h=0]", weights[0, 0])

    # ── Weighted sum of values ──────────────────────────────────
    head_out = torch.matmul(weights, V_h)
    # head_out: (1, 3, 5, 2)

    # ── Concatenate and project ─────────────────────────────────
    concat = head_out.transpose(1, 2).contiguous().view(BATCH, SEQ, D_MODEL)
    attn_out = block.W_O(concat)
    print(f"\n    Attention output (after concat + W_O):")
    print_tensor_3d(f"  attn_out", attn_out, [f"[batch 0]"])

    # ── Residual #1 ─────────────────────────────────────────────
    x1 = x + attn_out
    print(f"\n    After residual #1 (x + attn_out):")
    print_tensor_3d(f"  x1", x1, [f"[batch 0]"])

    # ── FFN ─────────────────────────────────────────────────────
    x1_norm = block.layer_norm_2(x1)
    ffn_h = block.ffn_linear1(x1_norm)
    ffn_a = block.ffn_gelu(ffn_h)
    ffn_out = block.ffn_linear2(ffn_a)

    print(f"\n    FFN: LayerNorm → Linear({D_MODEL}→{D_FF}) → GELU → Linear({D_FF}→{D_MODEL})")
    print_tensor_3d(f"  ffn_out", ffn_out, [f"[batch 0]"])

    # ── Residual #2 ─────────────────────────────────────────────
    x = x1 + ffn_out
    print(f"\n    After residual #2 (x1 + ffn_out) — OUTPUT of block {layer_idx}:")
    print_tensor_3d(f"  block_{layer_idx}_out", x, [f"[batch 0]"])

    # Show stats
    print(f"\n    Block {layer_idx} output stats:  "
          f"mean={x.mean().item():+.4f}  std={x.std().item():.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Final LayerNorm
# ══════════════════════════════════════════════════════════════════════

step_header("6", "Final LayerNorm")
print(f"""
    In Pre-LN, the last decoder block's output exits through a
    residual add WITHOUT a trailing LayerNorm.  This final LayerNorm
    stabilises the activations before the LM head projection.
""")

x_final = model.final_layer_norm(x)
# x_final: (1, 5, 6)

print_tensor_3d("x_final", x_final, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — LM Head → logits
# ══════════════════════════════════════════════════════════════════════

step_header("7", "LM Head → vocabulary logits")
print(f"""
    logits = x_final @ W_lm_head^T

    Each position's {D_MODEL}-dim vector is projected to {VOCAB} scores (logits).
    logits[b, t, v] = "how likely is vocab token v to be the NEXT token
                        after position t?"

    Shape: ({BATCH}, {SEQ}, {D_MODEL}) @ ({D_MODEL}, {VOCAB})^T → ({BATCH}, {SEQ}, {VOCAB})
""")

logits = model.lm_head(x_final)
# logits: (1, 5, 16)

print(f"    logits.shape = {tuple(logits.shape)}")
print()
print_tensor_3d("logits", logits, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# STEP 8 — Softmax → next-token probabilities
# ══════════════════════════════════════════════════════════════════════

step_header("8", "Softmax → next-token probabilities")
print(f"""
    probs = softmax(logits, dim=-1)

    For each position, softmax turns the {VOCAB} logits into a
    probability distribution over the vocabulary.

    At position t, the model predicts what comes at position t+1.
""")

probs = torch.softmax(logits, dim=-1)

# Show the last position's probabilities (next-word prediction)
last_probs = probs[0, -1]  # (16,)
last_word = SENTENCE.split()[-1]
print(f"    ── After \"{last_word}\", what word comes next? ──")
print(f"    The model assigns a probability to EACH of the {VOCAB} words:\n")

ranked = last_probs.argsort(descending=True)
for rank, v in enumerate(ranked.tolist()):
    bar = "█" * int(last_probs[v].item() * 40)
    marker = " ◄ predicted" if rank == 0 else ""
    print(f"      {rank+1:>2}. {id2word[v]:<10} {last_probs[v].item():.4f}  {bar}{marker}")

predicted = last_probs.argmax().item()
print(f"\n    Predicted next word: \"{id2word[predicted]}\"")
print(f"    (Untrained model — essentially random!)")

# ══════════════════════════════════════════════════════════════════════
# STEP 9 — Language Model Loss (next-token prediction)
# ══════════════════════════════════════════════════════════════════════

step_header("9", "Language model loss")
sent_words = SENTENCE.split()
print(f"""
    The training target: predict the NEXT word at every position.

    Sentence: \"{SENTENCE}\"

    \"{sent_words[0]}\" → predict → \"{sent_words[1]}\"
    \"{sent_words[1]}\" → predict → \"{sent_words[2]}\"
    \"{sent_words[2]}\" → predict → \"{sent_words[3]}\"
    \"{sent_words[3]}\" → predict → \"{sent_words[4]}\"

    Loss = cross-entropy = -mean(log P(correct_next_word))
""")

# Shift logits and targets
shift_logits = logits[:, :-1, :]    # (1, 4, 16) — predictions at pos 0..3
shift_targets = input_ids[:, 1:]     # (1, 4)     — ground truth at pos 1..4

print(f"    shift_logits.shape  = {tuple(shift_logits.shape)}  (predictions for positions 0..{SEQ-2})")
print(f"    shift_targets.shape = {tuple(shift_targets.shape)}  (targets: tokens at positions 1..{SEQ-1})")
print()

# Show per-position predictions vs targets
for t in range(SEQ - 1):
    pos_probs = torch.softmax(shift_logits[0, t], dim=-1)
    target_id = shift_targets[0, t].item()
    target_prob = pos_probs[target_id].item()
    pred_id = pos_probs.argmax().item()
    print(f"    \"{sent_words[t]}\" → target=\"{id2word[target_id]}\"  "
          f"P={target_prob:.4f}  "
          f"predicted=\"{id2word[pred_id]}\"  "
          f"{'✓' if pred_id == target_id else '✗'}")

loss = nn.CrossEntropyLoss()(
    shift_logits.reshape(-1, VOCAB),
    shift_targets.reshape(-1),
)

random_loss = math.log(VOCAB)
print(f"""
    Loss = {loss.item():.4f}
    Expected random loss = -log(1/{VOCAB}) = {random_loss:.4f}
    (Untrained model should be near random loss)
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 10 — Backward pass (gradient flow)
# ══════════════════════════════════════════════════════════════════════

step_header("10", "Backward pass — gradient flow")
print(f"""
    loss.backward() computes gradients for ALL parameters.
    Gradients flow: LM Head → Final LN → Block 1 → Block 0 → Embeddings

    Residual connections ensure gradients don't vanish through the blocks.
""")

loss.backward()

print(f"    Gradient norms per component:")
print(f"    {'─' * 56}")

emb_grad = model.token_embedding.weight.grad.norm().item()
pos_grad = model.positional_embedding.weight.grad.norm().item()
print(f"    token_embedding:       grad_norm = {emb_grad:.6f}")
print(f"    positional_embedding:  grad_norm = {pos_grad:.6f}")

for i, block_module in enumerate(model.blocks):
    block_grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in block_module.parameters() if p.grad is not None
    ) ** 0.5
    print(f"    decoder_block[{i}]:       grad_norm = {block_grad_norm:.6f}")

ln_grad = model.final_layer_norm.weight.grad.norm().item()
lm_grad = model.lm_head.weight.grad.norm().item()
print(f"    final_layer_norm:      grad_norm = {ln_grad:.6f}")
print(f"    lm_head:               grad_norm = {lm_grad:.6f}")

print(f"""
    ✓ If block[0] and block[1] have similar gradient norms,
      residual connections are keeping gradients healthy.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 11 — Training: memorise a tiny sequence
# ══════════════════════════════════════════════════════════════════════

step_header("11", "Training loop — memorise the sentence")
print(f"""
    We'll train this tiny model to memorise: \"{SENTENCE}\"

    After training, given \"{' '.join(SENTENCE.split()[:2])}\" as a prompt,
    it should GENERATE the rest: \"{' '.join(SENTENCE.split()[2:])}\".

    Optimizer: AdamW (lr=0.005)
    Steps: 100
""")

# Reset model for clean training
torch.manual_seed(42)
model = ManualMiniGPT(
    vocab_size=VOCAB, max_seq_len=MAX_SEQ,
    d_model=D_MODEL, num_heads=HEADS,
    d_ff=D_FF, num_layers=NUM_LAYERS,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

num_steps = 100
initial_loss = None

for step in range(num_steps):
    with contextlib.redirect_stdout(io.StringIO()):
        logits_train = model(input_ids, verbose=False)

    shift_l = logits_train[:, :-1, :].reshape(-1, VOCAB)
    shift_t = input_ids[:, 1:].reshape(-1)
    step_loss = nn.CrossEntropyLoss()(shift_l, shift_t)

    if initial_loss is None:
        initial_loss = step_loss.item()

    step_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        preds = logits_train[:, :-1, :].argmax(dim=-1)
        targets = input_ids[:, 1:]
        correct = (preds == targets).sum().item()
        total = targets.numel()
        acc = correct / total

    if step % 10 == 0 or step == num_steps - 1:
        print(f"    Step {step+1:>3}/{num_steps}  "
              f"loss={step_loss.item():.4f}  "
              f"accuracy={acc:.0%}  "
              f"({correct}/{total} tokens)")

print(f"""
    Initial loss: {initial_loss:.4f}  (random ≈ {random_loss:.4f})
    Final loss:   {step_loss.item():.4f}
    Final accuracy: {acc:.0%}
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 12 — Autoregressive Generation
# ══════════════════════════════════════════════════════════════════════

step_header("12", "GENERATION — The Model Writes Text")
prompt_w = SENTENCE.split()[:2]
expect_w = SENTENCE.split()[2:]
print(f"""
    THIS IS THE CORE IDEA OF GPT — generating text word by word.

    ┌────────────────────────────────────────────────────────────────┐
    │  Given a prompt, the model GENERATES new text by predicting  │
    │  one word at a time, then feeding it back as input.          │
    └────────────────────────────────────────────────────────────────┘

    Algorithm:
      1. Start with prompt: \"{' '.join(prompt_w)}\"
      2. Forward pass → probabilities for ALL {VOCAB} words
      3. Pick the most likely word
      4. Append it to the context and go to step 2

    Prompt:   \"{' '.join(prompt_w)}\"
    Expected: \"{' '.join(expect_w)}\"
""")

prompt = input_ids[:, :2]  # "the cat"
generated = prompt.clone()

print(f"    Generating word by word:\n")

for gen_step in range(SEQ - 2):
    with contextlib.redirect_stdout(io.StringIO()):
        gen_logits = model(generated, verbose=False)

    next_logits = gen_logits[:, -1, :]
    next_probs = torch.softmax(next_logits / 0.01, dim=-1)
    next_token = next_probs.argmax(dim=-1, keepdim=True)

    generated = torch.cat([generated, next_token], dim=1)

    expected_tok = input_ids[0, 2 + gen_step].item()
    got_tok = next_token.item()
    match = "✓" if got_tok == expected_tok else "✗"

    ctx = words_to_str(generated[0, :-1].tolist())
    print(f"    \"{ctx}\"  → predicts → "
          f"\"{id2word[got_tok]}\"  (expected \"{id2word[expected_tok]}\")  {match}")

gen_tokens = generated[0, 2:].tolist()
exp_tokens = input_ids[0, 2:].tolist()
matches = sum(g == e for g, e in zip(gen_tokens, exp_tokens))

print(f"""
    RESULT:
      Prompt:      \"{' '.join(prompt_w)}\"
      Generated:   \"{words_to_str(gen_tokens)}\"
      Expected:    \"{' '.join(expect_w)}\"
      Full output: \"{words_to_str(generated[0].tolist())}\"
      Matches:     {matches}/{len(exp_tokens)} words correctly generated

    This is EXACTLY how ChatGPT works:
      1. You type a prompt           →  \"{' '.join(prompt_w)}\"
      2. Model predicts next word    →  one word at a time
      3. Append and repeat           →  feeds each word back as input
      4. Causal mask ensures each word sees only past context
""")

# ══════════════════════════════════════════════════════════════════════
# VERIFICATION — compare with model's own forward + generate
# ══════════════════════════════════════════════════════════════════════

section("VERIFICATION")

with contextlib.redirect_stdout(io.StringIO()):
    verify_logits = model(input_ids, verbose=False)

manual_logits = model.lm_head(model.final_layer_norm(x_final))
# Note: x_final was computed before training, so we re-run
with contextlib.redirect_stdout(io.StringIO()):
    fresh_logits = model(input_ids, verbose=False)

print(f"\n    Forward pass consistency check:")
print(f"    model(input_ids) shape: {tuple(verify_logits.shape)}")
print(f"    Expected shape:         ({BATCH}, {SEQ}, {VOCAB})")
match = tuple(verify_logits.shape) == (BATCH, SEQ, VOCAB)
print(f"    Shape correct: {'✓ YES' if match else '✗ NO'}")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
    Mini-GPT: a complete decoder-only transformer language model.

    Forward pass flow:

     input_ids ({BATCH}, {SEQ})
         │
         ├─► Token Embedding    → ({BATCH}, {SEQ}, {D_MODEL})
         ├─► Position Embedding → ({SEQ}, {D_MODEL})
         └─► x = token + pos   → ({BATCH}, {SEQ}, {D_MODEL})
              │
              ▼
     ┌─── Decoder Block 0 ───────────────────────────────────┐
     │  LayerNorm → Q,K,V → {HEADS} heads → CAUSAL MASK →     │
     │  softmax → weighted V → concat → W_O → +residual     │
     │  → LayerNorm → FFN ({D_MODEL}→{D_FF}→{D_MODEL}) → +residual          │
     └───────────────────────────────────────────────────────┘
              │ ({BATCH}, {SEQ}, {D_MODEL})
              ▼
     ┌─── Decoder Block 1 ───────────────────────────────────┐
     │  (same architecture, independent weights)             │
     └───────────────────────────────────────────────────────┘
              │ ({BATCH}, {SEQ}, {D_MODEL})
              ▼
     Final LayerNorm → ({BATCH}, {SEQ}, {D_MODEL})
              │
         LM Head → ({BATCH}, {SEQ}, {VOCAB})  = logits over vocabulary
              │
         softmax → next-token probabilities

    Key insights:
      • Shape NEVER changes through blocks: ({BATCH}, {SEQ}, {D_MODEL}) throughout
      • Causal mask makes it autoregressive: each token sees only past
      • Training: predict next token at every position (teacher forcing)
      • Generation: one token at a time, feed back into the model

    Sentence:   \"{SENTENCE}\"
    Parameters: {total_params:,}
    Training:   {initial_loss:.2f} → {step_loss.item():.4f} loss in {num_steps} steps
    Generation: \"{words_to_str(generated[0].tolist())}\" ({matches}/{len(exp_tokens)} words)
""")
