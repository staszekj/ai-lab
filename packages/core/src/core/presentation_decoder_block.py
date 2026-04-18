"""
ASCII Art Presentation: Transformer Decoder Block — Step by Step
================================================================

Runs a REAL forward pass through ManualDecoderBlock with tiny tensors
so every value fits on screen.

    Source:   "the cat climbs the tree"  →  encoder_output  (1, 5, 6)
    Target:   "the dog eats a"         →  decoder input   (1, 4, 6)
    Predict:  "fish"                   ←  next token

    vocab_size   = 16    src_seq_len = 5
    tgt_seq_len  = 4     d_model     = 6
    num_heads    = 3     d_k         = 2     d_ff = 12

    KEY: the cross-attention score matrix is (4×5) — NOT square.
         Rows = decoder positions, Cols = encoder positions.

Run:  uv run python3 -m core.presentation_decoder_block
"""

import math
import torch
import torch.nn as nn
import io
import contextlib

# ══════════════════════════════════════════════════════════════════════
# Helpers — same convention as presentation_transformer_block.py
# ══════════════════════════════════════════════════════════════════════

def fmt(val: float, width: int = 7) -> str:
    s = f"{val:+.2f}"
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


def print_matrix_inf(name: str, t: torch.Tensor, indent: int = 4):
    """Print matrix, showing -inf instead of large negative numbers."""
    pad = " " * indent
    rows, cols = t.shape
    print(f"{pad}{name}  ({rows}×{cols})")
    print(f"{pad}┌{'─' * (8 * cols + 1)}┐")
    for r in range(rows):
        vals = []
        for c in range(cols):
            v = t[r, c].item()
            vals.append("   -inf" if v < -1e9 else fmt(v))
        print(f"{pad}│ {' '.join(vals)} │")
    print(f"{pad}└{'─' * (8 * cols + 1)}┘")


def print_tensor_3d(name: str, t: torch.Tensor, dim_labels=None, indent: int = 4):
    d0, d1, d2 = t.shape
    labels = dim_labels or [f"[{i}]" for i in range(d0)]
    for i in range(d0):
        print_matrix(f"{name}{labels[i]}", t[i], indent=indent)


def print_vector(name: str, t: torch.Tensor, indent: int = 4):
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

VOCAB    = 16
D_MODEL  = 6
HEADS    = 3
D_K      = D_MODEL // HEADS   # = 2
D_FF     = 12
BATCH    = 1
SRC_SEQ  = 5   # number of encoder output tokens
TGT_SEQ  = 4   # number of decoder input tokens (tgt so far)

torch.manual_seed(42)

VOCAB_WORDS = [
    "<pad>", "the", "cat", "dog", "climbs", "sits", "a", "tree",
    "house", "on", "big", "small", "runs", "eats", "fish", "bird",
]
word2id = {w: i for i, w in enumerate(VOCAB_WORDS)}
id2word  = VOCAB_WORDS

SRC_SENTENCE = "the cat climbs the tree"   # processed by the encoder
TGT_INPUT    = "the dog eats a"          # decoder has seen this so far
TGT_NEXT     = "fish"                    # decoder should predict this

# ══════════════════════════════════════════════════════════════════════
# Architecture overview
# ══════════════════════════════════════════════════════════════════════

section("ARCHITECTURE — Transformer Decoder Block")
print(f"""
    Source (encoder output): "{SRC_SENTENCE}"   ({BATCH}, {SRC_SEQ}, {D_MODEL})
    Target (decoder input):  "{TGT_INPUT}"         ({BATCH}, {TGT_SEQ}, {D_MODEL})
    Predict:                 "{TGT_NEXT}"

    A decoder block has THREE sub-layers (vs TWO in the encoder block):

    x  ({BATCH},{TGT_SEQ},{D_MODEL})
    │
    ├──► LayerNorm ──► SUB-LAYER 1: MASKED SELF-ATTENTION ──► + ──► x1
    │                  Q, K, V from x (decoder only)           │
    │                  Causal mask: future target = -inf        │ residual
    │                  Score shape: ({TGT_SEQ}×{TGT_SEQ})   (square)          │
    │                                                           │
    x1 ─────────────────────────────────────────────────────────
    │
    ├──► LayerNorm ──► SUB-LAYER 2: CROSS-ATTENTION ─────► + ──► x2
    │                  Q from x1 (decoder)                 │
    │                  K, V from encoder_output             │ residual
    │                  Score shape: ({TGT_SEQ}×{SRC_SEQ})   NOT square!       │
    │                  No mask — attend to all src tokens   │
    │                                                       │
    x2 ─────────────────────────────────────────────────────
    │
    └──► LayerNorm ──► SUB-LAYER 3: FFN ──────────────────► + ──► output
                       Linear({D_MODEL}→{D_FF}) → GELU → Linear({D_FF}→{D_MODEL})       │ residual
                                                               │
                                                               │
    ┌──────────────────────────────────────────────────────────────────┐
    │  GRADIENT FLOW (backward):                                      │
    │    dL/dself_W_Q,K,V → stays in decoder                          │
    │    dL/dcross_W_Q    → stays in decoder                          │
    │    dL/dcross_W_K,V  → flows back through encoder_output          │
    │                        → encoder parameters receive gradients!  │
    │    dL/dffn_weights  → stays in decoder                          │
    └──────────────────────────────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════
# Build the block and prepare inputs
# ══════════════════════════════════════════════════════════════════════

from core.manual_decoder_block import ManualDecoderBlock

block = ManualDecoderBlock(d_model=D_MODEL, num_heads=HEADS, d_ff=D_FF)

# Synthetic encoder output (in a real model this comes from the encoder stack)
torch.manual_seed(0)
encoder_output = torch.randn(BATCH, SRC_SEQ, D_MODEL)

# Decoder input embeddings (target tokens)
tgt_ids = torch.tensor([[word2id[w] for w in TGT_INPUT.split()]])
# Simple random-weight embeddings for this demo
embedding     = nn.Embedding(VOCAB, D_MODEL)
pos_embedding = nn.Embedding(20, D_MODEL)
pos_ids = torch.arange(TGT_SEQ)
x = embedding(tgt_ids) + pos_embedding(pos_ids)
# x: (1, 4, 6)

section("INPUTS")
print(f"""
    Source sentence:   "{SRC_SENTENCE}"
    Token IDs:          {[word2id[w] for w in SRC_SENTENCE.split()]}

    Encoder output (synthetic — would come from encoder stack in practice):
    Shape: ({BATCH}, {SRC_SEQ}, {D_MODEL})   ← src_len={SRC_SEQ}""")
print_tensor_3d("encoder_output", encoder_output, [f"[batch 0]"])

print(f"""
    Target seen so far: "{TGT_INPUT}"
    Token IDs:           {[word2id[w] for w in TGT_INPUT.split()]}
    Predict next:        "{TGT_NEXT}"

    Decoder input x (embeddings + positional):
    Shape: ({BATCH}, {TGT_SEQ}, {D_MODEL})   ← tgt_len={TGT_SEQ}""")
print_tensor_3d("x", x, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# SUB-LAYER 1 — Masked Self-Attention
# ══════════════════════════════════════════════════════════════════════

step_header("5.1", "Masked Self-Attention (decoder ↔ decoder, causal)")
print(f"""
    Q, K, V all come from x (the decoder input).
    Causal mask: each target token can only attend to itself and past tokens.
    This is IDENTICAL to the self-attention in GPT.

    Score matrix shape: ({TGT_SEQ}×{TGT_SEQ})  — square (tgt ↔ tgt)
""")

causal_mask = torch.triu(
    torch.ones(TGT_SEQ, TGT_SEQ) * float('-inf'), diagonal=1
)
tgt_words = TGT_INPUT.split()
print(f"    Causal mask ({TGT_SEQ}×{TGT_SEQ}) — rows and cols are target tokens:")
header = "              " + "   ".join(f'"{w[:4]}"' for w in tgt_words)
print(f"    {header}")
for r in range(TGT_SEQ):
    vals = ["  0  " if causal_mask[r, c] == 0 else " -inf" for c in range(TGT_SEQ)]
    print(f'    "{tgt_words[r]:<4}"   {"   ".join(vals)}')
print()

# LayerNorm
x_norm1 = block.layer_norm_1(x)
print(f"    After LayerNorm 1:")
print_tensor_3d("  x_norm1", x_norm1, [f"[batch 0]"])

# Q, K, V (all from decoder)
Q_s = block.self_W_Q(x_norm1)
K_s = block.self_W_K(x_norm1)
V_s = block.self_W_V(x_norm1)
print(f"\n    Q (from decoder): {Q_s.shape}")
print(f"    K (from decoder): {K_s.shape}")
print(f"    V (from decoder): {V_s.shape}  ← all same source (self-attention)")

# Split into heads
Q_sh = Q_s.view(BATCH, TGT_SEQ, HEADS, D_K).transpose(1, 2)
K_sh = K_s.view(BATCH, TGT_SEQ, HEADS, D_K).transpose(1, 2)
V_sh = V_s.view(BATCH, TGT_SEQ, HEADS, D_K).transpose(1, 2)
# (1, 3, 4, 2)
print(f"\n    After split into {HEADS} heads: each {tuple(Q_sh.shape)}")

# Scores
scores_s  = torch.matmul(Q_sh, K_sh.transpose(-2, -1)) / math.sqrt(D_K)
masked_s  = scores_s + causal_mask
weights_s = torch.softmax(masked_s, dim=-1)

print(f"\n    ── Head 0: scaled scores (Q·K^T / √{D_K}) ──")
print_matrix("  scores_self[b=0,h=0]", scores_s[0, 0])

print(f"\n    ── Head 0: after causal mask ──")
print_matrix_inf("  masked_self[b=0,h=0]", masked_s[0, 0])

print(f"\n    ── Head 0: attention weights after softmax ──")
print(f"    Future positions are exactly 0.00 — no information leaks forward!")
print_matrix("  weights_self[b=0,h=0]", weights_s[0, 0])

# Output
head_out_s  = torch.matmul(weights_s, V_sh)          # (1, 3, 4, 2)
concat_s    = head_out_s.transpose(1, 2).contiguous().view(BATCH, TGT_SEQ, D_MODEL)
self_attn_out = block.self_W_O(concat_s)             # (1, 4, 6)
x1 = x + self_attn_out

print(f"\n    Self-attention output (after concat + W_O):  {self_attn_out.shape}")
print(f"    After residual #1 (x1):                      {x1.shape}")
print_tensor_3d("  x1", x1, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# SUB-LAYER 2 — Cross-Attention
# ══════════════════════════════════════════════════════════════════════

step_header("5.2", "Cross-Attention (Q from decoder, K/V from encoder)")
print(f"""
    THIS IS THE NEW PART — UNIQUE TO THE DECODER BLOCK.

    Q comes from x1 (the decoder, after its own self-attention):
        shape: ({BATCH}, {TGT_SEQ}, {D_MODEL})   ← tgt_len = {TGT_SEQ}

    K and V come from encoder_output (the encoder's hidden states):
        shape: ({BATCH}, {SRC_SEQ}, {D_MODEL})   ← src_len = {SRC_SEQ}

    The score matrix Q @ K^T:
        ({BATCH}, {HEADS}, {TGT_SEQ}, {D_K}) @ ({BATCH}, {HEADS}, {D_K}, {SRC_SEQ})
        = ({BATCH}, {HEADS}, {TGT_SEQ}, {SRC_SEQ})

        Rows ({TGT_SEQ}) = decoder positions   → "which target token is asking?"
        Cols ({SRC_SEQ}) = encoder positions   → "which source token is being attended to?"
        NOT square because {TGT_SEQ} ≠ {SRC_SEQ}.

    No causal mask — the decoder can look at ALL source positions freely.
    (The source is fully processed; there's no autoregressive constraint here.)
""")

x1_norm = block.layer_norm_2(x1)
print(f"    After LayerNorm 2:")
print_tensor_3d("  x1_norm", x1_norm, [f"[batch 0]"])

# Q from decoder, K and V from encoder
Q_c = block.cross_W_Q(x1_norm)         # (1, TGT_SEQ=4, D_MODEL=6)
K_c = block.cross_W_K(encoder_output)  # (1, SRC_SEQ=5, D_MODEL=6)
V_c = block.cross_W_V(encoder_output)  # (1, SRC_SEQ=5, D_MODEL=6)

print(f"\n    Q (from decoder x1_norm): {Q_c.shape}")
print(f"    K (from encoder output):  {K_c.shape}  ← DIFFERENT source!")
print(f"    V (from encoder output):  {V_c.shape}  ← DIFFERENT source!")
print(f"\n    Q has {TGT_SEQ} rows, K/V have {SRC_SEQ} rows — cross-sequence attention.")

# Split into heads
Q_ch = Q_c.view(BATCH, TGT_SEQ, HEADS, D_K).transpose(1, 2)  # (1, 3, 4, 2)
K_ch = K_c.view(BATCH, SRC_SEQ, HEADS, D_K).transpose(1, 2)  # (1, 3, 5, 2)
V_ch = V_c.view(BATCH, SRC_SEQ, HEADS, D_K).transpose(1, 2)  # (1, 3, 5, 2)

print(f"\n    After split into heads:")
print(f"      Q_ch: {tuple(Q_ch.shape)}   [batch, heads, tgt_len, d_k]")
print(f"      K_ch: {tuple(K_ch.shape)}   [batch, heads, src_len, d_k]")
print(f"      V_ch: {tuple(V_ch.shape)}   [batch, heads, src_len, d_k]")

# Scores: (1, 3, 4, 2) @ (1, 3, 2, 5) = (1, 3, 4, 5)
scores_c  = torch.matmul(Q_ch, K_ch.transpose(-2, -1)) / math.sqrt(D_K)
weights_c = torch.softmax(scores_c, dim=-1)

print(f"\n    Cross-attention score matrix: {tuple(scores_c.shape)}  ← ({TGT_SEQ}×{SRC_SEQ}) NOT square!")
print(f"    After softmax (weights):       {tuple(weights_c.shape)}")

src_words = SRC_SENTENCE.split()

print(f"\n    ── Head 0: cross-attention SCORES  ({TGT_SEQ} target × {SRC_SEQ} source) ──")
print(f"    Rows = target tokens: {tgt_words}")
print(f"    Cols = source tokens: {src_words}")
print_matrix("  scores_cross[b=0,h=0]", scores_c[0, 0])

print(f"\n    ── Head 0: cross-attention WEIGHTS (softmax, dim=-1) ──")
print(f"    Each row is a probability distribution over the {SRC_SEQ} source tokens.")
print_matrix("  weights_cross[b=0,h=0]", weights_c[0, 0])

print(f"\n    Interpreting weights: which source token does each target token attend to?")
for t in range(TGT_SEQ):
    row = weights_c[0, 0, t]
    top = row.argmax().item()
    top2 = row.argsort(descending=True)[1].item()
    print(f"      \"{tgt_words[t]:<7}\" attends most to "
          f"\"{src_words[top]:<7}\" ({row[top].item():.3f})  "
          f"  2nd: \"{src_words[top2]:<7}\" ({row[top2].item():.3f})")

# Output
head_out_c    = torch.matmul(weights_c, V_ch)  # (1, 3, 4, 2)
concat_c      = head_out_c.transpose(1, 2).contiguous().view(BATCH, TGT_SEQ, D_MODEL)
cross_attn_out = block.cross_W_O(concat_c)
x2 = x1 + cross_attn_out

print(f"\n    Cross-attention output (after concat + W_O):  {cross_attn_out.shape}")
print(f"    After residual #2 (x2):                       {x2.shape}")
print_tensor_3d("  x2", x2, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# SUB-LAYER 3 — FFN
# ══════════════════════════════════════════════════════════════════════

step_header("5.3", "Feed-Forward Network (FFN)")
print(f"""
    Identical to the encoder's FFN.
    Applied independently to each of the {TGT_SEQ} target positions.
    Tokens do NOT interact with each other here.

    Structure:
      Linear({D_MODEL} → {D_FF})  — expand
      GELU            — non-linearity
      Linear({D_FF} → {D_MODEL})  — compress back
""")

x2_norm    = block.layer_norm_3(x2)
ffn_hidden = block.ffn_linear1(x2_norm)    # (1, 4, 12)
ffn_act    = block.ffn_gelu(ffn_hidden)
ffn_out    = block.ffn_linear2(ffn_act)    # (1, 4,  6)
output     = x2 + ffn_out

print(f"    After LayerNorm 3:         {x2_norm.shape}")
print(f"    After Linear({D_MODEL}→{D_FF}):    {ffn_hidden.shape}")
print(f"    After GELU:                {ffn_act.shape}")
print(f"    After Linear({D_FF}→{D_MODEL}):     {ffn_out.shape}")
print(f"    After residual #3 (output): {output.shape}")
print()
print_tensor_3d("  output", output, [f"[batch 0]"])

# ══════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════

section("VERIFICATION — match against ManualDecoderBlock.forward()")
print(f"""
    Running block.forward() to confirm manually computed values match.
""")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    output_ref = block.forward(x, encoder_output, tgt_mask=causal_mask)

match = torch.allclose(output, output_ref, atol=1e-5)
print(f"    Manual computation == block.forward():  {'✓ YES' if match else '✗ NO'}")
print(f"    Max absolute difference: {(output - output_ref).abs().max().item():.2e}")

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

section("SUMMARY — Decoder Block Data Flow")
print(f"""
    INPUTS:
      x (decoder)       {tuple(x.shape)}   "{TGT_INPUT}"
      encoder_output    {tuple(encoder_output.shape)}   "{SRC_SENTENCE}"

    SUB-LAYER 1: Masked Self-Attention
      x_norm1           {tuple(x_norm1.shape)}
      Q, K, V from x  — scores ({BATCH}×{HEADS}×{TGT_SEQ}×{TGT_SEQ}) — causal mask — softmax — weighted V
      self_attn_out     {tuple(self_attn_out.shape)}
      x1 = x + self_attn_out

    SUB-LAYER 2: Cross-Attention  ← THE KEY
      x1_norm           {tuple(x1_norm.shape)}
      Q from x1_norm  → ({BATCH}×{HEADS}×{TGT_SEQ}×{D_K})
      K from encoder  → ({BATCH}×{HEADS}×{SRC_SEQ}×{D_K})   ← different length!
      V from encoder  → ({BATCH}×{HEADS}×{SRC_SEQ}×{D_K})
      scores           ({BATCH}×{HEADS}×{TGT_SEQ}×{SRC_SEQ})  ← NOT square
      weights          softmax(scores, dim=-1)
      cross_attn_out    {tuple(cross_attn_out.shape)}
      x2 = x1 + cross_attn_out

    SUB-LAYER 3: FFN
      ffn_out           {tuple(ffn_out.shape)}
      output = x2 + ffn_out

    OUTPUT:  {tuple(output.shape)}

    ┌─────────────────────────────────────────────────────────────────┐
    │  ENCODER BLOCK vs DECODER BLOCK                                │
    │                                                                │
    │  EncoderBlock:  LayerNorm → Self-Attn (no mask) → residual    │
    │                 LayerNorm → FFN → residual                     │
    │                 Parameters: 4 attention matrices + FFN         │
    │                                                                │
    │  DecoderBlock:  LayerNorm → Masked Self-Attn → residual       │
    │                 LayerNorm → Cross-Attn (Q from dec,            │
    │                                         K/V from enc) → res.  │
    │                 LayerNorm → FFN → residual                     │
    │                 Parameters: 8 attention matrices + FFN         │
    │                             (2× more attention weights)        │
    └─────────────────────────────────────────────────────────────────┘
""")
