"""
ASCII Art Presentation: Transformer Encoder Block — Step by Step
================================================================

Runs a REAL forward pass through ManualTransformerEncoderBlock
with tiny tensors so every value fits on screen.

    batch_size = 2    (two sentences)
    seq_len    = 5    (five tokens each)
    d_model    = 6    (embedding width)
    num_heads  = 3    (attention heads)
    d_k        = 2    (per-head dimension = 6 / 3)
    d_ff       = 12   (feed-forward hidden size)

Run:  uv run python3 -m core.presentation_transformer_block
"""

import torch
import torch.nn as nn
import math

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


def print_tensor_4d(name: str, t: torch.Tensor, batch_labels=None):
    """Print a 4D tensor: (batch, heads, rows, cols)."""
    b, h, r, c = t.shape
    for bi in range(b):
        bl = batch_labels[bi] if batch_labels else f"batch={bi}"
        for hi in range(h):
            print_matrix(f"{name}[{bl}, head={hi}]", t[bi, hi])


def section(title: str):
    print(f"\n{'━' * 70}")
    print(f"  {title}")
    print(f"{'━' * 70}")


def step_header(num: str, title: str):
    print(f"\n  ┌─── STEP {num} {'─' * (52 - len(num) - len(title))} {title} ───┐")


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

BATCH   = 2
SEQ     = 5
D_MODEL = 6
HEADS   = 3
D_K     = D_MODEL // HEADS   # = 2
D_FF    = 12

torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════════════
# Vocabulary — 16 real English words (same as presentation_mini_gpt)
# ══════════════════════════════════════════════════════════════════════

VOCAB_WORDS = [
    "<pad>", "the", "cat", "dog", "climbs", "sits", "a", "tree",
    "house", "on", "big", "small", "runs", "eats", "fish", "bird",
]

VOCAB = len(VOCAB_WORDS)
word2id = {w: i for i, w in enumerate(VOCAB_WORDS)}

SENTENCES = ["the cat climbs a tree", "the dog eats a fish"]
SENT_LABELS = [f'["{s}"]' for s in SENTENCES]
WORD_LABELS = [[w for w in s.split()] for s in SENTENCES]

# ══════════════════════════════════════════════════════════════════════
# Build the block (with real weights)
# ══════════════════════════════════════════════════════════════════════

section("ARCHITECTURE")
print(f"""
    batch_size  = {BATCH}   ("{SENTENCES[0]}", "{SENTENCES[1]}")
    seq_len     = {SEQ}   (five words per sentence)
    d_model     = {D_MODEL}   (embedding dimension)
    num_heads   = {HEADS}   (parallel attention heads)
    d_k         = {D_K}   (per-head dim = d_model / num_heads)
    d_ff        = {D_FF}  (FFN hidden size)

    Architecture (Pre-LN):

        Input X ──────────────────────────────── shape: (2, 5, 6)
          │
          ├──► LayerNorm ──► W_Q ──► Q          shape: (2, 5, 6)
          │                  W_K ──► K          reshape → (2, 5, 6)
          │                  W_V ──► V                    (2, 5, 6)
          │                    │
          │         ┌──────── Split into {HEADS} heads ────────┐
          │         │  Q,K,V: (2, {HEADS}, 5, {D_K})              │
          │         │                                         │
          │         │  scores = Q @ K^T / √{D_K}               │
          │         │  weights = softmax(scores)              │
          │         │  head_out = weights @ V                 │
          │         └──── Concatenate heads ──────────────────┘
          │                    │
          │                  W_O ──► attention_output  (2, 5, 6)
          │                    │
          ╰──── + ◄────────────╯   ← residual connection
               │
               x1  (2, 5, 6)
               │
               ├──► LayerNorm ──► FFN_linear1 ──► GELU ──► FFN_linear2
               │                                              │
               ╰──── + ◄──────────────────────────────────────╯
                     │
                   output  (2, 5, 6)
""")

# Create the block
block = nn.Module.__new__(nn.Module)  # we'll build manually for clarity
nn.Module.__init__(block)

# We use the real ManualTransformerEncoderBlock but intercept each step
from core.manual_transformer_block import ManualTransformerEncoderBlock
real_block = ManualTransformerEncoderBlock(
    d_model=D_MODEL,
    num_heads=HEADS,
    d_ff=D_FF,
)

# ══════════════════════════════════════════════════════════════════════
# Input tensor X
# ══════════════════════════════════════════════════════════════════════

section("INPUT X — Word Embeddings for Two Real Sentences")

token_embedding = nn.Embedding(VOCAB, D_MODEL)
input_ids = torch.tensor([[word2id[w] for w in s.split()] for s in SENTENCES])
X = token_embedding(input_ids).detach()

print(f"\n    Sentences:")
for si, sent in enumerate(SENTENCES):
    words = sent.split()
    ids = [word2id[w] for w in words]
    mapping = ", ".join(f'"{w}"={word2id[w]}' for w in words)
    print(f'      {si}: "{sent}"  →  {ids}  ({mapping})')
print(f"\n    X.shape = ({BATCH}, {SEQ}, {D_MODEL})")
print(f"    Each word ID → row in embedding table → {D_MODEL}-dim vector\n")
print_tensor_3d("X", X, dim_labels=SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LayerNorm
# ══════════════════════════════════════════════════════════════════════

step_header("1", "LayerNorm (Pre-LN)")
print("""
    For each token vector (length 6), compute:
        mean  = average of 6 values
        std   = standard deviation
        x_norm = (x - mean) / std  ×  γ  +  β

    This normalises each token independently so activations stay stable.
    γ (gamma) and β (beta) are learnable — initialized to 1 and 0.
""")

x_norm1 = real_block.layer_norm_1(X)
print(f"    x_norm1.shape = {tuple(x_norm1.shape)}")
print()

# Show one token in detail
tok = X[0, 0]
mean_val = tok.mean().item()
std_val = tok.std(unbiased=False).item()
print(f"    Example: X[\"{SENTENCES[0]}\", word \"{WORD_LABELS[0][0]}\"] = [{', '.join(fmt(v.item(), 5) for v in tok)}]")
print(f"    mean = {mean_val:+.3f},  std = {std_val:.3f}")
normed = (tok - mean_val) / (std_val + 1e-5)
print(f"    (x - mean) / std           = [{', '.join(fmt(v.item(), 5) for v in normed)}]")
print()
print_tensor_3d("x_norm1", x_norm1, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Linear projections Q, K, V
# ══════════════════════════════════════════════════════════════════════

step_header("2", "Q, K, V projections")
print(f"""
    Q = x_norm1 @ W_Q^T + b_Q     shape: ({BATCH}, {SEQ}, {D_MODEL})
    K = x_norm1 @ W_K^T + b_K     shape: ({BATCH}, {SEQ}, {D_MODEL})
    V = x_norm1 @ W_V^T + b_V     shape: ({BATCH}, {SEQ}, {D_MODEL})

    Each is a ({D_MODEL}→{D_MODEL}) linear transformation.
    Same input, three different projections:
      Q = "what am I looking for?"
      K = "what do I contain?"
      V = "what information do I carry?"
""")

Q = real_block.W_Q(x_norm1)
K = real_block.W_K(x_norm1)
V = real_block.W_V(x_norm1)

print(f"    W_Q.weight.shape = {tuple(real_block.W_Q.weight.shape)}  (transposed internally)")
print()

print("    ── Q (Queries) ──")
print_tensor_3d("Q", Q, SENT_LABELS)
print("    ── K (Keys) ──")
print_tensor_3d("K", K, SENT_LABELS)
print("    ── V (Values) ──")
print_tensor_3d("V", V, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Reshape into multiple heads
# ══════════════════════════════════════════════════════════════════════

step_header("3", "Split into heads")
print(f"""
    Reshape: ({BATCH}, {SEQ}, {D_MODEL}) → ({BATCH}, {SEQ}, {HEADS}, {D_K}) → transpose → ({BATCH}, {HEADS}, {SEQ}, {D_K})

    The {D_MODEL}-dim vector is split into {HEADS} heads of {D_K} dims each.
    Each head processes its own {D_K}-dimensional subspace independently.

    Visually for one sentence:
      token 0: [a b | c d | e f]  →  head 0: [a b]   head 1: [c d]   head 2: [e f]
      token 1: [g h | i j | k l]  →  head 0: [g h]   head 1: [i j]   head 2: [k l]
      token 2: [m n | o p | q r]  →  head 0: [m n]   head 1: [o p]   head 2: [q r]
      token 3: [s t | u v | w x]  →  head 0: [s t]   head 1: [u v]   head 2: [w x]
      token 4: [y z | A B | C D]  →  head 0: [y z]   head 1: [A B]   head 2: [C D]
""")

Q_heads = Q.view(BATCH, SEQ, HEADS, D_K).transpose(1, 2)
K_heads = K.view(BATCH, SEQ, HEADS, D_K).transpose(1, 2)
V_heads = V.view(BATCH, SEQ, HEADS, D_K).transpose(1, 2)

print(f"    Q_heads.shape = {tuple(Q_heads.shape)}  →  (batch, heads, seq, d_k)")
print()
print("    ── Q per head ──")
print_tensor_4d("Q", Q_heads, SENT_LABELS)
print("    ── K per head ──")
print_tensor_4d("K", K_heads, SENT_LABELS)
print("    ── V per head ──")
print_tensor_4d("V", V_heads, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Attention scores: Q @ K^T
# ══════════════════════════════════════════════════════════════════════

step_header("4", "Attention scores  Q @ K^T")
print(f"""
    For each (batch, head):
      scores = Q @ K^T    →  ({SEQ}, {D_K}) @ ({D_K}, {SEQ})  =  ({SEQ}, {SEQ})

    scores[i][j] = "how much should word i attend to word j?"

    For \"{SENTENCES[0]}\":
      scores[0][1] = how much does \"{WORD_LABELS[0][0]}\" attend to \"{WORD_LABELS[0][1]}\"?
""")

K_T = K_heads.transpose(-2, -1)
scores = torch.matmul(Q_heads, K_T)

print(f"    scores.shape = {tuple(scores.shape)}")
print()
print_tensor_4d("scores", scores, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Scale
# ══════════════════════════════════════════════════════════════════════

step_header("5", f"Scale by √d_k = √{D_K} = {math.sqrt(D_K):.2f}")
print(f"""
    scaled = scores / √{D_K}

    Without scaling, large dot products push softmax into saturation
    (all weight on one token, zero gradient for others).
    Dividing by √d_k keeps variance ≈ 1.
""")

scale = math.sqrt(D_K)
scaled = scores / scale

print_tensor_4d("scaled", scaled, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Softmax → attention weights
# ══════════════════════════════════════════════════════════════════════

step_header("6", "Softmax → attention weights")
print(f"""
    weights = softmax(scaled, dim=-1)

    Each ROW sums to 1.0 — it's a probability distribution:
    "how much should this word attend to each other word?"
""")

weights = torch.softmax(scaled, dim=-1)

print_tensor_4d("weights", weights, SENT_LABELS)

# Verify rows sum to 1
print("    ── Row sums (should all be 1.00) ──")
for bi in range(BATCH):
    for hi in range(HEADS):
        sums = [f"{weights[bi, hi, r].sum().item():.2f}" for r in range(SEQ)]
        print(f"      [\"{SENTENCES[bi]}\", head={hi}]: {sums}")

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — Weighted sum: weights @ V
# ══════════════════════════════════════════════════════════════════════

step_header("7", "Weighted sum of values  weights @ V")
print(f"""
    head_out = weights @ V    →  ({SEQ}, {SEQ}) @ ({SEQ}, {D_K})  =  ({SEQ}, {D_K})

    Each output token = weighted blend of all value vectors.
    The weights tell us HOW MUCH of each value to mix in.
""")

head_out = torch.matmul(weights, V_heads)

print(f"    head_out.shape = {tuple(head_out.shape)}")
print()
print_tensor_4d("head_out", head_out, SENT_LABELS)

# Show one detailed calculation
w0 = WORD_LABELS[0]
print(f'    ── Detailed: "{SENTENCES[0]}", head=0, word "{w0[0]}" ──')
print(f"    V vectors:")
for t in range(SEQ):
    v = V_heads[0, 0, t]
    print(f"      \"{w0[t]}\": [{fmt(v[0].item(), 5)}, {fmt(v[1].item(), 5)}]")
w = weights[0, 0, 0]
wts_str = ", ".join(fmt(w[t].item(), 5) for t in range(SEQ))
print(f'    attention weights for "{w0[0]}": [{wts_str}]')
result = head_out[0, 0, 0]
terms = " + ".join(f'{fmt(w[t].item(),5)} × V["{w0[t]}"]' for t in range(SEQ))
print(f'    output = {terms}')
print(f"           = [{fmt(result[0].item(), 5)}, {fmt(result[1].item(), 5)}]")

# ══════════════════════════════════════════════════════════════════════
# STEP 8 — Concatenate heads
# ══════════════════════════════════════════════════════════════════════

step_header("8", "Concatenate heads back")
print(f"""
    ({BATCH}, {HEADS}, {SEQ}, {D_K}) → transpose → ({BATCH}, {SEQ}, {HEADS}, {D_K}) → reshape → ({BATCH}, {SEQ}, {D_MODEL})

    The {HEADS} heads × {D_K} dims = {D_MODEL} dims — back to original width.
""")

concatenated = head_out.transpose(1, 2).contiguous().view(BATCH, SEQ, D_MODEL)

print(f"    concatenated.shape = {tuple(concatenated.shape)}")
print()
print_tensor_3d("concat", concatenated, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 9 — Output projection W_O
# ══════════════════════════════════════════════════════════════════════

step_header("9", "Output projection W_O")
print(f"""
    attention_output = concatenated @ W_O^T + b_O

    This mixes information ACROSS heads.
    Without W_O, each head's output stays in its own {D_K}-dim subspace.
""")

attn_output = real_block.W_O(concatenated)

print(f"    attn_output.shape = {tuple(attn_output.shape)}")
print()
print_tensor_3d("attn_out", attn_output, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 10 — Residual connection #1
# ══════════════════════════════════════════════════════════════════════

step_header("10", "Residual connection #1")
print(f"""
    x1 = X + attention_output

    The SKIP CONNECTION: add the original input back.
    This lets gradients flow directly through, preventing
    vanishing gradients in deep networks.

    Pre-LN formula:  x1 = X + Attention(LayerNorm(X))
""")

x1 = X + attn_output

print_tensor_3d("x1 = X + attn", x1, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 11 — LayerNorm #2
# ══════════════════════════════════════════════════════════════════════

step_header("11", "LayerNorm #2 (before FFN)")

x1_norm = real_block.layer_norm_2(x1)
print()
print_tensor_3d("x1_norm", x1_norm, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 12 — Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════

step_header("12", "Feed-Forward Network (FFN)")
print(f"""
    FFN processes each token INDEPENDENTLY (no cross-token interaction).

    Linear1:  ({D_MODEL} → {D_FF})    expand
    GELU:     element-wise        non-linearity
    Linear2:  ({D_FF} → {D_MODEL})    compress back
""")

ffn_hidden = real_block.ffn_linear1(x1_norm)
print(f"    After Linear1:  shape = {tuple(ffn_hidden.shape)}")
print_tensor_3d("ffn_hid", ffn_hidden, SENT_LABELS)

ffn_activated = real_block.ffn_gelu(ffn_hidden)
print(f"\n    After GELU:  shape = {tuple(ffn_activated.shape)}  (same shape, values clipped)")

ffn_output = real_block.ffn_linear2(ffn_activated)
print(f"\n    After Linear2:  shape = {tuple(ffn_output.shape)}  (back to d_model={D_MODEL})")
print_tensor_3d("ffn_out", ffn_output, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# STEP 13 — Residual connection #2
# ══════════════════════════════════════════════════════════════════════

step_header("13", "Residual connection #2")
print(f"""
    output = x1 + FFN(LayerNorm(x1))

    Same idea: add back the input to the FFN branch.
""")

output = x1 + ffn_output

print_tensor_3d("output", output, SENT_LABELS)

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
    Input:   X  ({BATCH}, {SEQ}, {D_MODEL})   ← "{SENTENCES[0]}", "{SENTENCES[1]}"
    Output:  Y  ({BATCH}, {SEQ}, {D_MODEL})   ← same shape, refined values

    What happened inside:

     X ──► LayerNorm ──► Q, K, V projections ──► Split {HEADS} heads
                                                      │
                ┌─── for each head (d_k={D_K}): ──────┘
                │    scores = Q·K^T / √{D_K}
                │    weights = softmax(scores)     ← ({SEQ}×{SEQ}) matrix per head
                │    head_out = weights · V
                └──► Concatenate ──► W_O
                                      │
     X ──────────────────── + ◄───────┘   (residual #1)
                            │
                           x1
                            │
                ├──► LayerNorm ──► FFN ({D_MODEL}→{D_FF}→{D_MODEL})
                │                         │
                └──────── + ◄─────────────┘   (residual #2)
                          │
                        output

    Key insight: shape NEVER changes — ({BATCH}, {SEQ}, {D_MODEL}) throughout.
    The block REFINES the representation, it doesn't change its structure.

    Total parameters in this block: {sum(p.numel() for p in real_block.parameters()):,}
""")

# ══════════════════════════════════════════════════════════════════════
# Verify against real block
# ══════════════════════════════════════════════════════════════════════

section("VERIFICATION")
# Silence the real block's prints for verification
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    real_output = real_block(X)

match = torch.allclose(output, real_output, atol=1e-5)
print(f"\n    Our step-by-step output matches real block: {'✓ YES' if match else '✗ NO'}")
print(f"    Max difference: {(output - real_output).abs().max().item():.2e}")
print()
