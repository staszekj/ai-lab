# Math Refresher

## Matrix Multiplication — shapes matter, not the arithmetic

**Vector · Vector = Scalar** (dot product)
```
[a  b  c]  ·   [x]  =  ax + by + cz  =  42
               [y]
               [z]

(1×3)      (3×1)         (1×1) = scalar
```
In transformers this appears everywhere as Q·Kᵀ per head — one score per token pair.

---

**Matrix × Vector = Vector**
```
⎡ a  b  c ⎤   ⎡x⎤   ⎡ ax+by+cz ⎤
⎢ d  e  f ⎥ × ⎢y⎥ = ⎢ dx+ey+fz ⎥
⎣ g  h  i ⎦   ⎣z⎦   ⎣ gx+hy+iz ⎦

  (3×3)      (3×1)      (3×1)
```
Example: embedding lookup — weight matrix (vocab × d_model) × one-hot vector = one row.

---

**Matrix × Matrix = Matrix** — inner dimensions must match
```
  A         B          C
(m × k) × (k × n)  =  (m × n)
     ↑_____↑
  must match!
            ↑               ↑
         from A           from B
```
Example in attention: Q(seq×d_k) @ Kᵀ(d_k×seq) = scores(seq×seq)

---

## PyTorch Tensor Notation — (batch, [heads], sequence, vocab)

When working with transformers, we often describe tensor shapes using this pattern:

**`(batch, seq_len, vocab_size)`** — standard seq2seq model output
```
batch     = number of examples in the mini-batch (e.g., 32)
seq_len   = sequence length (e.g., 128 tokens)
vocab_size= number of tokens in vocabulary (e.g., 50000)

Example: logits from decoder
  shape (32, 128, 50000)
  logits[0, 5, 3] = raw score for "is token 3 likely at position 5 in example 0?"
```

**`(batch, num_heads, seq_len, head_dim)`** — attention heads split up the computation
```
batch     = mini-batch size
num_heads = how many parallel attention heads (e.g., 8)
seq_len   = sequence length
head_dim  = d_model / num_heads (e.g., 512 / 8 = 64)

Example: query vectors before attention
  shape (32, 8, 128, 64)
  Each of 32×8 = 256 separate attention heads processes 128 tokens independently.
  Later: Q @ K.T produces attention scores shape (32, 8, 128, 128)
         one score per (token_i, token_j) pair in each head.
```

**`[heads]` means optional** — some layers don't split heads:
```
(batch, seq_len, vocab_size)           — no heads (e.g., final output layer)
(batch, num_heads, seq_len, head_dim)  — with heads (e.g., attention layer)
```

**Real example: a training mini-batch through the model:**
```
Input (tokens):
  src_ids  shape (batch=2, src_len=4)
  tgt_in   shape (batch=2, tgt_len=3)

Embeddings:
  src_embed = (2, 4, d_model=512)
  tgt_embed = (2, 3, 512)

After encoder self-attention (with 8 heads):
  Q, K, V  each shape (2, 8, 4, 64)        ← split into 8 heads
  scores   shape (2, 8, 4, 4)              ← attention: which source positions matter?
  context  shape (2, 8, 4, 64)             ← weighted average of values
  output   shape (2, 4, 512)               ← concatenate heads back

Final logits:
  shape (2, 3, 50000)                      ← ready for cross-entropy loss
```

---

## Derivatives — gradient = slope = "how steep is the hill?"

**Ordinary derivative** — one input, one output:
```
f(x) = x²    →    f'(x) = 2x

  f(x)
   |        /
   |      /       steep here → large gradient
   |    /         → big weight update during training
   |  /
   |/____________
          x
```
If f'(x) = 0 → flat → model stopped learning (local minimum or saddle point).

---

**Partial derivative** — many inputs, differentiate w.r.t. one, hold the rest fixed:
```
f(x, y) = x² + 3y

∂f/∂x = 2x   (treat y as constant)
∂f/∂y = 3    (treat x as constant)
```
In a neural network every weight wᵢ gets its own ∂Loss/∂wᵢ — that's what backprop computes.

---

**Jacobian** — when both input AND output are vectors:
```
input:  x = [x₁, x₂, x₃]    (3 values)
output: f = [f₁, f₂]         (2 values)

Jacobian J = ⎡ ∂f₁/∂x₁  ∂f₁/∂x₂  ∂f₁/∂x₃ ⎤   shape: (2 × 3)
             ⎣ ∂f₂/∂x₁  ∂f₂/∂x₂  ∂f₂/∂x₃ ⎦

Each row = "how does output fᵢ change when I nudge each input?"
```
In transformers: the Jacobian of the attention layer tells us how gradients flow back through Q, K, V projections during training.
