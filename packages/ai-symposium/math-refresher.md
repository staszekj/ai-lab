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
    ↑___↑
  must match!
            ↑               ↑
         from A           from B
```
Example in attention: Q(seq×d_k) @ Kᵀ(d_k×seq) = scores(seq×seq)

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
