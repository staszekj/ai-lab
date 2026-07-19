# Glossary

**Repository:** [github.com/staszekj/ai-lab](https://github.com/staszekj/ai-lab)

---

**Token** — the basic unit of text for the model. Neither a word nor a character, but the result of the BPE algorithm. `MouseEventHandler` may be one or several tokens depending on the vocabulary.

**BPE (Byte-Pair Encoding)** — a vocabulary-building algorithm: starts with individual bytes, iteratively merges the most frequent adjacent pair into a new single token. The vocabulary reflects the statistical structure of the training corpus.

**Embedding** — a vector of numbers representing a token. The embedding table has shape `(vocab_size × d_model)` — each token is one row. The table is learnable: entries are updated by backpropagation.

**Positional embedding** — a second learnable table `(max_seq_len × d_model)`. Row `i` is added to the embedding of any token at position `i`. Without it the model is permutation-invariant and cannot distinguish word order.

**d_model** — the dimension of every representation vector throughout the model. Micro = 6, production = 256. All sub-layers preserve this dimension (their outputs have the same shape as their inputs).

**d_k** — the dimension per attention head: `d_model / num_heads`. Micro = 2, production = 32. Used as the scaling factor: `scores / √d_k`.

**d_ff** — the inner dimension of the Feed-Forward Network. Conventionally `4 × d_model`. Micro = 12, production = 1024.

**Tensor** — a generalisation of a matrix to more dimensions. A vector is a 1D tensor, a matrix is a 2D tensor. In PyTorch every operation works on tensors.

**Transpose** — swap two dimensions. `(4,5).T = (5,4)`. In code: `K.transpose(-2,-1)` swaps the last two dimensions.

**Q, K, V (Query, Key, Value)** — three independent linear projections of the same vector. Q = "what am I looking for", K = "what do I represent", V = "what I return if selected". Each is a learned `(d_model × d_model)` weight matrix.

**Scaled dot-product attention** — `softmax(Q @ Kᵀ / √d_k) @ V`. The scaling prevents softmax saturation when `d_k` is large.

**Self-attention** — Q, K, V all come from the same sequence. Each token computes similarity to every other token in the same sequence and mixes their value vectors proportionally.

**Cross-attention** — Q comes from the decoder; K and V come from the encoder output. The decoder can "query" any position of the encoded source sequence.

**Multi-head attention** — run scaled dot-product attention in parallel for `num_heads` subspaces of dimension `d_k`, then concatenate and project. Each head can learn to attend along a different dimension of meaning.

**Causal mask** — a `(seq_len × seq_len)` matrix of `-inf` on the upper triangle, added to scores before softmax. Ensures token at position `i` cannot attend to positions `i+1, i+2, …`. Required in the decoder for valid autoregressive generation.

**Residual connection** — `x_out = x_in + sublayer(x_in)`. Creates a gradient highway that allows gradients to flow unimpeded through many layers.

**Pre-LN** — applying LayerNorm before each sub-layer (`x_norm = LayerNorm(x)`, then compute the sub-layer on `x_norm`). Produces more stable gradients than the original Post-LN formulation used in "Attention Is All You Need".

**Feed-Forward Network (FFN)** — a two-layer MLP applied independently to each token position: `Linear(d_model, d_ff)` → GELU → `Linear(d_ff, d_model)`. Adds non-linear per-position transformation after attention mixes information across positions.

**Encoder** — reads the full source sequence with no mask (bidirectional). Produces `encoder_output (batch, src_len, d_model)` — the "memory" of the source, passed as K and V to every cross-attention layer.

**Decoder** — generates output one token at a time using causal self-attention (no future leakage) and cross-attention to the encoder.

**LM head** — a linear layer `(d_model → vocab_size, bias=False)` applied after the decoder final norm. Projects each decoder position to logits over the entire vocabulary.

**Logits** — raw, unnormalised scores before softmax. Shape `(batch, tgt_len, vocab_size)`. One number per vocabulary token per decoder position.

**Softmax** — converts logits to a probability distribution: all values ≥ 0, sum = 1. `softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)`.

**Jacobian** — the matrix of all partial derivatives of a vector-valued function. For softmax of size `n`, a dense `(n×n)` matrix. Dense because every output depends on every input: changing one logit shifts the normalisation for all others.

**GELU** — Gaussian Error Linear Unit. Smooth non-linearity used in FFN. Approximated as `x · Φ(x)` where `Φ` is the standard normal CDF. Similar to ReLU but with no sharp corner at zero.

**Autoregressive generation** — generating a sequence one token at a time, appending each generated token to the context before generating the next.

**Teacher forcing** — at training time, feed the model the correct previous tokens rather than its own predictions. Enables parallelised loss across all positions. Creates a train/inference gap (exposure bias).

**Exposure bias** — the mismatch between teacher forcing (always sees ground-truth context) and autoregressive inference (sees its own potentially wrong predictions). Can cause error accumulation at inference.

**KV-cache** — precomputed Key and Value tensors stored across generation steps. `precompute_cross_kv`: computed once from `encoder_output` before the generation loop. Self-attention KV grows by one step at each decode step.

**Cross-entropy loss** — `-log P(correct token)` averaged over all non-padding positions. A random model achieves `log(vocab_size)`.

**Teacher-forced accuracy (tf_acc)** — `argmax(logits) == tgt_target` measured during training with teacher-forced inputs. Does not measure autoregressive accuracy.

**Exact match (em)** — predicted string equals target string (whitespace-stripped). The strict metric.

**Validator pass (vp)** — `VALIDATORS[rule](prediction)` returns True. Measures whether the prediction has the correct structural shape for the rule. Only meaningful for positive (non-negative) pairs.

**Acceptable (acc)** — negative-aware composite: for positives it is `validator_pass`; for negatives it is `exact_match` (correctly preserved the already-degraded type).

**Macro accuracy** — average of per-rule acceptable rate. Weights all rules equally regardless of how many examples each has. The primary training and checkpoint metric.

**Micro accuracy** — `correct / total` across all pairs. Dominated by the most frequent rules.

**Gradient** — vector of all partial derivatives `∂loss/∂param`. Points toward steepest increase in loss; the optimiser steps the opposite way.

**Gradient clipping** — if the global gradient norm exceeds `max_grad_norm`, rescale all gradients proportionally. Prevents exploding gradients early in training when softmax outputs are still flat and loss landscape is steep.

**AdamW** — Adaptive Moment estimation with weight decay. Maintains per-parameter running mean (first moment) and running variance (second moment) of gradients. Effective learning rate per parameter = `lr / √(second_moment)`. Weight decay: `w -= weight_decay * w` at each step.

**Degradation** — programmatic widening of a precise type to a supertype according to a rule (e.g. `"on" | "off"` → `string`). Creates training pairs and defines what the model must undo.

**Hard negative** — a training pair where `degraded = typeText = target` and `isNegative = true`. The model must learn to output the input unchanged when it is already in the degraded shape.

**Validator** — a function checking whether the model's suggestion satisfies the structural constraint implied by a degradation rule. The inverse of the degradation. Filters hallucinations before writing back to source code.

**Hallucination** — a suggestion containing a non-trivial identifier (not a TypeScript builtin) that does not appear in the siblings string. The model invented a type name it had no evidence for in the context.

**Siblings** — a compact string summarising the other type annotations that share the same declaring scope as the target annotation. E.g. `[value: string, index: number, -> void]`. Provides in-context evidence for the target type. Must be identical at training and inference time.

**`@in:ContainingDecl` boost** — a prefix added to siblings for rules where the target identifier is typically declared in a parent scope (indexed access, utility types, `SetStateAction`). Gives the model the enclosing declaration's name as an additional signal.

**PROMPT_VERSION** — an integer stamped into the checkpoint and checked at inference time. Ensures the prompt format at training and inference match. Currently 2. Defined in both `prompt.py` and `degrade.ts` — both implementations must stay in sync.

**Content-hash split** — train/val assignment via `SHA1(input) % 100`. Guarantees identical prompts always land in the same split, preventing data leakage across the boundary.

**`train_batches` factory** — a zero-argument callable that returns a fresh iterable of batches on each call. Must be a factory (not a generator) so that re-shuffling happens at every epoch.

**`EncoderDecoderConfig`** — a dataclass of six integers (`vocab_size`, `max_seq_len`, `d_model`, `num_heads`, `d_ff`, `num_layers`) that fully determine all tensor shapes in the model. The only thing that must be stored alongside `state_dict` in a checkpoint for it to be reloadable.

**mean_logprob** — average of `log P(token | previous tokens)` over all tokens in the generated output. A confidence measure. Negative number — closer to zero means more confident. Random baseline: `log(1/vocab_size) ≈ -7.6` for vocab=2048.

**Partial derivative** — how much the output changes with respect to one variable while all others are held fixed. Notation: `∂L/∂w`.

**`EpochStats`** — a dataclass returned by each training epoch: `epoch`, `train_loss`, `train_tf_acc`, `val_metric`, `elapsed_s`. Passed to the `on_epoch_end` callback — the trainer never interprets `val_metric` itself.

**Rule-balanced sampler** — samples training pairs with weight `1/√count(rule)`. Prevents dominant rules from monopolising batches; tail rules are seen proportionally more often.

**`MUTED_RULES`** — a shared set of rule names disabled both in `degrade.ts` (no training pairs generated) and `validators.py` (no validator registered). Rules are muted when they have too few training examples to train reliably.

**Temperature** — a scalar dividing logits before softmax during generation: `logits / temperature`. Temperature < 1 → peaked distribution (near-greedy). Temperature > 1 → flat distribution (more random). `temperature=0.01` in `Predictor` default; `temperature=0.7` for `predict_n` to sample diverse candidates.
