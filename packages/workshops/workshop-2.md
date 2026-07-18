# Training 2 — Inside the Model
### 2 days · 7 h each

**Repository:** [github.com/staszekj/ai-lab](https://github.com/staszekj/ai-lab)

## Abstract

This training is a ground-up walkthrough of a complete encoder-decoder Transformer, from raw token IDs to trained weights. The same TypeScript type refinement system from Training 1 provides the running example, but again **the domain is incidental**. The model, the trainer, and the checkpoint module in `packages/ts-type-refiner` know nothing about TypeScript — they are entirely domain-agnostic. The type-system specifics that made Training 1 concrete (validators, prompt format, degradation rules) are not revisited here.

The subject of this training is the model itself:

- **Day 1 — Forward pass.** We build a micro-model with parameters small enough that every intermediate matrix fits on a terminal screen (`d_model=6`, `vocab=12`, 3 heads, 2 layers — the same values used in the inline comments of `encoder_decoder_model.py`). We then trace a single example through every operation: embedding lookup, multi-head self-attention with scaled dot-product scores, the causal mask, cross-attention between encoder and decoder, the feed-forward network, and the LM head that maps hidden states to a probability distribution over the vocabulary.

- **Day 2 — Backward pass and training.** We observe how `loss.backward()` propagates gradients through the entire graph, including back into the encoder which has no loss of its own. We examine why gradient clipping matters, how AdamW's adaptive per-parameter learning rates work, what teacher forcing is and what price it extracts at inference time. We then run a full training loop on the single micro-model example and watch the cross-attention weights shift from random to purposeful. The day closes with a comparison between the micro-model's six configuration numbers and the production model's, and with the evaluation metrics that tell us whether the trained model is actually useful.

The math refresher in Training 1 is assumed knowledge here. No TypeScript knowledge is needed. A working understanding of Python and basic familiarity with PyTorch (tensors, `nn.Module`) is sufficient.

**Prerequisites:** Training 1 (or equivalent familiarity with the inference pipeline), basic PyTorch.

---

## Day 1 — Forward Pass

### The Micro-Model as a Teaching Tool *(20 min)*

- Production model: `d_model=256, vocab=2048, 8 heads, 4 layers` → matrices are thousands of numbers
- Micro-model: `d_model=6, vocab=12, 3 heads, 2 layers` → every matrix fits in the terminal
- The comments in `encoder_decoder_model.py` use exactly these values throughout — the code is the lecture material
- Every `6` in the comments means `d_model`, every `12` means `vocab`, every `3` means `num_heads`, every `2` means `d_k`
- Instantiate live:
  ```python
  EncoderDecoderModel(EncoderDecoderModel.Config(
      vocab_size=12, d_model=6, num_heads=3, d_ff=12, num_layers=2
  ))
  ```

---

### STEP 1 — Embeddings *(40 min)*

- Token embedding table: `nn.Embedding(vocab_size=12, d_model=6)` — a learnable `(12×6)` matrix
- `src_ids = [[2, 5, 6, 7]]` for `"const enabled : string"` — print the ids
- Each ID selects one row: `token_emb = model.token_embedding(src_ids)` → shape `(1,4,6)` — print the `(4×6)` slice
- Positional embedding table: `nn.Embedding(max_seq_len=16, d_model=6)` — another learnable `(16×6)` matrix
- `pos_ids = [0,1,2,3]` → `pos_emb` shape `(4,6)` — print it
- `enc_x = token_emb + pos_emb` → shape `(1,4,6)` — elementwise add, broadcast over batch — print it
- Why add rather than concatenate: adding preserves `d_model`, concatenation would double it; both are information-theoretically equivalent with learnable weights
- Both tables are shared between encoder and decoder — source and target use the same vocabulary

---

### STEP 2 — Encoder: Self-Attention in Full Detail *(120 min)*

- **Pre-LN placement** *(10 min)*
  - Original "Post-LN" Transformer: LayerNorm after each sub-layer
  - Pre-LN (used here, as in GPT-2, T5, BART): `x_norm1 = layer_norm_1(x)`, then compute the sub-layer on `x_norm1`
  - Advantage: gradients flow more stably to early layers; the residual stream is always un-normalised, preserving magnitude information

- **Q, K, V projections** *(20 min)*
  - Three independent `nn.Linear(d_model, d_model)` layers: `W_Q`, `W_K`, `W_V`
  - Applied to `x_norm1`: `Q = W_Q(x_norm1)`, `K = W_K(x_norm1)`, `V = W_V(x_norm1)`
  - All three have shape `(1,4,6)` — print Q
  - They are different because `W_Q`, `W_K`, `W_V` have different learned weights
  - Why three projections: Q and K compute similarity; V is what gets mixed; keeping them separate lets the model learn asymmetric "asking" vs "answering"

- **Multi-head split** *(20 min)*
  - `d_model=6, num_heads=3, d_k=2` — each head operates in a 2-dim subspace
  - Reshape: `(1,4,6)` → `(1,4,3,2)` → transpose → `(1,3,4,2)` — print the shape at each step
  - Meaning: batch=1, heads=3, seq=4, d_k=2 — three independent `(4,2)` matrices
  - Each head learns a different "aspect" of attention: one might focus on syntax, another on type names

- **Scaled dot-product attention** *(30 min)*
  - `scores = Q @ Kᵀ / √d_k`
    - `Q`: `(1,3,4,2)`, `Kᵀ`: `(1,3,2,4)` → `scores (1,3,4,4)` — print `scores[0,0]` (head 0, the `(4×4)` matrix)
    - `scores[i,j]` = "how much does position `i` want to attend to position `j`"
    - Scaling by `1/√2`: without it, for large `d_k` the dot products grow large, pushing softmax into saturation where gradients vanish
  - `weights = softmax(scores, dim=-1)` → `(1,3,4,4)` — print `weights[0,0]`; every row sums to 1
  - No mask: `attn_mask=None` — full grid, every source token attends to every other source token
  - `head_out = weights @ V`: `(1,3,4,4) @ (1,3,4,2) = (1,3,4,2)` — print `head_out[0,0]`

- **Concatenation and output projection** *(10 min)*
  - Merge heads: `(1,3,4,2)` → transpose → `(1,4,3,2)` → reshape → `(1,4,6)` — print shape
  - `attn_out = W_O(merged)` — final `nn.Linear(d_model, d_model)` mixes information from all heads
  - Residual: `x1 = x + attn_out` — the original input is added back

- **Feed-Forward Network** *(10 min)*
  - Same two-layer MLP applied independently to each token position: `ffn_linear1 (6→12)` → GELU → `ffn_linear2 (12→6)`
  - `d_ff = 12 = 2 × d_model` in micro-model; production uses `4 × d_model = 1024`
  - `x1_norm = layer_norm_2(x1)` → `ffn_out = ffn_linear2(GELU(ffn_linear1(x1_norm)))` → `output = x1 + ffn_out`
  - Second residual connection; encoder block output has same shape as input: `(1,4,6)`
  - Two encoder blocks chained; `encoder_final_norm` applied after the last block → `encoder_output (1,4,6)`

---

☕ **Break** *(15 min)*

---

### STEP 3 — Decoder: Causal Self-Attention + Cross-Attention *(120 min)*

- **Decoder input** *(10 min)*
  - Teacher-forced input for `"ON | OFF"`: `tgt_input = [[1, 9, 10]]` = `[<bos>, ON, |]`, shape `(1,3)`
  - Same embedding lookup as encoder: `dec_x = token_emb(tgt_input) + pos_emb([0,1,2])` → `(1,3,6)`
  - Target to predict: `tgt_target = [[9, 10, 11]]` = `[ON, |, OFF]`

- **Causal mask** *(20 min)*
  - `_create_causal_mask(seq_len=3)` → `(3,3)` matrix — print it:
    ```
    ┌                    ┐
    │  0.   -inf   -inf  │   ← <bos> may only attend to itself
    │  0.    0.    -inf  │   ← ON  sees <bos> and ON
    │  0.    0.     0.   │   ← |   sees <bos>, ON, |
    └                    ┘
    ```
  - Added to scores before softmax: `-inf` cells become 0 after softmax — effectively zero attention
  - Why required: at inference time when predicting token `t` we must not "see" tokens `t+1, t+2, …`
  - At training time it enforces the same constraint: parallel loss at all positions is only valid if each position's prediction is based solely on its true past

- **Masked self-attention** *(20 min)*
  - Same mechanism as encoder self-attention — same `_multi_head_attention` helper in the code
  - `Q, K, V` from `x_norm1` (decoder hidden state)
  - `scores (1,3,3,3)` — print `scores[0,0]` before mask, then after mask, then after softmax
  - The causal mask turns the upper triangle to zero; each decoder position only mixes its own past
  - `x1 = x + self_attn_out` — first residual

- **Cross-attention** *(40 min)*
  - The bridge between encoder and decoder — the most important sub-layer in the model
  - `Q` from `layer_norm_2(x1)` — the decoder's current state after self-attention
  - `K, V` from `encoder_output` — constant across all decoder steps, never changes during generation
  - `Q`: `(1,3,6)`, `K`: `(1,4,6)`, `V`: `(1,4,6)`
  - After head split: `Q (1,3,3,2)`, `K (1,3,4,2)`, `V (1,3,4,2)`
  - `scores = Q @ Kᵀ`: `(1,3,3,2) @ (1,3,2,4)` = `(1,3,3,4)` — NOT square: 3 decoder × 4 source positions — print `scores[0,0]`
  - `weights (1,3,3,4)` — print; each decoder token's row is "how much does this decoder position attend to each of the 4 source tokens?"
  - No mask in cross-attention: the decoder may freely read any source position
  - Before training: weights are nearly uniform (random initialisation) — print them
  - After training: when generating `"ON"`, the `<bos>→ON` row peaks on `"enabled"` and `"string"` — show this
  - `x2 = x1 + cross_attn_out` — second residual

- **FFN and final norm** *(10 min)*
  - Same MLP as encoder FFN, applied per decoder token position
  - `output = x2 + ffn_out` — third residual
  - Two decoder blocks chained; `decoder_final_norm` applied after — `(1,3,6)`

---

### STEP 4 — LM Head and Logits *(40 min)*

- `lm_head = nn.Linear(d_model, vocab_size, bias=False)` = `(6→12)`
- `x = decoder_final_norm(dec_x_cur)` → `logits = lm_head(x)` → `(1,3,12)` — print all three rows
- Each row: 12 scores, one per vocabulary token
- Point to the column for `"ON"` (id=9), `"|"` (id=10), `"OFF"` (id=11) in each row — before training these are essentially random
- `softmax(logits[0,0])` → probabilities for position 0 — print; each adds to 1.0
- `argmax` → greedy prediction; `multinomial` with temperature → stochastic sampling
- Before training: `P(target) ≈ 1/12 = 8.3%` per position

---

### `EncoderDecoderConfig` and Checkpoint Format *(30 min)*

- Six numbers fully determine all tensor shapes: `vocab_size`, `max_seq_len`, `d_model`, `num_heads`, `d_ff`, `num_layers`
- `state_dict` alone is not sufficient to rebuild the model: `num_heads`, `num_layers`, `max_seq_len` are not recoverable from weight shapes — must be stored explicitly
- `checkpoint.py`: the single module that touches `torch.save` / `torch.load` — centralised so format migrations happen in one place
- `save`: writes `model_state_dict` + `model_config` + arbitrary extras (`epoch`, `val_accuracy`, `prompt_version`, …)
- `load` → `LoadedCheckpoint.extras`: everything beyond state and config, including `prompt_version` checked by `infer.py`
- `build_model(model_config)`: `EncoderDecoderConfig(**model_config)` — extra keys raise an error so typo'd hyper-parameters never silently disappear
- `max_seq_len` in production: `max(observed src_len, observed tgt_len) + 4` — sized exactly to fit the data

---

## Day 2 — Loss, Backward, Training, Evaluation

### STEP 5 — Cross-Entropy Loss *(45 min)*

- **Formula** *(15 min)*
  - For each target position `t`: `loss_t = -log P(target_t | context_t)`
  - P is read from `softmax(logits[0, t])` at the index of the correct token
  - Mean over all `T` positions: `loss = (loss_0 + loss_1 + ... + loss_{T-1}) / T`
  - Random baseline: if all vocab tokens are equally likely, `P = 1/vocab_size`, `loss = -log(1/12) = log(12) ≈ 2.48`

- **Implementation in `trainer.py`** *(15 min)*
  - `loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)` — PAD tokens are excluded from loss and accuracy
  - Reshape trick: `logits (batch, T, vocab) → (batch*T, vocab)`, `tgt_target (batch, T) → (batch*T,)`
  - PyTorch's `CrossEntropyLoss` takes raw logits, applies softmax internally — numerically more stable than manual `log(softmax(...))`

- **Teacher-forced accuracy** *(15 min)*
  - `preds = logits.argmax(dim=-1)` — greedy prediction from teacher-forced logits
  - `mask = tgt_target != pad_id` — ignore padding
  - `train_tf_acc = ((preds == tgt_target) & mask).sum() / mask.sum()`
  - This is NOT the same as autoregressive accuracy: at training time the model always sees the correct previous token; at inference it sees its own (possibly wrong) previous token
  - High teacher-forced accuracy does not guarantee high autoregressive accuracy — measured separately in `evaluate_exact_match`

---

### STEP 6 — Backward Pass: How the Encoder Learns Without Its Own Loss *(90 min)*

- **PyTorch autograd** *(20 min)*
  - Every tensor operation records itself in a computation graph
  - `loss.backward()` traverses the graph in reverse, computing `∂loss/∂param` for every parameter
  - `requires_grad=True` by default for `nn.Module` parameters
  - `model.zero_grad()` before every batch: gradients accumulate by default; zeroing ensures each batch starts clean

- **The gradient flow path** *(30 min)*
  - Trace the path from `loss` back to the encoder — code open on screen:
    ```
    loss
     └─► lm_head.weight
          └─► decoder_final_norm
               └─► decoder_blocks[1]
                    ├─► ffn weights
                    ├─► cross_W_O, cross_W_Q
                    └─► cross_W_K, cross_W_V  ← multiply encoder_output
                         └─► encoder_output   ← gradient flows INTO the encoder
                              └─► encoder_final_norm
                                   └─► encoder_blocks[1] → encoder_blocks[0]
                                        └─► token_embedding.weight
                                            positional_embedding.weight
    ```
  - The encoder has no loss term of its own — it trains because `cross_W_K` and `cross_W_V` multiply `encoder_output`, so `∂loss/∂encoder_output` is non-zero
  - One `backward()` call trains both encoder and decoder simultaneously

- **Live demonstration** *(20 min)*
  - `model.zero_grad()` → encode → decode → loss → `loss.backward()`
  - Print gradient norms for six groups: shared embeddings / encoder block 0 / decoder self-attention / decoder cross-attention / decoder FFN / LM head
  - Observe: encoder block 0 has non-zero gradient norms — the encoder trains through decoder loss

- **Gradient vanishing and exploding** *(20 min)*
  - Vanishing: gradients shrink propagating through many layers — Pre-LN and residual connections mitigate this
  - Exploding: gradients grow uncontrollably early in training — gradient clipping prevents it
  - `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`: rescales all gradients proportionally if their norm exceeds 1.0
  - Must happen AFTER `loss.backward()` and BEFORE `optimizer.step()`

---

☕ **Break** *(15 min)*

---

### STEP 7 — Training Loop *(90 min)*

- **`trainer.py` design: pure function, no domain knowledge** *(15 min)*
  - A function, not a class: no temptation to store dataset paths, tokenizer, checkpoint logic as state
  - The trainer owns exactly: forward → cross-entropy → backward → grad-clip → step
  - Everything else (dataset, eval semantics, logging, saving) is the caller's responsibility via callbacks
  - `train_batches` MUST be a callable (factory), not a generator: a one-shot generator would silently train only epoch 1

- **`TrainConfig` dataclass** *(10 min)*
  - `epochs=50`, `lr=3e-4`, `weight_decay=0.01`, `max_grad_norm=1.0`
  - `eval_every=10`: run `eval_fn` every 10 epochs and on the final epoch
  - `log_every_batches=25`: heartbeat within an epoch — on slow GPUs an epoch can take minutes; without this the user has no idea if the run is alive

- **AdamW** *(20 min)*
  - Adam = Adaptive Moment estimation: per-parameter running mean (momentum) and running variance of gradients
  - Update: `w -= lr * (first_moment / √second_moment)`
  - Effect: consistently large-gradient parameters get smaller effective lr (don't overshoot); sparse or small-gradient parameters get larger effective lr
  - Weight decay: `w -= weight_decay * w` at each step — L2 regularisation
  - Standard configuration: `lr=3e-4, weight_decay=0.01, batch_size=96`

- **Rule-balanced sampler** *(15 min)*
  - Problem: `string_literal_union` has thousands of examples; `dom_shadow_root_init` has tens — dominant rules monopolise every batch
  - Solution (`iter_balanced_batches`): weight per sample = `1/√count(rule)` — tail rules amplified, dominant rules suppressed
  - Cumulative weights + binary search for O(log n) sampling
  - `--balance-rules` flag in `train.py`; default is uniform sampling

- **200-step single-example demo** *(20 min)*
  - Run the micro-model training loop — watch `loss` and `predictions` every 20 steps
  - Step 1: predictions are random, `loss ≈ log(12) ≈ 2.48`, accuracy 0%
  - Step ~60: loss drops below 1.0, accuracy begins rising
  - Step 200: loss ≈ 0, accuracy = 100%, model memorised the single example
  - Why this matters: verifies the complete stack (encoder + decoder + cross-attention + LM head + cross-entropy + AdamW) can learn any mapping before scaling up

- **`eval_fn` and `on_epoch_end` callbacks** *(10 min)*
  - `eval_fn(model) → float`: caller defines what "good" means; trainer records it in `EpochStats.val_metric`
  - `on_epoch_end(EpochStats)`: triggered after every epoch; used in `train.py` to save best checkpoint
  - `EpochStats`: `epoch`, `train_loss`, `train_tf_acc`, `val_metric`, `elapsed_s`

---

☕ **Break** *(15 min)*

---

### STEP 8 — Generation After Training *(45 min)*

- **KV-cache walkthrough** *(20 min)*
  - `precompute_cross_kv(encoder_output)`: projects to `(K, V)` in multi-head format `(1,3,4,2)` — done once before generation loop
  - `forward_cached(x, cross_kv, self_kv)`:
    - Processes ONE new token `(1,1,d_model)` at a time
    - Self-attention: computes `new_K, new_V` for the single token, concatenates to `self_kv` along `dim=2`
    - Cross-attention: uses precomputed `cross_kv` directly — no recomputation
    - Returns `(output, updated_self_kv)` — cache grows by one step

- **Before vs after training** *(15 min)*
  - Before training: cross-attention weights `(3×4)` are nearly uniform
  - After 200 steps: weights for generating `"ON"` concentrate on `"enabled"` and `"string"` columns
  - Print both matrices side by side — the model learned to look at the relevant source tokens

- **Autoregressive decode** *(10 min)*
  - Step-by-step: `[<bos>]` → `ON` → `|` → `OFF`
  - `generated[:, 1:]` — strip the leading `<bos>` from output
  - Confirm: `✓ CORRECT — model recovered the literal union type`

---

### Evaluation: Measuring What Matters *(45 min)*

- **Three metrics in `evaluate_exact_match`** *(20 min)*
  - `exact_match (em)`: predicted string == target string (whitespace-stripped) — works for both positives and negatives
  - `validator_pass (vp)`: `VALIDATORS[rule](prediction)` returns True — meaningful only for positive pairs
  - `acceptable (acc)`: negative-aware composite
    - Positive pair: `VALIDATORS[rule](prediction)` passes
    - Negative pair: `prediction == degraded` (correctly preserved)
  - `acceptable` is the primary optimisation target

- **Macro vs micro accuracy** *(15 min)*
  - Micro: `correct / total` — dominated by the most frequent rules
  - Macro: average of per-rule `acceptable` rate — every rule contributes equally
  - `macro_acc` is what the trainer tracks and the checkpoint stores as `val_accuracy`
  - `print_rule_breakdown`: sorted by count, shows `em / vp / acc` per rule — identifies which rules collapse first

- **Reading the per-rule breakdown** *(10 min)*
  - High `em` but low `vp`: exact matches on training examples but wrong shapes on novel ones — signs of overfitting
  - Low `em` but decent `vp`: semantically correct but minor wording differences
  - `n_neg` column: how many preserve-type (negative) examples are in the eval slice

---

### From Micro-Model to Production *(30 min)*

- **Scaling up** *(15 min)*
  - `d_model`: 6 → 256 — representations are 43× richer
  - `vocab_size`: 12 → 2048 — full TypeScript subword vocabulary
  - `num_heads`: 3 → 8 — more parallel attention subspaces
  - `num_layers`: 2 → 4 — deeper composition of attention + FFN
  - `d_ff`: 12 → 1024 — standard `4 × d_model` ratio
  - `max_seq_len`: 16 → derived from `max(observed src_len, observed tgt_len) + 4`
  - Parameters: micro ≈ 4,500; production ≈ 4,200,000 (~4M) — still small compared to GPT-2 (117M)

- **What changes and what stays the same** *(10 min)*
  - Architecture: identical code path, just different numbers in `EncoderDecoderConfig`
  - Training: same `trainer.py`, same `train()` function — domain knowledge is in `train.py`, not in `trainer.py`
  - The six config numbers are the only difference between micro and production checkpoint

- **What limits accuracy** *(5 min)*
  - Data quality and quantity: copy-pasted props, unusual patterns, missing siblings context
  - Muted rules: too few training examples to enable reliably
  - Train/inference mismatch: teacher forcing vs autoregressive — the model never saw its own prediction errors during training

---

### Q&A *(30 min)*
